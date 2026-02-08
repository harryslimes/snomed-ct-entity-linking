#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from scripts.glinker.io import has_alnum, normalize_alias
from scripts.glinker.l2_dictionary import L2Candidate


def _resolve_device(device: str) -> str:
    d = str(device).strip().lower()
    if d and d != "auto":
        return device
    try:
        import torch  # type: ignore

        if bool(torch.cuda.is_available()):
            return "cuda"
    except Exception:
        pass
    return "cpu"


class _TextEncoder:
    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class _HashingEncoder(_TextEncoder):
    def __init__(self, *, dim: int = 512):
        self.dim = max(64, int(dim))

    def _embed_one(self, text: str) -> np.ndarray:
        v = np.zeros((self.dim,), dtype=np.float32)
        tokens = text.lower().split()
        if not tokens:
            return v
        for tok in tokens:
            t = f"#{tok}#"
            if len(t) < 3:
                h = hash(t) % self.dim
                v[h] += 1.0
                continue
            for i in range(len(t) - 2):
                tri = t[i : i + 3]
                h = hash(tri) % self.dim
                v[h] += 1.0
        n = float(np.linalg.norm(v))
        if n > 0.0:
            v /= n
        return v

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32, copy=False)


class _HFEncoder(_TextEncoder):
    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        batch_size: int,
        max_length: int,
        load_dtype: str = "auto",
    ):
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(8, int(max_length))
        self.device = _resolve_device(device)

        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Transformers + torch are required for HF L3 encoder. "
                "Install them or use backend=hash."
            ) from e

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_kwargs = {}
        raw_dtype = str(load_dtype).strip().lower()
        if raw_dtype in {"float16", "fp16"}:
            model_kwargs["torch_dtype"] = torch.float16
        elif raw_dtype in {"bfloat16", "bf16"}:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif raw_dtype in {"float32", "fp32"}:
            model_kwargs["torch_dtype"] = torch.float32
        elif raw_dtype in {"auto", "", "none"}:
            pass
        else:
            raise ValueError(f"Unsupported load_dtype: {load_dtype}")

        self._model = AutoModel.from_pretrained(model_path, **model_kwargs)
        self._model.eval()
        self._model.to(self.device)

    def _mean_pool(self, hidden, mask):
        mask_f = mask.unsqueeze(-1).to(dtype=hidden.dtype)
        summed = (hidden * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tok = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with self._torch.no_grad():
                res = self._model(**tok)
            hidden = res.last_hidden_state
            pooled = self._mean_pool(hidden, tok["attention_mask"])
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.detach().cpu().float().numpy())
        return np.vstack(out).astype(np.float32, copy=False)


def _make_encoder(
    *,
    backend: str,
    model_path: str,
    device: str,
    batch_size: int,
    max_length: int,
    load_dtype: str,
) -> _TextEncoder:
    b = str(backend).strip().lower()
    if b in {"hash", "hashed"}:
        return _HashingEncoder()
    if b in {"hf", "transformers", "auto"}:
        return _HFEncoder(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            load_dtype=load_dtype,
        )
    raise ValueError(f"Unsupported L3 backend: {backend}")


@dataclass(frozen=True)
class L3AliasRow:
    concept_id: str
    l1_type: str
    alias: str
    priority: int


def iter_alias_rows(
    super_dict_jsonl: Path,
    *,
    min_alias_len: int = 2,
    max_aliases_per_concept: int = 0,
) -> Iterable[L3AliasRow]:
    with super_dict_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            concept_id = str(obj.get("id") or "").strip()
            l1_type = str(obj.get("l1_type") or obj.get("hierarchy") or "").strip()
            names = obj.get("names") or []
            if not concept_id or not names:
                continue

            seen: set[str] = set()
            kept = 0
            for prio, raw_alias in enumerate(names):
                alias = normalize_alias(str(raw_alias))
                if not alias or len(alias) < int(min_alias_len) or not has_alnum(alias):
                    continue
                if alias in seen:
                    continue
                seen.add(alias)
                yield L3AliasRow(
                    concept_id=concept_id,
                    l1_type=l1_type,
                    alias=alias,
                    priority=int(prio),
                )
                kept += 1
                if int(max_aliases_per_concept) > 0 and kept >= int(max_aliases_per_concept):
                    break


def build_l3_index(
    *,
    super_dict_jsonl: Path,
    out_index_npz: Path,
    backend: str,
    model_path: str,
    device: str = "auto",
    batch_size: int = 128,
    max_length: int = 64,
    min_alias_len: int = 2,
    max_aliases_per_concept: int = 0,
    load_dtype: str = "auto",
) -> dict[str, int | str]:
    rows = list(
        iter_alias_rows(
            super_dict_jsonl,
            min_alias_len=min_alias_len,
            max_aliases_per_concept=max_aliases_per_concept,
        )
    )
    if not rows:
        raise RuntimeError(f"No alias rows produced from {super_dict_jsonl}")

    encoder = _make_encoder(
        backend=backend,
        model_path=model_path,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        load_dtype=load_dtype,
    )
    aliases = [r.alias for r in rows]
    embs = encoder.encode(aliases)
    if embs.ndim != 2 or embs.shape[0] != len(rows):
        raise RuntimeError("Invalid embedding matrix shape while building L3 index")
    embs = embs.astype(np.float32, copy=False)

    concept_ids = np.array([r.concept_id for r in rows], dtype=object)
    l1_types = np.array([r.l1_type for r in rows], dtype=object)
    alias_arr = np.array(aliases, dtype=object)
    priority_arr = np.array([int(r.priority) for r in rows], dtype=np.int32)

    out_index_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_index_npz,
        embeddings=embs,
        concept_ids=concept_ids,
        l1_types=l1_types,
        aliases=alias_arr,
        priorities=priority_arr,
        backend=np.array([str(backend)], dtype=object),
        model_path=np.array([str(model_path)], dtype=object),
    )
    return {
        "rows": int(len(rows)),
        "concepts": int(len(set(str(x) for x in concept_ids.tolist()))),
        "dim": int(embs.shape[1]),
        "backend": str(backend),
    }


def _safe_name(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
    return s or "unknown"


def build_l3_faiss_ann_index(
    *,
    index_npz: Path,
    out_dir: Path,
    mode: str = "faiss_hnsw",
    ivf_nlist: int = 4096,
    ivf_nprobe: int = 16,
    hnsw_m: int = 32,
    hnsw_ef_search: int = 64,
) -> dict[str, int | str]:
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError("faiss is required to build persisted L3 ANN index.") from e

    obj = np.load(index_npz, allow_pickle=True)
    embs = obj["embeddings"].astype(np.float32, copy=False)
    l1_types = [str(x) for x in obj["l1_types"].tolist()]
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = str(mode).strip().lower()
    if mode not in {"faiss_flat", "faiss_hnsw", "faiss_ivf"}:
        raise ValueError(f"Unsupported faiss mode: {mode}")

    def _build_one(vectors: np.ndarray):
        dim = int(vectors.shape[1])
        if mode == "faiss_flat":
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)
            return index
        if mode == "faiss_hnsw":
            index = faiss.IndexHNSWFlat(dim, int(max(8, hnsw_m)), faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = int(max(8, hnsw_ef_search))
            index.add(vectors)
            return index
        quantizer = faiss.IndexFlatIP(dim)
        nlist = min(int(max(8, ivf_nlist)), max(1, int(vectors.shape[0])))
        nlist = min(nlist, max(8, int(np.sqrt(max(1, vectors.shape[0])) * 8)))
        nlist = max(1, min(nlist, int(vectors.shape[0])))
        index = faiss.IndexIVFFlat(quantizer, dim, int(nlist), faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            index.train(vectors[: max(1, min(vectors.shape[0], 200000))])
        index.add(vectors)
        index.nprobe = int(max(1, ivf_nprobe))
        return index

    meta = {
        "mode": mode,
        "ivf_nlist": int(ivf_nlist),
        "ivf_nprobe": int(ivf_nprobe),
        "hnsw_m": int(hnsw_m),
        "hnsw_ef_search": int(hnsw_ef_search),
        "global_index_file": "global.faiss",
        "per_l1": {},
    }

    global_index = _build_one(embs)
    faiss.write_index(global_index, str(out_dir / "global.faiss"))

    l1_unique = sorted(set(l1_types))
    for l1 in l1_unique:
        idx = np.array([i for i, v in enumerate(l1_types) if v == l1], dtype=np.int32)
        if idx.size == 0:
            continue
        sub = embs[idx]
        index = _build_one(sub)
        tag = _safe_name(l1)
        index_file = f"l1_{tag}.faiss"
        idx_file = f"l1_{tag}_orig_idx.npy"
        faiss.write_index(index, str(out_dir / index_file))
        np.save(out_dir / idx_file, idx)
        meta["per_l1"][l1] = {
            "index_file": index_file,
            "orig_idx_file": idx_file,
        }

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "rows": int(embs.shape[0]),
        "dim": int(embs.shape[1]),
        "l1_indexes": int(len(meta["per_l1"])),
        "mode": str(mode),
    }


class L3BiEncoderRetriever:
    def __init__(
        self,
        *,
        index_npz: Path,
        model_path: str,
        backend: str = "auto",
        device: str = "auto",
        batch_size: int = 128,
        max_length: int = 64,
        load_dtype: str = "auto",
        search_backend: str = "auto",
        ann_candidate_pool: int = 256,
        ann_ivf_nlist: int = 4096,
        ann_ivf_nprobe: int = 16,
        ann_hnsw_m: int = 32,
        ann_hnsw_ef_search: int = 64,
        ann_index_dir: str = "",
    ):
        obj = np.load(index_npz, allow_pickle=True)
        self.embeddings = obj["embeddings"].astype(np.float32, copy=False)
        self.concept_ids = [str(x) for x in obj["concept_ids"].tolist()]
        self.l1_types = [str(x) for x in obj["l1_types"].tolist()]
        self.aliases = [str(x) for x in obj["aliases"].tolist()]
        self.priorities = [int(x) for x in obj["priorities"].tolist()]
        self.ann_candidate_pool = max(16, int(ann_candidate_pool))

        backend_in_index = ""
        if "backend" in obj:
            backend_in_index = str(obj["backend"].tolist()[0])
        use_backend = str(backend).strip().lower()
        if use_backend in {"", "auto"} and backend_in_index:
            use_backend = backend_in_index
        if use_backend in {"", "auto"}:
            use_backend = "hf"
        self.encoder = _make_encoder(
            backend=use_backend,
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            load_dtype=load_dtype,
        )

        self._idx_by_l1: dict[str, np.ndarray] = {}
        unique_l1 = sorted(set(self.l1_types))
        for l1 in unique_l1:
            idx = [i for i, v in enumerate(self.l1_types) if v == l1]
            self._idx_by_l1[l1] = np.array(idx, dtype=np.int32)

        self.search_backend = self._resolve_search_backend(str(search_backend))
        self._faiss_global = None
        self._faiss_by_l1: dict[str, tuple[object, np.ndarray]] = {}
        self._faiss_nprobe = max(1, int(ann_ivf_nprobe))
        self._faiss_ef_search = max(8, int(ann_hnsw_ef_search))
        self._faiss_ivf_nlist = max(8, int(ann_ivf_nlist))
        self._faiss_hnsw_m = max(8, int(ann_hnsw_m))
        self._ann_index_dir = Path(ann_index_dir) if str(ann_index_dir).strip() else None
        if self.search_backend.startswith("faiss_"):
            loaded = self._load_faiss_indexes() if self._ann_index_dir is not None else False
            if not loaded:
                self._build_faiss_indexes()

    @staticmethod
    def _faiss_available() -> bool:
        try:
            import faiss  # type: ignore  # noqa: F401
        except Exception:
            return False
        return True

    def _resolve_search_backend(self, backend: str) -> str:
        b = str(backend).strip().lower()
        if b in {"", "auto"}:
            return "faiss_hnsw" if self._faiss_available() else "bruteforce"
        if b in {"brute", "bruteforce", "exact"}:
            return "bruteforce"
        if b in {"faiss_flat", "faiss_hnsw", "faiss_ivf"}:
            if not self._faiss_available():
                raise RuntimeError(
                    f"Requested {b} search backend but faiss is unavailable in this environment."
                )
            return b
        raise ValueError(f"Unsupported L3 search backend: {backend}")

    def _build_faiss_one(self, vecs: np.ndarray):
        import faiss  # type: ignore

        dim = int(vecs.shape[1])
        mode = self.search_backend
        if mode == "faiss_flat":
            index = faiss.IndexFlatIP(dim)
            index.add(vecs)
            self._configure_faiss_index(index)
            return index
        if mode == "faiss_hnsw":
            index = faiss.IndexHNSWFlat(dim, int(self._faiss_hnsw_m), faiss.METRIC_INNER_PRODUCT)
            index.add(vecs)
            self._configure_faiss_index(index)
            return index
        if mode == "faiss_ivf":
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(int(self._faiss_ivf_nlist), max(8, int(np.sqrt(max(1, vecs.shape[0])) * 8)))
            nlist = min(nlist, max(1, int(vecs.shape[0])))
            index = faiss.IndexIVFFlat(quantizer, dim, int(nlist), faiss.METRIC_INNER_PRODUCT)
            if not index.is_trained:
                if vecs.shape[0] < nlist:
                    nlist = max(1, vecs.shape[0])
                index.train(vecs[: max(1, min(vecs.shape[0], 200000))])
            index.add(vecs)
            self._configure_faiss_index(index)
            return index
        raise RuntimeError(f"Unsupported faiss mode: {mode}")

    def _configure_faiss_index(self, index) -> None:
        if hasattr(index, "nprobe"):
            index.nprobe = int(self._faiss_nprobe)
        hnsw = getattr(index, "hnsw", None)
        if hnsw is not None and hasattr(hnsw, "efSearch"):
            hnsw.efSearch = int(self._faiss_ef_search)

    def _build_faiss_indexes(self) -> None:
        self._faiss_global = self._build_faiss_one(self.embeddings)
        for l1, idx in self._idx_by_l1.items():
            if idx.size == 0:
                continue
            self._faiss_by_l1[l1] = (self._build_faiss_one(self.embeddings[idx]), idx)

    def _load_faiss_indexes(self) -> bool:
        if self._ann_index_dir is None:
            return False
        try:
            import faiss  # type: ignore
        except Exception:
            return False
        meta_path = self._ann_index_dir / "metadata.json"
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mode = str(meta.get("mode") or "").strip().lower()
        if mode != self.search_backend:
            return False
        global_file = str(meta.get("global_index_file") or "")
        if not global_file:
            return False
        gpath = self._ann_index_dir / global_file
        if not gpath.exists():
            return False
        self._faiss_global = faiss.read_index(str(gpath))
        self._configure_faiss_index(self._faiss_global)
        self._faiss_by_l1 = {}
        for l1, cfg in dict(meta.get("per_l1") or {}).items():
            idx_file = self._ann_index_dir / str(cfg.get("index_file") or "")
            map_file = self._ann_index_dir / str(cfg.get("orig_idx_file") or "")
            if not idx_file.exists() or not map_file.exists():
                continue
            index = faiss.read_index(str(idx_file))
            self._configure_faiss_index(index)
            orig_idx = np.load(map_file).astype(np.int32, copy=False)
            self._faiss_by_l1[str(l1)] = (index, orig_idx)
        return True

    @staticmethod
    def _search_bruteforce(
        *,
        embeddings: np.ndarray,
        qv: np.ndarray,
        base_idx: np.ndarray | None,
    ) -> list[tuple[int, float]]:
        if base_idx is not None:
            if base_idx.size == 0:
                return []
            scores = embeddings[base_idx] @ qv
            return [(int(base_idx[i]), float(scores[i])) for i in range(len(scores))]
        scores = embeddings @ qv
        return [(i, float(scores[i])) for i in range(len(scores))]

    def _search_faiss(
        self,
        *,
        qv: np.ndarray,
        l1_type: str | None,
        top_k: int,
    ) -> list[tuple[int, float]]:
        if self._faiss_global is None:
            return []
        import numpy as _np

        k = max(int(top_k), int(self.ann_candidate_pool))
        q = _np.ascontiguousarray(qv.reshape(1, -1).astype(_np.float32))
        if l1_type is not None and l1_type in self._faiss_by_l1:
            index, base_idx = self._faiss_by_l1[l1_type]
            scores, ids = index.search(q, int(k))
            out: list[tuple[int, float]] = []
            for lid, sc in zip(ids[0], scores[0]):
                if int(lid) < 0:
                    continue
                out.append((int(base_idx[int(lid)]), float(sc)))
            return out
        scores, ids = self._faiss_global.search(q, int(k))
        out = []
        for gid, sc in zip(ids[0], scores[0]):
            if int(gid) < 0:
                continue
            out.append((int(gid), float(sc)))
        return out

    def _search(
        self,
        *,
        qv: np.ndarray,
        l1_type: str | None,
        top_k: int,
    ) -> list[tuple[int, float]]:
        if self.search_backend.startswith("faiss_"):
            return self._search_faiss(qv=qv, l1_type=l1_type, top_k=top_k)
        base_idx = None
        if l1_type is not None and l1_type in self._idx_by_l1:
            base_idx = self._idx_by_l1[l1_type]
        return self._search_bruteforce(embeddings=self.embeddings, qv=qv, base_idx=base_idx)

    def retrieve(
        self,
        mention: str,
        *,
        top_k: int = 50,
        l1_type: str | None = None,
    ) -> list[L2Candidate]:
        m = normalize_alias(mention)
        if not m:
            return []
        q = self.encoder.encode([m])
        if q.shape[0] != 1:
            return []
        qv = q[0]

        cand_rows = self._search(
            qv=qv,
            l1_type=l1_type,
            top_k=max(int(top_k), self.ann_candidate_pool),
        )
        if not cand_rows:
            return []

        # Keep best alias row per concept_id.
        best_by_concept: dict[str, tuple[int, float, int]] = {}
        for idx, score in cand_rows:
            cid = self.concept_ids[idx]
            prio = int(self.priorities[idx])
            prev = best_by_concept.get(cid)
            cur = (idx, score, prio)
            if prev is None or (score > prev[1]) or (score == prev[1] and prio < prev[2]):
                best_by_concept[cid] = cur
        rows = list(best_by_concept.values())
        rows.sort(key=lambda x: (-x[1], x[2], x[0]))
        if int(top_k) > 0:
            rows = rows[: int(top_k)]

        out: list[L2Candidate] = []
        for idx, score, _prio in rows:
            out.append(
                L2Candidate(
                    concept_id=self.concept_ids[idx],
                    l1_type=self.l1_types[idx],
                    matched_alias=self.aliases[idx],
                    score=float(score),
                    method="l3_biencoder",
                )
            )
        return out
