#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from scripts.glinker.io import normalize_alias


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


class _PairScorer:
    def score_pairs(self, mentions: list[str], aliases: list[str]) -> np.ndarray:
        raise NotImplementedError


class _HashPairScorer(_PairScorer):
    @staticmethod
    def _score_one(m: str, a: str) -> float:
        m_toks = set(normalize_alias(m).split())
        a_toks = set(normalize_alias(a).split())
        if not m_toks and not a_toks:
            return 0.0
        if not m_toks or not a_toks:
            return 0.0
        inter = len(m_toks & a_toks)
        union = len(m_toks | a_toks)
        return float(inter / max(1, union))

    def score_pairs(self, mentions: list[str], aliases: list[str]) -> np.ndarray:
        vals = [self._score_one(m, a) for m, a in zip(mentions, aliases)]
        return np.array(vals, dtype=np.float32)


class _HFPairScorer(_PairScorer):
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
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Transformers + torch are required for HF L4 reranker. "
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

        self._model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)
        self._model.eval()
        self._model.to(self.device)

    def score_pairs(self, mentions: list[str], aliases: list[str]) -> np.ndarray:
        if not mentions:
            return np.zeros((0,), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(mentions), self.batch_size):
            m_batch = mentions[i : i + self.batch_size]
            a_batch = aliases[i : i + self.batch_size]
            tok = self._tokenizer(
                m_batch,
                a_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with self._torch.no_grad():
                logits = self._model(**tok).logits
            if logits.ndim == 2 and logits.shape[1] > 1:
                scores = logits[:, 1]
            else:
                scores = logits.squeeze(-1)
            out.append(scores.detach().cpu().float().numpy())
        return np.concatenate(out, axis=0).astype(np.float32, copy=False)


class _GLiNERPairScorer(_PairScorer):
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
            from gliner import GLiNER  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "GLiNER + torch are required for gliner L4 backend. "
                "Install them or use backend=hash."
            ) from e

        self._torch = torch
        raw_dtype = str(load_dtype).strip().lower()
        if raw_dtype not in {"auto", "", "none", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"}:
            raise ValueError(f"Unsupported load_dtype: {load_dtype}")

        # Keep attention path conservative for ModernBERT + FA2 compatibility.
        attn_impl = "eager" if str(self.device).startswith("cuda") else "sdpa"
        self._model = GLiNER.from_pretrained(
            model_path,
            max_length=self.max_length,
            _attn_implementation=attn_impl,
        )
        if hasattr(self._model, "model"):
            self._model.model.eval()

        self._autocast_dtype = None
        if str(self.device).startswith("cuda"):
            if raw_dtype in {"auto", "", "none", "float16", "fp16"}:
                if hasattr(self._model, "model"):
                    self._model.model.half()
                self._autocast_dtype = torch.float16
            elif raw_dtype in {"bfloat16", "bf16"}:
                if hasattr(self._model, "model"):
                    self._model.model.to(dtype=torch.bfloat16)
                self._autocast_dtype = torch.bfloat16
            elif raw_dtype in {"float32", "fp32"}:
                self._autocast_dtype = None

        self._model.to(self.device)

    def _score_one_mention(self, mention: str, aliases: list[str]) -> np.ndarray:
        if not aliases:
            return np.zeros((0,), dtype=np.float32)

        mention_text = str(mention or "")
        if not mention_text:
            return np.zeros((len(aliases),), dtype=np.float32)

        seen: set[str] = set()
        deduped: list[str] = []
        for raw in aliases:
            a = str(raw or "").strip()
            if not a or a in seen:
                continue
            seen.add(a)
            deduped.append(a)

        if not deduped:
            return np.zeros((len(aliases),), dtype=np.float32)

        input_spans = [[{"start": 0, "end": len(mention_text)}]]
        label_to_score: dict[str, float] = {}
        autocast_ctx = (
            self._torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
            if str(self.device).startswith("cuda") and self._autocast_dtype is not None
            else nullcontext()
        )

        with self._torch.no_grad():
            with autocast_ctx:
                for i in range(0, len(deduped), self.batch_size):
                    chunk = deduped[i : i + self.batch_size]
                    ents = self._model.predict_entities(
                        mention_text,
                        chunk,
                        threshold=0.0,
                        flat_ner=True,
                        multi_label=True,
                        return_class_probs=True,
                        input_spans=input_spans,
                    )
                    for ent in ents:
                        lbl = str(ent.get("label") or "").strip()
                        if not lbl:
                            continue
                        sc = float(ent.get("score") or 0.0)
                        prev = label_to_score.get(lbl)
                        if prev is None or sc > prev:
                            label_to_score[lbl] = sc

        out = np.zeros((len(aliases),), dtype=np.float32)
        for i, raw in enumerate(aliases):
            a = str(raw or "").strip()
            out[i] = float(label_to_score.get(a, 0.0))
        return out

    def score_pairs(self, mentions: list[str], aliases: list[str]) -> np.ndarray:
        if not mentions:
            return np.zeros((0,), dtype=np.float32)
        if len(mentions) != len(aliases):
            raise ValueError("mentions and aliases must be the same length")

        if len(set(mentions)) == 1:
            return self._score_one_mention(mentions[0], aliases).astype(np.float32, copy=False)

        groups: dict[str, list[int]] = defaultdict(list)
        for i, m in enumerate(mentions):
            groups[str(m)].append(i)

        out = np.zeros((len(mentions),), dtype=np.float32)
        for mention, idxs in groups.items():
            sub_aliases = [aliases[i] for i in idxs]
            sub_scores = self._score_one_mention(mention, sub_aliases)
            for j, src_idx in enumerate(idxs):
                out[src_idx] = float(sub_scores[j])
        return out.astype(np.float32, copy=False)


def _make_pair_scorer(
    *,
    backend: str,
    model_path: str,
    device: str,
    batch_size: int,
    max_length: int,
    load_dtype: str,
) -> _PairScorer:
    b = str(backend).strip().lower()
    if b in {"hash", "hashed"}:
        return _HashPairScorer()
    if b in {"hf", "transformers"}:
        return _HFPairScorer(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            load_dtype=load_dtype,
        )
    if b in {"gliner", "gliner_rerank"}:
        return _GLiNERPairScorer(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            load_dtype=load_dtype,
        )
    if b in {"auto"}:
        if "gliner-linker-rerank" in str(model_path).lower():
            return _GLiNERPairScorer(
                model_path=model_path,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                load_dtype=load_dtype,
            )
        try:
            return _HFPairScorer(
                model_path=model_path,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                load_dtype=load_dtype,
            )
        except Exception:
            return _GLiNERPairScorer(
                model_path=model_path,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                load_dtype=load_dtype,
            )
    raise ValueError(f"Unsupported L4 backend: {backend}")


@dataclass(frozen=True)
class L4RerankConfig:
    top_n: int = 1


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_path: str,
        backend: str = "auto",
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 128,
        load_dtype: str = "auto",
        config: L4RerankConfig | None = None,
    ):
        self.config = config or L4RerankConfig()
        self.scorer = _make_pair_scorer(
            backend=backend,
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            load_dtype=load_dtype,
        )

    def rerank(
        self,
        mention: str,
        candidates: list[dict],
        *,
        top_n: int | None = None,
    ) -> list[dict]:
        if not candidates:
            return []
        use_top_n = int(top_n if top_n is not None else self.config.top_n)
        if use_top_n <= 0:
            use_top_n = len(candidates)

        aliases = []
        for c in candidates:
            alias = str(c.get("matched_alias") or "").strip()
            if not alias:
                alias = str(c.get("canonical_name") or c.get("concept_name") or c.get("concept_id") or "")
            aliases.append(alias)
        mentions = [str(mention)] * len(candidates)
        scores = self.scorer.score_pairs(mentions, aliases)

        ranked = []
        for i, c in enumerate(candidates):
            row = dict(c)
            row["base_score"] = float(c.get("score", 0.0))
            row["base_method"] = str(c.get("method") or "")
            row["score"] = float(scores[i])
            row["method"] = "l4_cross_rerank"
            ranked.append(row)
        ranked.sort(key=lambda x: (-float(x.get("score", 0.0)), -float(x.get("base_score", 0.0))))
        return ranked[:use_top_n]
