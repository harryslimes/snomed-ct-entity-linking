#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.l2_dictionary import L2Candidate
from scripts.glinker.l3_biencoder import L3BiEncoderRetriever
from scripts.glinker.l4_reranker import CrossEncoderReranker, L4RerankConfig


@dataclass(frozen=True)
class L3Config:
    enabled: bool = False
    trigger_mode: str = "no_exact"
    top_k: int = 50
    max_merge_k: int = 50


@dataclass(frozen=True)
class L4Config:
    enabled: bool = False
    top_n: int = 1
    max_pool_k: int = 50
    trigger_mode: str = "ambiguous"
    min_candidates: int = 2
    no_exact_high_risk_max_top1: float = -1.0
    no_exact_high_risk_max_margin: float = -1.0


@dataclass(frozen=True)
class L3L4Config:
    l3: L3Config
    l4: L4Config


def _trigger_l3(record: dict, mode: str) -> bool:
    m = str(mode).strip().lower()
    n_exact = int(record.get("n_exact") or 0)
    route = str(record.get("route") or "")
    if m in {"always", "all"}:
        return True
    if m in {"off", "none"}:
        return False
    if m in {"no_exact", "on_no_exact"}:
        return n_exact == 0 or route in {"none", "fuzzy_only"}
    if m in {"ambiguous_or_no_exact", "ambiguous"}:
        if n_exact == 0:
            return True
        return route in {"exact_plus_fuzzy", "exact_ambiguous_no_fallback"}
    raise ValueError(f"Unsupported l3 trigger mode: {mode}")


def _trigger_l4(record: dict, candidates: list[dict], mode: str, *, min_candidates: int) -> bool:
    m = str(mode).strip().lower()
    n_exact = int(record.get("n_exact") or 0)
    route = str(record.get("route") or "")
    ambiguous = len(candidates) >= max(2, int(min_candidates))
    no_exact = (n_exact == 0) or (route in {"none", "fuzzy_only"})
    if m in {"always", "all"}:
        return True
    if m in {"off", "none"}:
        return False
    if m in {"ambiguous", "on_ambiguous"}:
        return ambiguous
    if m in {"ambiguous_exact", "on_ambiguous_exact"}:
        return ambiguous and not no_exact
    if m in {"no_exact_high_risk", "on_no_exact_high_risk"}:
        if not no_exact:
            return False
        return False
    if m in {"no_exact", "on_no_exact"}:
        return no_exact
    if m in {"ambiguous_or_no_exact", "ambiguous_no_exact"}:
        return ambiguous or no_exact
    raise ValueError(f"Unsupported l4 trigger mode: {mode}")


def _top1_top2_scores(candidates: list[dict]) -> tuple[float, float]:
    if not candidates:
        return 0.0, 0.0
    s = sorted((float(c.get("score") or 0.0) for c in candidates), reverse=True)
    top1 = float(s[0]) if s else 0.0
    top2 = float(s[1]) if len(s) > 1 else 0.0
    return top1, top2


def _is_no_exact_high_risk(record: dict, candidates: list[dict], cfg: L4Config) -> bool:
    route = str(record.get("route") or "")
    n_exact = int(record.get("n_exact") or 0)
    no_exact = (n_exact == 0) or (route in {"none", "fuzzy_only", "none+l3"})
    if not no_exact:
        return False

    top1, top2 = _top1_top2_scores(candidates)
    margin = top1 - top2

    max_top1 = float(cfg.no_exact_high_risk_max_top1)
    max_margin = float(cfg.no_exact_high_risk_max_margin)
    by_top1 = max_top1 >= 0.0 and top1 <= max_top1
    by_margin = max_margin >= 0.0 and margin <= max_margin
    return bool(by_top1 or by_margin)


def _cand_dict(c: L2Candidate) -> dict:
    return {
        "concept_id": c.concept_id,
        "l1_type": c.l1_type,
        "score": float(c.score),
        "method": c.method,
        "matched_alias": c.matched_alias,
    }


def _merge_candidates(existing: list[dict], extra: list[dict], *, top_k: int) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for c in existing + extra:
        cid = str(c.get("concept_id") or "").strip()
        if not cid or cid in seen:
            continue
        out.append(c)
        seen.add(cid)
        if int(top_k) > 0 and len(out) >= int(top_k):
            break
    return out


def enrich_record(
    record: dict,
    *,
    cfg: L3L4Config,
    l3_retriever: L3BiEncoderRetriever | None,
    l4_reranker: CrossEncoderReranker | None,
) -> dict:
    out = dict(record)
    mention = str(record.get("mention") or "")
    l1_type = str(record.get("l1_type") or "").strip() or None

    final = list(record.get("final_candidates") or [])
    added_l3 = []
    if cfg.l3.enabled and l3_retriever is not None and _trigger_l3(record, cfg.l3.trigger_mode):
        l3 = l3_retriever.retrieve(mention, top_k=int(cfg.l3.top_k), l1_type=l1_type)
        added_l3 = [_cand_dict(c) for c in l3]
        final = _merge_candidates(final, added_l3, top_k=int(cfg.l3.max_merge_k))
        out["route"] = f"{str(record.get('route') or 'none')}+l3"

    should_l4 = False
    l4_mode = str(cfg.l4.trigger_mode).strip().lower()
    if l4_mode in {"no_exact_high_risk", "on_no_exact_high_risk"}:
        should_l4 = _is_no_exact_high_risk(record, final, cfg.l4)
    else:
        should_l4 = _trigger_l4(record, final, cfg.l4.trigger_mode, min_candidates=int(cfg.l4.min_candidates))

    if cfg.l4.enabled and l4_reranker is not None and final and should_l4:
        pool = final[: int(cfg.l4.max_pool_k)] if int(cfg.l4.max_pool_k) > 0 else final
        final = l4_reranker.rerank(mention, pool, top_n=int(cfg.l4.top_n))
        out["route"] = f"{str(out.get('route') or str(record.get('route') or 'none'))}+l4"

    out["final_candidates"] = final
    out["n_l3"] = int(len(added_l3))
    return out


def enrich_jsonl(
    *,
    in_jsonl: Path,
    out_jsonl: Path,
    cfg: L3L4Config,
    l3_retriever: L3BiEncoderRetriever | None,
    l4_reranker: CrossEncoderReranker | None,
) -> dict[str, int]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    route_counts: Counter[str] = Counter()
    with in_jsonl.open("r", encoding="utf-8") as fp_in, out_jsonl.open("w", encoding="utf-8") as fp_out:
        for line in fp_in:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            upd = enrich_record(rec, cfg=cfg, l3_retriever=l3_retriever, l4_reranker=l4_reranker)
            fp_out.write(json.dumps(upd, ensure_ascii=True) + "\n")
            n += 1
            route_counts[str(upd.get("route") or "")] += 1
    return {"rows": n, **{f"route[{k}]": int(v) for k, v in sorted(route_counts.items())}}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Enrich L2 candidate JSONL with optional L3 bi-encoder retrieval and "
            "optional L4 cross-encoder reranking."
        )
    )
    ap.add_argument("--in-jsonl", required=True)
    ap.add_argument("--out-jsonl", required=True)

    ap.add_argument("--enable-l3", action="store_true")
    ap.add_argument("--l3-index-npz", default="")
    ap.add_argument("--l3-model-path", default="")
    ap.add_argument("--l3-backend", default="auto", help="auto|hf|hash")
    ap.add_argument("--l3-device", default="auto")
    ap.add_argument("--l3-batch-size", type=int, default=128)
    ap.add_argument("--l3-max-length", type=int, default=64)
    ap.add_argument("--l3-load-dtype", default="auto")
    ap.add_argument("--l3-search-backend", default="auto")
    ap.add_argument("--l3-ann-index-dir", default="")
    ap.add_argument("--l3-ann-candidate-pool", type=int, default=256)
    ap.add_argument("--l3-ann-ivf-nlist", type=int, default=4096)
    ap.add_argument("--l3-ann-ivf-nprobe", type=int, default=16)
    ap.add_argument("--l3-ann-hnsw-m", type=int, default=32)
    ap.add_argument("--l3-ann-hnsw-ef-search", type=int, default=64)
    ap.add_argument("--l3-trigger", default="no_exact", help="no_exact|ambiguous_or_no_exact|always")
    ap.add_argument("--l3-top-k", type=int, default=50)
    ap.add_argument("--l3-max-merge-k", type=int, default=50)

    ap.add_argument("--enable-l4", action="store_true")
    ap.add_argument("--l4-model-path", default="")
    ap.add_argument("--l4-backend", default="auto", help="auto|hf|gliner|hash")
    ap.add_argument("--l4-device", default="auto")
    ap.add_argument("--l4-batch-size", type=int, default=64)
    ap.add_argument("--l4-max-length", type=int, default=128)
    ap.add_argument("--l4-load-dtype", default="auto")
    ap.add_argument("--l4-top-n", type=int, default=1)
    ap.add_argument("--l4-max-pool-k", type=int, default=50)
    ap.add_argument(
        "--l4-trigger",
        default="ambiguous",
        help=(
            "ambiguous|ambiguous_exact|no_exact|no_exact_high_risk|"
            "ambiguous_or_no_exact|always"
        ),
    )
    ap.add_argument("--l4-min-candidates", type=int, default=2)
    ap.add_argument("--l4-no-exact-high-risk-max-top1", type=float, default=-1.0)
    ap.add_argument("--l4-no-exact-high-risk-max-margin", type=float, default=-1.0)
    args = ap.parse_args(argv)

    l3 = None
    if args.enable_l3:
        if not args.l3_index_npz or not args.l3_model_path:
            raise ValueError("--enable-l3 requires --l3-index-npz and --l3-model-path")
        l3 = L3BiEncoderRetriever(
            index_npz=Path(args.l3_index_npz),
            model_path=str(args.l3_model_path),
            backend=str(args.l3_backend),
            device=str(args.l3_device),
            batch_size=int(args.l3_batch_size),
            max_length=int(args.l3_max_length),
            load_dtype=str(args.l3_load_dtype),
            search_backend=str(args.l3_search_backend),
            ann_candidate_pool=int(args.l3_ann_candidate_pool),
            ann_ivf_nlist=int(args.l3_ann_ivf_nlist),
            ann_ivf_nprobe=int(args.l3_ann_ivf_nprobe),
            ann_hnsw_m=int(args.l3_ann_hnsw_m),
            ann_hnsw_ef_search=int(args.l3_ann_hnsw_ef_search),
            ann_index_dir=str(args.l3_ann_index_dir),
        )
    l4 = None
    if args.enable_l4:
        if not args.l4_model_path:
            raise ValueError("--enable-l4 requires --l4-model-path")
        l4 = CrossEncoderReranker(
            model_path=str(args.l4_model_path),
            backend=str(args.l4_backend),
            device=str(args.l4_device),
            batch_size=int(args.l4_batch_size),
            max_length=int(args.l4_max_length),
            load_dtype=str(args.l4_load_dtype),
            config=L4RerankConfig(top_n=int(args.l4_top_n)),
        )
    cfg = L3L4Config(
        l3=L3Config(
            enabled=bool(args.enable_l3),
            trigger_mode=str(args.l3_trigger),
            top_k=int(args.l3_top_k),
            max_merge_k=int(args.l3_max_merge_k),
        ),
        l4=L4Config(
            enabled=bool(args.enable_l4),
            top_n=int(args.l4_top_n),
            max_pool_k=int(args.l4_max_pool_k),
            trigger_mode=str(args.l4_trigger),
            min_candidates=int(args.l4_min_candidates),
            no_exact_high_risk_max_top1=float(args.l4_no_exact_high_risk_max_top1),
            no_exact_high_risk_max_margin=float(args.l4_no_exact_high_risk_max_margin),
        ),
    )
    stats = enrich_jsonl(
        in_jsonl=Path(args.in_jsonl),
        out_jsonl=Path(args.out_jsonl),
        cfg=cfg,
        l3_retriever=l3,
        l4_reranker=l4,
    )
    print(f"rows: {stats['rows']:,}")
    for k, v in sorted(stats.items()):
        if k == "rows":
            continue
        print(f"{k}={v}")
    print(f"out_jsonl: {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
