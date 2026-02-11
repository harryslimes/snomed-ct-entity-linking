#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.io import detect_delimiter
from scripts.glinker.l2_dictionary import ExactDictionaryMatcher
from scripts.glinker.l2_elasticsearch import ElasticsearchAliasRetriever, ElasticsearchConfig
from scripts.glinker.l2_hybrid import HybridL2Config, HybridL2Retriever
from scripts.glinker.l3_biencoder import L3BiEncoderRetriever
from scripts.glinker.l3_l4_enrich import L3Config, L3L4Config, L4Config, enrich_record
from scripts.glinker.l4_reranker import CrossEncoderReranker, L4RerankConfig


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run hybrid L2 candidate generation over a mentions CSV: exact dictionary "
            "short-circuit + ES fuzzy fallback with ambiguity gating."
        )
    )
    ap.add_argument("--mentions-csv", required=True)
    ap.add_argument("--mention-col", default="mention")
    ap.add_argument("--mention-id-col", default="mention_id")
    ap.add_argument("--l1-type-col", default="l1_type")
    ap.add_argument("--exact-dict-tsv", default="data/interim/glinker/l2_exact_dictionary.tsv")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-flat-csv", default="")
    ap.add_argument("--no-es", action="store_true")
    ap.add_argument("--es-url", default="http://127.0.0.1:9200")
    ap.add_argument("--es-index-name", default="snomed_super_dict_v1")
    ap.add_argument("--es-api-key", default="")
    ap.add_argument("--es-timeout-s", type=float, default=10.0)
    ap.add_argument("--top-k-exact", type=int, default=50)
    ap.add_argument("--top-k-fuzzy", type=int, default=50)
    ap.add_argument("--top-k-final", type=int, default=50)
    ap.add_argument("--exact-ambiguity-max-candidates", type=int, default=1)
    ap.add_argument("--exact-decisive-min-score", type=float, default=0.0)
    ap.add_argument("--exact-decisive-min-margin", type=float, default=0.0)
    ap.add_argument("--disable-exact-short-circuit", action="store_true")
    ap.add_argument("--no-fallback-on-no-exact", action="store_true")
    ap.add_argument("--no-fallback-on-ambiguous", action="store_true")
    ap.add_argument("--fuzziness", default="AUTO")
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

    mentions_path = Path(args.mentions_csv)
    exact_path = Path(args.exact_dict_tsv)
    out_jsonl = Path(args.out_jsonl)
    out_flat = Path(args.out_flat_csv) if args.out_flat_csv else None

    exact = ExactDictionaryMatcher.from_tsv(exact_path)
    es_client = None
    if not args.no_es:
        es_client = ElasticsearchAliasRetriever(
            ElasticsearchConfig(
                base_url=args.es_url,
                index_name=args.es_index_name,
                timeout_s=float(args.es_timeout_s),
                api_key=(args.es_api_key or None),
            )
        )

    cfg = HybridL2Config(
        top_k_exact=int(args.top_k_exact),
        top_k_fuzzy=int(args.top_k_fuzzy),
        top_k_final=int(args.top_k_final),
        exact_short_circuit_unique=not bool(args.disable_exact_short_circuit),
        exact_decisive_min_score=float(args.exact_decisive_min_score),
        exact_decisive_min_margin=float(args.exact_decisive_min_margin),
        exact_ambiguity_max_candidates=int(args.exact_ambiguity_max_candidates),
        fallback_on_no_exact=not bool(args.no_fallback_on_no_exact),
        fallback_on_ambiguous_exact=not bool(args.no_fallback_on_ambiguous),
        fuzziness=str(args.fuzziness),
    )
    retriever = HybridL2Retriever(exact_matcher=exact, fuzzy_retriever=es_client, config=cfg)
    l3_retriever = None
    if args.enable_l3:
        if not args.l3_index_npz or not args.l3_model_path:
            raise ValueError("--enable-l3 requires --l3-index-npz and --l3-model-path")
        l3_retriever = L3BiEncoderRetriever(
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
    l4_reranker = None
    if args.enable_l4:
        if not args.l4_model_path:
            raise ValueError("--enable-l4 requires --l4-model-path")
        l4_reranker = CrossEncoderReranker(
            model_path=str(args.l4_model_path),
            backend=str(args.l4_backend),
            device=str(args.l4_device),
            batch_size=int(args.l4_batch_size),
            max_length=int(args.l4_max_length),
            load_dtype=str(args.l4_load_dtype),
            config=L4RerankConfig(top_n=int(args.l4_top_n)),
        )
    l3_l4_cfg = L3L4Config(
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

    delim = detect_delimiter(mentions_path)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_flat is not None:
        out_flat.parent.mkdir(parents=True, exist_ok=True)

    route_counter: Counter[str] = Counter()
    rows = 0
    with mentions_path.open("r", encoding="utf-8", newline="") as fp_in, out_jsonl.open(
        "w", encoding="utf-8"
    ) as fp_out:
        reader = csv.DictReader(fp_in, delimiter=delim)
        flat_writer = None
        flat_fp = None
        if out_flat is not None:
            flat_fp = out_flat.open("w", encoding="utf-8", newline="")
            flat_writer = csv.writer(flat_fp)
            flat_writer.writerow(
                [
                    "mention_id",
                    "mention",
                    "l1_type",
                    "route",
                    "rank",
                    "concept_id",
                    "candidate_l1_type",
                    "score",
                    "method",
                    "matched_alias",
                ]
            )

        try:
            for row in reader:
                mention = str(row.get(args.mention_col) or "").strip()
                mention_id = str(row.get(args.mention_id_col) or rows)
                l1_type = str(row.get(args.l1_type_col) or "").strip() or None
                note_id = str(row.get("note_id") or "").strip()
                start_char_raw = str(row.get("start_char") or "").strip()
                end_char_raw = str(row.get("end_char") or "").strip()
                if not mention:
                    continue

                result = retriever.retrieve(mention, l1_type=l1_type)
                rows += 1

                payload = {
                    "mention_id": mention_id,
                    "mention": mention,
                    "l1_type": l1_type,
                    "route": result.route,
                    "used_fallback": result.used_fallback,
                    "final_candidates": [
                        {
                            "concept_id": c.concept_id,
                            "l1_type": c.l1_type,
                            "score": c.score,
                            "method": c.method,
                            "matched_alias": c.matched_alias,
                        }
                        for c in result.final_candidates
                    ],
                    "n_exact": len(result.exact_candidates),
                    "n_fuzzy": len(result.fuzzy_candidates),
                }
                if note_id:
                    payload["note_id"] = note_id
                if start_char_raw:
                    try:
                        payload["start_char"] = int(start_char_raw)
                    except Exception:
                        pass
                if end_char_raw:
                    try:
                        payload["end_char"] = int(end_char_raw)
                    except Exception:
                        pass
                payload = enrich_record(
                    payload,
                    cfg=l3_l4_cfg,
                    l3_retriever=l3_retriever,
                    l4_reranker=l4_reranker,
                )
                route_counter[str(payload.get("route") or result.route)] += 1
                fp_out.write(json.dumps(payload, ensure_ascii=True) + "\n")

                if flat_writer is not None:
                    for rank, cand in enumerate(payload.get("final_candidates") or []):
                        flat_writer.writerow(
                            [
                                mention_id,
                                mention,
                                l1_type or "",
                                str(payload.get("route") or result.route),
                                rank,
                                str(cand.get("concept_id") or ""),
                                str(cand.get("l1_type") or ""),
                                f"{float(cand.get('score') or 0.0):.6f}",
                                str(cand.get("method") or ""),
                                str(cand.get("matched_alias") or ""),
                            ]
                        )
        finally:
            if flat_fp is not None:
                flat_fp.close()
            if es_client is not None:
                es_client.close()

    print(f"processed mentions: {rows:,}")
    for route, n in sorted(route_counter.items()):
        print(f"route[{route}]={n}")
    print(f"jsonl: {out_jsonl}")
    if out_flat is not None:
        print(f"flat_csv: {out_flat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
