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
                if not mention:
                    continue

                result = retriever.retrieve(mention, l1_type=l1_type)
                route_counter[result.route] += 1
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
                fp_out.write(json.dumps(payload, ensure_ascii=True) + "\n")

                if flat_writer is not None:
                    for rank, cand in enumerate(result.final_candidates):
                        flat_writer.writerow(
                            [
                                mention_id,
                                mention,
                                l1_type or "",
                                result.route,
                                rank,
                                cand.concept_id,
                                cand.l1_type,
                                f"{cand.score:.6f}",
                                cand.method,
                                cand.matched_alias,
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
