#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker import resolve_l2_links, run_l1_inference, run_l1_l2_pipeline


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run end-to-end L1->L2 candidate generation: GLiNER span extraction "
            "followed by hybrid dictionary+ES candidate retrieval."
        )
    )
    ap.add_argument("--notes-csv", required=True)
    ap.add_argument("--l1-model-path", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-flat-csv", default="")
    ap.add_argument("--out-resolved-csv", default="")
    ap.add_argument("--out-decisions-csv", default="")
    ap.add_argument("--work-dir", default="outputs/glinker/tmp")
    ap.add_argument("--l1-spans-csv", default="")
    ap.add_argument("--reuse-existing-l1-spans", action="store_true")

    ap.add_argument("--note-id-col", default="note_id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--entity-types", default="finding,procedure,body_structure")
    ap.add_argument("--l1-threshold", type=float, default=0.4)
    ap.add_argument("--l1-window-chars", type=int, default=0)
    ap.add_argument("--l1-window-overlap-chars", type=int, default=256)
    ap.add_argument("--l1-device", default="auto")
    ap.add_argument("--l1-attn-impl", default="auto")
    ap.add_argument("--l1-autocast-dtype", default="auto")
    ap.add_argument("--l1-window-chars", type=int, default=0)
    ap.add_argument("--l1-window-overlap-chars", type=int, default=256)
    ap.add_argument("--l1-limit-notes", type=int, default=0)
    ap.add_argument("--no-strict-l1-label-filter", action="store_true")

    ap.add_argument("--strict-l1-filter", action="store_true")
    ap.add_argument("--exact-dict-tsv", default="data/interim/glinker/l2_exact_dictionary.tsv")
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

    ap.add_argument("--resolver-allowed-concepts", default="")
    ap.add_argument("--resolver-no-fuzzy-top1", action="store_true")
    ap.add_argument("--resolver-require-l1-type-match", action="store_true")
    ap.add_argument("--resolver-min-top1-score-exact", type=float, default=0.2)
    ap.add_argument("--resolver-min-top1-score-fuzzy", type=float, default=6.0)
    ap.add_argument("--resolver-min-score-margin", type=float, default=0.0)
    ap.add_argument("--resolver-max-second-to-first-ratio", type=float, default=1.0)
    args = ap.parse_args(argv)

    out_jsonl = Path(args.out_jsonl)
    out_flat = Path(args.out_flat_csv) if args.out_flat_csv else None
    out_resolved = Path(args.out_resolved_csv) if args.out_resolved_csv else None
    out_decisions = Path(args.out_decisions_csv) if args.out_decisions_csv else None

    if args.l1_spans_csv:
        l1_spans_path = Path(args.l1_spans_csv)
    else:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        l1_spans_path = work_dir / "l1_spans.csv"

    if not (args.reuse_existing_l1_spans and l1_spans_path.exists()):
        l1_argv = [
            "--notes-csv",
            str(args.notes_csv),
            "--model-path",
            str(args.l1_model_path),
            "--out-spans-csv",
            str(l1_spans_path),
            "--note-id-col",
            str(args.note_id_col),
            "--text-col",
            str(args.text_col),
            "--entity-types",
            str(args.entity_types),
            "--threshold",
            str(args.l1_threshold),
            "--window-chars",
            str(max(0, int(args.l1_window_chars))),
            "--window-overlap-chars",
            str(max(0, int(args.l1_window_overlap_chars))),
            "--device",
            str(args.l1_device),
            "--attn-impl",
            str(args.l1_attn_impl),
            "--autocast-dtype",
            str(args.l1_autocast_dtype),
            "--window-chars",
            str(args.l1_window_chars),
            "--window-overlap-chars",
            str(args.l1_window_overlap_chars),
            "--limit-notes",
            str(args.l1_limit_notes),
        ]
        if args.no_strict_l1_label_filter:
            l1_argv.append("--no-strict-label-filter")
        rc_l1 = run_l1_inference.main(l1_argv)
        if rc_l1 != 0:
            return rc_l1
    else:
        print(f"reusing l1 spans: {l1_spans_path}")

    l2_argv = [
        "--l1-spans-csv",
        str(l1_spans_path),
        "--notes-csv",
        str(args.notes_csv),
        "--note-id-col",
        str(args.note_id_col),
        "--notes-text-col",
        str(args.text_col),
        "--exact-dict-tsv",
        str(args.exact_dict_tsv),
        "--out-jsonl",
        str(out_jsonl),
        "--es-url",
        str(args.es_url),
        "--es-index-name",
        str(args.es_index_name),
        "--es-api-key",
        str(args.es_api_key),
        "--es-timeout-s",
        str(args.es_timeout_s),
        "--top-k-exact",
        str(args.top_k_exact),
        "--top-k-fuzzy",
        str(args.top_k_fuzzy),
        "--top-k-final",
        str(args.top_k_final),
        "--exact-ambiguity-max-candidates",
        str(args.exact_ambiguity_max_candidates),
        "--exact-decisive-min-score",
        str(args.exact_decisive_min_score),
        "--exact-decisive-min-margin",
        str(args.exact_decisive_min_margin),
        "--fuzziness",
        str(args.fuzziness),
    ]
    if out_flat is not None:
        l2_argv += ["--out-flat-csv", str(out_flat)]
    if args.strict_l1_filter:
        l2_argv.append("--strict-l1-filter")
    if args.no_es:
        l2_argv.append("--no-es")
    if args.disable_exact_short_circuit:
        l2_argv.append("--disable-exact-short-circuit")
    if args.no_fallback_on_no_exact:
        l2_argv.append("--no-fallback-on-no-exact")
    if args.no_fallback_on_ambiguous:
        l2_argv.append("--no-fallback-on-ambiguous")

    rc_l2 = run_l1_l2_pipeline.main(l2_argv)
    if rc_l2 != 0:
        return rc_l2

    if out_resolved is not None:
        resolver_argv = [
            "--candidates-jsonl",
            str(out_jsonl),
            "--out-resolved-csv",
            str(out_resolved),
            "--min-top1-score-exact",
            str(args.resolver_min_top1_score_exact),
            "--min-top1-score-fuzzy",
            str(args.resolver_min_top1_score_fuzzy),
            "--min-score-margin",
            str(args.resolver_min_score_margin),
            "--max-second-to-first-ratio",
            str(args.resolver_max_second_to_first_ratio),
        ]
        if out_decisions is not None:
            resolver_argv += ["--out-decisions-csv", str(out_decisions)]
        if args.resolver_allowed_concepts:
            resolver_argv += ["--allowed-concepts", str(args.resolver_allowed_concepts)]
        if args.resolver_no_fuzzy_top1:
            resolver_argv.append("--no-fuzzy-top1")
        if args.resolver_require_l1_type_match:
            resolver_argv.append("--require-l1-type-match")

        rc_resolve = resolve_l2_links.main(resolver_argv)
        if rc_resolve != 0:
            return rc_resolve

    print(f"l1_spans_csv: {l1_spans_path}")
    print(f"l1_l2_jsonl: {out_jsonl}")
    if out_flat is not None:
        print(f"l1_l2_flat_csv: {out_flat}")
    if out_resolved is not None:
        print(f"resolved_csv: {out_resolved}")
    if out_decisions is not None:
        print(f"decisions_csv: {out_decisions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
