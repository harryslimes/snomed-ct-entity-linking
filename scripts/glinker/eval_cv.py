#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker import analyze_l1_errors, resolve_l2_links, run_l1_inference, run_l1_l2_pipeline
from scripts.glinker.io import detect_delimiter
from scripts.super_dictionary.runtime_scoring import class_char_iou, macro_char_iou


def _load_allowed_l1_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists() and path.suffix.lower() == ".parquet":
        fallback = path.with_suffix(".csv")
        if fallback.exists():
            path = fallback
    if not path.exists():
        return {}

    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception:
            return {}
    else:
        df = pd.read_csv(path)
    if "concept_id" not in df.columns or "l1_type" not in df.columns:
        return {}
    return {
        str(cid).strip(): str(l1).strip()
        for cid, l1 in df[["concept_id", "l1_type"]].itertuples(index=False, name=None)
        if str(cid).strip()
    }


def _notes_text_map(notes_csv: Path, note_id_col: str, text_col: str) -> dict[str, str]:
    delim = detect_delimiter(notes_csv)
    out: dict[str, str] = {}
    with notes_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter=delim)
        for row in reader:
            note_id = str(row.get(note_id_col) or "").strip()
            if note_id:
                out[note_id] = str(row.get(text_col) or "")
    return out


def _build_gold_l1_spans(
    *,
    val_annotations_csv: Path,
    val_notes_csv: Path,
    out_spans_csv: Path,
    note_id_col: str,
    notes_text_col: str,
    l1_map: dict[str, str],
) -> int:
    ann = pd.read_csv(val_annotations_csv)
    note_text = _notes_text_map(val_notes_csv, note_id_col=note_id_col, text_col=notes_text_col)

    out_spans_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with out_spans_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "mention_id",
                "note_id",
                "start_char",
                "end_char",
                "mention",
                "l1_type",
            ]
        )
        for i, row in ann.iterrows():
            note_id = str(row.get("note_id") or "").strip()
            if not note_id:
                continue
            try:
                start = int(row.get("start"))
                end = int(row.get("end"))
            except Exception:
                continue
            if end <= start:
                continue

            mention = str(row.get("span") or "").strip()
            if not mention:
                txt = note_text.get(note_id, "")
                if txt:
                    mention = txt[max(0, start) : min(len(txt), end)].strip()
            if not mention:
                continue

            concept_id = str(row.get("concept_id") or "").strip()
            l1_type = l1_map.get(concept_id, "")
            mention_id = f"{note_id}:{start}:{end}:{i}"
            writer.writerow([mention_id, note_id, start, end, mention, l1_type])
            rows += 1
    return rows


def _load_gold_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[["note_id", "start", "end", "concept_id"]].copy()
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    out["concept_id"] = out["concept_id"].astype(int)
    out["note_id"] = out["note_id"].astype(str)
    return out


def _load_pred_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id"])
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id"])
    cols = {"start_char": "start", "end_char": "end"}
    df = df.rename(columns=cols)
    out = df[["note_id", "start", "end", "concept_id"]].copy()
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    out["concept_id"] = out["concept_id"].astype(int)
    out["note_id"] = out["note_id"].astype(str)
    return out


def _note_concept_set_iou(pred: pd.DataFrame, gold: pd.DataFrame) -> float:
    pred_by_note = (
        pred.groupby("note_id")["concept_id"].apply(lambda s: set(s.astype(int).tolist())).to_dict()
        if not pred.empty
        else {}
    )
    gold_by_note = (
        gold.groupby("note_id")["concept_id"].apply(lambda s: set(s.astype(int).tolist())).to_dict()
        if not gold.empty
        else {}
    )
    all_notes = sorted(set(pred_by_note.keys()) | set(gold_by_note.keys()))
    if not all_notes:
        return 0.0
    vals: list[float] = []
    for nid in all_notes:
        p = pred_by_note.get(nid, set())
        g = gold_by_note.get(nid, set())
        union = len(p | g)
        vals.append((len(p & g) / union) if union > 0 else 0.0)
    return float(sum(vals) / len(vals))


def _route_stats(candidates_jsonl: Path) -> dict[str, int]:
    routes = Counter()
    if not candidates_jsonl.exists():
        return {}
    with candidates_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            route = str(rec.get("route") or "")
            if route:
                routes[route] += 1
    return dict(routes)


def _run_fold(
    *,
    fold_dir: Path,
    out_dir: Path,
    l1_source: str,
    l1_model_path: str,
    exact_dict_tsv: Path,
    note_id_col: str,
    notes_text_col: str,
    entity_types: str,
    l1_threshold: float,
    l1_window_chars: int,
    l1_window_overlap_chars: int,
    l1_section_header_lookback_chars: int,
    l1_device: str,
    l1_attn_impl: str,
    l1_autocast_dtype: str,
    l1_boundary_refine: bool,
    l1_boundary_expand_mid_token: bool,
    no_es: bool,
    es_url: str,
    es_index_name: str,
    es_api_key: str,
    es_timeout_s: float,
    top_k_exact: int,
    top_k_fuzzy: int,
    top_k_final: int,
    exact_ambiguity_max_candidates: int,
    exact_decisive_min_score: float,
    exact_decisive_min_margin: float,
    disable_exact_short_circuit: bool,
    no_fallback_on_no_exact: bool,
    no_fallback_on_ambiguous: bool,
    fuzziness: str,
    enable_l3: bool,
    l3_index_npz: Path | None,
    l3_model_path: str,
    l3_backend: str,
    l3_device: str,
    l3_batch_size: int,
    l3_max_length: int,
    l3_load_dtype: str,
    l3_search_backend: str,
    l3_ann_index_dir: str,
    l3_ann_candidate_pool: int,
    l3_ann_ivf_nlist: int,
    l3_ann_ivf_nprobe: int,
    l3_ann_hnsw_m: int,
    l3_ann_hnsw_ef_search: int,
    l3_trigger: str,
    l3_top_k: int,
    l3_max_merge_k: int,
    enable_l4: bool,
    l4_model_path: str,
    l4_backend: str,
    l4_device: str,
    l4_batch_size: int,
    l4_max_length: int,
    l4_load_dtype: str,
    l4_top_n: int,
    l4_max_pool_k: int,
    l4_trigger: str,
    l4_min_candidates: int,
    resolver_allowed_concepts: Path | None,
    resolver_no_fuzzy_top1: bool,
    resolver_require_l1_type_match: bool,
    resolver_min_top1_score_exact: float,
    resolver_min_top1_score_fuzzy: float,
    resolver_min_top1_score_l3: float,
    resolver_min_top1_score_l4: float,
    resolver_min_score_margin: float,
    resolver_max_second_to_first_ratio: float,
    resolver_route_min_top1_score: list[str],
    resolver_route_min_score_margin: list[str],
    resolver_route_max_second_to_first_ratio: list[str],
    resolver_no_trim_non_alnum_edges: bool,
    resolver_no_trim_history_of_prefix: bool,
    l1_map: dict[str, str],
    emit_l1_error_report: bool,
) -> dict:
    val_notes = fold_dir / "val_notes.csv"
    val_ann = fold_dir / "val_annotations.csv"
    if not val_notes.exists() or not val_ann.exists():
        raise FileNotFoundError(f"Missing val notes/annotations under {fold_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    l1_spans_csv = out_dir / "l1_spans.csv"
    candidates_jsonl = out_dir / "candidates.jsonl"
    resolved_csv = out_dir / "resolved.csv"
    decisions_csv = out_dir / "decisions.csv"
    flat_csv = out_dir / "candidates_flat.csv"

    t0 = time.perf_counter()

    if l1_source == "gold":
        _build_gold_l1_spans(
            val_annotations_csv=val_ann,
            val_notes_csv=val_notes,
            out_spans_csv=l1_spans_csv,
            note_id_col=note_id_col,
            notes_text_col=notes_text_col,
            l1_map=l1_map,
        )
    elif l1_source == "model":
        run_l1_argv = [
            "--notes-csv",
            str(val_notes),
            "--model-path",
            l1_model_path,
            "--out-spans-csv",
            str(l1_spans_csv),
            "--note-id-col",
            note_id_col,
            "--text-col",
            notes_text_col,
            "--entity-types",
            entity_types,
            "--threshold",
            str(l1_threshold),
            "--window-chars",
            str(l1_window_chars),
            "--window-overlap-chars",
            str(l1_window_overlap_chars),
            "--section-header-lookback-chars",
            str(l1_section_header_lookback_chars),
            "--device",
            str(l1_device),
            "--attn-impl",
            str(l1_attn_impl),
            "--autocast-dtype",
            str(l1_autocast_dtype),
        ]
        if not bool(l1_boundary_refine):
            run_l1_argv.append("--no-boundary-refine")
        if not bool(l1_boundary_expand_mid_token):
            run_l1_argv.append("--no-boundary-mid-token-expand")
        rc_l1 = run_l1_inference.main(run_l1_argv)
        if rc_l1 != 0:
            raise RuntimeError(f"L1 inference failed for {fold_dir.name}")
    else:
        raise ValueError(f"Unsupported l1_source: {l1_source}")

    l1_error_summary: dict | None = None
    l1_error_summary_path = ""
    if emit_l1_error_report:
        l1_error_dir = out_dir / "l1_error_report"
        l1_error_summary = analyze_l1_errors.run_analysis(
            gold_annotations_csv=val_ann,
            l1_spans_csv=l1_spans_csv,
            notes_csv=val_notes,
            out_dir=l1_error_dir,
            allowed_concepts_path=resolver_allowed_concepts,
            note_id_col=note_id_col,
            notes_text_col=notes_text_col,
            section_lookback_chars=max(0, int(l1_section_header_lookback_chars)),
        )
        l1_error_summary_path = str(l1_error_dir / "summary.json")

    l2_argv = [
        "--l1-spans-csv",
        str(l1_spans_csv),
        "--notes-csv",
        str(val_notes),
        "--note-id-col",
        note_id_col,
        "--notes-text-col",
        notes_text_col,
        "--exact-dict-tsv",
        str(exact_dict_tsv),
        "--out-jsonl",
        str(candidates_jsonl),
        "--out-flat-csv",
        str(flat_csv),
        "--es-url",
        es_url,
        "--es-index-name",
        es_index_name,
        "--es-api-key",
        es_api_key,
        "--es-timeout-s",
        str(es_timeout_s),
        "--top-k-exact",
        str(top_k_exact),
        "--top-k-fuzzy",
        str(top_k_fuzzy),
        "--top-k-final",
        str(top_k_final),
        "--exact-ambiguity-max-candidates",
        str(exact_ambiguity_max_candidates),
        "--exact-decisive-min-score",
        str(exact_decisive_min_score),
        "--exact-decisive-min-margin",
        str(exact_decisive_min_margin),
        "--fuzziness",
        fuzziness,
    ]
    if no_es:
        l2_argv.append("--no-es")
    if disable_exact_short_circuit:
        l2_argv.append("--disable-exact-short-circuit")
    if no_fallback_on_no_exact:
        l2_argv.append("--no-fallback-on-no-exact")
    if no_fallback_on_ambiguous:
        l2_argv.append("--no-fallback-on-ambiguous")
    if enable_l3:
        if l3_index_npz is None:
            raise ValueError("L3 enabled but no l3 index path provided")
        l2_argv += [
            "--enable-l3",
            "--l3-index-npz",
            str(l3_index_npz),
            "--l3-model-path",
            str(l3_model_path),
            "--l3-backend",
            str(l3_backend),
            "--l3-device",
            str(l3_device),
            "--l3-batch-size",
            str(l3_batch_size),
            "--l3-max-length",
            str(l3_max_length),
            "--l3-load-dtype",
            str(l3_load_dtype),
            "--l3-search-backend",
            str(l3_search_backend),
            "--l3-ann-index-dir",
            str(l3_ann_index_dir),
            "--l3-ann-candidate-pool",
            str(l3_ann_candidate_pool),
            "--l3-ann-ivf-nlist",
            str(l3_ann_ivf_nlist),
            "--l3-ann-ivf-nprobe",
            str(l3_ann_ivf_nprobe),
            "--l3-ann-hnsw-m",
            str(l3_ann_hnsw_m),
            "--l3-ann-hnsw-ef-search",
            str(l3_ann_hnsw_ef_search),
            "--l3-trigger",
            str(l3_trigger),
            "--l3-top-k",
            str(l3_top_k),
            "--l3-max-merge-k",
            str(l3_max_merge_k),
        ]
    if enable_l4:
        l2_argv += [
            "--enable-l4",
            "--l4-model-path",
            str(l4_model_path),
            "--l4-backend",
            str(l4_backend),
            "--l4-device",
            str(l4_device),
            "--l4-batch-size",
            str(l4_batch_size),
            "--l4-max-length",
            str(l4_max_length),
            "--l4-load-dtype",
            str(l4_load_dtype),
            "--l4-top-n",
            str(l4_top_n),
            "--l4-max-pool-k",
            str(l4_max_pool_k),
            "--l4-trigger",
            str(l4_trigger),
            "--l4-min-candidates",
            str(l4_min_candidates),
        ]

    rc_l2 = run_l1_l2_pipeline.main(l2_argv)
    if rc_l2 != 0:
        raise RuntimeError(f"L2 pipeline failed for {fold_dir.name}")

    resolver_argv = [
        "--candidates-jsonl",
        str(candidates_jsonl),
        "--out-resolved-csv",
        str(resolved_csv),
        "--out-decisions-csv",
        str(decisions_csv),
        "--min-top1-score-exact",
        str(resolver_min_top1_score_exact),
        "--min-top1-score-fuzzy",
        str(resolver_min_top1_score_fuzzy),
        "--min-top1-score-l3",
        str(resolver_min_top1_score_l3),
        "--min-top1-score-l4",
        str(resolver_min_top1_score_l4),
        "--min-score-margin",
        str(resolver_min_score_margin),
        "--max-second-to-first-ratio",
        str(resolver_max_second_to_first_ratio),
    ]
    if resolver_allowed_concepts is not None:
        resolver_argv += ["--allowed-concepts", str(resolver_allowed_concepts)]
    if resolver_no_fuzzy_top1:
        resolver_argv.append("--no-fuzzy-top1")
    if resolver_require_l1_type_match:
        resolver_argv.append("--require-l1-type-match")
    for token in resolver_route_min_top1_score:
        resolver_argv += ["--route-min-top1-score", str(token)]
    for token in resolver_route_min_score_margin:
        resolver_argv += ["--route-min-score-margin", str(token)]
    for token in resolver_route_max_second_to_first_ratio:
        resolver_argv += ["--route-max-second-to-first-ratio", str(token)]
    if resolver_no_trim_non_alnum_edges:
        resolver_argv.append("--no-trim-non-alnum-edges")
    if resolver_no_trim_history_of_prefix:
        resolver_argv.append("--no-trim-history-of-prefix")

    rc_resolve = resolve_l2_links.main(resolver_argv)
    if rc_resolve != 0:
        raise RuntimeError(f"Resolver failed for {fold_dir.name}")

    runtime_sec = time.perf_counter() - t0

    gold_df = _load_gold_df(val_ann)
    pred_df = _load_pred_df(resolved_csv)
    class_df = class_char_iou(pred_df, gold_df)
    macro = (
        float(class_df.loc[class_df["union"] > 0, "iou"].mean())
        if not class_df.empty and (class_df["union"] > 0).any()
        else 0.0
    )

    route_counts = _route_stats(candidates_jsonl)
    l1_recall_overlap = (
        float(l1_error_summary.get("recall_overlap")) if l1_error_summary is not None else float("nan")
    )
    l1_recall_iou50 = (
        float(l1_error_summary.get("recall_iou50")) if l1_error_summary is not None else float("nan")
    )
    l1_exact_match_rate = (
        float(l1_error_summary.get("exact_match_rate")) if l1_error_summary is not None else float("nan")
    )
    return {
        "fold": fold_dir.name,
        "n_notes": int(gold_df["note_id"].astype(str).nunique()),
        "n_gold_spans": int(len(gold_df)),
        "n_pred_spans": int(len(pred_df)),
        "macro_char_iou": macro,
        "note_concept_iou": _note_concept_set_iou(pred_df, gold_df),
        "runtime_sec": float(runtime_sec),
        "sec_per_note": float(runtime_sec / max(1, int(gold_df["note_id"].astype(str).nunique()))),
        "route_counts": route_counts,
        "l1_recall_overlap": l1_recall_overlap,
        "l1_recall_iou50": l1_recall_iou50,
        "l1_exact_match_rate": l1_exact_match_rate,
        "l1_error_summary_json": l1_error_summary_path,
    }


def _select_folds(folds_dir: Path, pattern: str, fold_limit: int) -> list[Path]:
    folds = sorted([p for p in folds_dir.glob(pattern) if p.is_dir()])
    if fold_limit > 0:
        folds = folds[:fold_limit]
    return folds


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "fold",
                "n_notes",
                "n_gold_spans",
                "n_pred_spans",
                "macro_char_iou",
                "note_concept_iou",
                "runtime_sec",
                "sec_per_note",
                "l1_recall_overlap",
                "l1_recall_iou50",
                "l1_exact_match_rate",
                "l1_error_summary_json",
                "route_counts_json",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["fold"],
                    r["n_notes"],
                    r["n_gold_spans"],
                    r["n_pred_spans"],
                    f"{r['macro_char_iou']:.6f}",
                    f"{r['note_concept_iou']:.6f}",
                    f"{r['runtime_sec']:.6f}",
                    f"{r['sec_per_note']:.6f}",
                    (
                        ""
                        if math.isnan(float(r["l1_recall_overlap"]))
                        else f"{float(r['l1_recall_overlap']):.6f}"
                    ),
                    (
                        ""
                        if math.isnan(float(r["l1_recall_iou50"]))
                        else f"{float(r['l1_recall_iou50']):.6f}"
                    ),
                    (
                        ""
                        if math.isnan(float(r["l1_exact_match_rate"]))
                        else f"{float(r['l1_exact_match_rate']):.6f}"
                    ),
                    str(r.get("l1_error_summary_json") or ""),
                    json.dumps(r["route_counts"], sort_keys=True),
                ]
            )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Cross-validation evaluation harness for GLinker pipeline stages currently implemented "
            "(L1 spans, L2 hybrid retrieval, optional L3/L4 enrichment, resolver)."
        )
    )
    ap.add_argument("--folds-dir", default="data/interim/glinker/folds")
    ap.add_argument("--fold-pattern", default="fold_*")
    ap.add_argument("--fold-limit", type=int, default=0)
    ap.add_argument("--output-dir", default="outputs/glinker/eval_cv")

    ap.add_argument("--l1-source", choices=["gold", "model"], default="gold")
    ap.add_argument("--l1-model-path", default="")
    ap.add_argument("--note-id-col", default="note_id")
    ap.add_argument("--notes-text-col", default="text")
    ap.add_argument("--entity-types", default="finding,procedure,body_structure")
    ap.add_argument("--l1-threshold", type=float, default=0.4)
    ap.add_argument("--l1-window-chars", type=int, default=0)
    ap.add_argument("--l1-window-overlap-chars", type=int, default=256)
    ap.add_argument("--l1-section-header-lookback-chars", type=int, default=0)
    ap.add_argument("--l1-device", default="auto")
    ap.add_argument("--l1-attn-impl", default="auto")
    ap.add_argument("--l1-autocast-dtype", default="auto")
    ap.add_argument("--l1-no-boundary-refine", action="store_true")
    ap.add_argument("--l1-no-boundary-mid-token-expand", action="store_true")

    ap.add_argument("--exact-dict-tsv", required=True)
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
    ap.add_argument("--l3-backend", default="auto")
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
    ap.add_argument("--l3-trigger", default="no_exact")
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
        help="ambiguous|ambiguous_exact|no_exact|ambiguous_or_no_exact|always",
    )
    ap.add_argument("--l4-min-candidates", type=int, default=2)

    ap.add_argument("--allowed-concepts", default="")
    ap.add_argument("--resolver-no-fuzzy-top1", action="store_true")
    ap.add_argument("--resolver-require-l1-type-match", action="store_true")
    ap.add_argument("--resolver-min-top1-score-exact", type=float, default=0.2)
    ap.add_argument("--resolver-min-top1-score-fuzzy", type=float, default=6.0)
    ap.add_argument("--resolver-min-top1-score-l3", type=float, default=0.0)
    ap.add_argument("--resolver-min-top1-score-l4", type=float, default=0.0)
    ap.add_argument("--resolver-min-score-margin", type=float, default=0.0)
    ap.add_argument("--resolver-max-second-to-first-ratio", type=float, default=1.0)
    ap.add_argument(
        "--resolver-route-min-top1-score",
        action="append",
        default=[],
        help="Per-route score threshold override. Format: route=score",
    )
    ap.add_argument(
        "--resolver-route-min-score-margin",
        action="append",
        default=[],
        help="Per-route margin override. Format: route=margin",
    )
    ap.add_argument(
        "--resolver-route-max-second-to-first-ratio",
        action="append",
        default=[],
        help="Per-route 2nd/1st score ratio override. Format: route=ratio",
    )
    ap.add_argument("--resolver-no-trim-non-alnum-edges", action="store_true")
    ap.add_argument("--resolver-no-trim-history-of-prefix", action="store_true")
    ap.add_argument("--emit-l1-error-report", action="store_true")
    args = ap.parse_args(argv)

    folds_dir = Path(args.folds_dir)
    output_dir = Path(args.output_dir)
    exact_dict_tsv = Path(args.exact_dict_tsv)
    allowed_concepts = Path(args.allowed_concepts) if args.allowed_concepts else None

    if args.l1_source == "model" and not args.l1_model_path:
        raise ValueError("--l1-model-path is required when --l1-source=model")
    if args.enable_l3:
        if not args.l3_index_npz:
            raise ValueError("--enable-l3 requires --l3-index-npz")
        if not args.l3_model_path:
            raise ValueError("--enable-l3 requires --l3-model-path")
    if args.enable_l4 and not args.l4_model_path:
        raise ValueError("--enable-l4 requires --l4-model-path")

    folds = _select_folds(folds_dir, args.fold_pattern, int(args.fold_limit))
    if not folds:
        raise FileNotFoundError(f"No folds matched under {folds_dir} with pattern {args.fold_pattern}")

    l1_map = _load_allowed_l1_map(allowed_concepts)

    metrics: list[dict] = []
    for fold_dir in folds:
        print(f"[eval] running {fold_dir.name} ...")
        fold_out = output_dir / fold_dir.name
        m = _run_fold(
            fold_dir=fold_dir,
            out_dir=fold_out,
            l1_source=args.l1_source,
            l1_model_path=str(args.l1_model_path),
            exact_dict_tsv=exact_dict_tsv,
            note_id_col=str(args.note_id_col),
            notes_text_col=str(args.notes_text_col),
            entity_types=str(args.entity_types),
            l1_threshold=float(args.l1_threshold),
            l1_window_chars=max(0, int(args.l1_window_chars)),
            l1_window_overlap_chars=max(0, int(args.l1_window_overlap_chars)),
            l1_section_header_lookback_chars=max(0, int(args.l1_section_header_lookback_chars)),
            l1_device=str(args.l1_device),
            l1_attn_impl=str(args.l1_attn_impl),
            l1_autocast_dtype=str(args.l1_autocast_dtype),
            l1_boundary_refine=not bool(args.l1_no_boundary_refine),
            l1_boundary_expand_mid_token=not bool(args.l1_no_boundary_mid_token_expand),
            no_es=bool(args.no_es),
            es_url=str(args.es_url),
            es_index_name=str(args.es_index_name),
            es_api_key=str(args.es_api_key),
            es_timeout_s=float(args.es_timeout_s),
            top_k_exact=int(args.top_k_exact),
            top_k_fuzzy=int(args.top_k_fuzzy),
            top_k_final=int(args.top_k_final),
            exact_ambiguity_max_candidates=int(args.exact_ambiguity_max_candidates),
            exact_decisive_min_score=float(args.exact_decisive_min_score),
            exact_decisive_min_margin=float(args.exact_decisive_min_margin),
            disable_exact_short_circuit=bool(args.disable_exact_short_circuit),
            no_fallback_on_no_exact=bool(args.no_fallback_on_no_exact),
            no_fallback_on_ambiguous=bool(args.no_fallback_on_ambiguous),
            fuzziness=str(args.fuzziness),
            enable_l3=bool(args.enable_l3),
            l3_index_npz=(Path(args.l3_index_npz) if args.l3_index_npz else None),
            l3_model_path=str(args.l3_model_path),
            l3_backend=str(args.l3_backend),
            l3_device=str(args.l3_device),
            l3_batch_size=int(args.l3_batch_size),
            l3_max_length=int(args.l3_max_length),
            l3_load_dtype=str(args.l3_load_dtype),
            l3_search_backend=str(args.l3_search_backend),
            l3_ann_index_dir=str(args.l3_ann_index_dir),
            l3_ann_candidate_pool=int(args.l3_ann_candidate_pool),
            l3_ann_ivf_nlist=int(args.l3_ann_ivf_nlist),
            l3_ann_ivf_nprobe=int(args.l3_ann_ivf_nprobe),
            l3_ann_hnsw_m=int(args.l3_ann_hnsw_m),
            l3_ann_hnsw_ef_search=int(args.l3_ann_hnsw_ef_search),
            l3_trigger=str(args.l3_trigger),
            l3_top_k=int(args.l3_top_k),
            l3_max_merge_k=int(args.l3_max_merge_k),
            enable_l4=bool(args.enable_l4),
            l4_model_path=str(args.l4_model_path),
            l4_backend=str(args.l4_backend),
            l4_device=str(args.l4_device),
            l4_batch_size=int(args.l4_batch_size),
            l4_max_length=int(args.l4_max_length),
            l4_load_dtype=str(args.l4_load_dtype),
            l4_top_n=int(args.l4_top_n),
            l4_max_pool_k=int(args.l4_max_pool_k),
            l4_trigger=str(args.l4_trigger),
            l4_min_candidates=int(args.l4_min_candidates),
            resolver_allowed_concepts=allowed_concepts,
            resolver_no_fuzzy_top1=bool(args.resolver_no_fuzzy_top1),
            resolver_require_l1_type_match=bool(args.resolver_require_l1_type_match),
            resolver_min_top1_score_exact=float(args.resolver_min_top1_score_exact),
            resolver_min_top1_score_fuzzy=float(args.resolver_min_top1_score_fuzzy),
            resolver_min_top1_score_l3=float(args.resolver_min_top1_score_l3),
            resolver_min_top1_score_l4=float(args.resolver_min_top1_score_l4),
            resolver_min_score_margin=float(args.resolver_min_score_margin),
            resolver_max_second_to_first_ratio=float(args.resolver_max_second_to_first_ratio),
            resolver_route_min_top1_score=list(args.resolver_route_min_top1_score),
            resolver_route_min_score_margin=list(args.resolver_route_min_score_margin),
            resolver_route_max_second_to_first_ratio=list(
                args.resolver_route_max_second_to_first_ratio
            ),
            resolver_no_trim_non_alnum_edges=bool(args.resolver_no_trim_non_alnum_edges),
            resolver_no_trim_history_of_prefix=bool(args.resolver_no_trim_history_of_prefix),
            l1_map=l1_map,
            emit_l1_error_report=bool(args.emit_l1_error_report),
        )
        metrics.append(m)
        print(
            f"[eval] {m['fold']}: macro_char_iou={m['macro_char_iou']:.4f} "
            f"note_concept_iou={m['note_concept_iou']:.4f} sec_per_note={m['sec_per_note']:.3f}"
        )

    macro_vals = [m["macro_char_iou"] for m in metrics]
    note_vals = [m["note_concept_iou"] for m in metrics]
    runtime_vals = [m["runtime_sec"] for m in metrics]
    sec_per_note_vals = [m["sec_per_note"] for m in metrics]
    l1_overlap_vals = [float(m["l1_recall_overlap"]) for m in metrics if not math.isnan(float(m["l1_recall_overlap"]))]
    l1_iou50_vals = [float(m["l1_recall_iou50"]) for m in metrics if not math.isnan(float(m["l1_recall_iou50"]))]
    l1_exact_vals = [float(m["l1_exact_match_rate"]) for m in metrics if not math.isnan(float(m["l1_exact_match_rate"]))]

    summary = {
        "config": {
            "folds_dir": str(folds_dir),
            "fold_pattern": str(args.fold_pattern),
            "fold_limit": int(args.fold_limit),
            "l1_source": str(args.l1_source),
            "l1_device": str(args.l1_device),
            "l1_attn_impl": str(args.l1_attn_impl),
            "l1_autocast_dtype": str(args.l1_autocast_dtype),
            "l1_boundary_refine": not bool(args.l1_no_boundary_refine),
            "l1_boundary_expand_mid_token": not bool(args.l1_no_boundary_mid_token_expand),
            "l1_window_chars": int(args.l1_window_chars),
            "l1_window_overlap_chars": int(args.l1_window_overlap_chars),
            "l1_section_header_lookback_chars": int(args.l1_section_header_lookback_chars),
            "no_es": bool(args.no_es),
            "emit_l1_error_report": bool(args.emit_l1_error_report),
            "enable_l3": bool(args.enable_l3),
            "enable_l4": bool(args.enable_l4),
            "resolver_route_min_top1_score": list(args.resolver_route_min_top1_score),
            "resolver_route_min_score_margin": list(args.resolver_route_min_score_margin),
            "resolver_route_max_second_to_first_ratio": list(
                args.resolver_route_max_second_to_first_ratio
            ),
            "resolver_no_trim_non_alnum_edges": bool(args.resolver_no_trim_non_alnum_edges),
            "resolver_no_trim_history_of_prefix": bool(args.resolver_no_trim_history_of_prefix),
        },
        "n_folds": len(metrics),
        "macro_char_iou_mean": float(statistics.mean(macro_vals)),
        "macro_char_iou_std": float(statistics.pstdev(macro_vals)) if len(macro_vals) > 1 else 0.0,
        "note_concept_iou_mean": float(statistics.mean(note_vals)),
        "note_concept_iou_std": float(statistics.pstdev(note_vals)) if len(note_vals) > 1 else 0.0,
        "runtime_sec_total": float(sum(runtime_vals)),
        "runtime_sec_mean": float(statistics.mean(runtime_vals)),
        "sec_per_note_mean": float(statistics.mean(sec_per_note_vals)),
        "l1_recall_overlap_mean": float(statistics.mean(l1_overlap_vals)) if l1_overlap_vals else None,
        "l1_recall_iou50_mean": float(statistics.mean(l1_iou50_vals)) if l1_iou50_vals else None,
        "l1_exact_match_rate_mean": float(statistics.mean(l1_exact_vals)) if l1_exact_vals else None,
        "folds": metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "fold_metrics.csv"
    summary_json = output_dir / "summary.json"
    _write_metrics_csv(metrics_csv, metrics)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("==== CV SUMMARY ====")
    print(f"folds: {summary['n_folds']}")
    print(f"macro_char_iou_mean: {summary['macro_char_iou_mean']:.4f}")
    print(f"macro_char_iou_std: {summary['macro_char_iou_std']:.4f}")
    print(f"note_concept_iou_mean: {summary['note_concept_iou_mean']:.4f}")
    print(f"runtime_sec_total: {summary['runtime_sec_total']:.2f}")
    print(f"sec_per_note_mean: {summary['sec_per_note_mean']:.4f}")
    if summary["l1_recall_overlap_mean"] is not None:
        print(f"l1_recall_overlap_mean: {summary['l1_recall_overlap_mean']:.4f}")
        print(f"l1_recall_iou50_mean: {summary['l1_recall_iou50_mean']:.4f}")
        print(f"l1_exact_match_rate_mean: {summary['l1_exact_match_rate_mean']:.4f}")
    print(f"fold_metrics_csv: {metrics_csv}")
    print(f"summary_json: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
