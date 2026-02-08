#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.io import detect_delimiter


def normalize_l1_type(value: str | None) -> str | None:
    if value is None:
        return None
    t = str(value).strip().lower().replace("-", " ").replace("_", " ")
    t = " ".join(t.split())
    if not t:
        return None
    if t in {"finding", "clinical finding", "disorder"}:
        return "finding"
    if t in {"procedure", "regime/therapy"}:
        return "procedure"
    if t in {"body structure", "body", "morphologic abnormality", "cell structure"}:
        return "body_structure"
    return None


def _load_notes_map(path: Path, note_id_col: str, text_col: str) -> dict[str, str]:
    delim = detect_delimiter(path)
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter=delim)
        for row in reader:
            note_id = str(row.get(note_id_col) or "").strip()
            if note_id:
                out[note_id] = str(row.get(text_col) or "")
    return out


def _load_concept_l1_map(path: Path | None) -> dict[str, str]:
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
    out: dict[str, str] = {}
    for cid, l1 in df[["concept_id", "l1_type"]].itertuples(index=False, name=None):
        key = str(cid).strip()
        typ = normalize_l1_type(str(l1))
        if key and typ:
            out[key] = typ
    return out


def _nearest_section_header(text: str, *, pos: int, lookback_chars: int) -> str:
    if pos <= 0 or lookback_chars <= 0:
        return ""
    lo = max(0, int(pos) - int(lookback_chars))
    segment = text[lo:pos]
    if not segment:
        return ""
    lines = [ln.strip() for ln in segment.splitlines()]
    for line in reversed(lines):
        if not line:
            continue
        if len(line) > 120:
            continue
        if line.endswith(":"):
            return line
        alpha = [ch for ch in line if ch.isalpha()]
        if alpha and all(ch.isupper() for ch in alpha):
            return line
    return ""


def _canonical_section(header: str) -> str:
    h = str(header or "").strip().lower().rstrip(":")
    if not h:
        return "unknown"
    if "history of present illness" in h or h == "hpi":
        return "hpi"
    if "past medical history" in h or h == "pmh":
        return "pmh"
    if "discharge medication" in h or "medications" in h:
        return "medications"
    if "hospital course" in h:
        return "hospital_course"
    if "physical exam" in h:
        return "physical_exam"
    if ("assessment" in h and "plan" in h) or h == "a/p":
        return "assessment_plan"
    if "diagnosis" in h:
        return "diagnosis"
    if "procedure" in h:
        return "procedures"
    if "allerg" in h:
        return "allergies"
    if "social history" in h:
        return "social_history"
    if "family history" in h:
        return "family_history"
    if "review of systems" in h or h == "ros":
        return "review_of_systems"
    if "lab" in h:
        return "labs"
    if "imaging" in h or "radiology" in h:
        return "imaging"
    return "other"


def _length_bin(n_chars: int) -> str:
    if n_chars <= 5:
        return "01_1_5"
    if n_chars <= 10:
        return "02_6_10"
    if n_chars <= 20:
        return "03_11_20"
    if n_chars <= 40:
        return "04_21_40"
    return "05_41_plus"


def _overlap_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> tuple[int, float]:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return (0, 0.0)
    union = max(a_end, b_end) - min(a_start, b_start)
    iou = (inter / union) if union > 0 else 0.0
    return (inter, iou)


def _boundary_pattern(*, gold_start: int, gold_end: int, pred_start: int, pred_end: int) -> str:
    if pred_start < gold_start:
        left = "left_expand"
    elif pred_start > gold_start:
        left = "left_shrink"
    else:
        left = "left_exact"

    if pred_end > gold_end:
        right = "right_expand"
    elif pred_end < gold_end:
        right = "right_shrink"
    else:
        right = "right_exact"
    return f"{left}|{right}"


def run_analysis(
    *,
    gold_annotations_csv: Path,
    l1_spans_csv: Path,
    notes_csv: Path,
    out_dir: Path,
    allowed_concepts_path: Path | None = None,
    note_id_col: str = "note_id",
    notes_text_col: str = "text",
    section_lookback_chars: int = 4000,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    notes_map = _load_notes_map(notes_csv, note_id_col=note_id_col, text_col=notes_text_col)
    concept_l1_map = _load_concept_l1_map(allowed_concepts_path)

    gold_df = pd.read_csv(gold_annotations_csv)
    gold_df = gold_df[["note_id", "start", "end", "concept_id"]].copy()
    gold_df["note_id"] = gold_df["note_id"].astype(str)
    gold_df["start"] = gold_df["start"].astype(int)
    gold_df["end"] = gold_df["end"].astype(int)
    gold_df = gold_df[gold_df["end"] > gold_df["start"]].copy()
    gold_df["concept_id"] = gold_df["concept_id"].astype(str)
    gold_df["gold_l1_type"] = gold_df["concept_id"].map(concept_l1_map).fillna("")

    pred_df = pd.read_csv(l1_spans_csv)
    pred_df = pred_df.rename(columns={"start_char": "start", "end_char": "end"})
    keep_cols = [c for c in ("note_id", "start", "end", "l1_type", "l1_score") if c in pred_df.columns]
    pred_df = pred_df[keep_cols].copy()
    if "l1_type" not in pred_df.columns:
        pred_df["l1_type"] = ""
    if "l1_score" not in pred_df.columns:
        pred_df["l1_score"] = 0.0
    pred_df["note_id"] = pred_df["note_id"].astype(str)
    pred_df["start"] = pred_df["start"].astype(int)
    pred_df["end"] = pred_df["end"].astype(int)
    pred_df = pred_df[pred_df["end"] > pred_df["start"]].copy()
    pred_df["l1_type"] = pred_df["l1_type"].map(lambda x: normalize_l1_type(str(x)) or "")
    pred_df["l1_score"] = pred_df["l1_score"].astype(float)

    preds_by_note: dict[str, list[dict]] = defaultdict(list)
    for row in pred_df.itertuples(index=False):
        preds_by_note[str(row.note_id)].append(
            {
                "start": int(row.start),
                "end": int(row.end),
                "l1_type": str(row.l1_type or ""),
                "l1_score": float(row.l1_score),
            }
        )
    for note_id in preds_by_note:
        preds_by_note[note_id].sort(key=lambda r: (r["start"], r["end"]))

    rows: list[dict] = []
    boundary_counter: Counter[str] = Counter()

    for rec in gold_df.itertuples(index=False):
        note_id = str(rec.note_id)
        g_start = int(rec.start)
        g_end = int(rec.end)
        g_len = int(g_end - g_start)
        gold_l1 = str(rec.gold_l1_type or "")

        text = notes_map.get(note_id, "")
        header = _nearest_section_header(
            text,
            pos=g_start,
            lookback_chars=max(0, int(section_lookback_chars)),
        )
        section = _canonical_section(header)

        best = None
        best_iou = 0.0
        best_inter = 0
        for p in preds_by_note.get(note_id, []):
            inter, iou = _overlap_iou(g_start, g_end, int(p["start"]), int(p["end"]))
            if inter <= 0:
                continue
            type_match = 1 if (gold_l1 and str(p["l1_type"] or "") == gold_l1) else 0
            rank = (iou, inter, type_match, float(p["l1_score"]))
            if best is None or rank > best:
                best = rank
                best_iou = float(iou)
                best_inter = int(inter)
                best_pred = p
        if best is None:
            has_overlap = False
            has_iou50 = False
            exact = False
            pred_start = ""
            pred_end = ""
            pred_l1 = ""
            type_match = False
            pattern = "none"
            err = "miss"
        else:
            has_overlap = best_inter > 0
            has_iou50 = best_iou >= 0.5
            pred_start = int(best_pred["start"])
            pred_end = int(best_pred["end"])
            pred_l1 = str(best_pred["l1_type"] or "")
            type_match = (not gold_l1) or (pred_l1 == gold_l1)
            exact = (best_iou == 1.0) and type_match
            pattern = _boundary_pattern(
                gold_start=g_start,
                gold_end=g_end,
                pred_start=pred_start,
                pred_end=pred_end,
            )
            if exact:
                err = "exact"
            elif (gold_l1 and pred_l1 and pred_l1 != gold_l1):
                err = "type_mismatch"
            else:
                err = "boundary"
                boundary_counter[pattern] += 1

        rows.append(
            {
                "note_id": note_id,
                "gold_start": g_start,
                "gold_end": g_end,
                "gold_len": g_len,
                "length_bin": _length_bin(g_len),
                "gold_l1_type": gold_l1,
                "section": section,
                "best_iou": float(best_iou),
                "has_overlap": int(has_overlap),
                "has_iou50": int(has_iou50),
                "exact_match": int(exact),
                "best_pred_start": pred_start,
                "best_pred_end": pred_end,
                "best_pred_l1_type": pred_l1,
                "type_match": int(type_match),
                "boundary_pattern": pattern,
                "error_type": err,
            }
        )

    detail_df = pd.DataFrame(rows)
    detail_tsv = out_dir / "per_gold_mention.tsv"
    detail_df.to_csv(detail_tsv, sep="\t", index=False)

    def _agg(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        grp = df.groupby(group_col, dropna=False, as_index=False).agg(
            n=("note_id", "size"),
            recall_overlap=("has_overlap", "mean"),
            recall_iou50=("has_iou50", "mean"),
            exact_match_rate=("exact_match", "mean"),
        )
        miss_counts = (
            df[df["error_type"] == "miss"]
            .groupby(group_col)
            .size()
            .rename("n_miss")
            .reset_index()
        )
        boundary_counts = (
            df[df["error_type"] == "boundary"]
            .groupby(group_col)
            .size()
            .rename("n_boundary")
            .reset_index()
        )
        mismatch_counts = (
            df[df["error_type"] == "type_mismatch"]
            .groupby(group_col)
            .size()
            .rename("n_type_mismatch")
            .reset_index()
        )
        grp = grp.merge(miss_counts, on=group_col, how="left")
        grp = grp.merge(boundary_counts, on=group_col, how="left")
        grp = grp.merge(mismatch_counts, on=group_col, how="left")
        grp = grp.fillna(0)
        for col in ("n", "n_miss", "n_boundary", "n_type_mismatch"):
            grp[col] = grp[col].astype(int)
        return grp.sort_values(group_col).reset_index(drop=True)

    length_df = _agg(detail_df, "length_bin")
    section_df = _agg(detail_df, "section")

    length_tsv = out_dir / "by_length.tsv"
    section_tsv = out_dir / "by_section.tsv"
    length_df.to_csv(length_tsv, sep="\t", index=False)
    section_df.to_csv(section_tsv, sep="\t", index=False)

    boundary_df = pd.DataFrame(
        [
            {"boundary_pattern": k, "n": int(v)}
            for k, v in boundary_counter.most_common()
        ]
    )
    boundary_tsv = out_dir / "boundary_patterns.tsv"
    boundary_df.to_csv(boundary_tsv, sep="\t", index=False)

    total = int(len(detail_df))
    summary = {
        "n_gold_mentions": total,
        "n_pred_mentions": int(len(pred_df)),
        "recall_overlap": float(detail_df["has_overlap"].mean()) if total else 0.0,
        "recall_iou50": float(detail_df["has_iou50"].mean()) if total else 0.0,
        "exact_match_rate": float(detail_df["exact_match"].mean()) if total else 0.0,
        "error_type_counts": {
            str(k): int(v) for k, v in detail_df["error_type"].value_counts().to_dict().items()
        },
        "files": {
            "per_gold_mention_tsv": str(detail_tsv),
            "by_length_tsv": str(length_tsv),
            "by_section_tsv": str(section_tsv),
            "boundary_patterns_tsv": str(boundary_tsv),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Analyze L1 span errors against gold annotations with section/length breakdowns."
    )
    ap.add_argument("--gold-annotations-csv", required=True)
    ap.add_argument("--l1-spans-csv", required=True)
    ap.add_argument("--notes-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allowed-concepts", default="")
    ap.add_argument("--note-id-col", default="note_id")
    ap.add_argument("--notes-text-col", default="text")
    ap.add_argument("--section-lookback-chars", type=int, default=4000)
    args = ap.parse_args(argv)

    allowed = Path(args.allowed_concepts) if str(args.allowed_concepts).strip() else None
    summary = run_analysis(
        gold_annotations_csv=Path(args.gold_annotations_csv),
        l1_spans_csv=Path(args.l1_spans_csv),
        notes_csv=Path(args.notes_csv),
        out_dir=Path(args.out_dir),
        allowed_concepts_path=allowed,
        note_id_col=str(args.note_id_col),
        notes_text_col=str(args.notes_text_col),
        section_lookback_chars=max(0, int(args.section_lookback_chars)),
    )
    print(f"n_gold_mentions: {summary['n_gold_mentions']:,}")
    print(f"n_pred_mentions: {summary['n_pred_mentions']:,}")
    print(f"recall_overlap: {summary['recall_overlap']:.4f}")
    print(f"recall_iou50: {summary['recall_iou50']:.4f}")
    print(f"exact_match_rate: {summary['exact_match_rate']:.4f}")
    print(f"summary_json: {Path(args.out_dir) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
