#!/usr/bin/env python3
"""Detailed error analysis of super dictionary predictions vs gold annotations.

Categorizes errors into:
1. FALSE NEGATIVES (recall failures): gold annotations with no matching prediction
2. FALSE POSITIVES: predictions with no matching gold annotation
3. WRONG INSTANCE: correct concept but matched at wrong location in note
4. BOUNDARY ERRORS: correct concept, correct instance, but span boundaries off
5. CORRECT: perfect or near-perfect matches
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def detect_section_header(text: str, char_pos: int) -> str:
    """Find the section header that contains the given character position."""
    COMMON_HEADERS = [
        "Allergies",
        "History of Present Illness",
        "Family History",
        "Major Surgical or Invasive Procedure",
        "Discharge Disposition",
        "Past Medical History",
        "Service",
        "Discharge Instructions",
        "Discharge Condition",
        "Chief Complaint",
        "Physical Exam",
        "Pertinent Results",
        "Discharge Medications",
        "Social History",
        "Followup Instructions",
        "Medications on Admission",
        "Discharge Diagnosis",
        "Brief Hospital Course",
    ]
    header_positions = []
    for header in COMMON_HEADERS:
        pattern = re.compile(rf"^{re.escape(header)}\s*:", re.MULTILINE | re.IGNORECASE)
        for m in pattern.finditer(text):
            header_positions.append((m.start(), header))

    if not header_positions:
        return "(no section)"

    header_positions.sort(key=lambda x: x[0])
    current_header = "(preamble)"
    for pos, header in header_positions:
        if pos <= char_pos:
            current_header = header
        else:
            break
    return current_header


def get_context(text: str, start: int, end: int, context_chars: int = 120) -> str:
    """Extract context around a span."""
    ctx_start = max(0, start - context_chars)
    ctx_end = min(len(text), end + context_chars)
    before = text[ctx_start:start]
    span = text[start:end]
    after = text[end:ctx_end]
    return f"...{before}<<<{span}>>>{after}..."


def compute_iou(start1, end1, start2, end2):
    """Character-level IoU between two spans."""
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    intersection = max(0, inter_end - inter_start)
    union = (end1 - start1) + (end2 - start2) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def analyze_errors(
    pred_path: str,
    gold_path: str,
    notes_path: str,
    *,
    iou_threshold: float = 0.5,
    wrong_instance_distance: int = 200,
):
    """Run full error analysis."""
    pred_df = pd.read_csv(pred_path)
    gold_df = pd.read_csv(gold_path)
    notes_df = pd.read_csv(notes_path).set_index("note_id")

    for df in [pred_df, gold_df]:
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        df["concept_id"] = df["concept_id"].astype(int)

    note_ids = notes_df.index.tolist()

    false_negatives = []
    false_positives = []
    wrong_instances = []
    boundary_errors = []
    correct_matches = []

    for note_id in note_ids:
        text = notes_df.loc[note_id, "text"]

        note_gold = gold_df[gold_df["note_id"] == note_id].copy()
        note_pred = pred_df[pred_df["note_id"] == note_id].copy()

        gold_matched = set()
        pred_matched = set()

        for g_idx, gold_row in note_gold.iterrows():
            g_start, g_end = int(gold_row["start"]), int(gold_row["end"])
            g_concept = int(gold_row["concept_id"])
            g_span = text[g_start:g_end]
            g_section = detect_section_header(text, g_start)

            same_concept = note_pred[note_pred["concept_id"] == g_concept]

            if len(same_concept) == 0:
                false_negatives.append({
                    "note_id": note_id,
                    "concept_id": g_concept,
                    "gold_start": g_start,
                    "gold_end": g_end,
                    "gold_span": g_span,
                    "section": g_section,
                    "context": get_context(text, g_start, g_end),
                    "error_type": "fn_no_concept",
                })
                continue

            best_iou = 0.0
            best_p_idx = None
            best_p_row = None
            for p_idx, pred_row in same_concept.iterrows():
                p_start, p_end = int(pred_row["start"]), int(pred_row["end"])
                iou = compute_iou(g_start, g_end, p_start, p_end)
                if iou > best_iou:
                    best_iou = iou
                    best_p_idx = p_idx
                    best_p_row = pred_row

            if best_iou >= 0.95:
                gold_matched.add(g_idx)
                pred_matched.add(best_p_idx)
                correct_matches.append({
                    "note_id": note_id,
                    "concept_id": g_concept,
                    "gold_start": g_start,
                    "gold_end": g_end,
                    "gold_span": g_span,
                    "pred_start": int(best_p_row["start"]),
                    "pred_end": int(best_p_row["end"]),
                    "pred_span": text[int(best_p_row["start"]):int(best_p_row["end"])],
                    "iou": best_iou,
                    "section": g_section,
                })
            elif best_iou >= iou_threshold:
                gold_matched.add(g_idx)
                pred_matched.add(best_p_idx)
                p_start = int(best_p_row["start"])
                p_end = int(best_p_row["end"])
                boundary_errors.append({
                    "note_id": note_id,
                    "concept_id": g_concept,
                    "gold_start": g_start,
                    "gold_end": g_end,
                    "gold_span": g_span,
                    "pred_start": p_start,
                    "pred_end": p_end,
                    "pred_span": text[p_start:p_end],
                    "delta_start": p_start - g_start,
                    "delta_end": p_end - g_end,
                    "iou": best_iou,
                    "section": g_section,
                    "context": get_context(text, min(g_start, p_start), max(g_end, p_end)),
                })
            elif best_iou > 0:
                gold_matched.add(g_idx)
                pred_matched.add(best_p_idx)
                p_start = int(best_p_row["start"])
                p_end = int(best_p_row["end"])
                distance = abs(g_start - p_start)
                if distance > wrong_instance_distance:
                    wrong_instances.append({
                        "note_id": note_id,
                        "concept_id": g_concept,
                        "gold_start": g_start,
                        "gold_end": g_end,
                        "gold_span": g_span,
                        "pred_start": p_start,
                        "pred_end": p_end,
                        "pred_span": text[p_start:p_end],
                        "distance": distance,
                        "iou": best_iou,
                        "section": g_section,
                        "gold_context": get_context(text, g_start, g_end),
                        "pred_context": get_context(text, p_start, p_end),
                    })
                else:
                    boundary_errors.append({
                        "note_id": note_id,
                        "concept_id": g_concept,
                        "gold_start": g_start,
                        "gold_end": g_end,
                        "gold_span": g_span,
                        "pred_start": p_start,
                        "pred_end": p_end,
                        "pred_span": text[p_start:p_end],
                        "delta_start": p_start - g_start,
                        "delta_end": p_end - g_end,
                        "iou": best_iou,
                        "section": g_section,
                        "context": get_context(text, min(g_start, p_start), max(g_end, p_end)),
                    })
            else:
                distances = same_concept.apply(
                    lambda r: abs(int(r["start"]) - g_start), axis=1
                )
                nearest_idx = distances.idxmin()
                nearest = same_concept.loc[nearest_idx]
                n_start, n_end = int(nearest["start"]), int(nearest["end"])
                distance = abs(g_start - n_start)

                gold_matched.add(g_idx)
                pred_matched.add(nearest_idx)

                if distance > wrong_instance_distance:
                    wrong_instances.append({
                        "note_id": note_id,
                        "concept_id": g_concept,
                        "gold_start": g_start,
                        "gold_end": g_end,
                        "gold_span": g_span,
                        "pred_start": n_start,
                        "pred_end": n_end,
                        "pred_span": text[n_start:n_end],
                        "distance": distance,
                        "iou": 0.0,
                        "section": g_section,
                        "gold_context": get_context(text, g_start, g_end),
                        "pred_context": get_context(text, n_start, n_end),
                    })
                else:
                    false_negatives.append({
                        "note_id": note_id,
                        "concept_id": g_concept,
                        "gold_start": g_start,
                        "gold_end": g_end,
                        "gold_span": g_span,
                        "section": g_section,
                        "context": get_context(text, g_start, g_end),
                        "error_type": "fn_no_overlap_nearby",
                        "nearest_pred_start": n_start,
                        "nearest_pred_end": n_end,
                        "nearest_pred_span": text[n_start:n_end],
                        "nearest_distance": distance,
                    })

        for p_idx, pred_row in note_pred.iterrows():
            if p_idx in pred_matched:
                continue
            p_start, p_end = int(pred_row["start"]), int(pred_row["end"])
            p_concept = int(pred_row["concept_id"])
            p_span = text[p_start:p_end]
            p_section = detect_section_header(text, p_start)

            concept_in_gold = len(note_gold[note_gold["concept_id"] == p_concept]) > 0

            false_positives.append({
                "note_id": note_id,
                "concept_id": p_concept,
                "pred_start": p_start,
                "pred_end": p_end,
                "pred_span": p_span,
                "section": p_section,
                "context": get_context(text, p_start, p_end),
                "concept_in_gold": concept_in_gold,
            })

    return {
        "false_negatives": pd.DataFrame(false_negatives) if false_negatives else pd.DataFrame(),
        "false_positives": pd.DataFrame(false_positives) if false_positives else pd.DataFrame(),
        "wrong_instances": pd.DataFrame(wrong_instances) if wrong_instances else pd.DataFrame(),
        "boundary_errors": pd.DataFrame(boundary_errors) if boundary_errors else pd.DataFrame(),
        "correct_matches": pd.DataFrame(correct_matches) if correct_matches else pd.DataFrame(),
    }


def print_summary(results: dict):
    """Print summary statistics."""
    fn = results["false_negatives"]
    fp = results["false_positives"]
    wi = results["wrong_instances"]
    be = results["boundary_errors"]
    ok = results["correct_matches"]

    total_gold = len(fn) + len(wi) + len(be) + len(ok)
    total_pred = len(fp) + len(wi) + len(be) + len(ok)

    print("=" * 80)
    print("SUPER DICTIONARY ERROR ANALYSIS")
    print("=" * 80)
    print(f"\nTotal gold annotations:  {total_gold}")
    print(f"Total predictions:       {total_pred}")
    print()
    print(f"{'Category':<30} {'Count':>8} {'% of gold':>10} {'% of pred':>10}")
    print("-" * 60)
    print(f"{'Correct (IoU >= 0.95)':<30} {len(ok):>8} {100*len(ok)/total_gold:>9.1f}% {100*len(ok)/total_pred:>9.1f}%")
    print(f"{'Boundary errors':<30} {len(be):>8} {100*len(be)/total_gold:>9.1f}%")
    print(f"{'Wrong instance':<30} {len(wi):>8} {100*len(wi)/total_gold:>9.1f}%")
    print(f"{'False negatives (missed)':<30} {len(fn):>8} {100*len(fn)/total_gold:>9.1f}%")
    print(f"{'False positives (spurious)':<30} {len(fp):>8} {'':>10} {100*len(fp)/total_pred:>9.1f}%")

    if not fn.empty and "error_type" in fn.columns:
        print(f"\n--- False Negative Breakdown ---")
        for etype, group in fn.groupby("error_type"):
            print(f"  {etype}: {len(group)}")

    if not be.empty:
        print(f"\n--- Boundary Error Statistics ---")
        print(f"  Mean IoU:          {be['iou'].mean():.3f}")
        print(f"  Median IoU:        {be['iou'].median():.3f}")
        print(f"  Mean |delta_start|: {be['delta_start'].abs().mean():.1f} chars")
        print(f"  Mean |delta_end|:   {be['delta_end'].abs().mean():.1f} chars")

    if not wi.empty:
        print(f"\n--- Wrong Instance Statistics ---")
        print(f"  Mean distance:  {wi['distance'].mean():.0f} chars")
        print(f"  Median distance: {wi['distance'].median():.0f} chars")
        print(f"  Max distance:    {wi['distance'].max():.0f} chars")

    print(f"\n--- Errors by Section ---")
    all_errors = []
    if not fn.empty:
        fn_copy = fn.copy()
        fn_copy["_error_type"] = "false_negative"
        all_errors.append(fn_copy[["section", "_error_type"]])
    if not wi.empty:
        wi_copy = wi.copy()
        wi_copy["_error_type"] = "wrong_instance"
        all_errors.append(wi_copy[["section", "_error_type"]])
    if not be.empty:
        be_copy = be.copy()
        be_copy["_error_type"] = "boundary_error"
        all_errors.append(be_copy[["section", "_error_type"]])

    if all_errors:
        all_err_df = pd.concat(all_errors, ignore_index=True)
        section_counts = all_err_df.groupby("section").size().sort_values(ascending=False)
        for section, count in section_counts.head(15).items():
            print(f"  {section:<40} {count:>5}")


def print_examples(results: dict, n: int = 5):
    """Print concrete examples of each error type."""
    print("\n" + "=" * 80)
    print("CONCRETE EXAMPLES")
    print("=" * 80)

    fn = results["false_negatives"]
    if not fn.empty:
        print(f"\n--- FALSE NEGATIVES (missed by dictionary) ---")
        sample = fn.head(n) if len(fn) <= n else fn.sample(n, random_state=42)
        for _, row in sample.iterrows():
            print(f"\n  Note: {row['note_id']}")
            print(f"  Section: {row['section']}")
            print(f"  Concept: {row['concept_id']}")
            print(f"  Gold span: [{row['gold_start']}-{row['gold_end']}] \"{row['gold_span']}\"")
            print(f"  Context: {row['context']}")
            if "error_type" in row and row["error_type"] == "fn_no_overlap_nearby":
                print(f"  Nearest pred: [{row.get('nearest_pred_start')}-{row.get('nearest_pred_end')}] \"{row.get('nearest_pred_span')}\" (distance={row.get('nearest_distance')})")

    be = results["boundary_errors"]
    if not be.empty:
        print(f"\n--- BOUNDARY ERRORS ---")
        sample = be.head(n) if len(be) <= n else be.sample(n, random_state=42)
        for _, row in sample.iterrows():
            print(f"\n  Note: {row['note_id']}")
            print(f"  Section: {row['section']}")
            print(f"  Concept: {row['concept_id']}")
            print(f"  Gold:     [{row['gold_start']}-{row['gold_end']}] \"{row['gold_span']}\"")
            print(f"  Predicted: [{row['pred_start']}-{row['pred_end']}] \"{row['pred_span']}\"")
            print(f"  Delta: start={row['delta_start']:+d}, end={row['delta_end']:+d}, IoU={row['iou']:.3f}")
            print(f"  Context: {row['context']}")

    wi = results["wrong_instances"]
    if not wi.empty:
        print(f"\n--- WRONG INSTANCE ---")
        sample = wi.head(n) if len(wi) <= n else wi.sample(n, random_state=42)
        for _, row in sample.iterrows():
            print(f"\n  Note: {row['note_id']}")
            print(f"  Section: {row['section']}")
            print(f"  Concept: {row['concept_id']}")
            print(f"  Gold:     [{row['gold_start']}-{row['gold_end']}] \"{row['gold_span']}\"")
            print(f"  Predicted: [{row['pred_start']}-{row['pred_end']}] \"{row['pred_span']}\"")
            print(f"  Distance: {row['distance']} chars")
            print(f"  Gold context:  {row['gold_context']}")
            print(f"  Pred context:  {row['pred_context']}")

    fp = results["false_positives"]
    if not fp.empty:
        print(f"\n--- FALSE POSITIVES ---")
        sample = fp.head(n) if len(fp) <= n else fp.sample(n, random_state=42)
        for _, row in sample.iterrows():
            print(f"\n  Note: {row['note_id']}")
            print(f"  Section: {row['section']}")
            print(f"  Concept: {row['concept_id']}")
            print(f"  Predicted: [{row['pred_start']}-{row['pred_end']}] \"{row['pred_span']}\"")
            print(f"  Concept in gold for this note: {row['concept_in_gold']}")
            print(f"  Context: {row['context']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detailed error analysis of predictions vs gold annotations."
    )
    parser.add_argument("--pred", required=True, help="Path to predictions CSV")
    parser.add_argument("--gold", required=True, help="Path to gold annotations CSV")
    parser.add_argument("--notes", required=True, help="Path to notes CSV (note_id, text)")
    parser.add_argument("--output-dir", default="outputs/error_analysis",
                        help="Directory to write detailed CSV outputs")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for boundary error classification")
    parser.add_argument("--examples", type=int, default=8,
                        help="Number of examples to print per error type")
    args = parser.parse_args(argv)

    print(f"Predictions: {args.pred}")
    print(f"Gold:        {args.gold}")
    print(f"Notes:       {args.notes}")

    results = analyze_errors(
        args.pred, args.gold, args.notes,
        iou_threshold=args.iou_threshold,
    )
    print_summary(results)
    print_examples(results, n=args.examples)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        if not df.empty:
            df.to_csv(out_dir / f"{name}.csv", index=False)
            print(f"\nSaved {name}: {len(df)} rows -> {out_dir / f'{name}.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
