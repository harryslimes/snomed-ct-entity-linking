#!/usr/bin/env python3
"""Class-agnostic span evaluation: measures whether the model detects
spans regardless of which class label it assigns.

Compares v12 (3-class) and v13 (9-class) on pure span detection.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from gliner2 import GLiNER2

sys.path.insert(0, str(Path(__file__).parent))
from eval_gliner2 import (
    ALL_ENTITY_TYPES,
    predict_note,
)
from prepare_gliner2_data import (
    load_sctid_to_tag, map_concept_to_fine_class,
    find_lab_regions, merge_adjacent_regions, annotation_in_lab_region,
)


def compute_agnostic_metrics(
    gold_spans: list[tuple[int, int]],
    pred_spans: list[tuple[int, int]],
    iou_threshold: float = 0.0,
) -> dict:
    """Span-level P/R/F1 ignoring class labels."""
    if iou_threshold == 0.0:
        gold_set = set(gold_spans)
        pred_set = set(pred_spans)
        tp = len(gold_set & pred_set)
    else:
        matched_gold = set()
        matched_pred = set()
        for i, (gs, ge) in enumerate(gold_spans):
            for j, (ps, pe) in enumerate(pred_spans):
                overlap_start = max(gs, ps)
                overlap_end = min(ge, pe)
                if overlap_start >= overlap_end:
                    continue
                overlap = overlap_end - overlap_start
                union = max(ge, pe) - min(gs, ps)
                iou = overlap / union if union > 0 else 0
                if iou >= iou_threshold and i not in matched_gold and j not in matched_pred:
                    matched_gold.add(i)
                    matched_pred.add(j)
        tp = len(matched_gold)

    precision = tp / len(pred_spans) if pred_spans else 0
    recall = tp / len(gold_spans) if gold_spans else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"tp": tp, "precision": precision, "recall": recall, "f1": f1,
            "n_gold": len(gold_spans), "n_pred": len(pred_spans)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="fastino/gliner2-large-v1")
    parser.add_argument("--data-dir", type=Path, default=Path("data/old-challenge-split"))
    parser.add_argument("--project-root", type=Path, default=Path("/workspaces/snomed-ct-entity-linking"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-tokens", type=int, default=200)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--use-3-class", action="store_true",
                        help="Use original 3-class entity types (for v12 baseline)")
    parser.add_argument("--filter-lab-tables", action="store_true",
                        help="Exclude gold annotations inside structured lab tables")
    args = parser.parse_args()

    # Override entity types for v12 baseline comparison
    if args.use_3_class:
        import eval_gliner2
        eval_gliner2.ALL_ENTITY_TYPES = [
            "medical finding, symptom, or disease",
            "procedure",
            "anatomical body part",
        ]
        eval_gliner2.ENTITY_TYPES = eval_gliner2.ALL_ENTITY_TYPES
        print("Using original 3-class entity types")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_dir}... (device={device})")
    try:
        model = GLiNER2.from_pretrained(args.model_dir).to(device)
    except Exception:
        print(f"Loading as adapter on {args.base_model}...")
        model = GLiNER2.from_pretrained(args.base_model).to(device)
        model.load_adapter(args.model_dir)

    sctid_to_tag = load_sctid_to_tag(args.project_root)

    test_notes = pd.read_csv(args.data_dir / "test_notes.csv")
    test_ann = pd.read_csv(args.data_dir / "test_annotations.csv", dtype={"concept_id": int})
    test_ann["start"] = test_ann["start"].astype(int)
    test_ann["end"] = test_ann["end"].astype(int)

    note_texts = dict(zip(test_notes["note_id"], test_notes["text"]))

    # Lab table filtering on gold annotations
    if args.filter_lab_tables:
        n_before = len(test_ann)
        keep_mask = pd.Series(True, index=test_ann.index)
        for note_id, text in note_texts.items():
            regions = merge_adjacent_regions(find_lab_regions(text))
            if not regions:
                continue
            note_mask = test_ann["note_id"] == note_id
            for idx in test_ann[note_mask].index:
                row = test_ann.loc[idx]
                if annotation_in_lab_region(row["start"], row["end"], regions):
                    keep_mask[idx] = False
        test_ann = test_ann[keep_mask]
        print(f"  Lab table filter: {n_before - len(test_ann):,} gold annotations removed")

    print(f"\nRunning inference on {len(test_notes)} test notes...")
    all_gold_spans_agnostic = []
    all_pred_spans_agnostic = []
    all_gold_spans_typed = []
    all_pred_spans_typed = []

    for idx, (note_id, text) in enumerate(note_texts.items()):
        note_ann = test_ann[test_ann["note_id"] == note_id]

        for _, row in note_ann.iterrows():
            s, e = int(row["start"]), int(row["end"])
            all_gold_spans_agnostic.append((s, e))
            cls = map_concept_to_fine_class(int(row["concept_id"]), sctid_to_tag)
            all_gold_spans_typed.append((s, e, cls))

        preds = predict_note(
            model, text,
            window_tokens=args.window_tokens,
            overlap_tokens=args.overlap_tokens,
            threshold=args.threshold,
        )

        for pred in preds:
            all_pred_spans_agnostic.append((pred["start"], pred["end"]))
            all_pred_spans_typed.append((pred["start"], pred["end"], pred["entity_type"]))

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(note_texts)} notes")

    # Class-agnostic metrics
    print(f"\n{'='*60}")
    print("CLASS-AGNOSTIC SPAN DETECTION")
    print(f"{'='*60}")

    # Deduplicate (same span may appear in gold with different classes)
    gold_unique = list(set(all_gold_spans_agnostic))
    pred_unique = list(set(all_pred_spans_agnostic))

    print(f"Gold unique spans: {len(gold_unique):,}")
    print(f"Pred unique spans: {len(pred_unique):,}")

    exact_ag = compute_agnostic_metrics(gold_unique, pred_unique, iou_threshold=0.0)
    partial_ag = compute_agnostic_metrics(gold_unique, pred_unique, iou_threshold=0.5)

    print(f"\n--- Exact Match (class-agnostic) ---")
    print(f"  Precision: {exact_ag['precision']:.4f}")
    print(f"  Recall:    {exact_ag['recall']:.4f}")
    print(f"  F1:        {exact_ag['f1']:.4f}")

    print(f"\n--- Partial Match IoU>=0.5 (class-agnostic) ---")
    print(f"  Precision: {partial_ag['precision']:.4f}")
    print(f"  Recall:    {partial_ag['recall']:.4f}")
    print(f"  F1:        {partial_ag['f1']:.4f}")

    # How many gold spans are detected by ANY class?
    gold_set = set(gold_unique)
    pred_set = set(pred_unique)
    detected = gold_set & pred_set
    print(f"\n--- Coverage ---")
    print(f"  Gold spans exactly detected: {len(detected):,} / {len(gold_set):,} ({100*len(detected)/len(gold_set):.1f}%)")

    # Per predicted class: how many unique (start,end) spans?
    class_spans = defaultdict(set)
    for s, e, cls in all_pred_spans_typed:
        class_spans[cls].add((s, e))
    all_detected = set()
    print(f"\n--- Unique spans per predicted class ---")
    for cls in ALL_ENTITY_TYPES:
        spans = class_spans.get(cls, set())
        matched = spans & gold_set
        all_detected |= matched
        print(f"  {cls:40s}: {len(spans):5d} pred, {len(matched):5d} exact gold matches")
    print(f"  {'UNION':40s}: {len(all_detected):5d} unique gold spans detected across all classes")


if __name__ == "__main__":
    main()
