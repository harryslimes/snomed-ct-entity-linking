#!/usr/bin/env python3
"""Grid search over window_tokens and overlap_tokens at inference time.

Uses the existing trained model (fixed weights) and only varies how we chunk
test notes and merge predictions. No retraining needed.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from gliner2 import GLiNER2

sys.path.insert(0, str(Path(__file__).parent))
from eval_gliner2 import (
    ALL_ENTITY_TYPES,
    ENTITY_TYPES,
    compute_span_metrics,
    predict_note,
)
from prepare_gliner2_data import load_sctid_to_tag, map_concept_to_fine_class


def run_eval(
    model,
    note_texts: dict,
    test_ann: pd.DataFrame,
    sctid_to_tag: dict,
    window_tokens: int,
    overlap_tokens: int,
    threshold: float,
) -> dict:
    """Run full evaluation with given window/overlap settings."""
    all_gold_spans = []
    all_pred_spans = []

    for note_id, text in note_texts.items():
        note_ann = test_ann[test_ann["note_id"] == note_id]

        # Gold spans
        for _, row in note_ann.iterrows():
            cls = map_concept_to_fine_class(int(row["concept_id"]), sctid_to_tag)
            all_gold_spans.append((int(row["start"]), int(row["end"]), cls))

        # Predictions
        preds = predict_note(
            model, text,
            window_tokens=window_tokens,
            overlap_tokens=overlap_tokens,
            threshold=threshold,
        )
        for pred in preds:
            all_pred_spans.append((pred["start"], pred["end"], pred["entity_type"]))

    exact = compute_span_metrics(all_gold_spans, all_pred_spans, iou_threshold=0.0)
    partial = compute_span_metrics(all_gold_spans, all_pred_spans, iou_threshold=0.5)

    return {
        "exact": exact,
        "partial": partial,
        "n_gold": len(all_gold_spans),
        "n_pred": len(all_pred_spans),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="fastino/gliner2-large-v1")
    parser.add_argument("--data-dir", type=Path, default=Path("data/old-challenge-split"))
    parser.add_argument("--project-root", type=Path, default=Path("/workspaces/snomed-ct-entity-linking"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("grid_search_results.json"))
    args = parser.parse_args()

    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_dir}... (device={device})")
    try:
        model = GLiNER2.from_pretrained(args.model_dir).to(device)
    except Exception:
        print(f"Loading as adapter on {args.base_model}...")
        model = GLiNER2.from_pretrained(args.base_model).to(device)
        model.load_adapter(args.model_dir)

    # Load data once
    sctid_to_tag = load_sctid_to_tag(args.project_root)
    test_notes = pd.read_csv(args.data_dir / "test_notes.csv")
    test_ann = pd.read_csv(args.data_dir / "test_annotations.csv", dtype={"concept_id": int})
    test_ann["start"] = test_ann["start"].astype(int)
    test_ann["end"] = test_ann["end"].astype(int)
    note_texts = dict(zip(test_notes["note_id"], test_notes["text"]))

    # Grid search parameters
    window_sizes = [75, 100, 125, 150, 200, 300]
    overlap_sizes = [15, 30, 50, 75]

    # Filter out invalid combos (overlap must be < window)
    combos = [(w, o) for w, o in itertools.product(window_sizes, overlap_sizes)
              if o < w]

    print(f"\nGrid search: {len(combos)} combinations at threshold={args.threshold}")
    print(f"{'Window':>8s} {'Overlap':>8s} | {'ExactP':>8s} {'ExactR':>8s} {'ExactF1':>8s} | {'PartP':>8s} {'PartR':>8s} {'PartF1':>8s} | {'Time':>6s}")
    print("-" * 95)

    results = []
    for w, o in combos:
        t0 = time.time()
        metrics = run_eval(
            model, note_texts, test_ann, sctid_to_tag,
            window_tokens=w, overlap_tokens=o, threshold=args.threshold,
        )
        elapsed = time.time() - t0

        row = {
            "window_tokens": w,
            "overlap_tokens": o,
            "threshold": args.threshold,
            "exact_precision": metrics["exact"]["precision"],
            "exact_recall": metrics["exact"]["recall"],
            "exact_f1": metrics["exact"]["f1"],
            "partial_precision": metrics["partial"]["precision"],
            "partial_recall": metrics["partial"]["recall"],
            "partial_f1": metrics["partial"]["f1"],
            "n_gold": metrics["n_gold"],
            "n_pred": metrics["n_pred"],
            "elapsed_s": round(elapsed, 1),
        }
        results.append(row)

        print(f"{w:>8d} {o:>8d} | {row['exact_precision']:>8.4f} {row['exact_recall']:>8.4f} {row['exact_f1']:>8.4f} | "
              f"{row['partial_precision']:>8.4f} {row['partial_recall']:>8.4f} {row['partial_f1']:>8.4f} | {elapsed:>5.1f}s")

        # Save incrementally
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    # Print summary sorted by exact F1
    print(f"\n{'='*95}")
    print("TOP 10 BY EXACT F1:")
    print(f"{'='*95}")
    sorted_results = sorted(results, key=lambda x: -x["exact_f1"])
    for i, r in enumerate(sorted_results[:10]):
        print(f"  {i+1:2d}. w={r['window_tokens']:3d} o={r['overlap_tokens']:2d} | "
              f"ExactF1={r['exact_f1']:.4f} PartialF1={r['partial_f1']:.4f} "
              f"P={r['exact_precision']:.4f} R={r['exact_recall']:.4f}")

    print(f"\nTOP 10 BY PARTIAL F1:")
    sorted_partial = sorted(results, key=lambda x: -x["partial_f1"])
    for i, r in enumerate(sorted_partial[:10]):
        print(f"  {i+1:2d}. w={r['window_tokens']:3d} o={r['overlap_tokens']:2d} | "
              f"PartialF1={r['partial_f1']:.4f} ExactF1={r['exact_f1']:.4f} "
              f"P={r['partial_precision']:.4f} R={r['partial_recall']:.4f}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
