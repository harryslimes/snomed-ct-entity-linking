#!/usr/bin/env python3
"""Per-entity-type threshold sweep for GLiNER2.

Runs inference once with threshold=0 to collect all predictions with confidence
scores, then sweeps per-type thresholds to find the optimal combination.

Usage:
    python scripts/threshold_sweep.py \
        --model-dir models/gliner2-snomed-large-v3/best \
        --data-dir data/old-challenge-split
"""

import argparse
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gliner2 import GLiNER2

sys.path.insert(0, str(Path(__file__).parent))
from eval_gliner2 import (
    ENTITY_DESCRIPTIONS,
    ENTITY_TYPES,
    compute_span_metrics,
    deduplicate_predictions,
)
from prepare_gliner2_data import (
    HEADER_DISPLAY,
    TAG_TO_CLASS,
    chunk_note_by_section,
    load_sctid_to_tag,
)


def predict_note_raw(
    model: GLiNER2,
    text: str,
    window_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[dict]:
    """Run GLiNER2 with NO threshold — return all predictions with confidence."""
    chunks = chunk_note_by_section(text, window_tokens, overlap_tokens)
    all_preds = []

    for section_header, chunk_start, chunk_end in chunks:
        raw_chunk_text = text[chunk_start:chunk_end].strip()
        if not raw_chunk_text:
            continue

        display_header = HEADER_DISPLAY.get(section_header, section_header.title())
        if display_header:
            chunk_text = f"[{display_header}] {raw_chunk_text}"
        else:
            chunk_text = raw_chunk_text

        result = model.extract_entities(
            chunk_text,
            ENTITY_TYPES,
            include_confidence=True,
            include_spans=True,
        )

        header_prefix_len = len(chunk_text) - len(raw_chunk_text)

        if isinstance(result, dict):
            entity_dict = result.get("entities", result)
            for etype, entities in entity_dict.items():
                if not isinstance(entities, list):
                    continue
                for ent in entities:
                    if not isinstance(ent, dict):
                        continue
                    conf = ent.get("confidence", 1.0)
                    ent_start_in_chunk = ent.get("start", 0)
                    ent_end_in_chunk = ent.get("end", 0)

                    if ent_end_in_chunk <= header_prefix_len:
                        continue

                    ent_start = ent_start_in_chunk - header_prefix_len + chunk_start
                    ent_end = ent_end_in_chunk - header_prefix_len + chunk_start
                    ent_start = max(ent_start, chunk_start)
                    ent_end = min(ent_end, chunk_end)

                    all_preds.append({
                        "start": ent_start,
                        "end": ent_end,
                        "span": ent.get("text", text[ent_start:ent_end]),
                        "entity_type": etype,
                        "confidence": conf,
                    })

    all_preds = deduplicate_predictions(all_preds)
    return all_preds


def filter_by_thresholds(
    raw_preds: list[dict],
    thresholds: dict[str, float],
) -> list[tuple[int, int, str]]:
    """Apply per-type thresholds to raw predictions."""
    filtered = []
    for pred in raw_preds:
        t = thresholds.get(pred["entity_type"], 0.5)
        if pred["confidence"] >= t:
            filtered.append((pred["start"], pred["end"], pred["entity_type"]))
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Per-type threshold sweep")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="fastino/gliner2-large-v1")
    parser.add_argument("--data-dir", type=Path, default=Path("data/old-challenge-split"))
    parser.add_argument("--project-root", type=Path,
                        default=Path("/workspaces/snomed-ct-entity-linking"))
    parser.add_argument("--window-tokens", type=int, default=400)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    # Sweep range
    parser.add_argument("--sweep-min", type=float, default=0.1)
    parser.add_argument("--sweep-max", type=float, default=0.9)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    args = parser.parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_dir}... (device={device})")
    try:
        model = GLiNER2.from_pretrained(args.model_dir).to(device)
    except Exception:
        print(f"Loading as adapter on {args.base_model}...")
        model = GLiNER2.from_pretrained(args.base_model).to(device)
        model.load_adapter(args.model_dir)

    # Load SNOMED tag mapping
    sctid_to_tag = load_sctid_to_tag(args.project_root)

    # Load test data
    print("Loading test data...")
    test_notes = pd.read_csv(args.data_dir / "test_notes.csv")
    test_ann = pd.read_csv(args.data_dir / "test_annotations.csv",
                           dtype={"concept_id": int})
    test_ann["start"] = test_ann["start"].astype(int)
    test_ann["end"] = test_ann["end"].astype(int)
    test_ann["cls"] = test_ann["concept_id"].apply(
        lambda cid: TAG_TO_CLASS.get(sctid_to_tag.get(int(cid), ""),
                                     "medical finding, symptom, or disease")
    )
    note_texts = dict(zip(test_notes["note_id"], test_notes["text"]))

    # Build gold spans
    all_gold_spans = []
    per_class_gold = defaultdict(list)
    for _, row in test_ann.iterrows():
        span_tuple = (int(row["start"]), int(row["end"]), row["cls"])
        all_gold_spans.append(span_tuple)
        per_class_gold[row["cls"]].append(span_tuple)

    print(f"Gold spans: {len(all_gold_spans):,}")
    for cls in ENTITY_TYPES:
        print(f"  {cls}: {len(per_class_gold.get(cls, []))}")

    # Phase 1: Collect all raw predictions (threshold=0)
    print(f"\nPhase 1: Running inference on {len(note_texts)} notes (threshold=0)...")
    all_raw_preds = []  # list of dicts with confidence
    for idx, (note_id, text) in enumerate(note_texts.items()):
        preds = predict_note_raw(
            model, text,
            window_tokens=args.window_tokens,
            overlap_tokens=args.overlap_tokens,
        )
        all_raw_preds.extend(preds)
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(note_texts)} notes")

    print(f"  Total raw predictions (no threshold): {len(all_raw_preds):,}")

    # Show confidence distribution per type
    print("\nConfidence distribution per type:")
    for cls in ENTITY_TYPES:
        confs = [p["confidence"] for p in all_raw_preds if p["entity_type"] == cls]
        if confs:
            arr = np.array(confs)
            print(f"  {cls}:")
            print(f"    count={len(confs)}, min={arr.min():.3f}, "
                  f"median={np.median(arr):.3f}, mean={arr.mean():.3f}, max={arr.max():.3f}")
            for pct in [10, 25, 50, 75, 90]:
                print(f"    p{pct}={np.percentile(arr, pct):.3f}", end="")
            print()

    # Phase 2: Sweep thresholds
    thresholds = np.arange(args.sweep_min, args.sweep_max + 0.001, args.sweep_step)
    thresholds = np.round(thresholds, 3)
    print(f"\nPhase 2: Sweeping {len(thresholds)} thresholds per type "
          f"({args.sweep_min} to {args.sweep_max} step {args.sweep_step})...")
    print(f"  Total grid: {len(thresholds)**3:,} combinations")

    # First: find optimal per-type thresholds independently
    print("\n--- Independent per-type optimization (exact match) ---")
    best_per_type = {}
    for cls in ENTITY_TYPES:
        cls_preds_raw = [p for p in all_raw_preds if p["entity_type"] == cls]
        cls_gold = per_class_gold.get(cls, [])
        best_f1 = -1
        best_t = 0.5
        best_metrics = {}

        for t in thresholds:
            cls_preds = [(p["start"], p["end"], p["entity_type"])
                         for p in cls_preds_raw if p["confidence"] >= t]
            m = compute_span_metrics(cls_gold, cls_preds, iou_threshold=0.0)
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_t = t
                best_metrics = m

        best_per_type[cls] = best_t
        print(f"  {cls}:")
        print(f"    Best threshold: {best_t:.3f}")
        print(f"    P={best_metrics['precision']:.4f} R={best_metrics['recall']:.4f} "
              f"F1={best_metrics['f1']:.4f}")
        print(f"    TP={best_metrics['tp']} FP={best_metrics['fp']} FN={best_metrics['fn']}")

    # Evaluate with independent best thresholds combined
    print(f"\n--- Combined independent thresholds ---")
    print(f"  Thresholds: {best_per_type}")
    combined_preds = filter_by_thresholds(all_raw_preds, best_per_type)

    exact = compute_span_metrics(all_gold_spans, combined_preds, iou_threshold=0.0)
    print(f"  Exact:   P={exact['precision']:.4f} R={exact['recall']:.4f} F1={exact['f1']:.4f}")
    partial = compute_span_metrics(all_gold_spans, combined_preds, iou_threshold=0.5)
    print(f"  Partial: P={partial['precision']:.4f} R={partial['recall']:.4f} F1={partial['f1']:.4f}")

    # Also do a coarse grid search on the combined F1
    # Use a coarser grid (0.05 step) for the 3-type joint search
    coarse_thresholds = np.arange(args.sweep_min, args.sweep_max + 0.001, 0.05)
    coarse_thresholds = np.round(coarse_thresholds, 3)
    print(f"\n--- Joint grid search (exact F1, step=0.05) ---")
    print(f"  Grid size: {len(coarse_thresholds)**3:,} combinations")

    best_joint_f1 = -1
    best_joint_thresholds = {}
    best_joint_metrics = {}

    types_list = list(ENTITY_TYPES)
    total_combos = len(coarse_thresholds) ** 3
    checked = 0

    for t0, t1, t2 in product(coarse_thresholds, repeat=3):
        threshs = {types_list[0]: t0, types_list[1]: t1, types_list[2]: t2}
        preds = filter_by_thresholds(all_raw_preds, threshs)
        m = compute_span_metrics(all_gold_spans, preds, iou_threshold=0.0)
        if m["f1"] > best_joint_f1:
            best_joint_f1 = m["f1"]
            best_joint_thresholds = threshs.copy()
            best_joint_metrics = m

        checked += 1
        if checked % 1000 == 0:
            print(f"  Checked {checked}/{total_combos}...", end="\r")

    print(f"\n  Best joint thresholds: {best_joint_thresholds}")
    print(f"  Exact:   P={best_joint_metrics['precision']:.4f} "
          f"R={best_joint_metrics['recall']:.4f} F1={best_joint_metrics['f1']:.4f}")

    # Evaluate joint best with partial match too
    preds_joint = filter_by_thresholds(all_raw_preds, best_joint_thresholds)
    partial_joint = compute_span_metrics(all_gold_spans, preds_joint, iou_threshold=0.5)
    print(f"  Partial: P={partial_joint['precision']:.4f} "
          f"R={partial_joint['recall']:.4f} F1={partial_joint['f1']:.4f}")

    # Per-class breakdown for joint best
    print(f"\n  Per-class breakdown (joint best, exact match):")
    for cls in ENTITY_TYPES:
        cls_preds = [(s, e, t) for s, e, t in preds_joint if t == cls]
        cls_gold = per_class_gold.get(cls, [])
        m = compute_span_metrics(cls_gold, cls_preds, iou_threshold=0.0)
        print(f"    {cls}: P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f}  (gold={len(cls_gold)} pred={len(cls_preds)})")

    # Comparison with flat 0.5
    print(f"\n{'='*60}")
    print("COMPARISON: flat 0.5 vs optimized thresholds")
    print(f"{'='*60}")
    flat_preds = filter_by_thresholds(all_raw_preds, {t: 0.5 for t in ENTITY_TYPES})
    flat_exact = compute_span_metrics(all_gold_spans, flat_preds, iou_threshold=0.0)
    flat_partial = compute_span_metrics(all_gold_spans, flat_preds, iou_threshold=0.5)

    print(f"  Flat 0.5:    Exact F1={flat_exact['f1']:.4f}  Partial F1={flat_partial['f1']:.4f}")
    print(f"  Independent: Exact F1={exact['f1']:.4f}  Partial F1={partial['f1']:.4f}")
    print(f"  Joint best:  Exact F1={best_joint_metrics['f1']:.4f}  "
          f"Partial F1={partial_joint['f1']:.4f}")
    print(f"\n  Exact F1 improvement (joint): "
          f"{best_joint_metrics['f1'] - flat_exact['f1']:+.4f} "
          f"({(best_joint_metrics['f1'] - flat_exact['f1'])/flat_exact['f1']*100:+.1f}%)")


if __name__ == "__main__":
    main()
