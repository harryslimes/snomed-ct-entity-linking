#!/usr/bin/env python3
"""Analyze GLiNER2 performance per concept using the competition metric.

Computes macro-averaged character-level IoU (the actual competition metric),
then breaks down by:
- Concept frequency (rare → common)
- SNOMED semantic class (finding/procedure/body)
- Individual worst-performing concepts

This shows where the biggest scoring gains are.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path("/workspaces/snomed-ct-entity-linking")))

from eval_gliner2 import ALL_ENTITY_TYPES, predict_note
from gliner2 import GLiNER2
from prepare_gliner2_data import (
    TAG_TO_CLASS,
    annotation_in_lab_region,
    find_lab_regions,
    load_sctid_to_tag,
    merge_adjacent_regions,
)
from super_dictionary.runtime_scoring import class_char_iou, macro_char_iou


def freq_bucket(count):
    if count == 0:
        return "unseen (0)"
    elif count == 1:
        return "rare (1)"
    elif count <= 5:
        return "low (2-5)"
    elif count <= 20:
        return "medium (6-20)"
    elif count <= 100:
        return "high (21-100)"
    else:
        return "very_high (>100)"


BUCKET_ORDER = [
    "unseen (0)",
    "rare (1)",
    "low (2-5)",
    "medium (6-20)",
    "high (21-100)",
    "very_high (>100)",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="fastino/gliner2-large-v1")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/old-challenge-split")
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/workspaces/snomed-ct-entity-linking"),
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-tokens", type=int, default=200)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--filter-lab-tables", action="store_true")
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

    # Load SNOMED mappings
    sctid_to_tag = load_sctid_to_tag(args.project_root)

    # Load data
    test_notes = pd.read_csv(args.data_dir / "test_notes.csv")
    test_ann = pd.read_csv(
        args.data_dir / "test_annotations.csv", dtype={"concept_id": int}
    )
    test_ann["start"] = test_ann["start"].astype(int)
    test_ann["end"] = test_ann["end"].astype(int)

    train_ann = pd.read_csv(
        args.data_dir / "train_annotations.csv", dtype={"concept_id": int}
    )

    note_texts = dict(zip(test_notes["note_id"], test_notes["text"]))

    # Lab table filtering
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
        print(
            f"  Lab table filter: {n_before - len(test_ann):,} gold annotations removed"
        )

    # Concept frequency from training data
    train_concept_freq = train_ann["concept_id"].value_counts().to_dict()

    # -----------------------------------------------------------------------
    # Run GLiNER2 predictions
    # -----------------------------------------------------------------------
    print(f"\nRunning inference on {len(test_notes)} test notes...")

    # GLiNER2 doesn't predict concept_ids — it predicts entity types (finding/procedure/body).
    # To compute competition metric, we need to assign concept_ids to predictions.
    # We can't do this here (that's the downstream linking step).
    #
    # Instead, we analyze which GOLD annotations GLiNER2 detects (span detection)
    # and compute the competition metric impact per concept.

    # For each gold annotation, check if GLiNER2 found a matching span
    gold_matches = []  # (note_id, start, end, concept_id, matched, pred_start, pred_end)

    for idx, (note_id, text) in enumerate(note_texts.items()):
        note_ann = test_ann[test_ann["note_id"] == note_id]

        preds = predict_note(
            model,
            text,
            window_tokens=args.window_tokens,
            overlap_tokens=args.overlap_tokens,
            threshold=args.threshold,
        )
        pred_spans = [(p["start"], p["end"]) for p in preds]

        for _, row in note_ann.iterrows():
            gs, ge = int(row["start"]), int(row["end"])
            cid = int(row["concept_id"])

            # Check for exact match
            exact = (gs, ge) in set(pred_spans)

            # Check for best partial match (IoU)
            best_iou = 0.0
            best_ps, best_pe = -1, -1
            for ps, pe in pred_spans:
                overlap_start = max(gs, ps)
                overlap_end = min(ge, pe)
                if overlap_start >= overlap_end:
                    continue
                overlap = overlap_end - overlap_start
                union = max(ge, pe) - min(gs, ps)
                iou = overlap / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_ps, best_pe = ps, pe

            gold_matches.append(
                {
                    "note_id": note_id,
                    "start": gs,
                    "end": ge,
                    "concept_id": cid,
                    "span": text[gs:ge],
                    "exact_match": exact,
                    "best_iou": best_iou,
                    "pred_start": best_ps,
                    "pred_end": best_pe,
                }
            )

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(note_texts)} notes")

    matches_df = pd.DataFrame(gold_matches)

    # Add metadata
    matches_df["train_freq"] = matches_df["concept_id"].map(
        lambda c: train_concept_freq.get(c, 0)
    )
    matches_df["freq_bucket"] = matches_df["train_freq"].apply(freq_bucket)
    matches_df["semantic_tag"] = matches_df["concept_id"].map(
        lambda c: sctid_to_tag.get(c, "unknown")
    )
    matches_df["class"] = matches_df["semantic_tag"].map(
        lambda t: TAG_TO_CLASS.get(t, "unknown")
    )
    matches_df["char_len"] = matches_df["end"] - matches_df["start"]

    # For competition metric: character-level IoU per concept
    # If GLiNER2 detects a span exactly, we get full char overlap for that annotation.
    # If partial, we get partial. If missed, we get 0.
    # Compute "detected chars" vs "gold chars" per concept.

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("GLINER2 SPAN DETECTION — COMPETITION METRIC IMPACT")
    print(f"{'='*70}")
    print(f"Total gold annotations: {len(matches_df):,}")
    print(
        f"Exact matches: {matches_df['exact_match'].sum():,} ({100*matches_df['exact_match'].mean():.1f}%)"
    )
    print(
        f"Partial matches (IoU>0): {(matches_df['best_iou'] > 0).sum():,} ({100*(matches_df['best_iou'] > 0).mean():.1f}%)"
    )
    print(
        f"Complete misses: {(matches_df['best_iou'] == 0).sum():,} ({100*(matches_df['best_iou'] == 0).mean():.1f}%)"
    )

    # -----------------------------------------------------------------------
    # Per frequency bucket
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("BY CONCEPT FREQUENCY (training set)")
    print(f"{'='*70}")
    print(
        f"{'Bucket':<18s} {'Concepts':>8s} {'Annots':>8s} {'Exact%':>8s} {'IoU>0%':>8s} {'Miss%':>8s} {'AvgIoU':>8s}"
    )
    print("-" * 70)

    for bucket in BUCKET_ORDER:
        bdf = matches_df[matches_df["freq_bucket"] == bucket]
        if len(bdf) == 0:
            continue
        n_concepts = bdf["concept_id"].nunique()
        n_annots = len(bdf)
        exact_pct = 100 * bdf["exact_match"].mean()
        partial_pct = 100 * (bdf["best_iou"] > 0).mean()
        miss_pct = 100 * (bdf["best_iou"] == 0).mean()
        avg_iou = bdf["best_iou"].mean()
        print(
            f"{bucket:<18s} {n_concepts:>8,d} {n_annots:>8,d} {exact_pct:>7.1f}% {partial_pct:>7.1f}% {miss_pct:>7.1f}% {avg_iou:>8.3f}"
        )

    # -----------------------------------------------------------------------
    # Per SNOMED class
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("BY SNOMED CLASS")
    print(f"{'='*70}")
    print(
        f"{'Class':<45s} {'Annots':>8s} {'Exact%':>8s} {'Miss%':>8s} {'AvgIoU':>8s}"
    )
    print("-" * 70)

    for cls in sorted(matches_df["class"].unique()):
        cdf = matches_df[matches_df["class"] == cls]
        exact_pct = 100 * cdf["exact_match"].mean()
        miss_pct = 100 * (cdf["best_iou"] == 0).mean()
        avg_iou = cdf["best_iou"].mean()
        print(
            f"{cls:<45s} {len(cdf):>8,d} {exact_pct:>7.1f}% {miss_pct:>7.1f}% {avg_iou:>8.3f}"
        )

    # -----------------------------------------------------------------------
    # Per class × frequency
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("BY CLASS × FREQUENCY")
    print(f"{'='*70}")

    for cls in sorted(matches_df["class"].unique()):
        print(f"\n  {cls}:")
        cdf = matches_df[matches_df["class"] == cls]
        for bucket in BUCKET_ORDER:
            bdf = cdf[cdf["freq_bucket"] == bucket]
            if len(bdf) == 0:
                continue
            n_concepts = bdf["concept_id"].nunique()
            exact_pct = 100 * bdf["exact_match"].mean()
            miss_pct = 100 * (bdf["best_iou"] == 0).mean()
            print(
                f"    {bucket:<18s}: {n_concepts:>5d} concepts, {len(bdf):>6d} annots, "
                f"exact={exact_pct:5.1f}%, miss={miss_pct:5.1f}%"
            )

    # -----------------------------------------------------------------------
    # Competition metric simulation
    # -----------------------------------------------------------------------
    # Compute per-concept "char IoU" assuming perfect concept linking
    # (i.e., if GLiNER2 detects a span, downstream linking assigns correct concept)
    print(f"\n{'='*70}")
    print("COMPETITION METRIC SIMULATION (assuming perfect concept linking)")
    print(f"{'='*70}")

    concept_stats = []
    for cid, grp in matches_df.groupby("concept_id"):
        n_annots = len(grp)
        total_gold_chars = grp["char_len"].sum()

        # Chars that would be correctly covered if linking is perfect
        detected_chars = 0
        for _, row in grp.iterrows():
            if row["exact_match"]:
                detected_chars += row["char_len"]
            elif row["best_iou"] > 0:
                # Partial: intersection chars
                gs, ge = row["start"], row["end"]
                ps, pe = row["pred_start"], row["pred_end"]
                overlap = min(ge, pe) - max(gs, ps)
                detected_chars += max(0, overlap)

        char_iou = detected_chars / total_gold_chars if total_gold_chars > 0 else 0
        train_freq = train_concept_freq.get(cid, 0)

        concept_stats.append(
            {
                "concept_id": cid,
                "n_annots": n_annots,
                "train_freq": train_freq,
                "freq_bucket": freq_bucket(train_freq),
                "total_gold_chars": total_gold_chars,
                "detected_chars": detected_chars,
                "char_recall": char_iou,
                "class": grp["class"].iloc[0],
                "example_span": grp["span"].iloc[0],
            }
        )

    concept_df = pd.DataFrame(concept_stats)
    concept_df = concept_df.sort_values("char_recall")

    # Overall macro char recall (proxy for competition metric upper bound from span detection)
    macro_recall = concept_df["char_recall"].mean()
    print(f"\nMacro-averaged char recall (span detection ceiling): {macro_recall:.4f}")
    print(f"  This is the max competition score assuming perfect concept linking")
    print(f"  and zero false positive concepts.")

    # Per bucket
    print(f"\n{'Bucket':<18s} {'Concepts':>8s} {'MacroRecall':>12s} {'Impact':>10s}")
    print("-" * 55)
    for bucket in BUCKET_ORDER:
        bdf = concept_df[concept_df["freq_bucket"] == bucket]
        if len(bdf) == 0:
            continue
        mr = bdf["char_recall"].mean()
        # Impact = concepts in bucket / total concepts * (1 - recall)
        # = potential macro score gain if we could detect all these spans
        impact = len(bdf) / len(concept_df) * (1 - mr)
        print(f"{bucket:<18s} {len(bdf):>8,d} {mr:>11.4f} {impact:>10.4f}")

    # -----------------------------------------------------------------------
    # Worst concepts (biggest scoring opportunities)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("TOP 50 WORST CONCEPTS (biggest scoring opportunities)")
    print(f"{'='*70}")

    # Weight by: concept gets equal weight in macro average, so concepts with
    # char_recall=0 are all equally bad. Sort by 0-recall first, then by frequency.
    zero_recall = concept_df[concept_df["char_recall"] == 0].sort_values(
        "n_annots", ascending=False
    )
    partial_recall = concept_df[
        (concept_df["char_recall"] > 0) & (concept_df["char_recall"] < 0.5)
    ].sort_values("char_recall")

    print(f"\n--- Concepts with ZERO recall ({len(zero_recall)}) ---")
    print(
        f"{'concept_id':>12s} {'class':<25s} {'annots':>6s} {'freq':>6s} {'span'}"
    )
    for _, row in zero_recall.head(30).iterrows():
        print(
            f"{row['concept_id']:>12d} {row['class']:<25s} {row['n_annots']:>6d} {row['train_freq']:>6d} {row['example_span'][:50]}"
        )

    print(f"\n--- Concepts with low recall (<0.5) ({len(partial_recall)}) ---")
    print(
        f"{'concept_id':>12s} {'class':<25s} {'recall':>7s} {'annots':>6s} {'freq':>6s} {'span'}"
    )
    for _, row in partial_recall.head(20).iterrows():
        print(
            f"{row['concept_id']:>12d} {row['class']:<25s} {row['char_recall']:>7.3f} {row['n_annots']:>6d} {row['train_freq']:>6d} {row['example_span'][:50]}"
        )

    # -----------------------------------------------------------------------
    # Summary: where to focus
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY: WHERE TO FOCUS")
    print(f"{'='*70}")

    n_total = len(concept_df)
    n_perfect = len(concept_df[concept_df["char_recall"] == 1.0])
    n_zero = len(concept_df[concept_df["char_recall"] == 0.0])
    n_partial = n_total - n_perfect - n_zero

    print(f"Total unique concepts in test: {n_total:,}")
    print(f"  Perfect recall (1.0):  {n_perfect:>5,d} ({100*n_perfect/n_total:.1f}%)")
    print(f"  Partial (0<r<1):       {n_partial:>5,d} ({100*n_partial/n_total:.1f}%)")
    print(f"  Zero recall (0.0):     {n_zero:>5,d} ({100*n_zero/n_total:.1f}%)")
    print()

    # Potential gains
    # Each concept with recall=0 contributes 0 to macro avg
    # If we could get them to recall=1, we'd gain 1/n_total per concept
    potential_from_zero = n_zero / n_total
    potential_from_partial = (
        concept_df[(concept_df["char_recall"] > 0) & (concept_df["char_recall"] < 1)][
            "char_recall"
        ]
        .apply(lambda r: (1 - r) / n_total)
        .sum()
    )

    print(
        f"Potential macro score gain from fixing zero-recall concepts: +{potential_from_zero:.4f}"
    )
    print(
        f"Potential macro score gain from perfecting partial concepts:  +{potential_from_partial:.4f}"
    )
    print(f"Current macro char recall (ceiling):                         {macro_recall:.4f}")
    print(f"Theoretical max (all concepts detected perfectly):           1.0000")


if __name__ == "__main__":
    main()
