#!/usr/bin/env python3
"""Generate out-of-fold KIRI predictions for realistic span correction training data.

Problem: Training KIRI on all data and predicting on the same data gives 98.5%
perfect spans (IoU=1.0) — the model memorized its own training data. But on unseen
test data, ~20% of concepts have imperfect spans.

Solution: K-fold cross-validation. For each fold:
  1. Train KIRI on K-1 folds
  2. Predict on the held-out fold
  3. These predictions have realistic boundary errors

The concatenated out-of-fold predictions provide proper training signal for
the span correction regression head.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
KIRI_SRC = REPO_ROOT / "1st Place" / "src"
if str(KIRI_SRC) not in sys.path:
    sys.path.append(str(KIRI_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from mimic_predict import make_predictions  # noqa: E402
from mimic_train import train  # noqa: E402
from mimic_common import common_headers  # noqa: E402


def main():
    n_folds = 4
    seed = 42

    # Set super dictionary path
    super_synonyms = "1st Place/data/interim/flattened_terminology_syn_super.csv"
    os.environ["KIRI_SYNONYMS_PATH"] = str(super_synonyms)
    os.environ["KIRI_LINGUISTIC_RULES"] = "1"
    os.environ["KIRI_STOPWORD_TRANSPARENT"] = "1"

    # Load data
    print("Loading training data...")
    split_dir = Path("data/old-challenge-split")
    all_notes = pd.read_csv(split_dir / "train_notes.csv")
    all_annotations = pd.read_csv(split_dir / "train_annotations.csv")

    # Rename 'span' to 'source' for KIRI compatibility
    if "span" in all_annotations.columns and "source" not in all_annotations.columns:
        all_annotations = all_annotations.rename(columns={"span": "source"})

    print(f"Total: {len(all_notes)} notes, {len(all_annotations)} annotations")

    # Create folds by note_id
    note_ids = np.array(sorted(all_notes["note_id"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(note_ids)
    folds = np.array_split(note_ids, n_folds)

    all_oof_preds = []

    for fold_idx, held_out_ids in enumerate(folds):
        held_out_set = set(held_out_ids)
        train_ids = [nid for nid in note_ids if nid not in held_out_set]

        # Split notes
        train_notes = all_notes[all_notes["note_id"].isin(train_ids)]
        held_out_notes = all_notes[all_notes["note_id"].isin(held_out_set)]

        # Split annotations
        train_ann = all_annotations[all_annotations["note_id"].isin(train_ids)]

        train_texts = train_notes.set_index("note_id")["text"]
        held_out_texts = held_out_notes.set_index("note_id")["text"]

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/{n_folds}: Train={len(train_texts)} notes, "
              f"Held-out={len(held_out_texts)} notes")
        print(f"{'='*60}")

        # Train KIRI on training fold
        print(f"Training KIRI on fold {fold_idx}...")
        d, uc_d = train(train_texts, train_ann, common_headers,
                        f"kiri_oof_fold{fold_idx}")

        # Predict on held-out fold
        print(f"Predicting on held-out fold {fold_idx}...")
        fold_pred = make_predictions(held_out_texts, d, uc_d,
                                     run_name=f"kiri_oof_fold{fold_idx}")

        print(f"Fold {fold_idx}: {len(fold_pred)} predictions on "
              f"{fold_pred['note_id'].nunique()} notes")
        all_oof_preds.append(fold_pred)

    # Concatenate all OOF predictions
    oof_pred = pd.concat(all_oof_preds, ignore_index=True)

    # Save
    output_dir = Path("outputs/old_challenge_split")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "kiri_super_oof_pred.csv"
    oof_pred.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"OUT-OF-FOLD PREDICTIONS COMPLETE")
    print(f"{'='*60}")
    print(f"Total predictions: {len(oof_pred)}")
    print(f"Notes covered: {oof_pred['note_id'].nunique()}")
    print(f"Unique concepts: {oof_pred['concept_id'].nunique()}")
    print(f"Saved to: {output_file}")

    # Quick IoU analysis vs gold
    gold = all_annotations.copy()
    if "source" in gold.columns and "span" not in gold.columns:
        gold = gold.rename(columns={"source": "span"})

    # Compute per-prediction IoU against gold
    from span_correction.data_prep import compute_span_iou

    matched_ious = []
    for _, pred_row in oof_pred.iterrows():
        note_id = pred_row["note_id"]
        concept_id = pred_row["concept_id"]
        pred_start, pred_end = int(pred_row["start"]), int(pred_row["end"])

        gold_matches = gold[(gold["note_id"] == note_id) &
                            (gold["concept_id"] == concept_id)]
        best_iou = 0.0
        for _, g in gold_matches.iterrows():
            iou = compute_span_iou((pred_start, pred_end),
                                   (int(g["start"]), int(g["end"])))
            best_iou = max(best_iou, iou)
        matched_ious.append(best_iou)

    oof_pred["best_iou"] = matched_ious
    matched = oof_pred[oof_pred["best_iou"] > 0.3]
    perfect = matched[matched["best_iou"] >= 0.999]
    imperfect = matched[(matched["best_iou"] > 0.3) & (matched["best_iou"] < 0.999)]
    fp = oof_pred[oof_pred["best_iou"] <= 0.3]

    print(f"\n=== OOF Prediction Quality ===")
    print(f"Matched (IoU>0.3):    {len(matched):>6,} ({len(matched)/len(oof_pred)*100:.1f}%)")
    print(f"  Perfect (IoU>=0.999): {len(perfect):>6,} ({len(perfect)/len(oof_pred)*100:.1f}%)")
    print(f"  Imperfect:            {len(imperfect):>6,} ({len(imperfect)/len(oof_pred)*100:.1f}%)")
    print(f"False positives:      {len(fp):>6,} ({len(fp)/len(oof_pred)*100:.1f}%)")

    if len(imperfect) > 0:
        print(f"\nImperfect span IoU distribution:")
        print(f"  Mean: {imperfect['best_iou'].mean():.3f}")
        print(f"  Median: {imperfect['best_iou'].median():.3f}")
        bins = [0.3, 0.5, 0.7, 0.9, 0.999]
        labels = ['0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-0.999']
        binned = pd.cut(imperfect['best_iou'], bins=bins, labels=labels)
        print(binned.value_counts().sort_index())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
