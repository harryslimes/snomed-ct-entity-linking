#!/usr/bin/env python3
"""Oracle span-correction analysis for KIRI super dictionary.

Answers the question: "If KIRI correctly identifies concepts but has imperfect
spans, how much IoU would improve if we perfectly aligned those spans?"

Approach:
  1. Baseline IoU: raw KIRI predictions vs gold annotations on test set.
  2. Oracle IoU: for each KIRI prediction whose (note_id, concept_id) pair
     matches a gold annotation, replace the predicted span(s) with the gold
     span(s). Unmatched predictions are kept as-is.
  3. The difference is the ceiling for span correction (given perfect concept
     identification by KIRI).

Uses the existing runtime_scoring.py for IoU computation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.super_dictionary.runtime_scoring import (  # noqa: E402
    class_char_iou,
    macro_char_iou,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
PRED_PATH = REPO_ROOT / "outputs" / "old_challenge_split" / "kiri_super_pred.csv"

GOLD_ANNOTATIONS = SPLIT_DIR / "test_annotations.csv"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
pred_raw = pd.read_csv(PRED_PATH)
gold_raw = pd.read_csv(GOLD_ANNOTATIONS)

# Normalise columns
pred = pred_raw[["note_id", "start", "end", "concept_id"]].copy()
pred["start"] = pred["start"].astype(int)
pred["end"] = pred["end"].astype(int)
pred["concept_id"] = pred["concept_id"].astype(int)

gold = gold_raw[["note_id", "start", "end", "concept_id"]].copy()
gold["start"] = gold["start"].astype(float).astype(int)
gold["end"] = gold["end"].astype(float).astype(int)
gold["concept_id"] = gold["concept_id"].astype(float).astype(int)

print(f"  Predictions:  {len(pred):,} rows, {pred['note_id'].nunique()} notes")
print(f"  Gold:         {len(gold):,} rows, {gold['note_id'].nunique()} notes")

# ---------------------------------------------------------------------------
# Build oracle predictions
# ---------------------------------------------------------------------------
print("\nBuilding oracle predictions...")

# Identify (note_id, concept_id) pairs that exist in BOTH pred and gold
pred_keys = set(zip(pred["note_id"], pred["concept_id"]))
gold_keys = set(zip(gold["note_id"], gold["concept_id"]))
matched_keys = pred_keys & gold_keys

# For matched pairs, we replace pred spans with gold spans.
# For unmatched pred pairs, we keep the original pred spans.

# Step 1: keep unmatched pred rows as-is
pred["_key"] = list(zip(pred["note_id"], pred["concept_id"]))
unmatched_pred = pred[~pred["_key"].isin(matched_keys)].drop(columns=["_key"])

# Step 2: for matched pairs, take gold spans
gold["_key"] = list(zip(gold["note_id"], gold["concept_id"]))
matched_gold = gold[gold["_key"].isin(matched_keys)].drop(columns=["_key"])

# Combine
oracle_pred = pd.concat([unmatched_pred, matched_gold], ignore_index=True)
oracle_pred = oracle_pred[["note_id", "start", "end", "concept_id"]]

print(f"  Matched (note, concept) pairs:   {len(matched_keys):,}")
print(f"  Unmatched pred pairs (kept):     {len(pred_keys - matched_keys):,}")
print(f"  Gold pairs not in pred (missed): {len(gold_keys - matched_keys):,}")
print(f"  Oracle prediction rows:          {len(oracle_pred):,}")

# ---------------------------------------------------------------------------
# Compute scores
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SCORING")
print("=" * 72)

print("\nComputing baseline IoU (raw KIRI super)...")
baseline_class = class_char_iou(pred.drop(columns=["_key"]), gold.drop(columns=["_key"]))
baseline_macro = float(baseline_class.loc[baseline_class["union"] > 0, "iou"].mean())

print("Computing oracle IoU (perfect span alignment for matched concepts)...")
oracle_class = class_char_iou(oracle_pred, gold.drop(columns=["_key"]))
oracle_macro = float(oracle_class.loc[oracle_class["union"] > 0, "iou"].mean())

print("\n" + "=" * 72)
print("RESULTS")
print("=" * 72)
print()
print(f"  Baseline macro char-IoU (KIRI super):  {baseline_macro:.6f}")
print(f"  Oracle macro char-IoU (perfect spans): {oracle_macro:.6f}")
print(f"  Absolute improvement:                  +{oracle_macro - baseline_macro:.6f}")
if baseline_macro > 0:
    pct_improvement = 100.0 * (oracle_macro - baseline_macro) / baseline_macro
    print(f"  Relative improvement:                  +{pct_improvement:.2f}%")
print(f"  Remaining gap to 1.0:                  {1.0 - oracle_macro:.6f}")
print()

# ---------------------------------------------------------------------------
# Breakdown analysis
# ---------------------------------------------------------------------------
print("=" * 72)
print("DETAILED BREAKDOWN")
print("=" * 72)

# Merge baseline and oracle class IoU
merged = pd.merge(
    baseline_class[["concept_id", "iou", "gt_chars", "pred_chars", "union"]],
    oracle_class[["concept_id", "iou"]],
    on="concept_id",
    how="outer",
    suffixes=("_baseline", "_oracle"),
)
merged["iou_baseline"] = merged["iou_baseline"].fillna(0.0)
merged["iou_oracle"] = merged["iou_oracle"].fillna(0.0)
merged["iou_delta"] = merged["iou_oracle"] - merged["iou_baseline"]

# Only consider concepts with union > 0 in either
valid = merged[(merged["union"].fillna(0) > 0)]

improved = valid[valid["iou_delta"] > 1e-9]
unchanged = valid[valid["iou_delta"].abs() <= 1e-9]
worsened = valid[valid["iou_delta"] < -1e-9]

print(f"\nConcepts with union > 0: {len(valid)}")
print(f"  Improved by oracle:  {len(improved)} ({100*len(improved)/len(valid):.1f}%)")
print(f"  Unchanged:           {len(unchanged)} ({100*len(unchanged)/len(valid):.1f}%)")
print(f"  Worsened:            {len(worsened)} ({100*len(worsened)/len(valid):.1f}%)")
if len(worsened) > 0:
    print("\n  NOTE: Worsened concepts should not occur unless there are overlapping")
    print("  spans with different concept_ids where last-write-wins semantics differ.")

# Show distribution of improvements
if len(improved) > 0:
    print(f"\nAmong improved concepts:")
    print(f"  Mean IoU gain:   +{improved['iou_delta'].mean():.6f}")
    print(f"  Median IoU gain: +{improved['iou_delta'].median():.6f}")
    print(f"  Max IoU gain:    +{improved['iou_delta'].max():.6f}")
    print(f"  Min IoU gain:    +{improved['iou_delta'].min():.6f}")

# Top 20 concepts by absolute improvement
print(f"\nTop 20 concepts with largest span-correction gains:")
print(f"  {'concept_id':>12}  {'baseline':>9}  {'oracle':>9}  {'delta':>9}  {'gt_chars':>10}")
print(f"  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}")
for _, row in improved.nlargest(20, "iou_delta").iterrows():
    print(
        f"  {int(row['concept_id']):>12}  "
        f"{row['iou_baseline']:>9.4f}  "
        f"{row['iou_oracle']:>9.4f}  "
        f"+{row['iou_delta']:>8.4f}  "
        f"{int(row.get('gt_chars', 0)):>10}"
    )

# ---------------------------------------------------------------------------
# Decompose the gap: concept recall vs span precision
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("GAP DECOMPOSITION")
print("=" * 72)

# Concepts that are in gold but NOT in pred at all => recall gap
gold_only_concepts = gold_keys - pred_keys
pred_only_concepts = pred_keys - gold_keys

# How many unique concept_ids does gold have that pred misses entirely?
gold_concept_ids = set(gold["concept_id"].unique())
pred_concept_ids = set(pred["concept_id"].unique())
missed_concepts = gold_concept_ids - pred_concept_ids

print(f"\n  Gold (note,concept) pairs:     {len(gold_keys):,}")
print(f"  Pred (note,concept) pairs:     {len(pred_keys):,}")
print(f"  Matched pairs:                 {len(matched_keys):,}")
print(f"  Gold-only pairs (FN):          {len(gold_only_concepts):,}")
print(f"  Pred-only pairs (FP):          {len(pred_only_concepts):,}")
print()
print(f"  Unique concept_ids in gold:    {len(gold_concept_ids):,}")
print(f"  Unique concept_ids in pred:    {len(pred_concept_ids):,}")
print(f"  Concepts in gold not in pred:  {len(missed_concepts):,}")
print(f"  Concepts in pred not in gold:  {len(pred_concept_ids - gold_concept_ids):,}")
print()

# What fraction of the remaining gap (1 - oracle) comes from concept-level
# FN (gold pairs not in pred) vs FP (pred pairs not in gold)?
# The oracle already handles FP and span misalignment, so the remaining
# gap is entirely due to FN (recall).
remaining_gap = 1.0 - oracle_macro
span_gap = oracle_macro - baseline_macro
print(f"  Total gap to perfect (1.0):       {1.0 - baseline_macro:.6f}")
print(f"  -- Span misalignment gap:         {span_gap:.6f}  ({100*span_gap/(1.0 - baseline_macro):.1f}%)")
print(f"  -- Remaining gap (recall + FP):   {remaining_gap:.6f}  ({100*remaining_gap/(1.0 - baseline_macro):.1f}%)")
print()
print("Interpretation:")
print("  The 'span misalignment gap' is the IoU improvement achievable by")
print("  perfectly aligning spans for concepts KIRI already found in the")
print("  correct note. The 'remaining gap' comes from:")
print("    - False negatives: gold (note,concept) pairs KIRI missed entirely")
print("    - False positives: pred (note,concept) pairs not in gold")
print("    - Overlapping span semantics (last-write-wins edge cases)")
