#!/usr/bin/env python3
"""Calculate the oracle score by taking the best of both approaches."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the note deltas
note_deltas = pd.read_csv("outputs/old_challenge_split/delta_report/note_deltas.csv")

print("=" * 80)
print("ORACLE SCORE ANALYSIS")
print("=" * 80)
print()

# Calculate mean IoUs
mean_default = note_deltas["iou_default"].mean()
mean_super = note_deltas["iou_super"].mean()
mean_oracle = note_deltas[["iou_default", "iou_super"]].max(axis=1).mean()

print(f"Mean Note-level IoU:")
print(f"  Default:  {mean_default:.4f}")
print(f"  Super:    {mean_super:.4f}")
print(f"  Oracle:   {mean_oracle:.4f}")
print()

# Calculate character-level oracle score
# Load the class IoU files
default_class = pd.read_csv("outputs/old_challenge_split/kiri_default_class_iou.csv")
super_class = pd.read_csv("outputs/old_challenge_split/kiri_super_class_iou.csv")

# Merge on concept_id
merged = pd.merge(
    default_class[["concept_id", "iou"]],
    super_class[["concept_id", "iou"]],
    on="concept_id",
    how="outer",
    suffixes=("_default", "_super")
)

# Fill NaN with 0 (concepts that appear in only one)
merged["iou_default"] = merged["iou_default"].fillna(0)
merged["iou_super"] = merged["iou_super"].fillna(0)

# Calculate oracle: take the max IoU for each concept
merged["iou_oracle"] = merged[["iou_default", "iou_super"]].max(axis=1)

# Calculate mean character-level IoU (only for concepts with union > 0)
# We need to check which concepts actually have unions
default_with_union = set(default_class[default_class["union"] > 0]["concept_id"])
super_with_union = set(super_class[super_class["union"] > 0]["concept_id"])
all_with_union = default_with_union | super_with_union

merged_with_union = merged[merged["concept_id"].isin(all_with_union)]

mean_char_default = merged_with_union["iou_default"].mean()
mean_char_super = merged_with_union["iou_super"].mean()
mean_char_oracle = merged_with_union["iou_oracle"].mean()

print(f"Mean Character-level IoU (runtime metric):")
print(f"  Default:  {mean_char_default:.4f}")
print(f"  Super:    {mean_char_super:.4f}")
print(f"  Oracle:   {mean_char_oracle:.4f}")
print()

# Calculate improvement
improvement_super = mean_char_super - mean_char_default
improvement_oracle = mean_char_oracle - mean_char_default
potential_gain = mean_char_oracle - mean_char_super

print(f"Improvements over default:")
print(f"  Super:    +{improvement_super:.4f} ({100*improvement_super/mean_char_default:.1f}%)")
print(f"  Oracle:   +{improvement_oracle:.4f} ({100*improvement_oracle/mean_char_default:.1f}%)")
print()
print(f"Potential gain from fixing super's losses: +{potential_gain:.4f}")
print()

# Analyze the losses
print("=" * 80)
print("LOSS ANALYSIS")
print("=" * 80)
print()

# Count notes with losses/gains
notes_improved = (note_deltas["iou_super"] > note_deltas["iou_default"]).sum()
notes_worsened = (note_deltas["iou_super"] < note_deltas["iou_default"]).sum()
notes_unchanged = (note_deltas["iou_super"] == note_deltas["iou_default"]).sum()

print(f"Notes breakdown:")
print(f"  Improved:  {notes_improved} ({100*notes_improved/len(note_deltas):.1f}%)")
print(f"  Worsened:  {notes_worsened} ({100*notes_worsened/len(note_deltas):.1f}%)")
print(f"  Unchanged: {notes_unchanged}")
print()

# Analyze concept-level losses
concepts_improved = (merged["iou_super"] > merged["iou_default"]).sum()
concepts_worsened = (merged["iou_super"] < merged["iou_default"]).sum()
concepts_unchanged = (merged["iou_super"] == merged["iou_default"]).sum()

print(f"Concepts breakdown:")
print(f"  Improved:  {concepts_improved} ({100*concepts_improved/len(merged):.1f}%)")
print(f"  Worsened:  {concepts_worsened} ({100*concepts_worsened/len(merged):.1f}%)")
print(f"  Unchanged: {concepts_unchanged} ({100*concepts_unchanged/len(merged):.1f}%)")
print()

# Show worst regressions
print("Top 10 concepts with worst regressions (super vs default):")
merged["iou_delta"] = merged["iou_super"] - merged["iou_default"]
worst_regressions = merged.nsmallest(10, "iou_delta")

# Load concept names if available
try:
    concept_names = pd.read_csv("1st Place/data/interim/flattened_terminology.csv")
    concept_names = concept_names[["concept_id", "concept_name"]].drop_duplicates("concept_id")
    worst_regressions = worst_regressions.merge(concept_names, on="concept_id", how="left")

    for _, row in worst_regressions.iterrows():
        name = row["concept_name"] if pd.notna(row["concept_name"]) else "Unknown"
        print(f"  {row['concept_id']:>10}  Δ={row['iou_delta']:>7.4f}  default={row['iou_default']:.4f} super={row['iou_super']:.4f}")
        print(f"              {name}")
except Exception as e:
    print(f"Could not load concept names: {e}")
    for _, row in worst_regressions.iterrows():
        print(f"  {row['concept_id']:>10}  Δ={row['iou_delta']:>7.4f}  default={row['iou_default']:.4f} super={row['iou_super']:.4f}")
print()

# Calculate loss contribution
# Losses are concepts where super performs worse than default
losses = merged[merged["iou_super"] < merged["iou_default"]].copy()
losses["loss_contribution"] = losses["iou_default"] - losses["iou_super"]

total_loss = losses["loss_contribution"].sum()
num_concepts_with_union = len(merged_with_union)

print(f"Total loss from regressions: {total_loss:.2f}")
print(f"Average loss per concept (over all {num_concepts_with_union} concepts): {total_loss/num_concepts_with_union:.4f}")
print()

# Save oracle results
merged[["concept_id", "iou_default", "iou_super", "iou_oracle", "iou_delta"]].to_csv(
    "outputs/old_challenge_split/oracle_concept_iou.csv", index=False
)
print("Saved oracle concept IoU to: outputs/old_challenge_split/oracle_concept_iou.csv")
