#!/usr/bin/env python3
"""
Split the combined training data back into the old challenge's train/test split.
"""

import pandas as pd
from pathlib import Path

# Paths
data_dir = Path("/workspaces/snomed-ct-entity-linking/data")
original_dir = data_dir / "original-challenge-data"
output_dir = data_dir / "old-challenge-split"

# Create output directory
output_dir.mkdir(exist_ok=True)

print("Loading data files...")

# Load original training data to get the note IDs
original_notes = pd.read_csv(original_dir / "mimic-iv_notes_training_set.csv")
original_train_note_ids = set(original_notes["note_id"])
print(f"Original training set has {len(original_train_note_ids)} notes")

# Load the combined data (old train + old test)
combined_notes = pd.read_csv(data_dir / "train_notes.csv")
combined_annotations = pd.read_csv(data_dir / "train_annotations.csv")
print(f"Combined data has {len(combined_notes)} notes and {len(combined_annotations)} annotations")

# Split notes
train_notes = combined_notes[combined_notes["note_id"].isin(original_train_note_ids)]
test_notes = combined_notes[~combined_notes["note_id"].isin(original_train_note_ids)]

print(f"\nSplit notes:")
print(f"  Train: {len(train_notes)} notes")
print(f"  Test: {len(test_notes)} notes")

# Split annotations
train_annotations = combined_annotations[combined_annotations["note_id"].isin(original_train_note_ids)]
test_annotations = combined_annotations[~combined_annotations["note_id"].isin(original_train_note_ids)]

print(f"\nSplit annotations:")
print(f"  Train: {len(train_annotations)} annotations")
print(f"  Test: {len(test_annotations)} annotations")

# Save the split data
print("\nSaving split data...")
train_notes.to_csv(output_dir / "train_notes.csv", index=False)
train_annotations.to_csv(output_dir / "train_annotations.csv", index=False)
test_notes.to_csv(output_dir / "test_notes.csv", index=False)
test_annotations.to_csv(output_dir / "test_annotations.csv", index=False)

print(f"\nDone! Files saved to {output_dir}")
print("\nFiles created:")
print(f"  - train_notes.csv ({len(train_notes)} rows)")
print(f"  - train_annotations.csv ({len(train_annotations)} rows)")
print(f"  - test_notes.csv ({len(test_notes)} rows)")
print(f"  - test_annotations.csv ({len(test_annotations)} rows)")
