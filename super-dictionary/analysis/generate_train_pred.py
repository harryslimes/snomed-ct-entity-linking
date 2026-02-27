#!/usr/bin/env python3
"""Generate KIRI super dictionary predictions for training set."""
from __future__ import annotations

import os
import sys
from pathlib import Path

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
    # Set super dictionary path
    super_synonyms = "1st Place/data/interim/flattened_terminology_syn_super.csv"
    os.environ["KIRI_SYNONYMS_PATH"] = str(super_synonyms)

    # Enable linguistic rules (matching test setup)
    os.environ["KIRI_LINGUISTIC_RULES"] = "1"
    os.environ["KIRI_STOPWORD_TRANSPARENT"] = "1"

    # Load data
    print("Loading training data...")
    split_dir = Path("data/old-challenge-split")
    train_texts = pd.read_csv(split_dir / "train_notes.csv").set_index("note_id")["text"]
    train_annotations = pd.read_csv(split_dir / "train_annotations.csv")

    # Rename 'span' to 'source' for KIRI compatibility
    if "span" in train_annotations.columns and "source" not in train_annotations.columns:
        train_annotations = train_annotations.rename(columns={"span": "source"})

    print(f"Train: {len(train_texts)} notes, {len(train_annotations)} annotations")

    # Train KIRI matcher
    print("Training KIRI matcher...")
    d, uc_d = train(train_texts, train_annotations, common_headers, "kiri_super_train")

    # Generate predictions on training set
    print("Generating predictions on training set...")
    pred = make_predictions(train_texts, d, uc_d, run_name="kiri_super_train")

    # Save predictions
    output_dir = Path("outputs/old_challenge_split")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "kiri_super_train_pred.csv"

    pred.to_csv(output_file, index=False)
    print(f"\nSaved {len(pred)} predictions to {output_file}")
    print(f"Unique notes: {pred['note_id'].nunique()}")
    print(f"Unique concepts: {pred['concept_id'].nunique()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
