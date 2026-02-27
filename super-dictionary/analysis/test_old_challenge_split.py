#!/usr/bin/env python3
"""Test super dictionary on the old challenge split."""
from __future__ import annotations

import argparse
import os
import sys
import time
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
from note_scoring import iou_per_note  # noqa: E402
from scripts.super_dictionary.runtime_scoring import class_char_iou  # noqa: E402


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Test super dictionary on old challenge split."
    )
    parser.add_argument(
        "--split-dir",
        default="data/old-challenge-split",
        help="Directory containing train/test split files",
    )
    parser.add_argument(
        "--default-synonyms",
        default="1st Place/data/interim/flattened_terminology_syn_snomed+omop_v5.csv",
        help="Path to default synonyms file",
    )
    parser.add_argument(
        "--super-synonyms",
        default="1st Place/data/interim/flattened_terminology_syn_super.csv",
        help="Path to super dictionary synonyms file",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/old_challenge_split",
        help="Directory to write results",
    )
    parser.add_argument(
        "--super-rules",
        action="store_true",
        help="Enable linguistic rules for super dictionary",
    )
    parser.add_argument(
        "--skip-default",
        action="store_true",
        help="Skip running default dictionary (only run super)",
    )
    args = parser.parse_args(argv)

    split_dir = Path(args.split_dir)

    # Load train and test data
    print("Loading train and test data...", flush=True)
    train_texts = pd.read_csv(split_dir / "train_notes.csv").set_index("note_id")["text"]
    test_texts = pd.read_csv(split_dir / "test_notes.csv").set_index("note_id")["text"]
    train_annotations = pd.read_csv(split_dir / "train_annotations.csv")
    test_annotations = pd.read_csv(split_dir / "test_annotations.csv")

    # Rename 'span' to 'source' if needed (for compatibility with KIRI)
    if "span" in train_annotations.columns and "source" not in train_annotations.columns:
        train_annotations = train_annotations.rename(columns={"span": "source"})
    if "span" in test_annotations.columns and "source" not in test_annotations.columns:
        test_annotations = test_annotations.rename(columns={"span": "source"})

    print(f"Train: {len(train_texts)} notes, {len(train_annotations)} annotations")
    print(f"Test: {len(test_texts)} notes, {len(test_annotations)} annotations")

    train_ids = train_texts.index.tolist()
    test_ids = test_texts.index.tolist()

    # Combine texts for processing
    all_texts = pd.concat([train_texts, test_texts])

    from mimic_common import common_headers  # noqa: E402

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _set_rule_env(*, linguistic: bool, stopword: bool):
        if linguistic:
            os.environ["KIRI_LINGUISTIC_RULES"] = "1"
        else:
            os.environ.pop("KIRI_LINGUISTIC_RULES", None)
        if stopword:
            os.environ["KIRI_STOPWORD_TRANSPARENT"] = "1"
        else:
            os.environ.pop("KIRI_STOPWORD_TRANSPARENT", None)

    def _run_once(synonyms_path, run_name, enable_rules=False):
        if synonyms_path:
            os.environ["KIRI_SYNONYMS_PATH"] = str(synonyms_path)
        else:
            os.environ.pop("KIRI_SYNONYMS_PATH", None)

        _set_rule_env(linguistic=enable_rules, stopword=enable_rules)

        t0 = time.perf_counter()
        print(f"[{run_name}] Training on {len(train_ids)} notes...", flush=True)
        d, uc_d = train(train_texts, train_annotations, common_headers, run_name)
        t1 = time.perf_counter()
        print(f"[{run_name}] Training done in {t1 - t0:.1f}s", flush=True)

        print(f"[{run_name}] Predicting on {len(test_ids)} notes...", flush=True)
        pred = make_predictions(test_texts, d, uc_d, run_name=run_name)
        t2 = time.perf_counter()
        print(f"[{run_name}] Prediction done in {t2 - t1:.1f}s (total {t2 - t0:.1f}s)", flush=True)

        return pred

    def _score_and_write(label: str, pred: pd.DataFrame):
        pred_out = out_dir / f"{label}_pred.csv"
        pred.to_csv(pred_out, index=False)

        cls = class_char_iou(
            pred[["note_id", "start", "end", "concept_id"]],
            test_annotations[["note_id", "start", "end", "concept_id"]],
        )
        cls_out = out_dir / f"{label}_class_iou.csv"
        cls.to_csv(cls_out, index=False)

        runtime_iou = float(cls.loc[cls["union"] > 0, "iou"].mean())
        note_iou = float(
            np.mean(
                list(
                    iou_per_note(
                        pred[["note_id", "concept_id"]],
                        test_annotations[["note_id", "concept_id"]],
                    ).values()
                )
            )
        )
        return note_iou, runtime_iou, pred_out, cls_out

    results = []

    # Run default dictionary
    if not args.skip_default:
        print("\n" + "="*80)
        print("Running DEFAULT dictionary...")
        print("="*80)
        default_pred = _run_once(args.default_synonyms, "kiri_default", enable_rules=False)
        print("Scoring default dictionary...", flush=True)
        default_note_iou, default_runtime_iou, default_pred_out, default_cls_out = _score_and_write(
            "kiri_default", default_pred
        )
        results.append(("Default", default_note_iou, default_runtime_iou, default_pred_out, default_cls_out))

    # Run super dictionary
    print("\n" + "="*80)
    print("Running SUPER dictionary...")
    print("="*80)
    super_pred = _run_once(args.super_synonyms, "kiri_super", enable_rules=args.super_rules)
    print("Scoring super dictionary...", flush=True)
    super_note_iou, super_runtime_iou, super_pred_out, super_cls_out = _score_and_write(
        "kiri_super", super_pred
    )
    results.append(("Super", super_note_iou, super_runtime_iou, super_pred_out, super_cls_out))

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Train: {len(train_ids)} notes")
    print(f"Test: {len(test_ids)} notes")
    print()
    print(f"{'Dictionary':<20} {'Note-level IoU':<20} {'Char-level IoU (mean)':<25}")
    print("-" * 80)
    for name, note_iou, runtime_iou, _, _ in results:
        print(f"{name:<20} {note_iou:<20.4f} {runtime_iou:<25.4f}")

    print("\nOutput files:")
    for name, _, _, pred_out, cls_out in results:
        print(f"{name}:")
        print(f"  Predictions: {pred_out}")
        print(f"  Class IoU:   {cls_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
