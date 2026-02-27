"""
Run 2nd place inference pipeline on old-challenge-split test data and evaluate.

Usage:
    cd "2nd Place"
    python run_old_challenge_eval.py
"""
import sys
from pathlib import Path

# Add submission directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "submission"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from loguru import logger

from submission.first_stage import fisrt_stage
from submission.first_stage_postprocess import clean_by_header
from submission.second_stage import second_stage
from submission.second_stage_postprocess import choose_concepts
from submission.static_dict import add_static_dict
from submission.iou import iou_per_class
import numpy as np


def cleanup(df):
    df = df[df.start < df.end]
    df = df[["note_id", "start", "end", "concept_id"]]
    return df


def main():
    ASSETS = Path("data")
    OLD_CHALLENGE = ASSETS / "old_challenge"

    # Paths
    TEST_NOTES_PATH = OLD_CHALLENGE / "test_notes.csv"
    TEST_ANNOTATIONS_PATH = OLD_CHALLENGE / "test_annotations.csv"
    TRAIN_NOTES_PATH = OLD_CHALLENGE / "cutmed_notes.csv"
    TRAIN_ANNOTATIONS_PATH = OLD_CHALLENGE / "cutmed_train_annotations.csv"
    STATIC_DICT_PATH = OLD_CHALLENGE / "most_common_concept.pkl"
    SUBMISSION_PATH = OLD_CHALLENGE / "submission.csv"

    # First stage checkpoints - use the trained models
    FIRST_STAGE_DIR = ASSETS / "first_stage"
    FIRST_STAGE_CHECKPOINTS = [p for p in FIRST_STAGE_DIR.iterdir() if p.is_dir()]
    logger.info(f"First stage checkpoints: {FIRST_STAGE_CHECKPOINTS}")

    # Second stage (SapBERT)
    SECOND_STAGE_CHECKPOINTS = [ASSETS / "second_stage" / "sapbert"]

    # Verify paths
    assert TEST_NOTES_PATH.exists(), f"Missing: {TEST_NOTES_PATH}"
    assert TRAIN_NOTES_PATH.exists(), f"Missing: {TRAIN_NOTES_PATH}"
    assert TRAIN_ANNOTATIONS_PATH.exists(), f"Missing: {TRAIN_ANNOTATIONS_PATH}"
    assert STATIC_DICT_PATH.exists(), f"Missing: {STATIC_DICT_PATH}"
    assert len(FIRST_STAGE_CHECKPOINTS) > 0, f"No checkpoints in {FIRST_STAGE_DIR}"

    # Load test notes
    note_df = pd.read_csv(TEST_NOTES_PATH)
    logger.info(f"Test notes: {note_df.shape}")

    # First stage - NER
    logger.info("Running first stage (NER)...")
    mentions_df = fisrt_stage(FIRST_STAGE_CHECKPOINTS, note_df, prepend_headers=False)
    logger.info(f"First stage mentions: {mentions_df.shape}")

    # First stage postprocess
    logger.info("Running first stage postprocess...")
    mentions_df = clean_by_header(mentions_df, note_df)
    logger.info(f"After header cleaning: {mentions_df.shape}")

    # Second stage - concept linking
    logger.info("Running second stage (concept linking)...")
    sap_checkpoint = SECOND_STAGE_CHECKPOINTS[0]
    tdf = second_stage(mentions_df, sap_checkpoint, topk=1)
    logger.info(f"After second stage: {tdf.shape}")

    # Second stage postprocess
    logger.info("Running second stage postprocess...")
    tdf = choose_concepts(tdf, note_df, TRAIN_NOTES_PATH, TRAIN_ANNOTATIONS_PATH)
    logger.info(f"After concept selection: {tdf.shape}")

    # Static dict
    logger.info("Adding static dict...")
    tdf = add_static_dict(tdf, STATIC_DICT_PATH, note_df)
    logger.info(f"After static dict: {tdf.shape}")

    # Cleanup and save
    tdf = cleanup(tdf)
    logger.info(f"Final predictions: {tdf.shape}")
    tdf.to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Saved to {SUBMISSION_PATH}")

    # Evaluate
    logger.info("Evaluating...")
    target = pd.read_csv(TEST_ANNOTATIONS_PATH)
    # Ensure correct dtypes
    tdf["start"] = tdf["start"].astype(int)
    tdf["end"] = tdf["end"].astype(int)
    tdf["concept_id"] = tdf["concept_id"].astype(int)
    target["start"] = target["start"].astype(int)
    target["end"] = target["end"].astype(int)
    target["concept_id"] = target["concept_id"].astype(int)

    iou_score = iou_per_class(tdf, target, mean=True)
    logger.info(f"Mean IoU: {iou_score:.4f}")

    # Also compute per-class IoU
    ious = iou_per_class(tdf, target, mean=False)
    logger.info(f"Per-concept IoU: {len(ious)} concepts, mean={np.mean(ious):.4f}")

    return iou_score


if __name__ == "__main__":
    main()
