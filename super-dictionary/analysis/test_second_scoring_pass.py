#!/usr/bin/env python3
"""Test the impact of a second scoring/removal pass after SNOMED additions.

Currently the pipeline:
1. Build dict from training annotations (~23K entries)
2. Score entries against training data
3. Remove bad keys (~2.3K removed)
4. Add SNOMED synonyms (~255K entries, UNSCORED)
5. Add word replacements + permutations (~450K entries, UNSCORED)

This script tests: what if we re-score the full dictionary (step 6.5)
after all additions? This catches:
- SNOMED entries that generate FPs (e.g., "to the")
- Permutations/replacements that are bad
- Entries whose precision changed after adding more entries (overlap effects)
"""
from __future__ import annotations

import copy
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from train_dictionary import (
    COMMON_HEADERS, BLACKLIST_THRESH, INTERNAL_BLACKLIST,
    IndexedDict, annotate_with_dict, remove_overlaps,
    CASE_SENSITIVE_DICT, get_pattern, train,
    build_dict_from_annotations, score_dict,
    count_correct, is_naive_key_remove, remove_bad_keys,
    load_snomed_synonyms_from_super_dict, process_term,
    load_concept_types, cond_update,
    extract_uppercase_mentions, get_word_replacements,
    get_permutations, get_allowed_sections,
    limit_any_to_allowed_sections,
    CORRECT_FRAC_FOR_DICT, CORRECT_FRAC_FOR_ANY,
)
from runtime_scoring import macro_char_iou, class_char_iou


def predict_with_dict(d, uc_d, notes):
    """Two-pass KIRI-style prediction."""
    uc_d_full = dict(uc_d)
    uc_d_full.update(CASE_SENSITIVE_DICT)

    headers_lc = [h.lower() for h in COMMON_HEADERS]
    headers_orig = list(COMMON_HEADERS)
    d_indexed = IndexedDict(d, prefilter="bigram")
    uc_indexed = IndexedDict(uc_d_full, prefilter="bigram")

    all_preds = []
    for _, note_row in notes.iterrows():
        note_id = str(note_row["note_id"])
        text = note_row["text"]
        text_lc = text.lower()
        ann_lc = annotate_with_dict(text_lc, d_indexed, headers_lc, note_id)
        ann_uc = annotate_with_dict(text, uc_indexed, headers_orig, note_id)
        combined = pd.concat([ann_lc, ann_uc], ignore_index=True)
        if not combined.empty:
            combined = remove_overlaps(combined)
            all_preds.append(combined)

    if not all_preds:
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id"])
    pred = pd.concat(all_preds, ignore_index=True)
    pred = pred[["note_id", "start", "end", "concept_id"]].copy()
    for c in ["start", "end", "concept_id"]:
        pred[c] = pred[c].astype(int)
    return pred


def score_full_dict(d, texts_lc, annotations, headers):
    """Score the entire dictionary against training data.

    Returns scores_by_note dict: key -> [1, -1, 1, ...]
    """
    print("    Building index...", flush=True)
    d_indexed = IndexedDict(d, prefilter="bigram")

    refs = {}
    for nid, df in annotations[annotations["note_id"].isin(texts_lc.index)].groupby("note_id", sort=False):
        refs[str(nid)] = df[["start", "end", "concept_id", "source"]].copy()

    scores = {}
    done = 0
    n_notes = len(texts_lc)
    for nid in texts_lc.index:
        ref = refs.get(str(nid))
        if ref is None:
            continue
        t_scores, _ = score_dict(texts_lc[nid], ref, d, d_indexed, headers)
        for k in t_scores:
            scores.setdefault(k, []).extend(t_scores[k])
            for s in [1, -1]:
                if s in t_scores[k]:
                    scores.setdefault(k, []).append(s)
        done += 1
        if done % 50 == 0:
            print(f"      {done}/{n_notes} notes scored...", flush=True)

    return scores


def main():
    t0 = time.perf_counter()
    split_dir = REPO_ROOT / "data" / "old-challenge-split"
    super_dict_path = REPO_ROOT / "data" / "interim" / "super_dictionary_full.tsv"
    flat_term_path = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"

    # Load data
    print("Loading data...", flush=True)
    train_notes_df = pd.read_csv(split_dir / "train_notes.csv")
    train_annotations_df = pd.read_csv(split_dir / "train_annotations.csv")
    test_notes = pd.read_csv(split_dir / "test_notes.csv")
    test_gold = pd.read_csv(split_dir / "test_annotations.csv")
    for col in ["start", "end", "concept_id"]:
        test_gold[col] = test_gold[col].astype(int)

    print(f"  Train: {len(train_notes_df)} notes, {len(train_annotations_df)} annotations")
    print(f"  Test:  {len(test_notes)} notes, {len(test_gold)} annotations")

    # ===================================================================
    # Experiment 1: Baseline (standard pipeline)
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Baseline (standard pipeline)")
    print("=" * 80)
    t1 = time.perf_counter()
    d_baseline, uc_baseline = train(
        train_notes_df, train_annotations_df,
        super_dict_path=super_dict_path,
        flat_terminology_path=flat_term_path,
    )
    print(f"  Dict: {len(d_baseline):,}, UC: {len(uc_baseline):,}, time: {time.perf_counter()-t1:.1f}s")

    pred_baseline = predict_with_dict(d_baseline, uc_baseline, test_notes)
    iou_baseline = macro_char_iou(
        pred_baseline[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"  Baseline IoU: {iou_baseline:.4f}  ({len(pred_baseline):,} predictions)")

    # ===================================================================
    # Prepare shared state for experiments
    # ===================================================================
    texts = train_notes_df.set_index("note_id")["text"]
    texts_lc = texts.str.lower()
    headers = [h.lower() for h in COMMON_HEADERS]

    annotations = train_annotations_df.copy()
    if "span" in annotations.columns:
        annotations["source orig"] = annotations["span"]
        annotations["source"] = annotations["span"].str.lower()

    # ===================================================================
    # Experiment 2: Second scoring pass on full dictionary
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Second scoring pass on full dictionary")
    print("=" * 80)

    d_exp2 = copy.copy(d_baseline)
    print(f"  Scoring full dictionary ({len(d_exp2):,} entries)...")
    t2 = time.perf_counter()
    scores_full = score_full_dict(d_exp2, texts_lc, annotations, headers)
    print(f"  Scoring done in {time.perf_counter()-t2:.1f}s, scored {len(scores_full):,} keys")

    # Remove bad keys using same thresholds
    bad_keys_2 = remove_bad_keys(d_exp2, scores_full)
    print(f"  After second removal: {len(d_exp2):,} entries (removed {len(bad_keys_2)})")

    pred_exp2 = predict_with_dict(d_exp2, uc_baseline, test_notes)
    iou_exp2 = macro_char_iou(
        pred_exp2[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"  IoU: {iou_exp2:.4f} (delta: {iou_exp2 - iou_baseline:+.4f}, {len(pred_exp2):,} preds)")

    # Show what was removed
    removed_mentions = Counter()
    for k in bad_keys_2:
        removed_mentions[k[1]] += 1
    print(f"\n  Top removed mentions in second pass:")
    for mention, count in removed_mentions.most_common(30):
        print(f"    {count:>3} entries  '{mention}'")

    # ===================================================================
    # Experiment 3: Stricter second pass (double threshold)
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Stricter second scoring pass (double threshold)")
    print("=" * 80)

    d_exp3 = copy.copy(d_baseline)
    # Use double thresholds for the second pass
    bad_keys_3 = []
    for k in scores_full:
        if is_naive_key_remove(count_correct(scores_full[k]), k, double_thr=True):
            bad_keys_3.append(k)

    n_in_d = len(set(bad_keys_3) & set(d_exp3.keys()))
    print(f"  Bad keys (strict): {len(bad_keys_3)} ({n_in_d} in dict)")
    for k in bad_keys_3:
        d_exp3.pop(k, None)
    print(f"  After strict removal: {len(d_exp3):,} entries")

    pred_exp3 = predict_with_dict(d_exp3, uc_baseline, test_notes)
    iou_exp3 = macro_char_iou(
        pred_exp3[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"  IoU: {iou_exp3:.4f} (delta: {iou_exp3 - iou_baseline:+.4f}, {len(pred_exp3):,} preds)")

    # ===================================================================
    # Experiment 4: Second pass ONLY on new entries (not re-scoring originals)
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Score only SNOMED+replacement entries (preserve originals)")
    print("=" * 80)

    # We need to know which keys were in the dict before SNOMED additions.
    # Re-train to capture the pre-SNOMED dict state.
    # Actually, we can approximate: entries that were in training annotations
    # are "original", everything else is "added".
    train_mentions = set()
    if "span" in annotations.columns:
        train_mentions = set(annotations["source"].unique())

    d_exp4 = copy.copy(d_baseline)
    # Only remove scored entries that are NOT from training data
    removed_exp4 = 0
    for k in list(scores_full.keys()):
        mention = k[1] if isinstance(k[1], str) else str(k[1])
        if mention in train_mentions:
            continue  # Preserve training-derived entries
        if is_naive_key_remove(count_correct(scores_full[k]), k):
            d_exp4.pop(k, None)
            removed_exp4 += 1

    print(f"  Removed {removed_exp4} non-training entries")
    print(f"  After selective removal: {len(d_exp4):,} entries")

    pred_exp4 = predict_with_dict(d_exp4, uc_baseline, test_notes)
    iou_exp4 = macro_char_iou(
        pred_exp4[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"  IoU: {iou_exp4:.4f} (delta: {iou_exp4 - iou_baseline:+.4f}, {len(pred_exp4):,} preds)")

    # ===================================================================
    # Experiment 5: Progressively tighter thresholds for new entries
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Sweep precision thresholds for non-training entries")
    print("=" * 80)

    for min_precision in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        d_exp5 = copy.copy(d_baseline)
        removed_5 = 0
        for k in scores_full:
            mention = k[1] if isinstance(k[1], str) else str(k[1])
            if mention in train_mentions:
                continue
            correct, incorrect = count_correct(scores_full[k])
            total = correct + incorrect
            if total == 0:
                continue
            precision = correct / total
            if precision < min_precision:
                d_exp5.pop(k, None)
                removed_5 += 1

        pred_5 = predict_with_dict(d_exp5, uc_baseline, test_notes)
        iou_5 = macro_char_iou(
            pred_5[["note_id", "start", "end", "concept_id"]],
            test_gold[["note_id", "start", "end", "concept_id"]],
        )
        print(f"  min_prec={min_precision:.1f}: removed={removed_5:>5}, "
              f"dict={len(d_exp5):,}, IoU={iou_5:.4f} ({iou_5-iou_baseline:+.4f}), "
              f"preds={len(pred_5):,}")

    # ===================================================================
    # Experiment 6: Same sweep but for ALL entries (including training)
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: Sweep precision thresholds for ALL scored entries")
    print("=" * 80)

    for min_precision in [0.0, 0.1, 0.2, 0.3, 0.5]:
        d_exp6 = copy.copy(d_baseline)
        removed_6 = 0
        for k in scores_full:
            correct, incorrect = count_correct(scores_full[k])
            total = correct + incorrect
            if total == 0:
                continue
            precision = correct / total
            if precision < min_precision:
                d_exp6.pop(k, None)
                removed_6 += 1

        pred_6 = predict_with_dict(d_exp6, uc_baseline, test_notes)
        iou_6 = macro_char_iou(
            pred_6[["note_id", "start", "end", "concept_id"]],
            test_gold[["note_id", "start", "end", "concept_id"]],
        )
        print(f"  min_prec={min_precision:.1f}: removed={removed_6:>5}, "
              f"dict={len(d_exp6):,}, IoU={iou_6:.4f} ({iou_6-iou_baseline:+.4f}), "
              f"preds={len(pred_6):,}")

    # ===================================================================
    # Experiment 7: Remove entries with zero precision AND high fire count
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: Remove zero-precision entries with high fire count")
    print("=" * 80)

    for min_fires in [1, 2, 3, 5, 10]:
        d_exp7 = copy.copy(d_baseline)
        removed_7 = 0
        for k in scores_full:
            correct, incorrect = count_correct(scores_full[k])
            if correct == 0 and incorrect >= min_fires:
                d_exp7.pop(k, None)
                removed_7 += 1

        pred_7 = predict_with_dict(d_exp7, uc_baseline, test_notes)
        iou_7 = macro_char_iou(
            pred_7[["note_id", "start", "end", "concept_id"]],
            test_gold[["note_id", "start", "end", "concept_id"]],
        )
        print(f"  zero_prec+fires>={min_fires}: removed={removed_7:>5}, "
              f"dict={len(d_exp7):,}, IoU={iou_7:.4f} ({iou_7-iou_baseline:+.4f}), "
              f"preds={len(pred_7):,}")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Baseline:                   IoU={iou_baseline:.4f}")
    print(f"  Second scoring pass:        IoU={iou_exp2:.4f} ({iou_exp2-iou_baseline:+.4f})")
    print(f"  Strict second pass:         IoU={iou_exp3:.4f} ({iou_exp3-iou_baseline:+.4f})")
    print(f"  Score new entries only:     IoU={iou_exp4:.4f} ({iou_exp4-iou_baseline:+.4f})")

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
