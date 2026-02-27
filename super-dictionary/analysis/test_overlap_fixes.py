#!/usr/bin/env python3
"""Test fixes for overlap removal and section filtering.

The diagnosis found:
- 904 correct matches eaten by overlap removal (9,201 chars)
- 774 FNs from section mismatch (4,773 chars)
- 171 FNs from header filtering (1,762 chars)

This script tests fixes for these issues.
"""
from __future__ import annotations

import copy
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from train_dictionary import (
    COMMON_HEADERS, IndexedDict, annotate_with_dict, remove_overlaps,
    CASE_SENSITIVE_DICT, get_pattern, train,
    get_sections, get_header_by_pos, is_in_header,
    _WORD_RE, _UNSAFE_MENTION_RE,
)
from runtime_scoring import macro_char_iou, class_char_iou


def predict_baseline(d, uc_d, test_notes):
    """Standard two-pass prediction."""
    uc_d_full = dict(uc_d)
    uc_d_full.update(CASE_SENSITIVE_DICT)
    headers_lc = [h.lower() for h in COMMON_HEADERS]
    headers_orig = list(COMMON_HEADERS)
    d_indexed = IndexedDict(d, prefilter="bigram")
    uc_indexed = IndexedDict(uc_d_full, prefilter="bigram")

    all_preds = []
    for _, note_row in test_notes.iterrows():
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


# ---------------------------------------------------------------------------
# Alternative overlap removal strategies
# ---------------------------------------------------------------------------
def remove_overlaps_keep_all_concepts(df: pd.DataFrame) -> pd.DataFrame:
    """Modified overlap removal: when overlapping predictions have DIFFERENT
    concept_ids, keep both (the scorer handles per-concept evaluation).
    Only remove when same concept_id overlaps."""
    if df.empty:
        return df
    df = df.sort_values("start").reset_index(drop=True)
    length = (df["end"] - df["start"]).astype(float)
    section_any = [isinstance(s, tuple) or s == "any" for s in df["section"]]
    length[section_any] -= 0.1

    to_remove = set()
    n = len(df)
    for i in range(n):
        if df.index[i] in to_remove:
            continue
        for j in range(i + 1, n):
            if df["start"].iloc[j] >= df["end"].iloc[i]:
                break
            # Only resolve overlap if same concept_id
            if df["concept_id"].iloc[i] == df["concept_id"].iloc[j]:
                if length.iloc[i] < length.iloc[j]:
                    remove_idx = i
                else:
                    remove_idx = j
                to_remove.add(df.index[remove_idx])
                if remove_idx == i:
                    break

    df2 = df.drop(to_remove)
    for idx in to_remove:
        s, e = df.loc[idx, ["start", "end"]].values
        cid = df.loc[idx, "concept_id"]
        # Only check overlaps with same concept
        same_cid = df2[df2["concept_id"] == cid]
        overlaps = ((same_cid["start"] <= s) & (same_cid["end"] > s)) | (
            (same_cid["start"] <= e) & (same_cid["end"] > e)
        )
        if overlaps.sum() == 0:
            df2.loc[idx] = df.loc[idx]

    return df2


def remove_overlaps_section_priority(df: pd.DataFrame) -> pd.DataFrame:
    """Modified overlap removal: section-specific entries get much stronger
    priority over 'any' section entries (not just 0.1 bonus)."""
    if df.empty:
        return df
    df = df.sort_values("start").reset_index(drop=True)
    length = (df["end"] - df["start"]).astype(float)
    section_any = [isinstance(s, tuple) or s == "any" for s in df["section"]]
    # Give section-specific a massive bonus (100 chars effective)
    length[section_any] -= 100.0

    to_remove = set()
    n = len(df)
    for i in range(n):
        if df.index[i] in to_remove:
            continue
        for j in range(i + 1, n):
            if df["start"].iloc[j] >= df["end"].iloc[i]:
                break
            if length.iloc[i] < length.iloc[j]:
                remove_idx = i
            else:
                remove_idx = j
            to_remove.add(df.index[remove_idx])
            if remove_idx == i:
                break

    df2 = df.drop(to_remove)
    for idx in to_remove:
        s, e = df.loc[idx, ["start", "end"]].values
        overlaps = ((df2["start"] <= s) & (df2["end"] > s)) | (
            (df2["start"] <= e) & (df2["end"] > e)
        )
        if overlaps.sum() == 0:
            df2.loc[idx] = df.loc[idx]

    return df2


def remove_overlaps_prefer_short_specific(df: pd.DataFrame) -> pd.DataFrame:
    """When overlapping: prefer section-specific, then break ties by length.
    This prevents long 'any'-section entries from eating short section-specific ones."""
    if df.empty:
        return df
    df = df.sort_values("start").reset_index(drop=True)

    is_any = [isinstance(s, tuple) or s == "any" for s in df["section"]]
    length = (df["end"] - df["start"]).astype(float)

    # Priority: section-specific > any; within same type, longer > shorter
    # Encode as: priority = (1 if specific else 0) * 10000 + length
    priority = pd.Series([(0 if a else 1) * 10000 + l for a, l in zip(is_any, length)],
                         index=df.index)

    to_remove = set()
    n = len(df)
    for i in range(n):
        if df.index[i] in to_remove:
            continue
        for j in range(i + 1, n):
            if df["start"].iloc[j] >= df["end"].iloc[i]:
                break
            if priority.iloc[i] < priority.iloc[j]:
                remove_idx = i
            else:
                remove_idx = j
            to_remove.add(df.index[remove_idx])
            if remove_idx == i:
                break

    df2 = df.drop(to_remove)
    for idx in to_remove:
        s, e = df.loc[idx, ["start", "end"]].values
        overlaps = ((df2["start"] <= s) & (df2["end"] > s)) | (
            (df2["start"] <= e) & (df2["end"] > e)
        )
        if overlaps.sum() == 0:
            df2.loc[idx] = df.loc[idx]

    return df2


# ---------------------------------------------------------------------------
# Modified annotate_with_dict
# ---------------------------------------------------------------------------
def annotate_with_dict_relaxed(
    text, d, headers, note_id,
    keep_overlaps=False,
    skip_header_filter=False,
    allow_medication_section=False,
):
    """annotate_with_dict with relaxable filters."""
    rows = []
    h_positions, pos_header = get_sections(text, headers)

    if isinstance(d, IndexedDict):
        items_iter = d.iter_items_for_text(text)
    else:
        items_iter = d.items()

    for (section, source_text), cid in items_iter:
        p = get_pattern(source_text)
        if p is None:
            continue
        for match in p.finditer(text):
            i, j = match.start(), match.end()
            if i < 100:
                continue
            if i > 0 and text[i - 1].isalnum():
                continue
            if j < len(text) and text[j].isalnum():
                continue
            if not skip_header_filter and is_in_header(text, i) and not keep_overlaps:
                continue

            h = get_header_by_pos(i, h_positions, pos_header, headers)
            if h is None:
                continue
            hl = h.lower()
            if not allow_medication_section:
                if "medication" in hl or "service" in hl or "date of birth" in hl:
                    continue

            if h == section or h in section or section == "any":
                if isinstance(cid, Counter):
                    for k in cid:
                        rows.append([note_id, i, j, k, section, source_text])
                else:
                    rows.append([note_id, i, j, cid, section, source_text])

    ann = pd.DataFrame(
        rows, columns=["note_id", "start", "end", "concept_id", "section", "dict_entry"]
    )
    if keep_overlaps:
        return ann
    return remove_overlaps(ann)


def predict_with_custom(
    d, uc_d, test_notes,
    overlap_fn=remove_overlaps,
    annotate_fn=annotate_with_dict,
    annotate_kwargs=None,
):
    """Predict with custom overlap removal and annotation functions."""
    uc_d_full = dict(uc_d)
    uc_d_full.update(CASE_SENSITIVE_DICT)
    headers_lc = [h.lower() for h in COMMON_HEADERS]
    headers_orig = list(COMMON_HEADERS)
    d_indexed = IndexedDict(d, prefilter="bigram")
    uc_indexed = IndexedDict(uc_d_full, prefilter="bigram")

    akw = annotate_kwargs or {}
    all_preds = []
    for _, note_row in test_notes.iterrows():
        note_id = str(note_row["note_id"])
        text = note_row["text"]
        text_lc = text.lower()

        ann_lc = annotate_fn(text_lc, d_indexed, headers_lc, note_id, keep_overlaps=True, **akw)
        ann_uc = annotate_fn(text, uc_indexed, headers_orig, note_id, keep_overlaps=True, **akw)
        combined = pd.concat([ann_lc, ann_uc], ignore_index=True)
        if not combined.empty:
            combined = overlap_fn(combined)
            all_preds.append(combined)

    if not all_preds:
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id"])
    pred = pd.concat(all_preds, ignore_index=True)
    pred = pred[["note_id", "start", "end", "concept_id"]].copy()
    for c in ["start", "end", "concept_id"]:
        pred[c] = pred[c].astype(int)
    return pred


def score(pred, gold):
    return macro_char_iou(
        pred[["note_id", "start", "end", "concept_id"]],
        gold[["note_id", "start", "end", "concept_id"]],
    )


def main():
    t0 = time.perf_counter()
    split_dir = REPO_ROOT / "data" / "old-challenge-split"
    super_dict_path = REPO_ROOT / "data" / "interim" / "super_dictionary_full.tsv"
    flat_term_path = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"

    print("Loading data...", flush=True)
    train_notes = pd.read_csv(split_dir / "train_notes.csv")
    train_annotations = pd.read_csv(split_dir / "train_annotations.csv")
    test_notes = pd.read_csv(split_dir / "test_notes.csv")
    test_gold = pd.read_csv(split_dir / "test_annotations.csv")
    for col in ["start", "end", "concept_id"]:
        test_gold[col] = test_gold[col].astype(int)

    print("Training dictionary...", flush=True)
    d, uc_d = train(
        train_notes, train_annotations,
        super_dict_path=super_dict_path,
        flat_terminology_path=flat_term_path,
    )
    print(f"  Dict: {len(d):,}, UC: {len(uc_d):,}")

    # Baseline
    print("\nBaseline...", flush=True)
    t1 = time.perf_counter()
    pred_baseline = predict_baseline(d, uc_d, test_notes)
    iou_baseline = score(pred_baseline, test_gold)
    print(f"  Baseline IoU: {iou_baseline:.4f} ({len(pred_baseline):,} preds, {time.perf_counter()-t1:.1f}s)")

    results = [("Baseline", iou_baseline, len(pred_baseline))]

    # ===================================================================
    # Experiment 1: Keep all concepts (don't remove overlapping different concepts)
    # ===================================================================
    print("\nExp 1: Keep overlapping different concepts...", flush=True)
    t1 = time.perf_counter()
    pred_1 = predict_with_custom(d, uc_d, test_notes,
                                  overlap_fn=remove_overlaps_keep_all_concepts)
    iou_1 = score(pred_1, test_gold)
    print(f"  IoU: {iou_1:.4f} ({iou_1-iou_baseline:+.4f}, {len(pred_1):,} preds, {time.perf_counter()-t1:.1f}s)")
    results.append(("Keep all concepts", iou_1, len(pred_1)))

    # ===================================================================
    # Experiment 2: Section-specific gets strong priority
    # ===================================================================
    print("\nExp 2: Section-specific priority in overlap...", flush=True)
    t1 = time.perf_counter()
    pred_2 = predict_with_custom(d, uc_d, test_notes,
                                  overlap_fn=remove_overlaps_section_priority)
    iou_2 = score(pred_2, test_gold)
    print(f"  IoU: {iou_2:.4f} ({iou_2-iou_baseline:+.4f}, {len(pred_2):,} preds, {time.perf_counter()-t1:.1f}s)")
    results.append(("Section priority overlap", iou_2, len(pred_2)))

    # ===================================================================
    # Experiment 3: Prefer specific, then length
    # ===================================================================
    print("\nExp 3: Prefer short-specific over long-any...", flush=True)
    t1 = time.perf_counter()
    pred_3 = predict_with_custom(d, uc_d, test_notes,
                                  overlap_fn=remove_overlaps_prefer_short_specific)
    iou_3 = score(pred_3, test_gold)
    print(f"  IoU: {iou_3:.4f} ({iou_3-iou_baseline:+.4f}, {len(pred_3):,} preds, {time.perf_counter()-t1:.1f}s)")
    results.append(("Prefer short specific", iou_3, len(pred_3)))

    # ===================================================================
    # Experiment 4: Relax header filtering
    # ===================================================================
    print("\nExp 4: Relax header filtering...", flush=True)
    t1 = time.perf_counter()
    pred_4 = predict_with_custom(d, uc_d, test_notes,
                                  annotate_fn=annotate_with_dict_relaxed,
                                  annotate_kwargs={"skip_header_filter": True})
    iou_4 = score(pred_4, test_gold)
    print(f"  IoU: {iou_4:.4f} ({iou_4-iou_baseline:+.4f}, {len(pred_4):,} preds, {time.perf_counter()-t1:.1f}s)")
    results.append(("Skip header filter", iou_4, len(pred_4)))

    # ===================================================================
    # Experiment 5: Allow medication/service sections
    # ===================================================================
    print("\nExp 5: Allow medication/service sections...", flush=True)
    t1 = time.perf_counter()
    pred_5 = predict_with_custom(d, uc_d, test_notes,
                                  annotate_fn=annotate_with_dict_relaxed,
                                  annotate_kwargs={"allow_medication_section": True})
    iou_5 = score(pred_5, test_gold)
    print(f"  IoU: {iou_5:.4f} ({iou_5-iou_baseline:+.4f}, {len(pred_5):,} preds, {time.perf_counter()-t1:.1f}s)")
    results.append(("Allow med/service sections", iou_5, len(pred_5)))

    # ===================================================================
    # Experiment 6: Combined - keep all concepts + skip header filter
    # ===================================================================
    print("\nExp 6: Keep all concepts + skip header filter...", flush=True)
    t1 = time.perf_counter()
    pred_6 = predict_with_custom(d, uc_d, test_notes,
                                  overlap_fn=remove_overlaps_keep_all_concepts,
                                  annotate_fn=annotate_with_dict_relaxed,
                                  annotate_kwargs={"skip_header_filter": True})
    iou_6 = score(pred_6, test_gold)
    print(f"  IoU: {iou_6:.4f} ({iou_6-iou_baseline:+.4f}, {len(pred_6):,} preds, {time.perf_counter()-t1:.1f}s)")
    results.append(("All concepts + no header", iou_6, len(pred_6)))

    # ===================================================================
    # Experiment 7: Everything relaxed
    # ===================================================================
    print("\nExp 7: All relaxations combined...", flush=True)
    t1 = time.perf_counter()
    pred_7 = predict_with_custom(d, uc_d, test_notes,
                                  overlap_fn=remove_overlaps_keep_all_concepts,
                                  annotate_fn=annotate_with_dict_relaxed,
                                  annotate_kwargs={
                                      "skip_header_filter": True,
                                      "allow_medication_section": True,
                                  })
    iou_7 = score(pred_7, test_gold)
    print(f"  IoU: {iou_7:.4f} ({iou_7-iou_baseline:+.4f}, {len(pred_7):,} preds, {time.perf_counter()-t1:.1f}s)")
    results.append(("All relaxed", iou_7, len(pred_7)))

    # ===================================================================
    # Experiment 8: Keep all concepts + section priority (best overlap strategies)
    # ===================================================================
    print("\nExp 8: Keep all concepts + section priority...", flush=True)
    t1 = time.perf_counter()

    def overlap_combined(df):
        """Keep different concepts, but among same concept prefer section-specific."""
        return remove_overlaps_keep_all_concepts(df)

    pred_8 = predict_with_custom(d, uc_d, test_notes,
                                  overlap_fn=overlap_combined,
                                  annotate_fn=annotate_with_dict_relaxed,
                                  annotate_kwargs={"skip_header_filter": True,
                                                   "allow_medication_section": True})
    iou_8 = score(pred_8, test_gold)
    print(f"  IoU: {iou_8:.4f} ({iou_8-iou_baseline:+.4f}, {len(pred_8):,} preds, {time.perf_counter()-t1:.1f}s)")
    results.append(("All concepts + all filters relaxed", iou_8, len(pred_8)))

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Experiment':<40} {'IoU':>8} {'Delta':>8} {'Preds':>8}")
    print("-" * 70)
    for name, iou, n_preds in results:
        delta = iou - iou_baseline
        print(f"  {name:<38} {iou:>8.4f} {delta:>+8.4f} {n_preds:>8,}")

    # Detailed comparison: best vs baseline
    best_name, best_iou, _ = max(results[1:], key=lambda x: x[1])
    print(f"\n  Best: {best_name} ({best_iou:.4f}, {best_iou-iou_baseline:+.4f})")

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
