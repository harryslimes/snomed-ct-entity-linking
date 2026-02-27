#!/usr/bin/env python3
"""Error analysis on baseline dictionary predictions.

Categorizes FPs and FNs to identify where IoU is lost and which
fixes would have the most impact:
1. Regex boundary issues (short terms matching inside words)
2. Unscored SNOMED entries causing FPs
3. Section mismatches
4. Missing recall (FN patterns)
"""
from __future__ import annotations

import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from train_dictionary import (
    COMMON_HEADERS, IndexedDict, annotate_with_dict, remove_overlaps,
    CASE_SENSITIVE_DICT, get_pattern, train, get_sections,
    get_header_by_pos, load_concept_types,
)
from runtime_scoring import macro_char_iou, class_char_iou


def predict_with_dict_detailed(
    d: dict, uc_d: dict, notes: pd.DataFrame,
) -> pd.DataFrame:
    """Two-pass prediction, preserving section and dict_entry columns."""
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
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id", "section", "dict_entry"])

    pred = pd.concat(all_preds, ignore_index=True)
    for col in ["start", "end", "concept_id"]:
        pred[col] = pred[col].astype(int)
    return pred


def classify_prediction(
    pred_row, gold_note, text, text_lc,
) -> dict:
    """Classify a single prediction as TP, FP, or partial match."""
    start, end, cid = int(pred_row["start"]), int(pred_row["end"]), int(pred_row["concept_id"])
    mention = pred_row.get("dict_entry", "")
    matched_text = text[start:end]

    # Check if this prediction overlaps with any gold annotation
    note_gold = gold_note
    if note_gold.empty:
        return {"type": "FP", "reason": "no_gold_in_note", "mention": mention,
                "matched_text": matched_text, "concept_id": cid}

    # Find overlapping gold annotations
    overlaps = note_gold[
        (note_gold["start"] < end) & (note_gold["end"] > start)
    ]

    if overlaps.empty:
        # No overlap at all - pure FP
        # Check if the matched text is inside a word (boundary issue)
        boundary_issue = False
        if start > 0 and text_lc[start - 1].isalnum():
            boundary_issue = True
        if end < len(text_lc) and text_lc[end].isalnum():
            boundary_issue = True

        return {"type": "FP", "reason": "boundary" if boundary_issue else "wrong_location",
                "mention": mention, "matched_text": matched_text, "concept_id": cid,
                "boundary_issue": boundary_issue}

    # Check if any overlap has the same concept_id
    concept_match = overlaps[overlaps["concept_id"] == cid]
    if not concept_match.empty:
        # Check span quality
        best = concept_match.iloc[0]
        gold_start, gold_end = int(best["start"]), int(best["end"])
        inter = max(0, min(end, gold_end) - max(start, gold_start))
        union = max(end, gold_end) - min(start, gold_start)
        span_iou = inter / max(union, 1)
        if span_iou > 0.5:
            return {"type": "TP", "reason": "good_match", "mention": mention,
                    "matched_text": matched_text, "concept_id": cid, "span_iou": span_iou}
        else:
            return {"type": "partial", "reason": "span_mismatch", "mention": mention,
                    "matched_text": matched_text, "concept_id": cid, "span_iou": span_iou}
    else:
        # Overlaps with different concept
        return {"type": "FP", "reason": "wrong_concept", "mention": mention,
                "matched_text": matched_text, "concept_id": cid,
                "gold_concepts": overlaps["concept_id"].tolist()}


def is_word_boundary_match(text: str, start: int, end: int) -> bool:
    """Check if match is at word boundaries."""
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def main():
    t0 = time.perf_counter()
    split_dir = REPO_ROOT / "data" / "old-challenge-split"
    super_dict_path = REPO_ROOT / "data" / "interim" / "super_dictionary_full.tsv"
    flat_term_path = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"

    # Load data
    print("Loading data...", flush=True)
    train_notes = pd.read_csv(split_dir / "train_notes.csv")
    train_annotations = pd.read_csv(split_dir / "train_annotations.csv")
    test_notes = pd.read_csv(split_dir / "test_notes.csv")
    test_gold = pd.read_csv(split_dir / "test_annotations.csv")
    for col in ["start", "end", "concept_id"]:
        test_gold[col] = test_gold[col].astype(int)

    print(f"  Train: {len(train_notes)} notes, {len(train_annotations)} annotations")
    print(f"  Test:  {len(test_notes)} notes, {len(test_gold)} annotations")

    # Train dictionary
    print("\nTraining dictionary...", flush=True)
    t1 = time.perf_counter()
    d, uc_d = train(
        train_notes, train_annotations,
        super_dict_path=super_dict_path,
        flat_terminology_path=flat_term_path,
    )
    print(f"Training done in {time.perf_counter() - t1:.1f}s")
    print(f"  Dict: {len(d):,}, UC dict: {len(uc_d):,}")

    # Categorize dict entries by source
    # Entries from training annotations have section != "any" or were in
    # the initial dict. SNOMED entries were added later as ("any", term).
    # We can't perfectly distinguish, but we can count.
    n_any = sum(1 for k in d if k[0] == "any" or isinstance(k[0], tuple))
    n_section = sum(1 for k in d if not (k[0] == "any" or isinstance(k[0], tuple)))
    print(f"  Section-specific entries: {n_section:,}")
    print(f"  'any'/tuple-section entries: {n_any:,}")

    # Predict with detailed info
    print("\nPredicting on test data...", flush=True)
    t2 = time.perf_counter()
    pred = predict_with_dict_detailed(d, uc_d, test_notes)
    print(f"  {len(pred):,} predictions in {time.perf_counter() - t2:.1f}s")

    # Score baseline
    pred_score = pred[["note_id", "start", "end", "concept_id"]].copy()
    baseline_iou, cls = score_predictions(pred_score, test_gold)
    print(f"  Baseline macro char IoU: {baseline_iou:.4f}")

    # -----------------------------------------------------------------------
    # Error Analysis Part 1: Classify every prediction
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS: Classifying predictions")
    print("=" * 80)

    texts = test_notes.set_index("note_id")["text"]
    texts_lc = texts.str.lower()
    gold_by_note = {
        str(nid): df for nid, df in test_gold.groupby("note_id", sort=False)
    }

    classifications = []
    for _, row in pred.iterrows():
        nid = str(row["note_id"])
        text = texts.get(nid, texts.get(int(nid) if nid.isdigit() else nid, ""))
        if isinstance(text, pd.Series):
            text = text.iloc[0] if len(text) > 0 else ""
        text_lc_note = text.lower()
        note_gold = gold_by_note.get(nid, pd.DataFrame())
        c = classify_prediction(row, note_gold, text, text_lc_note)
        c["note_id"] = nid
        c["start"] = int(row["start"])
        c["end"] = int(row["end"])
        c["section"] = row.get("section", "")
        c["n_chars_match"] = int(row["end"]) - int(row["start"])
        c["mention_len"] = len(str(row.get("dict_entry", "")))
        classifications.append(c)

    clf_df = pd.DataFrame(classifications)

    # Summary
    type_counts = clf_df["type"].value_counts()
    print(f"\n  Prediction classification:")
    for t, c in type_counts.items():
        print(f"    {t:<10} {c:>6} ({c / len(clf_df):.1%})")

    reason_counts = clf_df["reason"].value_counts()
    print(f"\n  FP reasons:")
    fp_df = clf_df[clf_df["type"] == "FP"]
    for r, c in fp_df["reason"].value_counts().items():
        print(f"    {r:<25} {c:>6} ({c / len(fp_df):.1%})")

    # -----------------------------------------------------------------------
    # Part 2: Boundary issues
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BOUNDARY ANALYSIS")
    print("=" * 80)

    boundary_fps = clf_df[(clf_df["type"] == "FP") & (clf_df.get("boundary_issue", False) == True)]
    non_boundary_fps = clf_df[(clf_df["type"] == "FP") & (clf_df.get("boundary_issue", False) != True)]

    print(f"\n  FPs with boundary issues: {len(boundary_fps)} ({len(boundary_fps) / max(len(fp_df), 1):.1%} of FPs)")
    print(f"  FPs without boundary issues: {len(non_boundary_fps)}")

    # Calculate IoU impact of boundary FPs
    # What mentions cause boundary FPs most often?
    if not boundary_fps.empty:
        mention_boundary = boundary_fps["mention"].value_counts().head(30)
        print(f"\n  Top mentions causing boundary FPs:")
        for mention, count in mention_boundary.items():
            mention_str = str(mention)
            chars_wasted = boundary_fps[boundary_fps["mention"] == mention]["n_chars_match"].sum()
            print(f"    {count:>4}x  ({chars_wasted:>6} chars)  '{mention_str}' (len={len(mention_str)})")

    # What would happen if we enforced word boundaries on short mentions?
    print("\n\n  FP analysis by mention length:")
    for max_len in [2, 3, 4, 5]:
        short_fps = fp_df[fp_df["mention_len"] <= max_len]
        short_tps = clf_df[(clf_df["type"] == "TP") & (clf_df["mention_len"] <= max_len)]
        print(f"    mention_len <= {max_len}: {len(short_fps)} FPs, {len(short_tps)} TPs")

    # -----------------------------------------------------------------------
    # Part 3: SNOMED entries causing FPs
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SNOMED vs TRAINING ENTRY FP ANALYSIS")
    print("=" * 80)

    # We can approximate: entries with section="any" and not in the training
    # annotations are likely SNOMED-sourced. Let's check which dict entries
    # cause FPs.
    train_mentions = set()
    if "span" in train_annotations.columns:
        train_mentions = set(train_annotations["span"].str.lower().unique())

    fp_from_snomed = []
    fp_from_training = []
    for _, row in fp_df.iterrows():
        mention = str(row.get("mention", "")).lower()
        if mention in train_mentions:
            fp_from_training.append(row)
        else:
            fp_from_snomed.append(row)

    print(f"\n  FPs from training-derived entries: {len(fp_from_training)}")
    print(f"  FPs from SNOMED/other entries:     {len(fp_from_snomed)}")

    if fp_from_snomed:
        snomed_fp_df = pd.DataFrame(fp_from_snomed)
        print(f"\n  Top SNOMED-source FP mentions:")
        for mention, count in snomed_fp_df["mention"].value_counts().head(30).items():
            chars = snomed_fp_df[snomed_fp_df["mention"] == mention]["n_chars_match"].sum()
            # Check if this has boundary issues
            boundary = snomed_fp_df[
                (snomed_fp_df["mention"] == mention) &
                (snomed_fp_df.get("boundary_issue", False) == True)
            ]
            bflag = f" [BOUNDARY:{len(boundary)}]" if len(boundary) > 0 else ""
            print(f"    {count:>4}x ({chars:>5} chars)  '{mention}'{bflag}")

    if fp_from_training:
        train_fp_df = pd.DataFrame(fp_from_training)
        print(f"\n  Top training-source FP mentions:")
        for mention, count in train_fp_df["mention"].value_counts().head(20).items():
            chars = train_fp_df[train_fp_df["mention"] == mention]["n_chars_match"].sum()
            print(f"    {count:>4}x ({chars:>5} chars)  '{mention}'")

    # -----------------------------------------------------------------------
    # Part 4: Char-level FP impact analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CHAR-LEVEL FP IMPACT")
    print("=" * 80)

    total_fp_chars = fp_df["n_chars_match"].sum()
    total_tp_chars = clf_df[clf_df["type"] == "TP"]["n_chars_match"].sum()
    total_pred_chars = pred_score.apply(lambda r: r["end"] - r["start"], axis=1).sum()

    print(f"\n  Total predicted chars: {total_pred_chars:,}")
    print(f"  TP chars:             {total_tp_chars:,} ({total_tp_chars/total_pred_chars:.1%})")
    print(f"  FP chars:             {total_fp_chars:,} ({total_fp_chars/total_pred_chars:.1%})")

    # Break down FP chars by category
    if not fp_df.empty:
        for reason in fp_df["reason"].unique():
            sub = fp_df[fp_df["reason"] == reason]
            chars = sub["n_chars_match"].sum()
            print(f"    {reason:<25} {chars:>8} chars ({chars/total_fp_chars:.1%} of FP chars)")

    # Boundary FPs by mention length
    if not boundary_fps.empty:
        print(f"\n  Boundary FP chars by mention length:")
        for ml in sorted(boundary_fps["mention_len"].unique()):
            sub = boundary_fps[boundary_fps["mention_len"] == ml]
            chars = sub["n_chars_match"].sum()
            print(f"    mention_len={ml}: {len(sub):>4} FPs, {chars:>6} chars")

    # -----------------------------------------------------------------------
    # Part 5: FN analysis - what are we missing?
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FALSE NEGATIVE ANALYSIS")
    print("=" * 80)

    # For each gold annotation, check if it's covered by any prediction
    fn_rows = []
    for _, gold_row in test_gold.iterrows():
        nid = str(gold_row["note_id"])
        gs, ge, gcid = int(gold_row["start"]), int(gold_row["end"]), int(gold_row["concept_id"])

        note_preds = pred[pred["note_id"].astype(str) == nid]
        if note_preds.empty:
            text = texts.get(nid, texts.get(int(nid) if nid.isdigit() else nid, ""))
            if isinstance(text, pd.Series):
                text = text.iloc[0] if len(text) > 0 else ""
            gold_text = text[gs:ge]
            fn_rows.append({
                "note_id": nid, "start": gs, "end": ge, "concept_id": gcid,
                "gold_text": gold_text, "reason": "no_preds_in_note",
            })
            continue

        # Check overlap
        overlaps = note_preds[
            (note_preds["start"].astype(int) < ge) &
            (note_preds["end"].astype(int) > gs) &
            (note_preds["concept_id"].astype(int) == gcid)
        ]
        if overlaps.empty:
            text = texts.get(nid, texts.get(int(nid) if nid.isdigit() else nid, ""))
            if isinstance(text, pd.Series):
                text = text.iloc[0] if len(text) > 0 else ""
            gold_text = text[gs:ge]
            # Check if there's ANY overlap (wrong concept)
            any_overlap = note_preds[
                (note_preds["start"].astype(int) < ge) &
                (note_preds["end"].astype(int) > gs)
            ]
            if not any_overlap.empty:
                reason = "wrong_concept_predicted"
            else:
                # Check if the gold span text is in the dictionary at all
                gold_mention = gold_text.lower().strip()
                gold_mention_norm = " ".join(gold_mention.split())
                in_dict = any(k[1] == gold_mention_norm for k in d)
                in_uc = any(k[1] == gold_text.strip() for k in uc_d)
                if in_dict or in_uc:
                    reason = "in_dict_but_not_matched"
                else:
                    reason = "not_in_dict"
            fn_rows.append({
                "note_id": nid, "start": gs, "end": ge, "concept_id": gcid,
                "gold_text": gold_text, "reason": reason,
            })

    fn_df = pd.DataFrame(fn_rows) if fn_rows else pd.DataFrame()

    if not fn_df.empty:
        print(f"\n  Total gold annotations: {len(test_gold):,}")
        print(f"  Missed (FN): {len(fn_df):,} ({len(fn_df)/len(test_gold):.1%})")
        print(f"\n  FN reasons:")
        for reason, count in fn_df["reason"].value_counts().items():
            chars = fn_df[fn_df["reason"] == reason].apply(lambda r: r["end"] - r["start"], axis=1).sum()
            print(f"    {reason:<30} {count:>6} ({count/len(fn_df):.1%})  {chars:>8} chars")

        # Top missed concepts
        print(f"\n  Top missed concept_ids (most FN chars):")
        fn_chars = fn_df.copy()
        fn_chars["chars"] = fn_chars["end"] - fn_chars["start"]
        concept_fn = fn_chars.groupby("concept_id").agg(
            count=("chars", "count"), total_chars=("chars", "sum")
        ).sort_values("total_chars", ascending=False)

        concept_types = load_concept_types(flat_term_path) if flat_term_path.exists() else None

        for cid, row in concept_fn.head(30).iterrows():
            samples = fn_df[fn_df["concept_id"] == cid]["gold_text"].head(3).tolist()
            ctype = concept_types.get(cid, "?") if concept_types is not None else "?"
            print(f"    {cid:<12} {row['count']:>4} FNs ({row['total_chars']:>6} chars) "
                  f"type={ctype}  e.g. {samples[:2]}")

        # FNs that are "not_in_dict" - these are the recall gap
        not_in_dict = fn_df[fn_df["reason"] == "not_in_dict"]
        if not not_in_dict.empty:
            print(f"\n  'not_in_dict' FN samples (could be added to dictionary):")
            for _, row in not_in_dict.head(30).iterrows():
                print(f"    concept={row['concept_id']:<12} '{row['gold_text']}'")

    # -----------------------------------------------------------------------
    # Part 6: Quantify the potential of each fix direction
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("POTENTIAL IMPACT OF FIX DIRECTIONS")
    print("=" * 80)

    # Direction 1: Fix boundary issues
    if not boundary_fps.empty:
        boundary_chars = boundary_fps["n_chars_match"].sum()
        print(f"\n  1. Fix regex boundaries:")
        print(f"     Boundary FPs: {len(boundary_fps)} predictions, {boundary_chars:,} FP chars")
        print(f"     {boundary_chars/total_pred_chars:.1%} of total predicted chars are boundary FPs")
        # But we'd also lose some TPs - check
        tp_with_boundary = clf_df[clf_df["type"] == "TP"]
        tp_boundary_risk = 0
        for _, row in tp_with_boundary.iterrows():
            nid = str(row["note_id"])
            text = texts.get(nid, texts.get(int(nid) if nid.isdigit() else nid, ""))
            if isinstance(text, pd.Series):
                text = text.iloc[0] if len(text) > 0 else ""
            s, e = int(row["start"]), int(row["end"])
            if not is_word_boundary_match(text.lower(), s, e):
                tp_boundary_risk += 1
        print(f"     TPs that would also be lost: {tp_boundary_risk}")

    # Direction 2: Score SNOMED entries
    if fp_from_snomed:
        snomed_fp_chars = sum(r["n_chars_match"] for r in fp_from_snomed)
        print(f"\n  2. Score/filter SNOMED entries:")
        print(f"     SNOMED FPs: {len(fp_from_snomed)} predictions, {snomed_fp_chars:,} FP chars")
        print(f"     {snomed_fp_chars/total_fp_chars:.1%} of FP chars come from SNOMED entries")

    # Direction 3: Section-aware fixes
    # Check if FPs tend to come from certain sections
    if not fp_df.empty and "section" in fp_df.columns:
        print(f"\n  3. Section-aware filtering:")
        section_fps = fp_df["section"].value_counts().head(10)
        for sec, count in section_fps.items():
            chars = fp_df[fp_df["section"] == sec]["n_chars_match"].sum()
            print(f"     {str(sec):<50} {count:>4} FPs ({chars:>6} chars)")

    print(f"\nTotal analysis time: {time.perf_counter() - t0:.1f}s")


def score_predictions(pred, gold):
    pred_cols = pred[["note_id", "start", "end", "concept_id"]].copy()
    gold_cols = gold[["note_id", "start", "end", "concept_id"]].copy()
    return macro_char_iou(pred_cols, gold_cols), class_char_iou(pred_cols, gold_cols)


if __name__ == "__main__":
    main()
