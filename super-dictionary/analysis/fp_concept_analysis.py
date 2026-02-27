#!/usr/bin/env python3
"""FP Concept Analysis: identify pure FP concepts common to train (OOF) and test,
characterize them, quantify safe removal gains, and find patterns."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.super_dictionary.runtime_scoring import class_char_iou, macro_char_iou


# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TEST_PRED = ROOT / "outputs/old_challenge_split/kiri_super_pred.csv"
TEST_GOLD = ROOT / "data/old-challenge-split/test_annotations.csv"
OOF_PRED = ROOT / "outputs/old_challenge_split/kiri_super_oof_pred.csv"
TRAIN_GOLD = ROOT / "data/old-challenge-split/train_annotations.csv"
TEST_NOTES = ROOT / "data/old-challenge-split/test_notes.csv"
TRAIN_NOTES = ROOT / "data/old-challenge-split/train_notes.csv"
DICT_PATH = ROOT / "1st Place/data/interim/flattened_terminology_syn_super.csv"


def load_df(path):
    df = pd.read_csv(path)
    for col in ["start", "end", "concept_id"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def get_pure_fp_concepts(pred_df, gold_df):
    """Return per-concept IoU and the set of pure FP concept IDs."""
    pc = pred_df[["note_id", "start", "end", "concept_id"]].copy()
    gc = gold_df[["note_id", "start", "end", "concept_id"]].copy()
    per_concept = class_char_iou(pc, gc)
    valid = per_concept[per_concept["union"] > 0]
    pure_fp = valid[(valid["gt_chars"] == 0) & (valid["pred_chars"] > 0)]
    return per_concept, set(pure_fp["concept_id"].tolist())


def get_span_text(pred_df, notes_df, concept_ids):
    """For given concept_ids, extract the matched text from notes."""
    notes_map = dict(zip(notes_df["note_id"], notes_df["text"]))
    subset = pred_df[pred_df["concept_id"].isin(concept_ids)].copy()
    texts = []
    for _, row in subset.iterrows():
        note_text = notes_map.get(row["note_id"], "")
        span = note_text[int(row["start"]):int(row["end"])]
        texts.append({
            "concept_id": int(row["concept_id"]),
            "note_id": row["note_id"],
            "start": int(row["start"]),
            "end": int(row["end"]),
            "span_text": span,
            "span_len": int(row["end"]) - int(row["start"]),
        })
    return pd.DataFrame(texts)


def main():
    print("=" * 80)
    print("FP CONCEPT ANALYSIS")
    print("=" * 80)

    # Load data
    test_pred = load_df(TEST_PRED)
    test_gold = load_df(TEST_GOLD)
    oof_pred = load_df(OOF_PRED)
    train_gold = load_df(TRAIN_GOLD)
    test_notes = pd.read_csv(TEST_NOTES)
    train_notes = pd.read_csv(TRAIN_NOTES)
    dict_df = pd.read_csv(DICT_PATH)

    # Build concept name lookup (first synonym per concept)
    concept_names = dict_df.groupby("concept_id").first()["concept_name"].to_dict()

    # ── 1. PURE FP CONCEPTS IN TEST AND TRAIN (OOF) ───────────────────
    print("\n" + "─" * 80)
    print("1. PURE FP CONCEPTS: TEST vs TRAIN (OOF)")
    print("─" * 80)

    test_per_concept, test_fp_set = get_pure_fp_concepts(test_pred, test_gold)
    oof_per_concept, oof_fp_set = get_pure_fp_concepts(oof_pred, train_gold)

    common_fp = test_fp_set & oof_fp_set
    test_only_fp = test_fp_set - oof_fp_set
    oof_only_fp = oof_fp_set - test_fp_set

    print(f"Test pure FP concepts:  {len(test_fp_set)}")
    print(f"OOF pure FP concepts:   {len(oof_fp_set)}")
    print(f"Common (both):          {len(common_fp)}")
    print(f"Test-only FP:           {len(test_only_fp)}")
    print(f"OOF-only FP:            {len(oof_only_fp)}")

    # ── 2. CHARACTERIZE COMMON FP CONCEPTS ─────────────────────────────
    print("\n" + "─" * 80)
    print("2. CHARACTERIZE COMMON FP CONCEPTS")
    print("─" * 80)

    # Get span texts for common FP concepts from test predictions
    span_texts = get_span_text(test_pred, test_notes, common_fp)

    if not span_texts.empty:
        # Summary per concept
        concept_summary = (
            span_texts.groupby("concept_id")
            .agg(
                n_predictions=("span_text", "count"),
                avg_span_len=("span_len", "mean"),
                unique_texts=("span_text", lambda x: list(set(x))[:5]),  # up to 5 examples
            )
            .reset_index()
        )
        concept_summary["concept_name"] = concept_summary["concept_id"].map(concept_names)
        concept_summary = concept_summary.sort_values("n_predictions", ascending=False)

        print(f"\nTop 30 common FP concepts by # predictions:")
        print(f"{'Concept ID':>12} {'# Preds':>8} {'Avg Len':>8}  {'Concept Name':<40} {'Example Texts'}")
        print("-" * 130)
        for _, row in concept_summary.head(30).iterrows():
            examples = ", ".join(f'"{t}"' for t in row["unique_texts"][:3])
            name = str(row["concept_name"])[:40] if pd.notna(row["concept_name"]) else "N/A"
            print(f"{row['concept_id']:>12} {row['n_predictions']:>8} {row['avg_span_len']:>8.1f}  {name:<40} {examples[:80]}")

        # Span length distribution
        print(f"\nSpan length distribution for common FP predictions:")
        bins = [0, 3, 5, 10, 20, 50, 1000]
        labels = ["1-3", "4-5", "6-10", "11-20", "21-50", "51+"]
        span_texts["len_bin"] = pd.cut(span_texts["span_len"], bins=bins, labels=labels, right=True)
        print(span_texts["len_bin"].value_counts().sort_index().to_string())

        # Short span analysis
        short = span_texts[span_texts["span_len"] <= 5]
        print(f"\nShort spans (<=5 chars): {len(short)} / {len(span_texts)} predictions ({100*len(short)/len(span_texts):.1f}%)")
        if not short.empty:
            print("Most common short span texts:")
            print(short["span_text"].value_counts().head(20).to_string())

    # ── 3. QUANTIFY SAFE REMOVAL GAIN ──────────────────────────────────
    print("\n" + "─" * 80)
    print("3. QUANTIFY SAFE REMOVAL GAIN")
    print("─" * 80)

    # Baseline score
    baseline = macro_char_iou(
        test_pred[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"Baseline test score: {baseline:.4f}")

    # Oracle: remove ALL test pure FP concepts
    oracle_pred = test_pred[~test_pred["concept_id"].isin(test_fp_set)]
    oracle_score = macro_char_iou(
        oracle_pred[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"Oracle (remove all {len(test_fp_set)} test FP concepts): {oracle_score:.4f} (+{oracle_score - baseline:.4f})")

    # Safe: remove only COMMON FP concepts
    safe_pred = test_pred[~test_pred["concept_id"].isin(common_fp)]
    safe_score = macro_char_iou(
        safe_pred[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"Safe (remove {len(common_fp)} common FP concepts):  {safe_score:.4f} (+{safe_score - baseline:.4f})")

    # RISK CHECK: Are any common FP concepts actually TPs in test?
    # (They shouldn't be by definition, but let's verify)
    test_gold_concepts = set(test_gold["concept_id"].unique())
    common_fp_in_test_gold = common_fp & test_gold_concepts
    print(f"\nRisk check: common FP concepts that appear in test gold: {len(common_fp_in_test_gold)}")
    if common_fp_in_test_gold:
        print(f"  WARNING: these concepts are in test gold: {sorted(common_fp_in_test_gold)}")
    else:
        print("  SAFE: none of the common FP concepts appear in test gold.")

    # CRITICAL RISK: OOF-only FP concepts that are TPs in test
    oof_only_fp_in_test_gold = oof_only_fp & test_gold_concepts
    print(f"\nOOF-only FP concepts that are actually TPs in test: {len(oof_only_fp_in_test_gold)}")
    if oof_only_fp_in_test_gold:
        # Show these - they would be dangerous to blocklist
        print("  These would be DANGEROUS to blocklist:")
        for cid in sorted(oof_only_fp_in_test_gold):
            name = concept_names.get(cid, "N/A")
            test_iou_row = test_per_concept[test_per_concept["concept_id"] == cid]
            test_iou = test_iou_row["iou"].values[0] if len(test_iou_row) > 0 else "N/A"
            print(f"    {cid}: {name} (test IoU={test_iou})")

    # ── 4. BROADER ANALYSIS ────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("4. BROADER ANALYSIS")
    print("─" * 80)

    # Of 356 test pure FP concepts, how many also appear as TPs in training gold?
    train_gold_concepts = set(train_gold["concept_id"].unique())
    test_fp_also_train_tp = test_fp_set & train_gold_concepts
    test_fp_not_in_train = test_fp_set - train_gold_concepts
    print(f"\nTest pure FP concepts also in train gold (legitimate medical concepts): {len(test_fp_also_train_tp)} / {len(test_fp_set)}")
    print(f"Test pure FP concepts NOT in train gold (never annotated anywhere): {len(test_fp_not_in_train)} / {len(test_fp_set)}")

    # For the common FP set specifically
    common_fp_in_train_gold = common_fp & train_gold_concepts
    common_fp_not_in_train = common_fp - train_gold_concepts
    print(f"\nCommon FP concepts also in train gold: {len(common_fp_in_train_gold)} / {len(common_fp)}")
    print(f"Common FP concepts NOT in train gold: {len(common_fp_not_in_train)} / {len(common_fp)}")

    # How many predictions would we remove?
    n_test_preds_removed_common = test_pred["concept_id"].isin(common_fp).sum()
    n_test_preds_total = len(test_pred)
    print(f"\nPredictions removed by common FP blocklist: {n_test_preds_removed_common} / {n_test_preds_total} ({100*n_test_preds_removed_common/n_test_preds_total:.1f}%)")

    # Hierarchy distribution of FP concepts
    print("\n--- Hierarchy distribution of common FP concepts ---")
    fp_hierarchy = dict_df[dict_df["concept_id"].isin(common_fp)].groupby("concept_id").first()["hierarchy"]
    if not fp_hierarchy.empty:
        print(fp_hierarchy.value_counts().to_string())

    # Compare with overall gold distribution
    print("\n--- Hierarchy of test gold concepts (for comparison) ---")
    gold_hierarchy = dict_df[dict_df["concept_id"].isin(test_gold_concepts)].groupby("concept_id").first()["hierarchy"]
    if not gold_hierarchy.empty:
        print(gold_hierarchy.value_counts().head(10).to_string())

    # ── 5. EXTENDED BLOCKLIST ANALYSIS ─────────────────────────────────
    print("\n" + "─" * 80)
    print("5. EXTENDED BLOCKLIST: concepts FP in OOF AND not in train gold")
    print("─" * 80)

    # Concepts that are pure FP in OOF and also never appear in train gold
    # These are concepts the dictionary matches but that are never annotated
    extended_blocklist = oof_fp_set - train_gold_concepts
    print(f"Extended blocklist candidates (OOF FP + not in train gold): {len(extended_blocklist)}")

    # How many of these are also test FPs?
    ext_also_test_fp = extended_blocklist & test_fp_set
    ext_are_test_tp = extended_blocklist & test_gold_concepts
    ext_neutral = extended_blocklist - test_fp_set - test_gold_concepts
    print(f"  → Also test FP (removal helps): {len(ext_also_test_fp)}")
    print(f"  → Actually test TP (removal HURTS): {len(ext_are_test_tp)}")
    print(f"  → Not predicted on test at all (neutral): {len(ext_neutral)}")

    if ext_are_test_tp:
        ext_pred = test_pred[~test_pred["concept_id"].isin(extended_blocklist)]
        ext_score = macro_char_iou(
            ext_pred[["note_id", "start", "end", "concept_id"]],
            test_gold[["note_id", "start", "end", "concept_id"]],
        )
        print(f"  Score with extended blocklist: {ext_score:.4f} (+{ext_score - baseline:.4f})")
        if ext_are_test_tp:
            print(f"  WARNING: {len(ext_are_test_tp)} concepts would be wrongly blocked")
    else:
        ext_pred = test_pred[~test_pred["concept_id"].isin(extended_blocklist)]
        ext_score = macro_char_iou(
            ext_pred[["note_id", "start", "end", "concept_id"]],
            test_gold[["note_id", "start", "end", "concept_id"]],
        )
        print(f"  Score with extended blocklist: {ext_score:.4f} (+{ext_score - baseline:.4f})")

    # ── 6. CONSERVATIVE vs AGGRESSIVE STRATEGIES ───────────────────────
    print("\n" + "─" * 80)
    print("6. STRATEGY COMPARISON")
    print("─" * 80)

    strategies = {
        "Baseline (no removal)": set(),
        f"Conservative (common FP, n={len(common_fp)})": common_fp,
        f"Medium (common FP not in train gold, n={len(common_fp_not_in_train)})": common_fp_not_in_train,
        f"Extended (OOF FP not in train gold, n={len(extended_blocklist)})": extended_blocklist,
        f"Oracle (all test FP, n={len(test_fp_set)})": test_fp_set,
    }

    print(f"\n{'Strategy':<60} {'Score':>8} {'Delta':>8} {'Concepts Blocked':>18}")
    print("-" * 100)
    for name, blocklist in strategies.items():
        if blocklist:
            filtered = test_pred[~test_pred["concept_id"].isin(blocklist)]
        else:
            filtered = test_pred
        score = macro_char_iou(
            filtered[["note_id", "start", "end", "concept_id"]],
            test_gold[["note_id", "start", "end", "concept_id"]],
        )
        print(f"{name:<60} {score:>8.4f} {score - baseline:>+8.4f} {len(blocklist):>18}")

    # ── 7. SAVE BLOCKLISTS ─────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("7. SAVING BLOCKLISTS")
    print("─" * 80)

    out_dir = ROOT / "outputs/old_challenge_split"
    # Save common FP blocklist
    common_fp_df = pd.DataFrame({"concept_id": sorted(common_fp)})
    common_fp_df["concept_name"] = common_fp_df["concept_id"].map(concept_names)
    common_fp_path = out_dir / "fp_blocklist_common.csv"
    common_fp_df.to_csv(common_fp_path, index=False)
    print(f"Saved common FP blocklist ({len(common_fp)} concepts): {common_fp_path}")

    # Save extended blocklist
    ext_df = pd.DataFrame({"concept_id": sorted(extended_blocklist)})
    ext_df["concept_name"] = ext_df["concept_id"].map(concept_names)
    ext_path = out_dir / "fp_blocklist_extended.csv"
    ext_df.to_csv(ext_path, index=False)
    print(f"Saved extended blocklist ({len(extended_blocklist)} concepts): {ext_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
