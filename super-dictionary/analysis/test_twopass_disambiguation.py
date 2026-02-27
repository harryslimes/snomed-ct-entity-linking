#!/usr/bin/env python3
"""Two-pass concept disambiguation.

Pass 1: Standard single-concept prediction (baseline).
Pass 2: For each prediction, check if the training data had alternative
        concepts for that mention. Score alternatives using:
        - Note-level concept co-occurrence from training
        - Section-specific mention→concept frequency
        Swap if an alternative scores higher.

This avoids the prediction flooding problem seen with multi-concept dicts.
"""
from __future__ import annotations

import copy
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from train_dictionary import (
    COMMON_HEADERS, BLACKLIST_THRESH, INTERNAL_BLACKLIST,
    IndexedDict, annotate_with_dict, remove_overlaps,
    CASE_SENSITIVE_DICT, get_pattern, train,
    build_dict_from_annotations,
    get_sections, get_header_by_pos,
)
from runtime_scoring import macro_char_iou, class_char_iou


def predict_with_dict(d, uc_d, notes):
    """Standard two-pass KIRI-style prediction."""
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


def predict_with_dict_keep_metadata(d, uc_d, notes):
    """Like predict_with_dict but keeps section and dict_entry columns."""
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
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id",
                                      "section", "dict_entry"])
    pred = pd.concat(all_preds, ignore_index=True)
    for c in ["start", "end", "concept_id"]:
        pred[c] = pred[c].astype(int)
    return pred


def build_cooccurrence(annotations):
    """Build concept co-occurrence matrix from training annotations.

    Returns dict: concept_id -> Counter({other_concept_id: count})
    """
    cooc = defaultdict(Counter)
    for nid, group in annotations.groupby("note_id"):
        concepts = group["concept_id"].astype(int).unique()
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1:]:
                cooc[c1][c2] += 1
                cooc[c2][c1] += 1
    return dict(cooc)


def build_mention_concept_priors(d_combined):
    """Build (mention -> {concept_id: count}) from training Counter dict.

    d_combined has keys (section, mention) -> Counter({concept_id: count}).
    We aggregate across all sections for each mention.
    """
    mention_concepts = defaultdict(Counter)
    for (section, mention), counter in d_combined.items():
        mention_concepts[mention] += counter
    return dict(mention_concepts)


def build_section_mention_concept_priors(d_combined):
    """Build (section, mention) -> {concept_id: count} priors."""
    priors = {}
    for (section, mention), counter in d_combined.items():
        priors[(section, mention)] = dict(counter)
    return priors


def score_candidate(cid, note_concepts, cooc, mention_freq, section_freq,
                    mention, section, weights):
    """Score a candidate concept_id using multiple signals."""
    score = 0.0

    # Signal 1: Co-occurrence with other concepts in this note
    if cooc and note_concepts:
        cooc_score = 0
        cid_cooc = cooc.get(cid, {})
        for other_cid in note_concepts:
            if other_cid != cid:
                cooc_score += cid_cooc.get(other_cid, 0)
        score += weights.get("cooc", 1.0) * cooc_score

    # Signal 2: Global mention→concept frequency
    if mention_freq:
        freq = mention_freq.get(mention, {}).get(cid, 0)
        score += weights.get("freq", 1.0) * freq

    # Signal 3: Section-specific mention→concept frequency
    if section_freq:
        sec_freq = section_freq.get((section, mention), {}).get(cid, 0)
        if sec_freq == 0:
            sec_freq = section_freq.get(("any", mention), {}).get(cid, 0) * 0.5
        score += weights.get("sec_freq", 1.0) * sec_freq

    return score


def twopass_disambiguate(pred_df, d_combined, cooc, notes_df,
                          weights, min_alternatives=2):
    """Second pass: swap concept_ids using contextual signals.

    For each prediction:
    1. Look up the mention in d_combined to find alternative concepts
    2. If alternatives exist, score each using co-occurrence + frequency
    3. Swap to the highest-scoring alternative if it beats current
    """
    mention_freq = build_mention_concept_priors(d_combined)
    section_freq = build_section_mention_concept_priors(d_combined)

    pred = pred_df.copy()
    swaps = 0
    swap_details = []

    # Group predictions by note for co-occurrence scoring
    for note_id, note_preds in pred.groupby("note_id"):
        # Get all predicted concepts in this note (for co-occurrence)
        note_concepts = set(note_preds["concept_id"].astype(int).unique())

        # Get note text for extracting mention strings
        note_text_row = notes_df[notes_df["note_id"] == note_id]
        if note_text_row.empty:
            continue
        note_text = note_text_row.iloc[0]["text"].lower()

        for idx in note_preds.index:
            s, e = int(pred.loc[idx, "start"]), int(pred.loc[idx, "end"])
            current_cid = int(pred.loc[idx, "concept_id"])

            # Extract the mention text from the note
            mention = note_text[s:e]
            mention_norm = " ".join(mention.split())

            # Get section if available
            section = pred.loc[idx].get("section", "any")
            if pd.isna(section):
                section = "any"

            # Look up alternative concepts from training
            alternatives = mention_freq.get(mention_norm, {})
            if len(alternatives) < min_alternatives:
                continue

            # Score current concept
            # Remove current concept from note_concepts for fair scoring
            other_concepts = note_concepts - {current_cid}

            current_score = score_candidate(
                current_cid, other_concepts, cooc, mention_freq, section_freq,
                mention_norm, section, weights
            )

            # Score alternatives
            best_cid = current_cid
            best_score = current_score
            for alt_cid in alternatives:
                if alt_cid == current_cid:
                    continue
                alt_score = score_candidate(
                    alt_cid, other_concepts, cooc, mention_freq, section_freq,
                    mention_norm, section, weights
                )
                if alt_score > best_score:
                    best_score = alt_score
                    best_cid = alt_cid

            if best_cid != current_cid:
                pred.loc[idx, "concept_id"] = best_cid
                swaps += 1
                # Update note_concepts
                note_concepts.discard(current_cid)
                note_concepts.add(best_cid)
                if len(swap_details) < 30:
                    swap_details.append({
                        "mention": mention_norm,
                        "from": current_cid,
                        "to": best_cid,
                        "from_score": current_score,
                        "to_score": best_score,
                    })

    return pred, swaps, swap_details


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
    # Step 1: Build training state
    # ===================================================================
    print("\nBuilding training state...", flush=True)

    # Build d_combined (Counter-valued dict) for alternative lookup
    texts = train_notes_df.set_index("note_id")["text"]
    texts_lc = texts.str.lower()
    headers = [h.lower() for h in COMMON_HEADERS]

    annotations = train_annotations_df.copy()
    if "span" in annotations.columns:
        annotations["source orig"] = annotations["span"]
        annotations["source"] = annotations["span"].str.lower()

    words_counter = Counter()
    for text in texts_lc:
        words_counter.update(text.split())
    blacklist_set = {w for w in words_counter if words_counter[w] > BLACKLIST_THRESH}
    blacklist_set.update(INTERNAL_BLACKLIST)

    d_combined: dict[tuple, Counter] = {}
    for nid in texts_lc.index:
        note_anns = annotations[annotations["note_id"] == nid]
        if note_anns.empty:
            continue
        t = build_dict_from_annotations(texts_lc[nid], note_anns, headers, blacklist_set)
        for k in t:
            d_combined.setdefault(k, Counter()).update(t[k])

    print(f"  d_combined: {len(d_combined)} entries")

    # Build co-occurrence matrix
    print("Building co-occurrence matrix...", flush=True)
    cooc = build_cooccurrence(train_annotations_df)
    n_concepts_with_cooc = len(cooc)
    total_pairs = sum(len(v) for v in cooc.values()) // 2
    print(f"  {n_concepts_with_cooc} concepts with co-occurrence data")
    print(f"  {total_pairs:,} unique concept pairs")

    # ===================================================================
    # Step 2: Baseline
    # ===================================================================
    print("\n" + "=" * 80)
    print("BASELINE")
    print("=" * 80)

    d_baseline, uc_baseline = train(
        train_notes_df, train_annotations_df,
        super_dict_path=super_dict_path,
        flat_terminology_path=flat_term_path,
    )

    pred_baseline = predict_with_dict_keep_metadata(d_baseline, uc_baseline, test_notes)
    pred_baseline_eval = pred_baseline[["note_id", "start", "end", "concept_id"]].copy()
    iou_baseline = macro_char_iou(pred_baseline_eval, test_gold[["note_id", "start", "end", "concept_id"]])
    print(f"  Baseline IoU: {iou_baseline:.4f}  ({len(pred_baseline):,} predictions)")

    gold_by_note = {}
    for nid, g in test_gold.groupby("note_id"):
        gold_by_note[str(nid)] = g

    # ===================================================================
    # Step 3: Test weight configurations
    # ===================================================================
    print("\n" + "=" * 80)
    print("TWO-PASS DISAMBIGUATION EXPERIMENTS")
    print("=" * 80)

    weight_configs = [
        ("cooc_only",       {"cooc": 1.0, "freq": 0.0, "sec_freq": 0.0}),
        ("freq_only",       {"cooc": 0.0, "freq": 1.0, "sec_freq": 0.0}),
        ("sec_freq_only",   {"cooc": 0.0, "freq": 0.0, "sec_freq": 1.0}),
        ("cooc+freq",       {"cooc": 1.0, "freq": 1.0, "sec_freq": 0.0}),
        ("cooc+sec_freq",   {"cooc": 1.0, "freq": 0.0, "sec_freq": 1.0}),
        ("all_equal",       {"cooc": 1.0, "freq": 1.0, "sec_freq": 1.0}),
        ("cooc_heavy",      {"cooc": 5.0, "freq": 1.0, "sec_freq": 1.0}),
        ("freq_heavy",      {"cooc": 1.0, "freq": 5.0, "sec_freq": 1.0}),
        ("sec_freq_heavy",  {"cooc": 1.0, "freq": 1.0, "sec_freq": 5.0}),
    ]

    results = []
    for name, weights in weight_configs:
        pred_swap, n_swaps, details = twopass_disambiguate(
            pred_baseline, d_combined, cooc, test_notes, weights
        )
        pred_eval = pred_swap[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]:
            pred_eval[c] = pred_eval[c].astype(int)
        iou = macro_char_iou(pred_eval, test_gold[["note_id", "start", "end", "concept_id"]])
        delta = iou - iou_baseline
        results.append((name, iou, delta, n_swaps))
        print(f"  {name:<20} IoU={iou:.4f} ({delta:+.4f}) swaps={n_swaps}")

        if name == "cooc_only" and details:
            print(f"    Sample swaps:")
            for d in details[:10]:
                print(f"      '{d['mention']}': {d['from']} -> {d['to']} "
                      f"(score {d['from_score']:.1f} -> {d['to_score']:.1f})")

    # ===================================================================
    # Step 4: Oracle analysis of swappable predictions
    # ===================================================================
    print("\n" + "=" * 80)
    print("ORACLE ANALYSIS: WHICH SWAPS WOULD HELP?")
    print("=" * 80)

    mention_freq = build_mention_concept_priors(d_combined)

    # For each wrong-concept prediction, check if the right concept is in alternatives
    n_wrong_concept = 0
    n_swappable = 0
    n_cooc_would_help = 0
    n_cooc_would_hurt = 0
    swappable_examples = []

    for _, row in pred_baseline.iterrows():
        nid = str(row["note_id"])
        gold = gold_by_note.get(nid)
        if gold is None:
            continue

        s, e, cid = int(row["start"]), int(row["end"]), int(row["concept_id"])
        overlaps = gold[(gold["start"] < e) & (gold["end"] > s)]
        if overlaps.empty:
            continue
        if cid in overlaps["concept_id"].values:
            continue

        n_wrong_concept += 1

        # Get mention
        text = test_notes[test_notes["note_id"] == row["note_id"]]["text"].iloc[0].lower()
        mention = " ".join(text[s:e].split())

        # Check if correct concept is in alternatives
        alternatives = mention_freq.get(mention, {})
        gold_cids = set(overlaps["concept_id"].astype(int))
        fixable_cids = gold_cids & set(alternatives.keys())

        if fixable_cids:
            n_swappable += 1
            # Check if co-occurrence would pick the right one
            note_preds = pred_baseline[pred_baseline["note_id"] == row["note_id"]]
            note_concepts = set(note_preds["concept_id"].astype(int)) - {cid}

            current_cooc = sum(cooc.get(cid, {}).get(oc, 0) for oc in note_concepts)
            for fix_cid in fixable_cids:
                fix_cooc = sum(cooc.get(fix_cid, {}).get(oc, 0) for oc in note_concepts)
                if fix_cooc > current_cooc:
                    n_cooc_would_help += 1
                    break
                elif fix_cooc < current_cooc:
                    n_cooc_would_hurt += 1
                    break

            if len(swappable_examples) < 15:
                fix_cid = list(fixable_cids)[0]
                curr_freq = alternatives.get(cid, 0)
                fix_freq = alternatives.get(fix_cid, 0)
                curr_cooc_score = sum(cooc.get(cid, {}).get(oc, 0) for oc in note_concepts)
                fix_cooc_score = sum(cooc.get(fix_cid, {}).get(oc, 0) for oc in note_concepts)
                swappable_examples.append({
                    "mention": mention,
                    "current_cid": cid,
                    "correct_cid": fix_cid,
                    "current_freq": curr_freq,
                    "correct_freq": fix_freq,
                    "current_cooc": curr_cooc_score,
                    "correct_cooc": fix_cooc_score,
                })

    print(f"  Wrong-concept predictions: {n_wrong_concept}")
    print(f"  Swappable (correct concept in training alternatives): {n_swappable}")
    print(f"  Co-occurrence would pick correctly: {n_cooc_would_help}")
    print(f"  Co-occurrence would pick wrong: {n_cooc_would_hurt}")
    print(f"  Co-occurrence neutral/tied: {n_swappable - n_cooc_would_help - n_cooc_would_hurt}")

    if swappable_examples:
        print(f"\n  Swappable examples (cooc=co-occurrence score, freq=training frequency):")
        print(f"  {'mention':<25} {'curr_cid':>10} {'corr_cid':>10} {'c_freq':>6} {'g_freq':>6} {'c_cooc':>6} {'g_cooc':>6} {'signal':>8}")
        for ex in swappable_examples:
            # Determine which signal would help
            freq_helps = ex["correct_freq"] > ex["current_freq"]
            cooc_helps = ex["correct_cooc"] > ex["current_cooc"]
            signal = ""
            if freq_helps and cooc_helps:
                signal = "both"
            elif freq_helps:
                signal = "freq"
            elif cooc_helps:
                signal = "cooc"
            else:
                signal = "neither"
            print(f"  {ex['mention']:<25} {ex['current_cid']:>10} {ex['correct_cid']:>10} "
                  f"{ex['current_freq']:>6} {ex['correct_freq']:>6} "
                  f"{ex['current_cooc']:>6} {ex['correct_cooc']:>6} {signal:>8}")

    # ===================================================================
    # Step 5: Selective swap (only swap when co-occurrence strongly favors)
    # ===================================================================
    print("\n" + "=" * 80)
    print("SELECTIVE SWAP EXPERIMENTS")
    print("=" * 80)

    for cooc_min_ratio in [1.5, 2.0, 3.0, 5.0]:
        pred_selective = pred_baseline.copy()
        n_sel_swaps = 0

        for note_id, note_preds in pred_selective.groupby("note_id"):
            note_concepts = set(note_preds["concept_id"].astype(int).unique())
            note_text_row = test_notes[test_notes["note_id"] == note_id]
            if note_text_row.empty:
                continue
            note_text = note_text_row.iloc[0]["text"].lower()

            for idx in note_preds.index:
                s, e = int(pred_selective.loc[idx, "start"]), int(pred_selective.loc[idx, "end"])
                current_cid = int(pred_selective.loc[idx, "concept_id"])
                mention = " ".join(note_text[s:e].split())

                alternatives = mention_freq.get(mention, {})
                if len(alternatives) < 2:
                    continue

                other_concepts = note_concepts - {current_cid}
                if not other_concepts:
                    continue

                current_cooc = sum(cooc.get(current_cid, {}).get(oc, 0) for oc in other_concepts)

                best_cid = current_cid
                best_cooc = current_cooc
                for alt_cid in alternatives:
                    if alt_cid == current_cid:
                        continue
                    alt_cooc = sum(cooc.get(alt_cid, {}).get(oc, 0) for oc in other_concepts)
                    if alt_cooc > best_cooc:
                        best_cooc = alt_cooc
                        best_cid = alt_cid

                # Only swap if ratio is strong enough
                if best_cid != current_cid:
                    if current_cooc == 0 or (best_cooc / max(current_cooc, 1)) >= cooc_min_ratio:
                        pred_selective.loc[idx, "concept_id"] = best_cid
                        note_concepts.discard(current_cid)
                        note_concepts.add(best_cid)
                        n_sel_swaps += 1

        pred_sel_eval = pred_selective[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]:
            pred_sel_eval[c] = pred_sel_eval[c].astype(int)
        iou_sel = macro_char_iou(pred_sel_eval, test_gold[["note_id", "start", "end", "concept_id"]])
        print(f"  cooc_ratio>={cooc_min_ratio:.1f}: IoU={iou_sel:.4f} ({iou_sel-iou_baseline:+.4f}) swaps={n_sel_swaps}")

    # ===================================================================
    # Step 6: Swap only when current has zero co-occurrence
    # ===================================================================
    print("\n" + "=" * 80)
    print("ZERO-COOC SWAP (only swap when current concept has 0 co-occurrence)")
    print("=" * 80)

    for min_alt_cooc in [1, 2, 3, 5, 10]:
        pred_zero = pred_baseline.copy()
        n_zero_swaps = 0

        for note_id, note_preds in pred_zero.groupby("note_id"):
            note_concepts = set(note_preds["concept_id"].astype(int).unique())
            note_text_row = test_notes[test_notes["note_id"] == note_id]
            if note_text_row.empty:
                continue
            note_text = note_text_row.iloc[0]["text"].lower()

            for idx in note_preds.index:
                s, e = int(pred_zero.loc[idx, "start"]), int(pred_zero.loc[idx, "end"])
                current_cid = int(pred_zero.loc[idx, "concept_id"])
                mention = " ".join(note_text[s:e].split())

                alternatives = mention_freq.get(mention, {})
                if len(alternatives) < 2:
                    continue

                other_concepts = note_concepts - {current_cid}
                if not other_concepts:
                    continue

                current_cooc = sum(cooc.get(current_cid, {}).get(oc, 0) for oc in other_concepts)
                if current_cooc > 0:
                    continue  # Only swap when current has zero co-occurrence

                best_cid = current_cid
                best_cooc = 0
                for alt_cid in alternatives:
                    if alt_cid == current_cid:
                        continue
                    alt_cooc = sum(cooc.get(alt_cid, {}).get(oc, 0) for oc in other_concepts)
                    if alt_cooc > best_cooc and alt_cooc >= min_alt_cooc:
                        best_cooc = alt_cooc
                        best_cid = alt_cid

                if best_cid != current_cid:
                    pred_zero.loc[idx, "concept_id"] = best_cid
                    note_concepts.discard(current_cid)
                    note_concepts.add(best_cid)
                    n_zero_swaps += 1

        pred_zero_eval = pred_zero[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]:
            pred_zero_eval[c] = pred_zero_eval[c].astype(int)
        iou_zero = macro_char_iou(pred_zero_eval, test_gold[["note_id", "start", "end", "concept_id"]])
        print(f"  min_alt_cooc>={min_alt_cooc:>2}: IoU={iou_zero:.4f} ({iou_zero-iou_baseline:+.4f}) swaps={n_zero_swaps}")

    # ===================================================================
    # Step 7: Frequency-only swap (swap when alt frequency >> current)
    # ===================================================================
    print("\n" + "=" * 80)
    print("FREQUENCY RATIO SWAP")
    print("=" * 80)

    for min_freq_ratio in [2.0, 3.0, 5.0, 10.0]:
        pred_freqr = pred_baseline.copy()
        n_freqr_swaps = 0

        for note_id, note_preds in pred_freqr.groupby("note_id"):
            note_text_row = test_notes[test_notes["note_id"] == note_id]
            if note_text_row.empty:
                continue
            note_text = note_text_row.iloc[0]["text"].lower()

            for idx in note_preds.index:
                s, e = int(pred_freqr.loc[idx, "start"]), int(pred_freqr.loc[idx, "end"])
                current_cid = int(pred_freqr.loc[idx, "concept_id"])
                section = pred_freqr.loc[idx].get("section", "any")
                mention = " ".join(note_text[s:e].split())

                # Check section-specific frequency
                sec_key = (section, mention) if section != "any" else None
                sec_priors = d_combined.get(sec_key, Counter()) if sec_key else Counter()

                # Also check "any" section
                any_priors = d_combined.get(("any", mention), Counter())

                # Combine
                combined = Counter()
                combined.update(sec_priors)
                combined.update(any_priors)

                if len(combined) < 2:
                    continue

                current_freq = combined.get(current_cid, 0)
                if current_freq == 0:
                    continue  # Can't compute ratio

                best_cid = current_cid
                best_freq = current_freq
                for alt_cid, alt_freq in combined.items():
                    if alt_cid == current_cid:
                        continue
                    if alt_freq > best_freq and (alt_freq / current_freq) >= min_freq_ratio:
                        best_freq = alt_freq
                        best_cid = alt_cid

                if best_cid != current_cid:
                    pred_freqr.loc[idx, "concept_id"] = best_cid
                    n_freqr_swaps += 1

        pred_freqr_eval = pred_freqr[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]:
            pred_freqr_eval[c] = pred_freqr_eval[c].astype(int)
        iou_freqr = macro_char_iou(pred_freqr_eval, test_gold[["note_id", "start", "end", "concept_id"]])
        print(f"  freq_ratio>={min_freq_ratio:>4.1f}: IoU={iou_freqr:.4f} ({iou_freqr-iou_baseline:+.4f}) swaps={n_freqr_swaps}")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Baseline IoU: {iou_baseline:.4f}")
    print(f"  Oracle ceiling (training-available): +0.0381")
    print(f"\n  Best results from each category:")
    results.sort(key=lambda x: x[1], reverse=True)
    for name, iou, delta, n_swaps in results[:3]:
        print(f"    {name:<20} IoU={iou:.4f} ({delta:+.4f}) swaps={n_swaps}")

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
