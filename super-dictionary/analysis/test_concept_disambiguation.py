#!/usr/bin/env python3
"""Analyze and test concept disambiguation strategies.

The current pipeline picks most_common(1) concept per (section, mention) key
during training. This script:
1. Quantifies the ambiguity problem (how many mentions map to multiple concepts)
2. Measures the ceiling from oracle disambiguation
3. Tests practical disambiguation strategies
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
    build_dict_from_annotations, score_dict,
    count_correct, is_naive_key_remove, remove_bad_keys,
    load_snomed_synonyms_from_super_dict, process_term,
    load_concept_types, cond_update,
    extract_uppercase_mentions, get_word_replacements,
    get_permutations, get_allowed_sections,
    limit_any_to_allowed_sections,
    CORRECT_FRAC_FOR_DICT, CORRECT_FRAC_FOR_ANY,
    get_sections, get_header_by_pos, is_in_header,
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


def predict_with_multi_concept_dict(d_multi, uc_d, notes, gold_by_note=None,
                                    strategy="oracle", concept_priors=None,
                                    section_concept_priors=None,
                                    cooccurrence=None):
    """Predict with a multi-concept dictionary using disambiguation strategy.

    d_multi: dict mapping (section, mention) -> Counter of {concept_id: count}
    strategy: "oracle", "frequency", "section_freq", "cooccurrence"
    """
    uc_d_full = dict(uc_d)
    uc_d_full.update(CASE_SENSITIVE_DICT)

    headers_lc = [h.lower() for h in COMMON_HEADERS]
    headers_orig = list(COMMON_HEADERS)

    all_preds = []
    for _, note_row in notes.iterrows():
        note_id = str(note_row["note_id"])
        text = note_row["text"]
        text_lc = text.lower()

        # Get raw annotations with keep_overlaps=True to see all candidates
        raw_lc = annotate_with_dict(text_lc, d_multi, headers_lc, note_id,
                                     keep_overlaps=True)
        # UC dict uses standard single-concept approach
        uc_indexed = IndexedDict(uc_d_full, prefilter="bigram")
        ann_uc = annotate_with_dict(text, uc_indexed, headers_orig, note_id)

        if raw_lc.empty and ann_uc.empty:
            continue

        # Disambiguate the LC annotations
        if not raw_lc.empty:
            disambiguated = disambiguate(
                raw_lc, strategy, note_id, gold_by_note,
                concept_priors, section_concept_priors, cooccurrence
            )
        else:
            disambiguated = raw_lc

        combined = pd.concat([disambiguated, ann_uc], ignore_index=True)
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


def disambiguate(ann, strategy, note_id, gold_by_note=None,
                 concept_priors=None, section_concept_priors=None,
                 cooccurrence=None):
    """Disambiguate annotations where multiple concepts match same span.

    For each unique (start, end), if multiple concept_ids exist, pick one.
    """
    if ann.empty:
        return ann

    # Group by (start, end) to find ambiguous positions
    groups = ann.groupby(["start", "end"])
    rows = []
    for (start, end), group in groups:
        if len(group) == 1:
            rows.append(group.iloc[0])
            continue

        # Multiple concepts at this position - disambiguate
        candidates = group.copy()

        if strategy == "oracle":
            # Pick the concept that matches gold (if available)
            gold = gold_by_note.get(str(note_id))
            if gold is not None:
                gold_at_pos = gold[
                    (gold["start"] <= start) & (gold["end"] >= end)
                ]
                if not gold_at_pos.empty:
                    gold_cids = set(gold_at_pos["concept_id"].astype(int))
                    matching = candidates[candidates["concept_id"].astype(int).isin(gold_cids)]
                    if not matching.empty:
                        rows.append(matching.iloc[0])
                        continue
            # Fall back to frequency
            best_idx = _pick_by_frequency(candidates, concept_priors)
            rows.append(candidates.loc[best_idx])

        elif strategy == "frequency":
            best_idx = _pick_by_frequency(candidates, concept_priors)
            rows.append(candidates.loc[best_idx])

        elif strategy == "section_freq":
            best_idx = _pick_by_section_freq(
                candidates, section_concept_priors, concept_priors
            )
            rows.append(candidates.loc[best_idx])

        elif strategy == "cooccurrence":
            # First pass: use frequency. Second pass will re-score.
            best_idx = _pick_by_frequency(candidates, concept_priors)
            rows.append(candidates.loc[best_idx])

        else:
            # Default: pick first (arbitrary)
            rows.append(group.iloc[0])

    result = pd.DataFrame(rows).reset_index(drop=True)
    return result


def _pick_by_frequency(candidates, concept_priors):
    """Pick the candidate with the highest training frequency."""
    if concept_priors is None:
        return candidates.index[0]
    best_idx = candidates.index[0]
    best_count = -1
    for idx in candidates.index:
        cid = int(candidates.loc[idx, "concept_id"])
        mention = candidates.loc[idx, "dict_entry"]
        section = candidates.loc[idx, "section"]
        # Look up (section, mention, concept) prior
        count = concept_priors.get((section, mention, cid), 0)
        if count == 0:
            count = concept_priors.get(("any", mention, cid), 0)
        if count > best_count:
            best_count = count
            best_idx = idx
    return best_idx


def _pick_by_section_freq(candidates, section_concept_priors, concept_priors):
    """Pick based on section-specific concept frequency."""
    if section_concept_priors is None:
        return _pick_by_frequency(candidates, concept_priors)

    best_idx = candidates.index[0]
    best_score = -1
    for idx in candidates.index:
        cid = int(candidates.loc[idx, "concept_id"])
        section = candidates.loc[idx, "section"]
        mention = candidates.loc[idx, "dict_entry"]
        # Section-specific prior
        score = section_concept_priors.get((section, cid), 0)
        if score == 0 and concept_priors:
            # Fall back to global prior
            score = concept_priors.get(("any", mention, cid), 0) * 0.01
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def build_multi_concept_dict(train_notes_df, train_annotations_df):
    """Build dictionary keeping ALL concepts per (section, mention), not just most_common.

    Also builds concept frequency priors.
    """
    texts = train_notes_df.set_index("note_id")["text"]
    texts_lc = texts.str.lower()
    headers = [h.lower() for h in COMMON_HEADERS]

    annotations = train_annotations_df.copy()
    if "span" in annotations.columns:
        annotations["source orig"] = annotations["span"]
        annotations["source"] = annotations["span"].str.lower()

    # Build word counter + blacklist
    words_counter = Counter()
    for text in texts_lc:
        words_counter.update(text.split())
    blacklist_set = {w for w in words_counter if words_counter[w] > BLACKLIST_THRESH}
    blacklist_set.update(INTERNAL_BLACKLIST)

    # Build training dictionary with Counter values
    d_combined: dict[tuple, Counter] = {}
    for nid in texts_lc.index:
        note_anns = annotations[annotations["note_id"] == nid]
        if note_anns.empty:
            continue
        t = build_dict_from_annotations(texts_lc[nid], note_anns, headers, blacklist_set)
        for k in t:
            d_combined.setdefault(k, Counter()).update(t[k])

    # Build concept priors: (section, mention, concept_id) -> count
    concept_priors = {}
    for k, counter in d_combined.items():
        section, mention = k
        for cid, count in counter.items():
            concept_priors[(section, mention, cid)] = count

    # Build section-concept priors from training annotations
    section_concept_priors = Counter()
    for nid in texts_lc.index:
        note_anns = annotations[annotations["note_id"] == nid]
        if note_anns.empty:
            continue
        text = texts_lc[nid]
        h_positions, pos_header = get_sections(text, headers)
        for _, row in note_anns.iterrows():
            h = get_header_by_pos(int(row["start"]), h_positions, pos_header, headers)
            if h:
                section_concept_priors[(h, int(row["concept_id"]))] += 1

    return d_combined, concept_priors, section_concept_priors, blacklist_set


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
    # Part 1: Quantify ambiguity in training data
    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 1: QUANTIFYING CONCEPT AMBIGUITY")
    print("=" * 80)

    d_combined, concept_priors, section_concept_priors, blacklist_set = \
        build_multi_concept_dict(train_notes_df, train_annotations_df)

    # Count ambiguous mentions
    n_single = 0
    n_multi = 0
    n_concepts_dist = Counter()
    ambig_examples = []

    for k, counter in d_combined.items():
        section, mention = k
        n_concepts = len(counter)
        n_concepts_dist[n_concepts] += 1
        if n_concepts > 1:
            n_multi += 1
            if section == "any" and len(ambig_examples) < 20:
                total = sum(counter.values())
                top = counter.most_common(3)
                top_pct = top[0][1] / total * 100
                ambig_examples.append((mention, n_concepts, total, top_pct, top))
        else:
            n_single += 1

    print(f"\n  Dictionary entries: {len(d_combined):,}")
    print(f"  Single concept:    {n_single:,} ({100*n_single/len(d_combined):.1f}%)")
    print(f"  Multi concept:     {n_multi:,} ({100*n_multi/len(d_combined):.1f}%)")
    print(f"\n  Concepts per entry distribution:")
    for n, count in sorted(n_concepts_dist.items()):
        print(f"    {n} concepts: {count:>5} entries")

    # Show ambiguous examples sorted by count
    ambig_examples.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  Top ambiguous mentions (section='any'):")
    print(f"  {'mention':<30} {'#cid':>4} {'total':>5} {'top%':>5}  top concepts")
    for mention, nc, total, top_pct, top in ambig_examples[:20]:
        top_str = ", ".join(f"{cid}({cnt})" for cid, cnt in top)
        print(f"  {mention:<30} {nc:>4} {total:>5} {top_pct:>4.0f}%  {top_str}")

    # ===================================================================
    # Part 2: How often does most_common(1) get it wrong on test data?
    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 2: WRONG-CONCEPT ERRORS IN BASELINE")
    print("=" * 80)

    # Build standard (single-concept) dictionary
    d_baseline, uc_baseline = train(
        train_notes_df, train_annotations_df,
        super_dict_path=super_dict_path,
        flat_terminology_path=flat_term_path,
    )

    pred_baseline = predict_with_dict(d_baseline, uc_baseline, test_notes)
    iou_baseline = macro_char_iou(
        pred_baseline[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"  Baseline IoU: {iou_baseline:.4f}  ({len(pred_baseline):,} predictions)")

    # Find wrong-concept errors
    gold_by_note = {}
    for nid, g in test_gold.groupby("note_id"):
        gold_by_note[str(nid)] = g

    wrong_concept_fp = 0
    wrong_concept_chars = 0
    wrong_concept_examples = []
    correct_concept = 0

    for _, pred_row in pred_baseline.iterrows():
        nid = str(pred_row["note_id"])
        gold = gold_by_note.get(nid)
        if gold is None:
            continue
        s, e, cid = int(pred_row["start"]), int(pred_row["end"]), int(pred_row["concept_id"])

        # Check if any gold annotation overlaps this prediction
        overlaps = gold[(gold["start"] < e) & (gold["end"] > s)]
        if overlaps.empty:
            continue  # No overlap = location FP, not concept issue

        # Check concept match
        if cid in overlaps["concept_id"].values:
            correct_concept += 1
        else:
            wrong_concept_fp += 1
            wrong_concept_chars += (e - s)
            # Get the mention text
            text = test_notes[test_notes["note_id"] == pred_row["note_id"]]["text"].iloc[0]
            mention = text[s:e].lower()
            mention_norm = " ".join(mention.split())
            gold_cids = overlaps["concept_id"].tolist()
            if len(wrong_concept_examples) < 30:
                wrong_concept_examples.append({
                    "mention": mention_norm,
                    "predicted_cid": cid,
                    "gold_cids": gold_cids,
                    "chars": e - s,
                })

    print(f"\n  Predictions overlapping gold: {correct_concept + wrong_concept_fp:,}")
    print(f"  Correct concept: {correct_concept:,}")
    print(f"  Wrong concept:   {wrong_concept_fp:,} ({wrong_concept_chars:,} chars)")

    # Check: for wrong-concept FPs, was the correct concept available?
    n_fixable = 0
    n_not_fixable = 0
    for ex in wrong_concept_examples:
        mention = ex["mention"]
        for gold_cid in ex["gold_cids"]:
            # Check if (section, mention) -> gold_cid exists in d_combined
            any_key = ("any", mention)
            if any_key in d_combined and gold_cid in d_combined[any_key]:
                n_fixable += 1
                break
        else:
            n_not_fixable += 1

    print(f"\n  Of {len(wrong_concept_examples)} sampled wrong-concept FPs:")
    print(f"    Correct concept available in training dict: {n_fixable}")
    print(f"    Correct concept NOT available:              {n_not_fixable}")

    if wrong_concept_examples:
        print(f"\n  Wrong-concept examples:")
        for ex in wrong_concept_examples[:15]:
            avail = "YES" if ("any", ex["mention"]) in d_combined and \
                any(gc in d_combined.get(("any", ex["mention"]), {}) for gc in ex["gold_cids"]) \
                else "NO"
            print(f"    '{ex['mention']}' predicted={ex['predicted_cid']} "
                  f"gold={ex['gold_cids']} fixable={avail} ({ex['chars']} chars)")

    # ===================================================================
    # Part 3: Check gold annotation overlap patterns
    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 3: GOLD ANNOTATION OVERLAP PATTERNS")
    print("=" * 80)

    # Do gold annotations overlap at same positions with different concepts?
    n_overlap_same_span = 0
    n_overlap_partial = 0
    multi_concept_positions = 0

    for nid, gold in gold_by_note.items():
        gold_sorted = gold.sort_values("start")
        for i in range(len(gold_sorted)):
            for j in range(i + 1, len(gold_sorted)):
                si, ei = int(gold_sorted.iloc[i]["start"]), int(gold_sorted.iloc[i]["end"])
                sj, ej = int(gold_sorted.iloc[j]["start"]), int(gold_sorted.iloc[j]["end"])
                if sj >= ei:
                    break
                ci = int(gold_sorted.iloc[i]["concept_id"])
                cj = int(gold_sorted.iloc[j]["concept_id"])
                if ci == cj:
                    continue
                if si == sj and ei == ej:
                    n_overlap_same_span += 1
                    multi_concept_positions += 1
                else:
                    n_overlap_partial += 1

    print(f"  Gold same-span multi-concept: {n_overlap_same_span}")
    print(f"  Gold partial-overlap different-concept: {n_overlap_partial}")
    print(f"  (These affect scoring since only last-write-wins per position)")

    # ===================================================================
    # Part 4: Oracle disambiguation ceiling
    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 4: ORACLE DISAMBIGUATION (CEILING)")
    print("=" * 80)

    # Strategy: for each prediction, if the predicted concept is wrong but
    # a correct concept exists in the dictionary for that mention, swap it.
    pred_oracle = pred_baseline.copy()
    swaps = 0
    for idx in pred_oracle.index:
        nid = str(pred_oracle.loc[idx, "note_id"])
        gold = gold_by_note.get(nid)
        if gold is None:
            continue
        s, e = int(pred_oracle.loc[idx, "start"]), int(pred_oracle.loc[idx, "end"])
        cid = int(pred_oracle.loc[idx, "concept_id"])

        # Check if any gold overlaps
        overlaps = gold[(gold["start"] < e) & (gold["end"] > s)]
        if overlaps.empty:
            continue
        if cid in overlaps["concept_id"].values:
            continue  # Already correct

        # Try to find a better concept from the gold
        for gold_cid in overlaps["concept_id"].values:
            pred_oracle.loc[idx, "concept_id"] = int(gold_cid)
            swaps += 1
            break

    iou_oracle = macro_char_iou(
        pred_oracle[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"  Oracle swaps: {swaps}")
    print(f"  Oracle IoU:   {iou_oracle:.4f} (delta: {iou_oracle - iou_baseline:+.4f})")
    print(f"  This is the ceiling from concept disambiguation alone.")

    # ===================================================================
    # Part 5: Oracle using only training-available concepts
    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 5: ORACLE USING ONLY TRAINING-AVAILABLE CONCEPTS")
    print("=" * 80)

    pred_oracle2 = pred_baseline.copy()
    swaps2 = 0
    for idx in pred_oracle2.index:
        nid = str(pred_oracle2.loc[idx, "note_id"])
        gold = gold_by_note.get(nid)
        if gold is None:
            continue
        s, e = int(pred_oracle2.loc[idx, "start"]), int(pred_oracle2.loc[idx, "end"])
        cid = int(pred_oracle2.loc[idx, "concept_id"])

        overlaps = gold[(gold["start"] < e) & (gold["end"] > s)]
        if overlaps.empty:
            continue
        if cid in overlaps["concept_id"].values:
            continue

        # Get the mention text
        text = test_notes[test_notes["note_id"] == pred_oracle2.loc[idx, "note_id"]]["text"].iloc[0]
        mention = text[s:e].lower()
        mention_norm = " ".join(mention.split())

        # Check if correct concept is in training dict
        any_key = ("any", mention_norm)
        if any_key in d_combined:
            available_cids = set(d_combined[any_key].keys())
            for gold_cid in overlaps["concept_id"].values:
                if int(gold_cid) in available_cids:
                    pred_oracle2.loc[idx, "concept_id"] = int(gold_cid)
                    swaps2 += 1
                    break

    iou_oracle2 = macro_char_iou(
        pred_oracle2[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"  Feasible oracle swaps: {swaps2}")
    print(f"  Feasible oracle IoU:   {iou_oracle2:.4f} (delta: {iou_oracle2 - iou_baseline:+.4f})")

    # ===================================================================
    # Part 6: Test practical disambiguation strategies
    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 6: PRACTICAL DISAMBIGUATION STRATEGIES")
    print("=" * 80)

    # Build the multi-concept dictionary (Counter values)
    # We need to also add SNOMED entries + replacements with Counter values
    # For now, start with just training dict as multi-concept

    # Strategy A: Use d_combined (Counter) directly for annotation
    # This emits all concepts at each position, then disambiguate
    print("\n  --- Strategy A: Training-frequency disambiguation ---")

    # We need the d_combined to pass through scoring/removal too
    # Build a version where we keep Counters through the pipeline
    d_multi = {}
    for k, counter in d_combined.items():
        d_multi[k] = counter  # Keep as Counter

    # Also build single-concept version for entries added after training
    # (SNOMED, replacements)
    d_single, uc_single = train(
        train_notes_df, train_annotations_df,
        super_dict_path=super_dict_path,
        flat_terminology_path=flat_term_path,
    )

    # Merge: for keys in d_single that aren't in d_multi, add them
    for k, v in d_single.items():
        if k not in d_multi:
            d_multi[k] = v  # Single concept int value

    pred_freq = predict_with_multi_concept_dict(
        d_multi, uc_single, test_notes,
        strategy="frequency",
        concept_priors=concept_priors,
    )
    iou_freq = macro_char_iou(
        pred_freq[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"  Frequency IoU:    {iou_freq:.4f} (delta: {iou_freq - iou_baseline:+.4f})")
    print(f"  Predictions: {len(pred_freq):,}")

    # Strategy B: Section-specific frequency
    print("\n  --- Strategy B: Section-frequency disambiguation ---")
    pred_secfreq = predict_with_multi_concept_dict(
        d_multi, uc_single, test_notes,
        strategy="section_freq",
        concept_priors=concept_priors,
        section_concept_priors=section_concept_priors,
    )
    iou_secfreq = macro_char_iou(
        pred_secfreq[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"  Section-freq IoU: {iou_secfreq:.4f} (delta: {iou_secfreq - iou_baseline:+.4f})")
    print(f"  Predictions: {len(pred_secfreq):,}")

    # Strategy C: Oracle disambiguation (only using training-available concepts)
    print("\n  --- Strategy C: Oracle disambiguation (training-available only) ---")
    pred_oracle_multi = predict_with_multi_concept_dict(
        d_multi, uc_single, test_notes,
        gold_by_note=gold_by_note,
        strategy="oracle",
        concept_priors=concept_priors,
    )
    iou_oracle_multi = macro_char_iou(
        pred_oracle_multi[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    print(f"  Oracle multi IoU: {iou_oracle_multi:.4f} (delta: {iou_oracle_multi - iou_baseline:+.4f})")
    print(f"  Predictions: {len(pred_oracle_multi):,}")

    # ===================================================================
    # Part 7: Per-concept IoU analysis for wrong-concept errors
    # ===================================================================
    print("\n" + "=" * 80)
    print("PART 7: PER-CONCEPT IoU IMPACT OF DISAMBIGUATION")
    print("=" * 80)

    baseline_class = class_char_iou(
        pred_baseline[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )
    oracle_class = class_char_iou(
        pred_oracle2[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]],
    )

    # Merge and find concepts with biggest improvement
    merged = baseline_class.merge(
        oracle_class, on="concept_id", suffixes=("_base", "_oracle")
    )
    merged["iou_delta"] = merged["iou_oracle"] - merged["iou_base"]
    improved = merged[merged["iou_delta"] > 0.001].sort_values("iou_delta", ascending=False)

    print(f"\n  Concepts improved by oracle disambiguation: {len(improved)}")
    if not improved.empty:
        print(f"  {'concept_id':>12} {'gt_chars':>8} {'base_iou':>8} {'oracle_iou':>10} {'delta':>8}")
        for _, row in improved.head(20).iterrows():
            print(f"  {int(row['concept_id']):>12} {int(row['gt_chars_base']):>8} "
                  f"{row['iou_base']:>8.4f} {row['iou_oracle']:>10.4f} {row['iou_delta']:>+8.4f}")

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Baseline IoU:                           {iou_baseline:.4f}")
    print(f"  Oracle (any concept):                   {iou_oracle:.4f} ({iou_oracle-iou_baseline:+.4f})")
    print(f"  Oracle (training-available):             {iou_oracle2:.4f} ({iou_oracle2-iou_baseline:+.4f})")
    print(f"  Oracle (multi-concept dict):             {iou_oracle_multi:.4f} ({iou_oracle_multi-iou_baseline:+.4f})")
    print(f"  Frequency disambiguation:               {iou_freq:.4f} ({iou_freq-iou_baseline:+.4f})")
    print(f"  Section-frequency disambiguation:       {iou_secfreq:.4f} ({iou_secfreq-iou_baseline:+.4f})")
    print(f"\n  Wrong-concept FPs in baseline: {wrong_concept_fp} ({wrong_concept_chars} chars)")
    print(f"  Ambiguous entries in training dict: {n_multi}")

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
