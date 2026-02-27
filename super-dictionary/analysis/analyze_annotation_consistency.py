#!/usr/bin/env python3
"""Analyze annotation consistency in the training data.

For each mention text that maps to multiple SNOMED concepts, determine:
1. NOISE: Same context → different concepts (annotator disagreement)
2. SIGNAL: Different context → different concepts (genuine disambiguation)

This tells us the true ceiling for concept disambiguation — noise cases
can only be resolved by majority vote, not by any model.
"""
from __future__ import annotations

import re
import sys
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from train_dictionary import (
    COMMON_HEADERS, BLACKLIST_THRESH, INTERNAL_BLACKLIST,
    IndexedDict, annotate_with_dict, remove_overlaps,
    CASE_SENSITIVE_DICT, train, build_dict_from_annotations,
)
from runtime_scoring import macro_char_iou


def normalize_context(text: str, n_words: int = 15) -> str:
    """Normalize a context string for comparison."""
    text = re.sub(r'\s+', ' ', text.lower().strip())
    text = re.sub(r'[_]{3,}', '___', text)  # Normalize deidentification markers
    text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', 'DATE', text)
    return text


def context_fingerprint(before: str, after: str, n_words: int = 10) -> str:
    """Create a fingerprint from surrounding context words."""
    before_words = normalize_context(before).split()[-n_words:]
    after_words = normalize_context(after).split()[:n_words:]
    return ' '.join(before_words) + ' | ' + ' '.join(after_words)


def main():
    split_dir = REPO_ROOT / "data" / "old-challenge-split"
    super_dict_path = REPO_ROOT / "data" / "interim" / "super_dictionary_full.tsv"
    flat_term_path = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"

    print("Loading data...", flush=True)
    train_notes_df = pd.read_csv(split_dir / "train_notes.csv")
    train_annotations_df = pd.read_csv(split_dir / "train_annotations.csv")
    test_notes = pd.read_csv(split_dir / "test_notes.csv")
    test_gold = pd.read_csv(split_dir / "test_annotations.csv")
    for col in ["start", "end", "concept_id"]:
        test_gold[col] = test_gold[col].astype(int)

    ft = pd.read_csv(REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv")
    concept_names = dict(zip(ft["concept_id"].astype(int), ft["concept_name"]))
    train_texts = {row["note_id"]: row["text"] for _, row in train_notes_df.iterrows()}
    test_text_by_id = {str(r["note_id"]): r["text"] for _, r in test_notes.iterrows()}

    # ================================================================
    # Step 1: Analyze training annotation consistency
    # ================================================================
    print("\n" + "=" * 80)
    print("STEP 1: TRAINING ANNOTATION CONSISTENCY")
    print("=" * 80)

    # Group annotations by normalized mention text
    mention_annotations = defaultdict(list)
    for _, row in train_annotations_df.iterrows():
        note_text = train_texts.get(row["note_id"], "")
        s, e = int(row["start"]), int(row["end"])
        mention = note_text[s:e].lower().strip()
        mention = re.sub(r'\s+', ' ', mention)
        cid = int(row["concept_id"])

        before = note_text[max(0, s - 120):s]
        after = note_text[e:min(len(note_text), e + 120)]
        fp = context_fingerprint(before, after)

        mention_annotations[mention].append({
            "concept_id": cid,
            "context_fp": fp,
            "before": before[-80:].replace('\n', ' '),
            "after": after[:80].replace('\n', ' '),
            "note_id": row["note_id"],
        })

    # Classify each multi-concept mention
    n_single_concept = 0
    n_multi_concept = 0
    noise_mentions = []  # Same context → different concepts
    signal_mentions = []  # Different context → different concepts
    mixed_mentions = []  # Some noise, some signal

    for mention, anns in mention_annotations.items():
        concepts = set(a["concept_id"] for a in anns)
        if len(concepts) < 2:
            n_single_concept += 1
            continue
        n_multi_concept += 1

        # Group by context fingerprint
        fp_concepts = defaultdict(set)
        fp_counts = defaultdict(Counter)
        for a in anns:
            fp_concepts[a["context_fp"]].add(a["concept_id"])
            fp_counts[a["context_fp"]][a["concept_id"]] += 1

        # Count contexts with multiple concepts (noise)
        noisy_fps = {fp for fp, cids in fp_concepts.items() if len(cids) > 1}
        clean_fps = {fp for fp, cids in fp_concepts.items() if len(cids) == 1}

        # Categorize
        n_noisy_anns = sum(
            sum(fp_counts[fp].values()) for fp in noisy_fps
        )
        n_total_anns = len(anns)

        entry = {
            "mention": mention,
            "n_annotations": n_total_anns,
            "n_concepts": len(concepts),
            "concepts": concepts,
            "n_noisy_contexts": len(noisy_fps),
            "n_clean_contexts": len(clean_fps),
            "noise_fraction": n_noisy_anns / n_total_anns if n_total_anns > 0 else 0,
            "concept_counts": Counter(a["concept_id"] for a in anns),
        }

        if len(noisy_fps) > 0 and len(clean_fps) == 0:
            noise_mentions.append(entry)
        elif len(noisy_fps) == 0:
            signal_mentions.append(entry)
        else:
            mixed_mentions.append(entry)

    print(f"\n  Total unique mentions: {len(mention_annotations):,}")
    print(f"  Single-concept mentions: {n_single_concept:,}")
    print(f"  Multi-concept mentions: {n_multi_concept:,}")
    print(f"    Pure noise (same ctx → diff concepts): {len(noise_mentions)}")
    print(f"    Pure signal (diff ctx → diff concepts): {len(signal_mentions)}")
    print(f"    Mixed (some noise, some signal):        {len(mixed_mentions)}")

    # Show top noise mentions
    noise_mentions.sort(key=lambda x: -x["n_annotations"])
    print(f"\n  TOP NOISE MENTIONS (same context → different concepts):")
    for entry in noise_mentions[:20]:
        ccs = ", ".join(
            f"{concept_names.get(c, c)[:40]}={cnt}"
            for c, cnt in entry["concept_counts"].most_common()
        )
        print(f"    \"{entry['mention']}\" ({entry['n_annotations']} anns, "
              f"{entry['n_concepts']} concepts, {entry['noise_fraction']:.0%} noisy): {ccs}")

    print(f"\n  TOP MIXED MENTIONS (some noise, some signal):")
    mixed_mentions.sort(key=lambda x: -x["n_annotations"])
    for entry in mixed_mentions[:20]:
        ccs = ", ".join(
            f"{concept_names.get(c, c)[:40]}={cnt}"
            for c, cnt in entry["concept_counts"].most_common()
        )
        print(f"    \"{entry['mention']}\" ({entry['n_annotations']} anns, "
              f"{entry['noise_fraction']:.0%} noisy): {ccs}")

    print(f"\n  TOP SIGNAL MENTIONS (genuine disambiguation):")
    signal_mentions.sort(key=lambda x: -x["n_annotations"])
    for entry in signal_mentions[:20]:
        ccs = ", ".join(
            f"{str(concept_names.get(c, c))[:40]}={cnt}"
            for c, cnt in entry["concept_counts"].most_common()
        )
        print(f"    \"{entry['mention']}\" ({entry['n_annotations']} anns, "
              f"{entry['n_concepts']} concepts): {ccs}")

    # ================================================================
    # Step 2: Quantify impact on test predictions
    # ================================================================
    print("\n" + "=" * 80)
    print("STEP 2: IMPACT ON TEST PREDICTIONS")
    print("=" * 80)

    # Build mention priors
    texts = train_notes_df.set_index("note_id")["text"]
    texts_lc = texts.str.lower()
    headers = [h.lower() for h in COMMON_HEADERS]
    annotations = train_annotations_df.copy()
    if "span" in annotations.columns:
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

    mention_freq = defaultdict(Counter)
    for (section, mention), counter in d_combined.items():
        mention_freq[mention] += counter
    mention_freq = dict(mention_freq)

    # Build baseline predictions
    print("  Building baseline...", flush=True)
    d_baseline, uc_baseline = train(
        train_notes_df, train_annotations_df,
        super_dict_path=super_dict_path,
        flat_terminology_path=flat_term_path,
    )

    uc_d_full = dict(uc_baseline)
    uc_d_full.update(CASE_SENSITIVE_DICT)
    headers_lc = [h.lower() for h in COMMON_HEADERS]
    headers_orig = list(COMMON_HEADERS)
    d_indexed = IndexedDict(d_baseline, prefilter="bigram")
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
    pred_baseline = pd.concat(all_preds, ignore_index=True)
    for c in ["start", "end", "concept_id"]:
        pred_baseline[c] = pred_baseline[c].astype(int)

    iou_baseline = macro_char_iou(
        pred_baseline[["note_id", "start", "end", "concept_id"]],
        test_gold[["note_id", "start", "end", "concept_id"]]
    )
    print(f"  Baseline IoU: {iou_baseline:.4f}")

    gold_by_note = {str(nid): g for nid, g in test_gold.groupby("note_id")}

    # Classify ambiguous test predictions
    # Build noise lookup: mention -> noise_fraction
    mention_noise = {}
    for entry in noise_mentions + mixed_mentions + signal_mentions:
        mention_noise[entry["mention"]] = entry["noise_fraction"]

    n_ambiguous = 0
    n_wrong_concept = 0
    noise_wrong = 0
    signal_wrong = 0
    mixed_wrong = 0
    unknown_wrong = 0
    noise_chars = 0
    signal_chars = 0
    mixed_chars = 0
    unknown_chars = 0

    noise_set = {e["mention"] for e in noise_mentions}
    signal_set = {e["mention"] for e in signal_mentions}
    mixed_set = {e["mention"] for e in mixed_mentions}

    for idx in pred_baseline.index:
        s, e = int(pred_baseline.loc[idx, "start"]), int(pred_baseline.loc[idx, "end"])
        cid = int(pred_baseline.loc[idx, "concept_id"])
        nid = str(pred_baseline.loc[idx, "note_id"])
        span_len = e - s

        note_text = test_text_by_id.get(nid, "")
        mention = re.sub(r'\s+', ' ', note_text[s:e].lower().strip())

        alternatives = mention_freq.get(mention, {})
        if len(alternatives) < 2:
            continue
        total = sum(alternatives.values())
        sorted_alts = sorted(alternatives.items(), key=lambda x: -x[1])
        top_share = sorted_alts[0][1] / total
        if top_share >= 0.95:
            continue

        n_ambiguous += 1

        # Check if prediction is wrong
        gold = gold_by_note.get(nid)
        gold_cids = set()
        if gold is not None:
            overlaps = gold[(gold["start"] < e) & (gold["end"] > s)]
            gold_cids = set(overlaps["concept_id"].astype(int))

        if gold_cids and cid not in gold_cids:
            n_wrong_concept += 1
            if mention in noise_set:
                noise_wrong += 1
                noise_chars += span_len
            elif mention in signal_set:
                signal_wrong += 1
                signal_chars += span_len
            elif mention in mixed_set:
                mixed_wrong += 1
                mixed_chars += span_len
            else:
                unknown_wrong += 1
                unknown_chars += span_len

    print(f"\n  Ambiguous test predictions: {n_ambiguous:,}")
    print(f"  Wrong concept predictions: {n_wrong_concept:,}")
    print(f"\n  Wrong concept breakdown:")
    print(f"    Noise (annotator disagreement):   {noise_wrong:4d} ({noise_chars:,} chars)")
    print(f"    Signal (genuine disambiguation):  {signal_wrong:4d} ({signal_chars:,} chars)")
    print(f"    Mixed (some noise, some signal):  {mixed_wrong:4d} ({mixed_chars:,} chars)")
    print(f"    Unknown (not in training):        {unknown_wrong:4d} ({unknown_chars:,} chars)")

    total_wrong_chars = noise_chars + signal_chars + mixed_chars + unknown_chars
    if total_wrong_chars > 0:
        print(f"\n  Char-level breakdown of wrong-concept errors:")
        print(f"    Noise:   {100*noise_chars/total_wrong_chars:.1f}%")
        print(f"    Signal:  {100*signal_chars/total_wrong_chars:.1f}%")
        print(f"    Mixed:   {100*mixed_chars/total_wrong_chars:.1f}%")
        print(f"    Unknown: {100*unknown_chars/total_wrong_chars:.1f}%")

    # ================================================================
    # Step 3: Oracle analysis — if we fix only signal errors
    # ================================================================
    print(f"\n{'=' * 80}")
    print("STEP 3: ORACLE ANALYSIS")
    print("=" * 80)

    # Oracle: fix all wrong-concept predictions (any available concept)
    pred_oracle_all = pred_baseline.copy()
    pred_oracle_signal = pred_baseline.copy()
    pred_oracle_noise_majority = pred_baseline.copy()
    n_fixed_all = 0
    n_fixed_signal = 0
    n_fixed_noise = 0

    for idx in pred_baseline.index:
        s, e = int(pred_baseline.loc[idx, "start"]), int(pred_baseline.loc[idx, "end"])
        cid = int(pred_baseline.loc[idx, "concept_id"])
        nid = str(pred_baseline.loc[idx, "note_id"])

        note_text = test_text_by_id.get(nid, "")
        mention = re.sub(r'\s+', ' ', note_text[s:e].lower().strip())

        alternatives = mention_freq.get(mention, {})
        if len(alternatives) < 2:
            continue

        gold = gold_by_note.get(nid)
        if gold is None:
            continue
        overlaps = gold[(gold["start"] < e) & (gold["end"] > s)]
        gold_cids = set(overlaps["concept_id"].astype(int))
        if not gold_cids or cid in gold_cids:
            continue

        # Try to find correct concept in alternatives
        correct = gold_cids & set(alternatives.keys())
        if correct:
            best = max(correct, key=lambda c: alternatives.get(c, 0))
            pred_oracle_all.loc[idx, "concept_id"] = best
            n_fixed_all += 1

            if mention in signal_set:
                pred_oracle_signal.loc[idx, "concept_id"] = best
                n_fixed_signal += 1

        # For noise mentions: swap to majority vote
        if mention in noise_set or mention in mixed_set:
            most_common = max(alternatives, key=alternatives.get)
            if most_common != cid:
                pred_oracle_noise_majority.loc[idx, "concept_id"] = most_common
                n_fixed_noise += 1

    for name, pred, n in [
        ("Oracle: fix all wrong concepts", pred_oracle_all, n_fixed_all),
        ("Oracle: fix only signal errors", pred_oracle_signal, n_fixed_signal),
        ("Majority vote on noise mentions", pred_oracle_noise_majority, n_fixed_noise),
    ]:
        ev = pred[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]:
            ev[c] = ev[c].astype(int)
        iou = macro_char_iou(ev, test_gold[["note_id", "start", "end", "concept_id"]])
        print(f"  {name}: IoU={iou:.4f} ({iou-iou_baseline:+.4f}) fixed={n}")

    # ================================================================
    # Step 4: Broader context fingerprint (looser matching)
    # ================================================================
    print(f"\n{'=' * 80}")
    print("STEP 4: CONTEXT SIMILARITY ANALYSIS (varying window sizes)")
    print("=" * 80)

    for n_words in [5, 10, 15, 20]:
        n_noisy = 0
        n_clean = 0
        n_multi = 0

        for mention, anns in mention_annotations.items():
            concepts = set(a["concept_id"] for a in anns)
            if len(concepts) < 2:
                continue
            n_multi += 1

            fp_concepts = defaultdict(set)
            for a in anns:
                fp = context_fingerprint(a["before"], a["after"], n_words=n_words)
                fp_concepts[fp].add(a["concept_id"])

            has_noise = any(len(cids) > 1 for cids in fp_concepts.values())
            if has_noise:
                n_noisy += 1
            else:
                n_clean += 1

        print(f"  Window={n_words} words: {n_noisy}/{n_multi} mentions have noise "
              f"({100*n_noisy/max(n_multi,1):.0f}%), "
              f"{n_clean} clean disambiguation")


if __name__ == "__main__":
    main()
