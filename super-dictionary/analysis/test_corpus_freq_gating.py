#!/usr/bin/env python3
"""Test corpus-frequency gating on the old-challenge-split test data.

End-to-end experiment:
1. Train dictionary on train split (same as baseline)
2. Score baseline on test data
3. Scan TEST notes for corpus frequency (simulating real eval)
4. Apply various gating strategies to filter the dictionary
5. Re-predict and re-score each variant
6. Report delta IoU

Key insight: at evaluation time, we see all test notes before classifying.
So we can count how often each dictionary entry fires in the test corpus,
and suppress high-frequency generic entries that are likely FPs.
"""
from __future__ import annotations

import copy
import pickle
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
)
from runtime_scoring import macro_char_iou, class_char_iou


# ---------------------------------------------------------------------------
# Prediction (same as test_old_challenge_split.predict_notes_trained, inlined)
# ---------------------------------------------------------------------------
def predict_with_dict(
    d: dict, uc_d: dict, notes: pd.DataFrame,
) -> pd.DataFrame:
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
    pred["start"] = pred["start"].astype(int)
    pred["end"] = pred["end"].astype(int)
    pred["concept_id"] = pred["concept_id"].astype(int)
    return pred


def score_predictions(pred: pd.DataFrame, gold: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Score predictions, return (macro_iou, class_df)."""
    pred_cols = pred[["note_id", "start", "end", "concept_id"]].copy()
    gold_cols = gold[["note_id", "start", "end", "concept_id"]].copy()
    score = macro_char_iou(pred_cols, gold_cols)
    cls = class_char_iou(pred_cols, gold_cols)
    return score, cls


# ---------------------------------------------------------------------------
# Corpus frequency counting on test notes
# ---------------------------------------------------------------------------
def count_mentions_in_corpus(
    mentions: set[str], corpus_text: str
) -> dict[str, int]:
    """Count regex pattern matches for each mention in the corpus."""
    counts = {}
    for mention in mentions:
        p = get_pattern(mention)
        if p is None:
            counts[mention] = 0
            continue
        counts[mention] = len(p.findall(corpus_text))
    return counts


# ---------------------------------------------------------------------------
# Gating strategies
# ---------------------------------------------------------------------------
def gate_by_corpus_count(
    d: dict, mention_counts: dict[str, int], threshold: int,
) -> dict:
    """Remove entries whose mention fires more than `threshold` times."""
    d_filtered = {}
    removed = 0
    for k, v in d.items():
        mention = k[1] if isinstance(k[1], str) else str(k[1])
        if mention_counts.get(mention, 0) > threshold:
            removed += 1
            continue
        d_filtered[k] = v
    return d_filtered


def gate_by_corpus_count_and_surface(
    d: dict, mention_counts: dict[str, int],
    count_threshold: int, min_word_len: float = 5.0,
) -> dict:
    """Remove entries that fire often AND have short/generic surface forms.

    Only removes entries where:
    - mention fires > count_threshold times in corpus
    - AND average word length < min_word_len (proxy for generic terms)
    """
    d_filtered = {}
    removed = 0
    for k, v in d.items():
        mention = k[1] if isinstance(k[1], str) else str(k[1])
        corpus_count = mention_counts.get(mention, 0)
        if corpus_count > count_threshold:
            words = mention.split()
            avg_wl = np.mean([len(w) for w in words]) if words else 0
            n_words = len(words)
            # Keep if: long medical-looking words, or multi-word phrases
            if avg_wl < min_word_len and n_words <= 2:
                removed += 1
                continue
        d_filtered[k] = v
    return d_filtered


def gate_by_frequency_rank(
    d: dict, mention_counts: dict[str, int], top_n_remove: int,
) -> dict:
    """Remove the top-N most frequent mentions from the dictionary."""
    # Rank mentions by corpus count
    mention_to_count = {}
    for k in d:
        mention = k[1] if isinstance(k[1], str) else str(k[1])
        mention_to_count[mention] = mention_counts.get(mention, 0)

    # Find top-N mentions
    ranked = sorted(mention_to_count.items(), key=lambda x: -x[1])
    to_remove = {m for m, c in ranked[:top_n_remove] if c > 0}

    d_filtered = {}
    for k, v in d.items():
        mention = k[1] if isinstance(k[1], str) else str(k[1])
        if mention in to_remove:
            continue
        d_filtered[k] = v
    return d_filtered


def gate_by_corpus_per_note(
    d: dict, mention_counts: dict[str, int],
    n_notes: int, fires_per_note_threshold: float,
) -> dict:
    """Remove entries that fire on average more than N times per note.

    This normalizes by corpus size - an entry firing 100 times across
    1000 notes (0.1/note) is fine, but 100 times across 10 notes (10/note)
    is suspicious.
    """
    d_filtered = {}
    for k, v in d.items():
        mention = k[1] if isinstance(k[1], str) else str(k[1])
        corpus_count = mention_counts.get(mention, 0)
        fires_per_note = corpus_count / max(n_notes, 1)
        if fires_per_note > fires_per_note_threshold:
            continue
        d_filtered[k] = v
    return d_filtered


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
        if col in test_gold.columns:
            test_gold[col] = test_gold[col].astype(int)

    print(f"  Train: {len(train_notes)} notes, {len(train_annotations)} annotations")
    print(f"  Test:  {len(test_notes)} notes, {len(test_gold)} annotations")

    # Step 1: Train dictionary
    print("\n" + "=" * 80)
    print("STEP 1: Training dictionary on train split")
    print("=" * 80)
    t1 = time.perf_counter()
    d, uc_d = train(
        train_notes, train_annotations,
        super_dict_path=super_dict_path,
        flat_terminology_path=flat_term_path,
    )
    print(f"Training done in {time.perf_counter() - t1:.1f}s")
    print(f"  Dict: {len(d):,} entries, UC dict: {len(uc_d):,} entries")

    # Step 2: Baseline predictions
    print("\n" + "=" * 80)
    print("STEP 2: Baseline predictions on test data")
    print("=" * 80)
    t2 = time.perf_counter()
    baseline_pred = predict_with_dict(d, uc_d, test_notes)
    baseline_iou, baseline_cls = score_predictions(baseline_pred, test_gold)
    print(f"Baseline predictions: {len(baseline_pred):,} in {time.perf_counter() - t2:.1f}s")
    print(f"Baseline macro char IoU: {baseline_iou:.4f}")

    # Step 3: Count mentions in TEST corpus
    print("\n" + "=" * 80)
    print("STEP 3: Counting mention frequencies in TEST notes")
    print("=" * 80)
    test_texts_lc = test_notes.set_index("note_id")["text"].str.lower()
    test_corpus = "\n".join(test_texts_lc.values)
    n_test_notes = len(test_notes)

    # Collect all unique mentions from the dictionary
    all_mentions = set()
    for k in d:
        all_mentions.add(k[1] if isinstance(k[1], str) else str(k[1]))
    for k in uc_d:
        all_mentions.add(k[1] if isinstance(k[1], str) else str(k[1]))
    print(f"  Unique mentions in dictionary: {len(all_mentions):,}")

    t3 = time.perf_counter()
    mention_counts = count_mentions_in_corpus(all_mentions, test_corpus)
    print(f"  Counting done in {time.perf_counter() - t3:.1f}s")

    # Stats
    counts = pd.Series(mention_counts)
    n_firing = (counts > 0).sum()
    print(f"  Mentions that fire in test corpus: {n_firing:,} / {len(counts):,}")
    print(f"  Top 20 most frequent mentions:")
    for mention, count in counts.nlargest(20).items():
        n_entries = sum(1 for k in d if (isinstance(k[1], str) and k[1] == mention) or str(k[1]) == mention)
        print(f"    {count:>6}x  ({n_entries} entries)  {mention}")

    # Also count in TRAIN corpus for the fire-without-annotation analysis
    print("\n  Also counting in TRAIN corpus for comparison...", flush=True)
    train_texts_lc = train_notes.set_index("note_id")["text"].str.lower()
    train_corpus = "\n".join(train_texts_lc.values)
    t3b = time.perf_counter()
    mention_counts_train = count_mentions_in_corpus(all_mentions, train_corpus)
    print(f"  Train counting done in {time.perf_counter() - t3b:.1f}s")

    # Build annotation counts
    if "span" in train_annotations.columns:
        ann_source = train_annotations["span"].str.lower()
    else:
        ann_source = train_annotations.apply(
            lambda r: train_texts_lc.get(r["note_id"], "")[int(r["start"]):int(r["end"])].lower(),
            axis=1,
        )
    mention_ann_counts = ann_source.value_counts().to_dict()

    # Step 4: Run gating experiments
    print("\n" + "=" * 80)
    print("STEP 4: Gating experiments")
    print("=" * 80)

    results = [("Baseline", baseline_iou, len(d), len(baseline_pred))]

    experiments = []

    # Experiment A: Simple corpus count threshold on TEST corpus
    for thr in [5, 10, 20, 50, 100, 200]:
        experiments.append((
            f"test_corpus_count>{thr}",
            lambda d_, thr_=thr: gate_by_corpus_count(d_, mention_counts, thr_),
        ))

    # Experiment B: Fires-per-note threshold (normalized by corpus size)
    for fpn in [0.5, 1.0, 2.0, 5.0]:
        experiments.append((
            f"fires_per_note>{fpn}",
            lambda d_, fpn_=fpn: gate_by_corpus_per_note(d_, mention_counts, n_test_notes, fpn_),
        ))

    # Experiment C: Corpus count + surface form filter
    for thr, min_wl in [(10, 5.0), (20, 5.0), (50, 4.0), (10, 4.0)]:
        experiments.append((
            f"corpus>{thr}_avgwl<{min_wl}",
            lambda d_, thr_=thr, mwl_=min_wl: gate_by_corpus_count_and_surface(
                d_, mention_counts, thr_, mwl_
            ),
        ))

    # Experiment D: Top-N removal
    for n in [10, 25, 50, 100, 200]:
        experiments.append((
            f"remove_top{n}",
            lambda d_, n_=n: gate_by_frequency_rank(d_, mention_counts, n_),
        ))

    # Experiment E: TRAIN corpus fire-without-annotation ratio
    # This uses train data (available at train time) for entries that exist
    # in the train dict, combined with test corpus count for entries that
    # only appear at test time
    for ratio_thr in [5, 10, 20, 50]:
        def _gate_fwa(d_, ratio_thr_=ratio_thr):
            d_filtered = {}
            for k, v in d_.items():
                mention = k[1] if isinstance(k[1], str) else str(k[1])
                train_count = mention_counts_train.get(mention, 0)
                ann_count = mention_ann_counts.get(mention, 0)
                test_count = mention_counts.get(mention, 0)

                # For entries with training signal: use fire/ann ratio
                if ann_count > 0:
                    ratio = train_count / max(ann_count, 1)
                    if ratio > ratio_thr_:
                        continue
                else:
                    # No annotation data - use test corpus count as proxy
                    # If it fires a lot in test notes, it's probably generic
                    if test_count > ratio_thr_ * 2:
                        continue
                d_filtered[k] = v
            return d_filtered
        experiments.append((f"fwa_ratio>{ratio_thr}", _gate_fwa))

    # Run all experiments
    for exp_name, gate_fn in experiments:
        d_gated = gate_fn(copy.copy(d))
        n_removed = len(d) - len(d_gated)
        pred = predict_with_dict(d_gated, uc_d, test_notes)
        iou, cls = score_predictions(pred, test_gold)
        delta = iou - baseline_iou
        results.append((exp_name, iou, len(d_gated), len(pred)))
        print(f"  {exp_name:<35} IoU={iou:.4f} ({delta:+.4f})  "
              f"dict={len(d_gated):,} (-{n_removed})  preds={len(pred):,}")

    # Summary table
    print("\n\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Experiment':<35} {'IoU':>8} {'Delta':>8} {'Dict size':>10} {'Preds':>8}")
    print("-" * 75)
    for name, iou, dict_size, n_preds in results:
        delta = iou - baseline_iou
        print(f"  {name:<33} {iou:>8.4f} {delta:>+8.4f} {dict_size:>10,} {n_preds:>8,}")

    # Detailed analysis of best experiment
    best = max(results[1:], key=lambda x: x[1])
    print(f"\n  Best: {best[0]} (IoU={best[1]:.4f}, delta={best[1] - baseline_iou:+.4f})")

    # Show what the best experiment removed
    print(f"\n\n--- Entries removed by best experiment ({best[0]}) ---")
    # Re-run to get the removed entries
    best_gate_fn = dict(experiments)[best[0]]
    d_best = best_gate_fn(copy.copy(d))
    removed_entries = {k: d[k] for k in d if k not in d_best}
    print(f"  Removed {len(removed_entries)} entries:")
    # Show them sorted by test corpus count
    removed_with_counts = []
    for k, v in removed_entries.items():
        mention = k[1] if isinstance(k[1], str) else str(k[1])
        removed_with_counts.append((
            mention, k[0], v,
            mention_counts.get(mention, 0),
            mention_counts_train.get(mention, 0),
            mention_ann_counts.get(mention, 0),
        ))
    removed_with_counts.sort(key=lambda x: -x[3])
    for mention, section, cid, tc, trc, ac in removed_with_counts[:50]:
        print(f"    test={tc:>5} train={trc:>5} ann={ac:>3}  "
              f"[{str(section):<20}] {mention:<40} -> {cid}")

    # FP/FN analysis: what changed between baseline and best?
    print(f"\n\n--- FP/FN analysis (best vs baseline) ---")
    best_pred = predict_with_dict(d_best, uc_d, test_notes)
    _, best_cls = score_predictions(best_pred, test_gold)
    _, baseline_cls = score_predictions(baseline_pred, test_gold)

    baseline_cls = baseline_cls.set_index("concept_id")
    best_cls = best_cls.set_index("concept_id")

    # Concepts that improved
    common = baseline_cls.index.intersection(best_cls.index)
    deltas = best_cls.loc[common, "iou"] - baseline_cls.loc[common, "iou"]
    improved = deltas[deltas > 0.01].sort_values(ascending=False)
    worsened = deltas[deltas < -0.01].sort_values()

    print(f"\n  Concepts that IMPROVED (IoU delta > 0.01): {len(improved)}")
    for cid, delta in improved.head(20).items():
        bi = baseline_cls.loc[cid, "iou"]
        ni = best_cls.loc[cid, "iou"]
        print(f"    concept={cid:<12} {bi:.3f} -> {ni:.3f} ({delta:+.3f})")

    print(f"\n  Concepts that WORSENED (IoU delta < -0.01): {len(worsened)}")
    for cid, delta in worsened.head(20).items():
        bi = baseline_cls.loc[cid, "iou"]
        ni = best_cls.loc[cid, "iou"]
        print(f"    concept={cid:<12} {bi:.3f} -> {ni:.3f} ({delta:+.3f})")

    # New FPs introduced by best (shouldn't be any since we only remove)
    best_only = best_cls.index.difference(baseline_cls.index)
    baseline_only = baseline_cls.index.difference(best_cls.index)
    print(f"\n  Concepts only in best: {len(best_only)}")
    print(f"  Concepts only in baseline: {len(baseline_only)}")

    print(f"\nTotal experiment time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
