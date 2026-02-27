#!/usr/bin/env python3
"""Refined negative example analysis — filters out very short spans and
focuses on the patterns most useful for rule generation training.

Categories of negative examples:
1. UNANNOTATED: Same span text appears in corpus but not annotated
   (span should NOT be linked to SNOMED in that context)
2. DIFFERENT CONCEPT: Same span text annotated with a different concept
   (span is ambiguous — concept depends on context)
3. PARTIAL OVERLAP: Span text matches but overlaps with a different annotation
   (annotation boundaries differ)
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def build_normalized_text_and_map(text: str) -> tuple[str, list[int]]:
    normalized_chars = []
    offset_map = []
    prev_was_space = False
    for i, ch in enumerate(text):
        if ch in (' ', '\t', '\n', '\r', '\f', '\v'):
            if not prev_was_space and normalized_chars:
                normalized_chars.append(' ')
                offset_map.append(i)
                prev_was_space = True
        else:
            normalized_chars.append(ch)
            offset_map.append(i)
            prev_was_space = False
    if normalized_chars and normalized_chars[-1] == ' ':
        normalized_chars.pop()
        offset_map.pop()
    return ''.join(normalized_chars), offset_map


def find_all_occurrences(text: str, pattern: str) -> list[tuple[int, int]]:
    if not pattern:
        return []
    results = []
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        results.append((pos, pos + len(pattern)))
        start = pos + 1
    return results


def is_word_boundary(text: str, start: int, end: int) -> bool:
    """Check if the occurrence is at word boundaries (not a substring of another word)."""
    if start > 0 and text[start - 1].isalnum():
        return False
    if end < len(text) and text[end].isalnum():
        return False
    return True


def main():
    t0 = time.time()
    MIN_SPAN_LEN = 3  # ignore 1-2 char spans

    print("Loading data...")
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    ann_df = pd.read_csv(SPLIT_DIR / "train_annotations.csv")
    print(f"  {len(notes_df)} notes, {len(ann_df)} annotations")

    # Build annotation index: note_id -> list of (start, end, concept_id)
    ann_by_note: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for _, row in ann_df.iterrows():
        ann_by_note[row["note_id"]].append((
            int(row["start"]), int(row["end"]), int(row["concept_id"])
        ))

    # Unique span texts -> concept_ids used for that span
    span_concept_map: dict[str, set[int]] = defaultdict(set)
    span_ann_count: Counter = Counter()
    for _, row in ann_df.iterrows():
        span_text = str(row["span"]) if pd.notna(row["span"]) else ""
        if len(span_text) >= MIN_SPAN_LEN:
            span_concept_map[span_text].add(int(row["concept_id"]))
            span_ann_count[span_text] += 1

    print(f"  {len(span_concept_map)} unique spans (>= {MIN_SPAN_LEN} chars)")

    # Pre-process notes
    note_texts: dict[str, str] = {}
    note_norm: dict[str, tuple[str, list[int]]] = {}
    for _, row in notes_df.iterrows():
        nid = row["note_id"]
        text = row["text"]
        note_texts[nid] = text
        note_norm[nid] = build_normalized_text_and_map(text)

    # --- Scan ---
    print("Scanning...")
    stats_exact_unannotated = 0
    stats_exact_unannotated_wordboundary = 0
    stats_exact_diff_concept = 0
    stats_exact_same_concept = 0
    stats_fuzzy_unannotated = 0
    stats_fuzzy_diff_concept = 0
    stats_partial_overlap = 0

    per_span_results = {}
    n_spans = len(span_concept_map)
    checkpoint = max(1, n_spans // 20)

    sorted_spans = sorted(span_concept_map.keys(), key=lambda s: -span_ann_count[s])

    for i, span_text in enumerate(sorted_spans):
        if (i + 1) % checkpoint == 0:
            print(f"  {i+1}/{n_spans} ({100*(i+1)/n_spans:.0f}%)")

        gold_cids = span_concept_map[span_text]
        span_lower = span_text.lower()
        span_norm = normalize_whitespace(span_text)
        span_norm_lower = span_norm.lower()

        sp_exact_unannotated = 0
        sp_exact_unannotated_wb = 0
        sp_exact_diff = 0
        sp_exact_same = 0
        sp_fuzzy_unannotated = 0
        sp_fuzzy_diff = 0
        sp_partial = 0
        sp_examples: list[dict] = []

        for nid, text in note_texts.items():
            anns = ann_by_note.get(nid, [])

            # Exact search
            exact_occs = find_all_occurrences(text, span_text)
            exact_pos_set = set()

            for os_, oe_ in exact_occs:
                exact_pos_set.add((os_, oe_))
                # Check against annotations
                matched = False
                for as_, ae_, acid in anns:
                    if os_ == as_ and oe_ == ae_:
                        matched = True
                        if acid in gold_cids:
                            sp_exact_same += 1
                        else:
                            sp_exact_diff += 1
                            if len(sp_examples) < 5:
                                sp_examples.append({
                                    "type": "diff_concept",
                                    "note_id": nid, "start": os_, "end": oe_,
                                    "found_cid": acid,
                                    "expected_cids": sorted(gold_cids),
                                    "context": text[max(0,os_-50):oe_+50].replace('\n','\\n'),
                                })
                        break
                    elif os_ < ae_ and oe_ > as_:
                        # Partial overlap
                        matched = True
                        sp_partial += 1
                        break

                if not matched:
                    sp_exact_unannotated += 1
                    wb = is_word_boundary(text, os_, oe_)
                    if wb:
                        sp_exact_unannotated_wb += 1
                        if len(sp_examples) < 5:
                            sp_examples.append({
                                "type": "unannotated_exact_wb",
                                "note_id": nid, "start": os_, "end": oe_,
                                "context": text[max(0,os_-50):oe_+50].replace('\n','\\n'),
                            })

            # Fuzzy search (case-insensitive + whitespace-normalized)
            norm_text, offset_map = note_norm[nid]
            if not offset_map:
                continue
            fuzzy_occs = find_all_occurrences(norm_text.lower(), span_norm_lower)
            for fpos, fend in fuzzy_occs:
                fo = offset_map[fpos]
                fe = offset_map[min(fend - 1, len(offset_map) - 1)] + 1
                # Skip if already found as exact
                if (fo, fe) in exact_pos_set:
                    continue
                # Check near-duplicates
                skip = False
                for ep in exact_pos_set:
                    if abs(fo - ep[0]) <= 2 and abs(fe - ep[1]) <= 2:
                        skip = True
                        break
                if skip:
                    continue

                matched = False
                for as_, ae_, acid in anns:
                    if fo == as_ and fe == ae_:
                        matched = True
                        if acid not in gold_cids:
                            sp_fuzzy_diff += 1
                        break
                    elif fo < ae_ and fe > as_:
                        matched = True
                        break

                if not matched:
                    sp_fuzzy_unannotated += 1

        stats_exact_unannotated += sp_exact_unannotated
        stats_exact_unannotated_wordboundary += sp_exact_unannotated_wb
        stats_exact_diff_concept += sp_exact_diff
        stats_exact_same_concept += sp_exact_same
        stats_fuzzy_unannotated += sp_fuzzy_unannotated
        stats_fuzzy_diff_concept += sp_fuzzy_diff
        stats_partial_overlap += sp_partial

        total_neg = sp_exact_unannotated_wb + sp_exact_diff + sp_fuzzy_unannotated + sp_fuzzy_diff
        if total_neg > 0:
            per_span_results[span_text] = {
                "ann_count": span_ann_count[span_text],
                "n_gold_concepts": len(gold_cids),
                "exact_unannotated_all": sp_exact_unannotated,
                "exact_unannotated_word_boundary": sp_exact_unannotated_wb,
                "exact_diff_concept": sp_exact_diff,
                "exact_same_concept": sp_exact_same,
                "fuzzy_unannotated": sp_fuzzy_unannotated,
                "fuzzy_diff_concept": sp_fuzzy_diff,
                "partial_overlap": sp_partial,
                "total_negative_wb": total_neg,
                "examples": sp_examples,
            }

    elapsed = time.time() - t0

    # --- Report ---
    print("\n" + "=" * 70)
    print("NEGATIVE EXAMPLE ANALYSIS (spans >= 3 chars)")
    print("=" * 70)

    print(f"\nTotal training annotations: {len(ann_df)}")
    print(f"Unique spans (>= 3 chars): {len(span_concept_map)}")
    print(f"Spans with negative examples: {len(per_span_results)}")

    print(f"\n--- Exact Match ---")
    print(f"  Positive (same concept): {stats_exact_same_concept}")
    print(f"  Unannotated (all): {stats_exact_unannotated}")
    print(f"  Unannotated (word boundary only): {stats_exact_unannotated_wordboundary}")
    print(f"  Different concept: {stats_exact_diff_concept}")
    print(f"  Partial overlap: {stats_partial_overlap}")

    print(f"\n--- Fuzzy Only (additional to exact) ---")
    print(f"  Unannotated: {stats_fuzzy_unannotated}")
    print(f"  Different concept: {stats_fuzzy_diff_concept}")

    total_useful_neg = stats_exact_unannotated_wordboundary + stats_exact_diff_concept + stats_fuzzy_unannotated + stats_fuzzy_diff_concept
    print(f"\n--- Useful Negative Examples (word-boundary exact + fuzzy) ---")
    print(f"  Total: {total_useful_neg}")
    print(f"  Ratio to positive annotations: {total_useful_neg / len(ann_df):.1f}x")

    # Top spans by word-boundary unannotated
    print(f"\n--- Top 40 Spans with Most Word-Boundary Unannotated Occurrences ---")
    top_wb = sorted(per_span_results.items(),
                    key=lambda x: -x[1]["exact_unannotated_word_boundary"])[:40]
    print(f"{'Span':<35} {'Ann#':>5} {'WB-Unann':>8} {'Diff':>5} {'Ratio':>7}")
    print("-" * 65)
    for span_text, st in top_wb:
        display = span_text[:33].replace('\n', '\\n')
        ratio = st["exact_unannotated_word_boundary"] / max(1, st["ann_count"])
        print(f"{display:<35} {st['ann_count']:>5} {st['exact_unannotated_word_boundary']:>8} "
              f"{st['exact_diff_concept']:>5} {ratio:>7.1f}")

    # Spans with multiple concepts (ambiguous)
    multi_concept = {k: v for k, v in per_span_results.items()
                     if v["n_gold_concepts"] > 1}
    print(f"\n--- Ambiguous Spans (annotated with >1 concept across corpus) ---")
    print(f"  Count: {len(multi_concept)}")
    top_ambig = sorted(multi_concept.items(),
                       key=lambda x: -x[1]["n_gold_concepts"])[:30]
    print(f"  {'Span':<35} {'#Concepts':>9} {'Ann#':>5} {'DiffAnn':>7}")
    print("  " + "-" * 60)
    for span_text, st in top_ambig:
        display = span_text[:33].replace('\n', '\\n')
        print(f"  {display:<35} {st['n_gold_concepts']:>9} {st['ann_count']:>5} "
              f"{st['exact_diff_concept'] + st['fuzzy_diff_concept']:>7}")

    # Show some concrete examples of unannotated occurrences for meaningful spans
    print(f"\n--- Sample Negative Examples (unannotated, word-boundary) ---")
    # Pick spans that are clearly clinical terms, not just common words
    clinical_spans = [(k, v) for k, v in per_span_results.items()
                      if v["exact_unannotated_word_boundary"] >= 3
                      and len(k) >= 5
                      and v["ann_count"] >= 3]
    clinical_spans.sort(key=lambda x: -(x[1]["exact_unannotated_word_boundary"] / max(1, x[1]["ann_count"])))

    for span_text, st in clinical_spans[:20]:
        examples = [e for e in st["examples"] if e["type"] == "unannotated_exact_wb"][:2]
        if examples:
            print(f"\n  Span: '{span_text}' (annotated {st['ann_count']}x, unannotated {st['exact_unannotated_word_boundary']}x)")
            for ex in examples:
                ctx = ex["context"]
                if len(ctx) > 120:
                    ctx = ctx[:120] + "..."
                print(f"    [{ex['note_id']}:{ex['start']}] ...{ctx}...")

    print(f"\nElapsed: {elapsed:.1f}s")

    # Save
    output = {
        "summary": {
            "total_annotations": int(len(ann_df)),
            "unique_spans_ge3": int(len(span_concept_map)),
            "spans_with_negatives": int(len(per_span_results)),
            "exact_positive_same_concept": int(stats_exact_same_concept),
            "exact_unannotated_all": int(stats_exact_unannotated),
            "exact_unannotated_word_boundary": int(stats_exact_unannotated_wordboundary),
            "exact_diff_concept": int(stats_exact_diff_concept),
            "fuzzy_unannotated": int(stats_fuzzy_unannotated),
            "fuzzy_diff_concept": int(stats_fuzzy_diff_concept),
            "partial_overlap": int(stats_partial_overlap),
            "total_useful_negative": int(total_useful_neg),
        },
        "ambiguous_spans_count": int(len(multi_concept)),
        "top_40_unannotated": [
            {"span": k, **{kk: vv for kk, vv in v.items() if kk != "examples"}}
            for k, v in top_wb
        ],
    }

    out_path = REPO_ROOT / "scripts" / "negative_example_analysis_v2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
