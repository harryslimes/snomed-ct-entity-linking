#!/usr/bin/env python3
"""Analyze negative training examples: places where an annotated span text
appears in the training corpus but was NOT annotated (or annotated differently).

This includes both exact and fuzzy matching (whitespace/newline normalization,
case-insensitive).
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
    """Collapse all whitespace (including newlines) to single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


def build_annotation_index(ann_df: pd.DataFrame) -> dict[str, set[tuple[str, int, int, int]]]:
    """Build index: note_id -> set of (note_id, start, end, concept_id)."""
    idx: dict[str, set[tuple[str, int, int, int]]] = defaultdict(set)
    for _, row in ann_df.iterrows():
        idx[row["note_id"]].add((
            row["note_id"],
            int(row["start"]),
            int(row["end"]),
            int(row["concept_id"]),
        ))
    return dict(idx)


def find_all_occurrences_exact(text: str, pattern: str) -> list[tuple[int, int]]:
    """Find all exact occurrences of pattern in text. Returns (start, end) pairs."""
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


def find_all_occurrences_fuzzy(text_normalized: str, pattern_normalized: str,
                                offset_map: list[int]) -> list[tuple[int, int]]:
    """Find all fuzzy occurrences (whitespace-normalized) in the text.

    text_normalized: the whitespace-normalized version of the original text
    pattern_normalized: the whitespace-normalized span
    offset_map: maps each char position in normalized text back to original text position

    Returns (orig_start, orig_end) pairs in the original text coordinates.
    """
    if not pattern_normalized:
        return []
    results = []
    start = 0
    while True:
        pos = text_normalized.find(pattern_normalized, start)
        if pos == -1:
            break
        end = pos + len(pattern_normalized)
        # Map back to original coordinates
        orig_start = offset_map[pos]
        orig_end = offset_map[end - 1] + 1  # +1 because end is exclusive
        results.append((orig_start, orig_end))
        start = pos + 1
    return results


def build_normalized_text_and_map(text: str) -> tuple[str, list[int]]:
    """Build whitespace-normalized text and a mapping from normalized positions
    back to original positions.

    Returns (normalized_text, offset_map) where offset_map[i] gives the
    original text position corresponding to normalized_text[i].
    """
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

    # Strip trailing space
    if normalized_chars and normalized_chars[-1] == ' ':
        normalized_chars.pop()
        offset_map.pop()

    return ''.join(normalized_chars), offset_map


def overlaps_annotation(start: int, end: int, annotations: set[tuple[str, int, int, int]]) -> tuple[bool, int | None]:
    """Check if a (start, end) span overlaps any existing annotation.

    Returns (exact_match, concept_id_if_exact_match).
    If it's an exact positional match, returns the concept_id.
    If it partially overlaps, returns (True, None) indicating overlap but not exact.
    If no overlap, returns (False, None).
    """
    for _, ann_start, ann_end, cid in annotations:
        if start == ann_start and end == ann_end:
            return True, cid
        # Check overlap
        if start < ann_end and end > ann_start:
            return True, None  # partial overlap
    return False, None


def main():
    t0 = time.time()

    print("Loading data...")
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    ann_df = pd.read_csv(SPLIT_DIR / "train_annotations.csv")

    print(f"  {len(notes_df)} notes, {len(ann_df)} annotations")

    # Build annotation index
    ann_index = build_annotation_index(ann_df)

    # Get unique span texts and their concept_ids
    span_concepts: dict[str, set[int]] = defaultdict(set)
    span_counts: Counter = Counter()
    for _, row in ann_df.iterrows():
        span_text = str(row["span"]) if pd.notna(row["span"]) else ""
        if span_text:
            span_concepts[span_text].add(int(row["concept_id"]))
            span_counts[span_text] += 1

    print(f"  {len(span_concepts)} unique span texts")

    # Pre-build normalized texts and offset maps for all notes
    print("Pre-processing notes (normalization + offset maps)...")
    note_texts: dict[str, str] = {}
    note_normalized: dict[str, tuple[str, list[int]]] = {}
    note_lower: dict[str, str] = {}

    for _, row in notes_df.iterrows():
        nid = row["note_id"]
        text = row["text"]
        note_texts[nid] = text
        note_normalized[nid] = build_normalized_text_and_map(text)
        note_lower[nid] = text.lower()

    # For each unique span, find all occurrences across all notes
    print("Scanning for span occurrences across all notes...")

    # Statistics
    total_exact_unannotated = 0
    total_exact_diff_concept = 0
    total_exact_same_concept = 0
    total_fuzzy_only_unannotated = 0
    total_fuzzy_only_diff_concept = 0
    total_partial_overlap = 0

    # Per-span details for the most interesting cases
    interesting_cases = []

    # Track per-span stats
    span_stats = {}

    n_spans = len(span_concepts)
    checkpoint = max(1, n_spans // 20)

    # Sort spans by frequency (most common first) for efficiency
    sorted_spans = sorted(span_concepts.keys(), key=lambda s: -span_counts[s])

    # Skip very short spans (1-2 chars) for fuzzy matching as they produce too many false positives
    MIN_FUZZY_LEN = 3
    # Skip very common single-word spans to avoid combinatorial explosion
    MAX_NOTES_TO_SCAN = len(notes_df)  # scan all notes

    for i, span_text in enumerate(sorted_spans):
        if (i + 1) % checkpoint == 0:
            print(f"  Progress: {i+1}/{n_spans} spans ({100*(i+1)/n_spans:.0f}%)")

        gold_concepts = span_concepts[span_text]
        span_norm = normalize_whitespace(span_text)
        span_lower = span_text.lower()
        span_norm_lower = span_norm.lower()

        exact_unannotated = 0
        exact_diff_concept = 0
        exact_same_concept = 0
        fuzzy_only_unannotated = 0
        fuzzy_only_diff_concept = 0
        partial_overlap = 0

        example_unannotated = []
        example_diff_concept = []

        for nid, text in note_texts.items():
            note_anns = ann_index.get(nid, set())

            # --- Exact matches ---
            exact_occs = find_all_occurrences_exact(text, span_text)
            exact_positions = set()  # track which positions we found exactly

            for occ_start, occ_end in exact_occs:
                exact_positions.add((occ_start, occ_end))
                is_annotated, matched_cid = overlaps_annotation(occ_start, occ_end, note_anns)

                if not is_annotated:
                    exact_unannotated += 1
                    if len(example_unannotated) < 3:
                        # Get some context
                        ctx_start = max(0, occ_start - 40)
                        ctx_end = min(len(text), occ_end + 40)
                        context = text[ctx_start:ctx_end].replace('\n', '\\n')
                        example_unannotated.append({
                            "note_id": nid,
                            "start": occ_start,
                            "end": occ_end,
                            "context": context,
                        })
                elif matched_cid is not None:
                    if matched_cid in gold_concepts:
                        exact_same_concept += 1
                    else:
                        exact_diff_concept += 1
                        if len(example_diff_concept) < 3:
                            example_diff_concept.append({
                                "note_id": nid,
                                "start": occ_start,
                                "end": occ_end,
                                "matched_cid": matched_cid,
                                "gold_cids": list(gold_concepts),
                            })
                else:
                    # Partial overlap with some annotation
                    partial_overlap += 1

            # --- Fuzzy matches (whitespace-normalized) ---
            if len(span_norm) >= MIN_FUZZY_LEN:
                norm_text, offset_map = note_normalized[nid]
                fuzzy_occs = find_all_occurrences_fuzzy(
                    norm_text.lower(), span_norm_lower, offset_map
                )

                for orig_start, orig_end in fuzzy_occs:
                    # Skip if this was already found as exact match
                    if (orig_start, orig_end) in exact_positions:
                        continue
                    # Also skip if very close to an exact match (off by 1-2 chars)
                    skip = False
                    for es, ee in exact_positions:
                        if abs(orig_start - es) <= 2 and abs(orig_end - ee) <= 2:
                            skip = True
                            break
                    if skip:
                        continue

                    is_annotated, matched_cid = overlaps_annotation(orig_start, orig_end, note_anns)

                    if not is_annotated:
                        fuzzy_only_unannotated += 1
                    elif matched_cid is not None:
                        if matched_cid not in gold_concepts:
                            fuzzy_only_diff_concept += 1

        total_exact_unannotated += exact_unannotated
        total_exact_diff_concept += exact_diff_concept
        total_exact_same_concept += exact_same_concept
        total_fuzzy_only_unannotated += fuzzy_only_unannotated
        total_fuzzy_only_diff_concept += fuzzy_only_diff_concept
        total_partial_overlap += partial_overlap

        total_neg = exact_unannotated + exact_diff_concept + fuzzy_only_unannotated + fuzzy_only_diff_concept

        if total_neg > 0:
            span_stats[span_text] = {
                "annotation_count": span_counts[span_text],
                "gold_concepts": list(gold_concepts),
                "exact_unannotated": exact_unannotated,
                "exact_diff_concept": exact_diff_concept,
                "exact_same_concept": exact_same_concept,
                "fuzzy_only_unannotated": fuzzy_only_unannotated,
                "fuzzy_only_diff_concept": fuzzy_only_diff_concept,
                "partial_overlap": partial_overlap,
                "total_negative": total_neg,
            }

            if total_neg >= 5 and len(interesting_cases) < 100:
                interesting_cases.append({
                    "span_text": span_text,
                    "total_negative": total_neg,
                    "exact_unannotated": exact_unannotated,
                    "exact_diff_concept": exact_diff_concept,
                    "fuzzy_only_unannotated": fuzzy_only_unannotated,
                    "examples_unannotated": example_unannotated,
                    "examples_diff_concept": example_diff_concept,
                })

    elapsed = time.time() - t0

    # --- Report ---
    print("\n" + "=" * 70)
    print("NEGATIVE EXAMPLE ANALYSIS RESULTS")
    print("=" * 70)

    print(f"\nTotal annotations in training set: {len(ann_df)}")
    print(f"Unique span texts: {len(span_concepts)}")
    print(f"Spans with at least 1 negative example: {len(span_stats)}")

    print(f"\n--- Exact Match Negatives ---")
    print(f"  Unannotated occurrences: {total_exact_unannotated}")
    print(f"  Different concept annotations: {total_exact_diff_concept}")
    print(f"  Same concept (positive): {total_exact_same_concept}")
    print(f"  Partial overlaps: {total_partial_overlap}")

    print(f"\n--- Fuzzy Match Negatives (whitespace-norm + case-insensitive, additional to exact) ---")
    print(f"  Unannotated fuzzy occurrences: {total_fuzzy_only_unannotated}")
    print(f"  Different concept fuzzy: {total_fuzzy_only_diff_concept}")

    total_neg = (total_exact_unannotated + total_exact_diff_concept +
                 total_fuzzy_only_unannotated + total_fuzzy_only_diff_concept)
    print(f"\n--- Total Negative Examples ---")
    print(f"  Total: {total_neg}")
    print(f"    Unannotated (exact + fuzzy): {total_exact_unannotated + total_fuzzy_only_unannotated}")
    print(f"    Different concept (exact + fuzzy): {total_exact_diff_concept + total_fuzzy_only_diff_concept}")

    # Top spans by negative count
    top_spans = sorted(span_stats.items(), key=lambda x: -x[1]["total_negative"])[:50]

    print(f"\n--- Top 50 Spans by Negative Example Count ---")
    print(f"{'Span':<40} {'Ann#':>5} {'ExUn':>6} {'ExDf':>6} {'FzUn':>6} {'FzDf':>6} {'Total':>6}")
    print("-" * 80)
    for span_text, stats in top_spans:
        display = span_text[:38].replace('\n', '\\n')
        print(f"{display:<40} {stats['annotation_count']:>5} "
              f"{stats['exact_unannotated']:>6} {stats['exact_diff_concept']:>6} "
              f"{stats['fuzzy_only_unannotated']:>6} {stats['fuzzy_only_diff_concept']:>6} "
              f"{stats['total_negative']:>6}")

    # Distribution of negative counts
    neg_counts = [s["total_negative"] for s in span_stats.values()]
    if neg_counts:
        print(f"\n--- Distribution of Negative Example Counts (per span) ---")
        buckets = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100), (101, 500), (501, 10000)]
        for lo, hi in buckets:
            count = sum(1 for n in neg_counts if lo <= n <= hi)
            if count:
                print(f"  {lo}-{hi}: {count} spans")

    # Different-concept analysis (most interesting for rules)
    diff_concept_spans = {k: v for k, v in span_stats.items()
                         if v["exact_diff_concept"] > 0 or v["fuzzy_only_diff_concept"] > 0}
    print(f"\n--- Spans Annotated with Different Concepts in Different Locations ---")
    print(f"  {len(diff_concept_spans)} unique spans have different concept annotations")

    top_diff = sorted(diff_concept_spans.items(),
                     key=lambda x: -(x[1]["exact_diff_concept"] + x[1]["fuzzy_only_diff_concept"]))[:30]
    print(f"\n  Top 30 ambiguous spans:")
    print(f"  {'Span':<40} {'#Concepts':>9} {'DiffAnn':>8}")
    print("  " + "-" * 60)
    for span_text, stats in top_diff:
        display = span_text[:38].replace('\n', '\\n')
        n_concepts = len(stats["gold_concepts"])
        diff_total = stats["exact_diff_concept"] + stats["fuzzy_only_diff_concept"]
        print(f"  {display:<40} {n_concepts:>9} {diff_total:>8}")

    print(f"\nAnalysis completed in {elapsed:.1f}s")

    # Save detailed results
    output = {
        "summary": {
            "total_annotations": len(ann_df),
            "unique_spans": len(span_concepts),
            "spans_with_negatives": len(span_stats),
            "exact_unannotated": total_exact_unannotated,
            "exact_diff_concept": total_exact_diff_concept,
            "exact_same_concept": total_exact_same_concept,
            "fuzzy_only_unannotated": total_fuzzy_only_unannotated,
            "fuzzy_only_diff_concept": total_fuzzy_only_diff_concept,
            "partial_overlap": total_partial_overlap,
            "total_negative": total_neg,
        },
        "interesting_cases": sorted(interesting_cases, key=lambda x: -x["total_negative"]),
        "top_50_spans": [
            {"span_text": k, **v} for k, v in top_spans
        ],
    }

    out_path = REPO_ROOT / "scripts" / "negative_example_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
