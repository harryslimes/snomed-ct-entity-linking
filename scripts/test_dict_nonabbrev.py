#!/usr/bin/env python3
"""Test how well direct dictionary lookup works for non-abbreviation spans.

For each non-abbreviation annotation, checks if the span text (exact or
case-insensitive) matches any description in the SNOMED terminology, and
whether the matched concept is the gold concept.

Reports:
  - Exact match rate (span text found in terminology descriptions)
  - Gold hit rate (matched concept IS the gold concept)
  - Combined with hybrid index: what retrieval recall looks like when
    dictionary matches are used first, falling back to hybrid search

Usage:
  python3 scripts/test_dict_nonabbrev.py
  python3 scripts/test_dict_nonabbrev.py --split test
  python3 scripts/test_dict_nonabbrev.py --with-index  # also test hybrid fallback
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from build_abbrev_dictionary_llm import extract_non_abbreviation_items  # noqa: E402

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
TERMINOLOGY_CSV = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"
MRCONSO_PATH = REPO_ROOT / "snomed_index" / "mrconso_terms.pkl"


def build_term_to_concepts(
    terminology_csv: Path,
    mrconso_path: Path,
) -> dict[str, set[int]]:
    """Build a lowercase term -> set of concept IDs lookup from all sources."""
    term_map: dict[str, set[int]] = defaultdict(set)

    # 1. Flattened terminology (concept_name column)
    ft = pd.read_csv(terminology_csv)
    for _, row in ft.iterrows():
        cid = int(row["concept_id"])
        name = str(row["concept_name"]).strip().lower()
        if name:
            term_map[name].add(cid)

    print(f"  Terminology CSV: {len(ft):,} entries → {len(term_map):,} unique terms")

    # 2. MRCONSO thesaurus (all synonyms per concept)
    if mrconso_path.exists():
        with open(mrconso_path, "rb") as f:
            mrconso = pickle.load(f)
        n_terms_before = len(term_map)
        for cid, terms in mrconso.items():
            cid = int(cid)
            for t in terms:
                term_map[t.strip().lower()].add(cid)
        print(f"  MRCONSO thesaurus: {len(mrconso):,} concepts → "
              f"{len(term_map) - n_terms_before:,} new terms "
              f"({len(term_map):,} total)")
    else:
        print(f"  MRCONSO not found at {mrconso_path}")

    return dict(term_map)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--split", default="train", choices=["train", "test", "both"])
    parser.add_argument("--context-before", type=int, default=100)
    parser.add_argument("--context-after", type=int, default=50)
    parser.add_argument("--with-index", action="store_true",
                        help="Also test hybrid index fallback for dictionary misses")
    parser.add_argument("--index-server", default="http://127.0.0.1:8421")
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.split_dir} ...")
    splits = ["train", "test"] if args.split == "both" else [args.split]
    ann_frames, note_frames = [], []
    for s in splits:
        ann_frames.append(pd.read_csv(args.split_dir / f"{s}_annotations.csv"))
        note_frames.append(pd.read_csv(args.split_dir / f"{s}_notes.csv"))
    ann_df = pd.concat(ann_frames, ignore_index=True)
    notes_df = pd.concat(note_frames, ignore_index=True)
    for col in ("start", "end", "concept_id", "annotation_id"):
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce").fillna(0).astype(int)
    print(f"  {len(ann_df):,} annotations, {len(notes_df):,} notes")

    # Extract non-abbreviation items
    print("\nExtracting non-abbreviation annotations ...")
    items = extract_non_abbreviation_items(
        ann_df, notes_df, args.context_before, args.context_after,
    )
    print(f"  {len(items):,} non-abbreviation instances")

    # Build dictionary
    print("\nBuilding term → concept dictionary ...")
    term_map = build_term_to_concepts(TERMINOLOGY_CSV, MRCONSO_PATH)

    # Test dictionary matching
    print(f"\nTesting dictionary lookup on {len(items):,} annotations ...")
    t0 = time.time()

    n_exact_match = 0      # span found in dictionary (any concept)
    n_gold_hit = 0         # span found AND gold concept is among matches
    n_gold_unique = 0      # gold hit AND only one concept matched (unambiguous)
    n_total = len(items)

    # Per-section stats
    section_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "match": 0, "gold_hit": 0}
    )

    # Track misses for analysis
    miss_examples: list[dict] = []
    ambiguous_examples: list[dict] = []

    for item in items:
        span = item["span"].strip()
        span_lower = span.lower()
        gold_cid = item["gold_concept_id"]
        section = item["section_header"]

        section_stats[section]["total"] += 1

        matched_cids = term_map.get(span_lower, set())

        if matched_cids:
            n_exact_match += 1
            section_stats[section]["match"] += 1

            if gold_cid in matched_cids:
                n_gold_hit += 1
                section_stats[section]["gold_hit"] += 1
                if len(matched_cids) == 1:
                    n_gold_unique += 1
            elif len(miss_examples) < 30:
                miss_examples.append({
                    "span": span,
                    "section": section,
                    "gold_cid": gold_cid,
                    "n_matched": len(matched_cids),
                    "type": "wrong_concept",
                })
        elif len(miss_examples) < 30:
            miss_examples.append({
                "span": span,
                "section": section,
                "gold_cid": gold_cid,
                "n_matched": 0,
                "type": "not_in_dict",
            })

        if matched_cids and len(matched_cids) > 5 and len(ambiguous_examples) < 20:
            ambiguous_examples.append({
                "span": span,
                "section": section,
                "n_concepts": len(matched_cids),
                "gold_in_matches": gold_cid in matched_cids,
            })

    elapsed = time.time() - t0

    # Report
    match_rate = n_exact_match / n_total if n_total else 0
    gold_rate = n_gold_hit / n_total if n_total else 0
    gold_of_matched = n_gold_hit / n_exact_match if n_exact_match else 0
    unique_rate = n_gold_unique / n_total if n_total else 0

    print(f"\n{'=' * 60}")
    print(f"DICTIONARY LOOKUP RESULTS ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Total annotations:     {n_total:,}")
    print(f"  Dictionary match:      {n_exact_match:,}  ({100 * match_rate:.1f}%)")
    print(f"  Gold concept hit:      {n_gold_hit:,}  ({100 * gold_rate:.1f}%)")
    print(f"  Gold | matched:        {n_gold_hit:,}/{n_exact_match:,}  ({100 * gold_of_matched:.1f}%)")
    print(f"  Unambiguous gold hit:  {n_gold_unique:,}  ({100 * unique_rate:.1f}%)")
    print(f"  Not in dictionary:     {n_total - n_exact_match:,}  ({100 * (1 - match_rate):.1f}%)")

    # Section breakdown
    print(f"\n  Section breakdown:")
    print(f"  {'Section':40s}  {'n':>5}  {'match%':>7}  {'gold%':>7}")
    for sec in sorted(section_stats.keys(),
                       key=lambda s: section_stats[s]["total"], reverse=True):
        s = section_stats[sec]
        m_pct = 100 * s["match"] / s["total"] if s["total"] else 0
        g_pct = 100 * s["gold_hit"] / s["total"] if s["total"] else 0
        print(f"  {sec:40s}  {s['total']:5d}  {m_pct:6.1f}%  {g_pct:6.1f}%")

    # Miss examples
    not_in_dict = [m for m in miss_examples if m["type"] == "not_in_dict"]
    wrong_concept = [m for m in miss_examples if m["type"] == "wrong_concept"]

    if not_in_dict:
        print(f"\n  Sample spans NOT in dictionary ({len(not_in_dict)}):")
        for m in not_in_dict[:15]:
            print(f"    {m['span']!r:40s}  section={m['section']}")

    if wrong_concept:
        print(f"\n  Sample spans matched but WRONG concept ({len(wrong_concept)}):")
        for m in wrong_concept[:15]:
            print(f"    {m['span']!r:40s}  section={m['section']}  "
                  f"gold={m['gold_cid']}  matched={m['n_matched']} concepts")

    if ambiguous_examples:
        print(f"\n  Highly ambiguous spans (>5 concepts matched, {len(ambiguous_examples)}):")
        for a in ambiguous_examples[:10]:
            gold_str = "HIT" if a["gold_in_matches"] else "MISS"
            print(f"    {a['span']!r:40s}  {a['n_concepts']:3d} concepts  {gold_str}")

    # --- Optional: test with hybrid index fallback ---
    if args.with_index:
        print(f"\n{'=' * 60}")
        print("HYBRID INDEX FALLBACK TEST")
        print(f"{'=' * 60}")
        print("For dictionary misses, fall back to hybrid FAISS+BM25 search")
        print("using the span text as the query.")

        import requests

        # Check index server
        try:
            resp = requests.get(f"{args.index_server}/health", timeout=5)
            resp.raise_for_status()
            print(f"  Index server OK at {args.index_server}")
        except Exception as e:
            print(f"  ERROR: Cannot reach index server: {e}")
            return

        # Load concept names for reporting
        concept_names = {}
        ft = pd.read_csv(TERMINOLOGY_CSV)
        for _, row in ft.iterrows():
            concept_names[int(row["concept_id"])] = str(row["concept_name"])

        # Collect all items that need hybrid search
        dict_misses: list[dict] = []  # items where dictionary didn't find gold
        for item in items:
            span_lower = item["span"].strip().lower()
            gold_cid = item["gold_concept_id"]
            matched_cids = term_map.get(span_lower, set())
            if gold_cid not in matched_cids:
                dict_misses.append(item)

        print(f"  Dictionary misses needing hybrid search: {len(dict_misses):,}")

        # Batch search: use span text as query
        batch_size = 500
        n_hybrid_hits = 0
        t0 = time.time()

        for batch_start in range(0, len(dict_misses), batch_size):
            batch = dict_misses[batch_start:batch_start + batch_size]
            queries = [item["span"].strip() for item in batch]

            resp = requests.post(
                f"{args.index_server}/search/hybrid",
                json={"queries": queries, "fusion_top_k": args.top_k},
                timeout=120,
            )
            resp.raise_for_status()
            results = resp.json()["results"]

            for item, hits in zip(batch, results):
                gold_cid = item["gold_concept_id"]
                hit_ids = {h["sctid"] for h in hits}
                if gold_cid in hit_ids:
                    n_hybrid_hits += 1

            done = min(batch_start + batch_size, len(dict_misses))
            print(f"  Searched {done:,}/{len(dict_misses):,} ...", end="\r")

        hybrid_elapsed = time.time() - t0
        print(f"  Hybrid search done in {hybrid_elapsed:.1f}s" + " " * 30)

        # Combined stats
        n_combined = n_gold_hit + n_hybrid_hits
        combined_rate = n_combined / n_total if n_total else 0
        hybrid_of_misses = n_hybrid_hits / len(dict_misses) if dict_misses else 0

        print(f"\n  Dictionary gold hits:  {n_gold_hit:,}  ({100 * gold_rate:.1f}%)")
        print(f"  Hybrid rescue hits:    {n_hybrid_hits:,}/{len(dict_misses):,}  "
              f"({100 * hybrid_of_misses:.1f}% of dict misses)")
        print(f"  Combined gold hits:    {n_combined:,}/{n_total:,}  "
              f"({100 * combined_rate:.1f}%)")
        print(f"\n  Hybrid-only baseline:  "
              f"~{100 * (n_total - len(dict_misses) + n_hybrid_hits) / n_total:.1f}% "
              f"(if we only used hybrid for everything)")


if __name__ == "__main__":
    main()
