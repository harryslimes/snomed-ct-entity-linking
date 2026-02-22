#!/usr/bin/env python3
"""Frequency analysis: how many concept IDs does the raw dictionary return
for each abbreviation span we need to look up?"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "super-dictionary"))
sys.path.insert(0, str(REPO / "rulebook"))

from build_abbrev_dictionary_llm import extract_abbreviation_items
import rule_testing as rt

SPLIT_DIR = REPO / "data" / "old-challenge-split"
RAW_DICT_PATH = REPO / "data" / "interim" / "super_dictionary_full.tsv"


def load_raw_dict() -> dict[str, set[int]]:
    raw: dict[str, set[int]] = {}
    with open(RAW_DICT_PATH, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            term = row["term"].strip().lower()
            cid = int(row["snomed_concept_id"])
            raw.setdefault(term, set()).add(cid)
    return raw


def main():
    # Load items
    print("Loading data ...")
    ann_df = pd.read_csv(SPLIT_DIR / "train_annotations.csv")
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    for col in ("start", "end", "concept_id", "annotation_id"):
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce").fillna(0).astype(int)

    all_items = extract_abbreviation_items(ann_df, notes_df, 200, 100)
    print(f"  {len(all_items):,} abbreviation instances")

    # Deduplicate by (span, section)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in all_items:
        groups[(item["span"], item["section_header"])].append(item)
    print(f"  {len(groups):,} unique (span, section) pairs")

    # Load dictionary
    print("Loading raw dictionary ...")
    raw = load_raw_dict()
    print(f"  {len(raw):,} unique terms")

    # Load concept names for reporting
    concept_names = rt.load_concept_names()

    # For each unique span, check how many concept IDs the dict returns
    unique_spans = sorted({item["span"] for item in all_items})
    print(f"  {len(unique_spans)} unique spans\n")

    results = []
    for span in unique_spans:
        mention = span.strip().lower()
        cids = raw.get(mention, set())
        results.append((span, len(cids), cids))

    # Distribution analysis
    counts = [r[1] for r in results]
    counts_arr = np.array(counts)

    print("=" * 60)
    print("DICTIONARY CANDIDATE COUNT DISTRIBUTION")
    print(f"  (across {len(unique_spans)} unique abbreviation spans)")
    print("=" * 60)
    print(f"  No match (0):     {sum(1 for c in counts if c == 0)}")
    print(f"  Exactly 1:        {sum(1 for c in counts if c == 1)}")
    print(f"  2-5:              {sum(1 for c in counts if 2 <= c <= 5)}")
    print(f"  6-10:             {sum(1 for c in counts if 6 <= c <= 10)}")
    print(f"  11-50:            {sum(1 for c in counts if 11 <= c <= 50)}")
    print(f"  51-100:           {sum(1 for c in counts if 51 <= c <= 100)}")
    print(f"  101-500:          {sum(1 for c in counts if 101 <= c <= 500)}")
    print(f"  500+:             {sum(1 for c in counts if c > 500)}")
    print()

    matched = [c for c in counts if c > 0]
    if matched:
        matched_arr = np.array(matched)
        print(f"  Among {len(matched)} spans WITH a match:")
        print(f"    Mean:   {matched_arr.mean():.1f}")
        print(f"    Median: {np.median(matched_arr):.0f}")
        print(f"    P25:    {np.percentile(matched_arr, 25):.0f}")
        print(f"    P75:    {np.percentile(matched_arr, 75):.0f}")
        print(f"    P90:    {np.percentile(matched_arr, 90):.0f}")
        print(f"    P95:    {np.percentile(matched_arr, 95):.0f}")
        print(f"    Max:    {matched_arr.max()}")
    print()

    # Show specific examples sorted by candidate count (worst offenders)
    results.sort(key=lambda x: x[1], reverse=True)
    print("TOP 30 — most ambiguous spans (highest candidate count):")
    print(f"  {'Span':25s} | {'#Cands':>7s} | Gold concept(s)")
    print(f"  {'-'*25}-+-{'-'*7}-+-{'-'*50}")
    for span, n_cands, cids in results[:30]:
        # Find gold concept IDs for this span
        gold_ids = {item["gold_concept_id"] for item in all_items if item["span"] == span}
        gold_names = [concept_names.get(gid, ("?", "?"))[0] for gid in gold_ids]
        gold_str = "; ".join(f"{gid} ({name})" for gid, name in zip(gold_ids, gold_names))
        if len(gold_str) > 60:
            gold_str = gold_str[:57] + "..."
        print(f"  {span!r:25s} | {n_cands:7,d} | {gold_str}")

    print()

    # Show the bottom — spans with 1-3 candidates (ideal)
    low_cand = [(s, n, c) for s, n, c in results if 1 <= n <= 3]
    low_cand.sort(key=lambda x: x[1])
    print(f"CLEAN LOOKUPS — spans with 1-3 candidates: {len(low_cand)}/{len(unique_spans)}")
    for span, n_cands, cids in low_cand[:20]:
        gold_ids = {item["gold_concept_id"] for item in all_items if item["span"] == span}
        gold_names = [concept_names.get(gid, ("?", "?"))[0] for gid in gold_ids]
        gold_in_dict = "HIT" if any(gid in cids for gid in gold_ids) else "MISS"
        print(f"  {span!r:25s} | {n_cands:3d} cands | {gold_in_dict} | {gold_names[0]}")

    # Instance-weighted: how many candidates does the average abbreviation *instance* face?
    print()
    inst_counts = []
    for item in all_items:
        mention = item["span"].strip().lower()
        n = len(raw.get(mention, set()))
        inst_counts.append(n)
    inst_arr = np.array(inst_counts)
    print("INSTANCE-WEIGHTED (10,481 instances):")
    print(f"  Mean candidates:  {inst_arr.mean():.1f}")
    print(f"  Median:           {np.median(inst_arr):.0f}")
    print(f"  P75:              {np.percentile(inst_arr, 75):.0f}")
    print(f"  P90:              {np.percentile(inst_arr, 90):.0f}")
    print(f"  P95:              {np.percentile(inst_arr, 95):.0f}")


if __name__ == "__main__":
    main()
