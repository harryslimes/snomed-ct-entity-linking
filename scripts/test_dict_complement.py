#!/usr/bin/env python3
"""Experiment: how much does the super-dictionary add on top of
the hybrid FAISS+BM25 index for abbreviation retrieval?

Runs the pipeline on ALL abbreviation instances (each with its own context),
then checks which retrieval misses the dictionary would have resolved.
"""
import csv
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "super-dictionary"))
sys.path.insert(0, str(REPO / "rulebook"))

from engine import segment_sections, get_section_for_pos
from build_abbrev_dictionary_llm import extract_abbreviation_items
import rule_testing as rt

# ── Config ──
SPLIT_DIR = REPO / "data" / "old-challenge-split"
RAW_DICT_PATH = REPO / "data" / "interim" / "super_dictionary_full.tsv"
VLLM_URL = "http://localhost:8000"
INDEX_URL = "http://127.0.0.1:8421"
MODEL = str(REPO / "models" / "Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")


def load_raw_dict() -> dict[str, set[int]]:
    """Load the untrained super dictionary TSV → {term_lower: {concept_ids}}."""
    raw: dict[str, set[int]] = {}
    with open(RAW_DICT_PATH, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            term = row["term"].strip().lower()
            cid = int(row["snomed_concept_id"])
            raw.setdefault(term, set()).add(cid)
    print(f"Loaded raw dict: {len(raw):,} unique terms")
    return raw


def raw_dict_lookup(raw: dict[str, set[int]], span: str, gold_id: int) -> bool:
    """Check if the raw dictionary maps this span to the gold concept ID."""
    mention = span.strip().lower()
    cids = raw.get(mention, set())
    return gold_id in cids


def build_all_reps(items: list[dict], rules: list[dict]) -> list[dict]:
    """Build one rep per item (no dedup) — each gets its own context."""
    from abbrev_rule_loop import rule_applies_search, _format_rules_block, BASE_SEARCH_RULES

    reps = []
    for item in items:
        section = item["section_header"]
        search_rules = [r for r in rules if rule_applies_search(r, section)]
        search_extra = _format_rules_block(search_rules, "SEARCH")
        search_rules_text = BASE_SEARCH_RULES + ("\n\n" + search_extra if search_extra else "")

        reps.append({
            "before": item["before"],
            "span": item["span"],
            "after": item["after"],
            "section_header": section,
            "search_rules_text": search_rules_text,
            "gold_start": item["start"],
            "gold_end": item["end"],
            "instance_count": 1,
            "gold_concept_ids_all": [item["gold_concept_id"]],
            "representative_note_id": item["note_id"],
        })
    return reps


def main():
    rt._index_server_url = INDEX_URL

    # Load data
    print("Loading data ...")
    ann_df = pd.read_csv(SPLIT_DIR / "train_annotations.csv")
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    for col in ("start", "end", "concept_id", "annotation_id"):
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce").fillna(0).astype(int)

    all_items = extract_abbreviation_items(ann_df, notes_df, 200, 100)
    print(f"  {len(all_items):,} abbreviation instances")

    # Load dictionary
    raw = load_raw_dict()

    # Load concept names
    concept_names = rt.load_concept_names()

    # Build one rep per instance
    from abbrev_rule_loop import load_rules
    rules = load_rules(REPO / "super-dictionary" / "abbrev_rules_fresh.json")
    reps = build_all_reps(all_items, rules)
    print(f"  Built {len(reps):,} reps (one per instance)")

    # Run the pipeline (search + retrieve) on ALL instances
    print(f"\nRunning pipeline on {len(reps)} instances ...")
    from abbrev_rule_loop import ABBREV_SEARCH_SYSTEM
    orig = rt.SEARCH_SYSTEM
    rt.SEARCH_SYSTEM = ABBREV_SEARCH_SYSTEM
    search_terms, candidates, _sel, timings = rt.vllm_pipeline(
        reps, base_url=VLLM_URL, model=MODEL,
        max_concurrent=32, reasoning_effort="none",
        search_only=True,
    )
    rt.SEARCH_SYSTEM = orig

    # Score every instance
    total = len(reps)
    index_hits = 0
    raw_hits = 0
    union_hits = 0

    # Track per-(span, section) for detailed reporting
    pair_stats: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"total": 0, "idx_hit": 0, "raw_hit": 0, "union_hit": 0, "gold_name": "?"}
    )

    for rep, cands in zip(reps, candidates):
        gold_id = rep["gold_concept_ids_all"][0]
        span = rep["span"]
        section = rep["section_header"]
        key = (span, section)

        pair_stats[key]["total"] += 1

        idx_hit = any(c["concept_id"] == gold_id for c in cands)
        raw_hit = raw_dict_lookup(raw, span, gold_id)

        if idx_hit:
            index_hits += 1
            pair_stats[key]["idx_hit"] += 1
        if raw_hit:
            raw_hits += 1
            pair_stats[key]["raw_hit"] += 1
        if idx_hit or raw_hit:
            union_hits += 1
            pair_stats[key]["union_hit"] += 1

        if pair_stats[key]["gold_name"] == "?":
            pair_stats[key]["gold_name"] = concept_names.get(gold_id, ("?", "?"))[0]

    # Report
    print(f"\n{'=' * 60}")
    print(f"RESULTS on {total:,} abbreviation instances:")
    print(f"{'=' * 60}")
    print(f"  Index only:        {index_hits:,}/{total:,}  = {100*index_hits/total:.1f}%")
    print(f"  Raw dict only:     {raw_hits:,}/{total:,}  = {100*raw_hits/total:.1f}%")
    print(f"  Index + Raw dict:  {union_hits:,}/{total:,}  = {100*union_hits/total:.1f}%")
    print()

    # Index misses resolved by raw dict (sorted by instance count)
    resolved = []
    neither = []
    for (span, section), stats in pair_stats.items():
        idx_miss = stats["total"] - stats["idx_hit"]
        raw_resolves = stats["raw_hit"]  # how many the raw dict hits in this pair
        union_miss = stats["total"] - stats["union_hit"]

        if idx_miss > 0 and raw_resolves > 0:
            resolved.append((idx_miss, span, section, stats["gold_name"]))
        if union_miss > 0:
            neither.append((union_miss, span, section, stats["gold_name"]))

    if resolved:
        resolved.sort(reverse=True)
        print(f"Raw dict resolves index misses for {len(resolved)} (span, section) pairs (top 30):")
        for n, span, section, name in resolved[:30]:
            print(f"  {span!r:20s} | {section:30s} | n={n:3d} | ({name})")

    if neither:
        neither.sort(reverse=True)
        print(f"\nNeither index NOR raw dict ({len(neither)} pairs, {sum(n for n,_,_,_ in neither)} instances, top 20):")
        for n, span, section, name in neither[:20]:
            print(f"  {span!r:20s} | {section:30s} | n={n:3d} | ({name})")

    # Per-instance index variability: cases where same (span, section) has mixed index results
    mixed = []
    for (span, section), stats in pair_stats.items():
        if 0 < stats["idx_hit"] < stats["total"]:
            mixed.append((stats["total"], stats["idx_hit"], span, section))
    if mixed:
        mixed.sort(reverse=True)
        print(f"\nMixed index results (same span+section, different contexts → different outcomes): {len(mixed)} pairs")
        for total_n, hit_n, span, section in mixed[:20]:
            print(f"  {span!r:20s} | {section:30s} | {hit_n}/{total_n} hit ({100*hit_n/total_n:.0f}%)")


if __name__ == "__main__":
    main()
