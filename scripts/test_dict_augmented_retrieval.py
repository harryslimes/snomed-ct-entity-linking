#!/usr/bin/env python3
"""Test whether adding dictionary lookups of LLM-generated terms to hybrid
retrieval improves gold concept recall at the candidate stage.

For each annotation:
  1. LLM generates 1-3 search terms
  2. Hybrid index retrieves top-K candidates from those terms
  3. Dictionary lookup: each LLM term checked against MRCONSO/terminology
  4. Merge dictionary concept hits into the candidate list
  5. Check if gold concept is in hybrid-only vs hybrid+dict candidates

Usage:
  python3 scripts/test_dict_augmented_retrieval.py --sample 2000
  python3 scripts/test_dict_augmented_retrieval.py --sample 0   # all
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))
sys.path.insert(0, str(REPO_ROOT))

import rulebook.rule_testing as rt  # noqa: E402
from build_abbrev_dictionary_llm import extract_non_abbreviation_items  # noqa: E402
from rulebook.general_rule_loop import (  # noqa: E402
    GENERAL_SEARCH_SYSTEM,
    BASE_SEARCH_RULES,
    load_rules,
    rule_applies,
    _format_rules_block,
)

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
TERMINOLOGY_CSV = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"
MRCONSO_PATH = REPO_ROOT / "snomed_index" / "mrconso_terms.pkl"
DEFAULT_RULES_FILE = REPO_ROOT / "rulebook" / "general_rules.json"


def build_term_to_concepts(
    terminology_csv: Path,
    mrconso_path: Path,
) -> dict[str, set[int]]:
    """Build a lowercase term -> set of concept IDs lookup."""
    term_map: dict[str, set[int]] = defaultdict(set)

    ft = pd.read_csv(terminology_csv)
    for _, row in ft.iterrows():
        cid = int(row["concept_id"])
        name = str(row["concept_name"]).strip().lower()
        if name:
            term_map[name].add(cid)
    print(f"  Terminology CSV: {len(ft):,} entries -> {len(term_map):,} unique terms")

    if mrconso_path.exists():
        with open(mrconso_path, "rb") as f:
            mrconso = pickle.load(f)
        n_before = len(term_map)
        for cid, terms in mrconso.items():
            cid = int(cid)
            for t in terms:
                term_map[t.strip().lower()].add(cid)
        print(f"  MRCONSO thesaurus: {len(mrconso):,} concepts -> "
              f"{len(term_map) - n_before:,} new terms ({len(term_map):,} total)")

    return dict(term_map)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--split", default="train", choices=["train", "test", "both"])
    parser.add_argument("--sample", type=int, default=2000,
                        help="Number of annotations to sample (0 = all)")
    parser.add_argument("--context-before", type=int, default=100)
    parser.add_argument("--context-after", type=int, default=50)
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--model", default="/workspaces/snomed-ct-entity-linking/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")
    parser.add_argument("--max-concurrent", type=int, default=0)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--rules-file", type=Path, default=DEFAULT_RULES_FILE,
                        help="Rules JSON file (use 'none' to skip rules)")
    parser.add_argument("--seed", type=int, default=42)
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

    # Sample
    if args.sample > 0 and args.sample < len(items):
        random.seed(args.seed)
        items = random.sample(items, args.sample)
        print(f"  Sampled {len(items):,} annotations")
    n_total = len(items)

    # Build dictionary
    print("\nBuilding term -> concept dictionary ...")
    term_map = build_term_to_concepts(TERMINOLOGY_CSV, MRCONSO_PATH)

    # Load concept names for reporting
    concept_names = {}
    ft = pd.read_csv(TERMINOLOGY_CSV)
    for _, row in ft.iterrows():
        concept_names[int(row["concept_id"])] = str(row["concept_name"])

    # Load rules
    rules = []
    if str(args.rules_file).lower() != "none" and args.rules_file.exists():
        rules = load_rules(args.rules_file)
    print(f"  Rules: {len(rules)}")

    # Build annotation dicts with rule injection
    print(f"\n{'='*60}")
    print("Running search + hybrid retrieval pipeline")
    print(f"{'='*60}")

    annotations = []
    for item in items:
        gold_cids = [item["gold_concept_id"]]
        section = item["section_header"]
        applicable = [r for r in rules if rule_applies(r, section, gold_cids)]
        rules_extra = _format_rules_block(applicable)
        search_rules_text = BASE_SEARCH_RULES + (
            "\n\n" + rules_extra if rules_extra else ""
        )
        annotations.append({
            "before": item["before"],
            "span": item["span"],
            "after": item["after"],
            "section_header": section,
            "search_rules_text": search_rules_text,
            "select_rules_text": "",  # not used in search_only
            "gold_start": item["start"],
            "gold_end": item["end"],
        })

    orig_search_system = rt.SEARCH_SYSTEM
    t0 = time.time()
    try:
        rt.SEARCH_SYSTEM = GENERAL_SEARCH_SYSTEM
        search_terms_list, candidates_list, _, timings = rt.vllm_pipeline(
            annotations,
            base_url=args.vllm_url,
            model=args.model,
            max_concurrent=args.max_concurrent,
            reasoning_effort=args.reasoning_effort,
            search_only=True,
        )
    finally:
        rt.SEARCH_SYSTEM = orig_search_system
    elapsed = time.time() - t0
    print(f"  Pipeline done in {elapsed:.1f}s")

    # Analyze results
    print(f"\n{'='*60}")
    print("RETRIEVAL RECALL ANALYSIS")
    print(f"{'='*60}")

    n_hybrid_hit = 0          # gold in hybrid candidates
    n_dict_hit = 0            # gold in dictionary lookup of LLM terms
    n_hybrid_plus_dict = 0    # gold in hybrid OR dict
    n_dict_rescued = 0        # gold in dict but NOT in hybrid
    n_hybrid_only = 0         # gold in hybrid but NOT in dict

    rescue_examples = []

    for i, (item, terms, candidates) in enumerate(
        zip(items, search_terms_list, candidates_list)
    ):
        gold_cid = item["gold_concept_id"]

        # Hybrid candidates
        hybrid_cids = {c["concept_id"] for c in candidates}
        hybrid_hit = gold_cid in hybrid_cids

        # Dictionary lookup of LLM terms
        dict_cids: set[int] = set()
        for t in terms:
            t_lower = t.strip().lower()
            dict_cids |= term_map.get(t_lower, set())
        dict_hit = gold_cid in dict_cids

        if hybrid_hit:
            n_hybrid_hit += 1
        if dict_hit:
            n_dict_hit += 1
        if hybrid_hit or dict_hit:
            n_hybrid_plus_dict += 1
        if dict_hit and not hybrid_hit:
            n_dict_rescued += 1
            if len(rescue_examples) < 30:
                # Find which term matched
                winning_term = None
                for t in terms:
                    t_lower = t.strip().lower()
                    if gold_cid in term_map.get(t_lower, set()):
                        winning_term = t
                        break
                rescue_examples.append({
                    "span": item["span"],
                    "section": item["section_header"],
                    "gold_cid": gold_cid,
                    "gold_name": concept_names.get(gold_cid, "?"),
                    "winning_term": winning_term,
                    "all_terms": terms,
                    "n_hybrid_cands": len(candidates),
                    "n_dict_cids": len(dict_cids),
                })
        if hybrid_hit and not dict_hit:
            n_hybrid_only += 1

    hybrid_rate = 100 * n_hybrid_hit / n_total
    dict_rate = 100 * n_dict_hit / n_total
    combined_rate = 100 * n_hybrid_plus_dict / n_total
    rescue_rate = 100 * n_dict_rescued / n_total

    print(f"  Total annotations:       {n_total:,}")
    print(f"  Hybrid-only recall:      {n_hybrid_hit:,} ({hybrid_rate:.1f}%)")
    print(f"  Dict-of-LLM-terms:      {n_dict_hit:,} ({dict_rate:.1f}%)")
    print(f"  Hybrid + Dict combined:  {n_hybrid_plus_dict:,} ({combined_rate:.1f}%)")
    print(f"  Dict rescued (not in hybrid): {n_dict_rescued:,} ({rescue_rate:.1f}%)")
    print(f"  Hybrid only (not in dict):    {n_hybrid_only:,} ({100*n_hybrid_only/n_total:.1f}%)")
    print(f"  Neither:                 {n_total - n_hybrid_plus_dict:,} "
          f"({100*(n_total - n_hybrid_plus_dict)/n_total:.1f}%)")
    print(f"\n  Delta from adding dict:  +{n_dict_rescued:,} "
          f"({hybrid_rate:.1f}% -> {combined_rate:.1f}%)")

    if rescue_examples:
        print(f"\n  Sample dict-rescued annotations ({len(rescue_examples)}):")
        for ex in rescue_examples[:20]:
            print(f"    span={ex['span']!r:35s}  gold={ex['gold_name']!r:40s}")
            print(f"      winner={ex['winning_term']!r}  terms={ex['all_terms']}")
            print(f"      hybrid had {ex['n_hybrid_cands']} cands, dict had {ex['n_dict_cids']} cids")


if __name__ == "__main__":
    main()
