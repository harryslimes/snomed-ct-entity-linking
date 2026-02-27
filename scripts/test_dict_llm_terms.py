#!/usr/bin/env python3
"""Test dictionary lookup using LLM-generated search terms.

For each non-abbreviation annotation:
  1. Run LLM search step to generate 1-3 search terms
  2. Check each term against the SNOMED dictionary (MRCONSO + terminology CSV)
  3. Compare gold-hit rates: span-only dict vs span+LLM-terms dict vs hybrid index

Usage:
  python3 scripts/test_dict_llm_terms.py --sample 2000
  python3 scripts/test_dict_llm_terms.py --sample 0   # all annotations
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

# Import rule machinery from general_rule_loop
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
    """Build a lowercase term -> set of concept IDs lookup from all sources."""
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
    else:
        print(f"  MRCONSO not found at {mrconso_path}")

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

    # Sample if requested
    if args.sample > 0 and args.sample < len(items):
        random.seed(args.seed)
        items = random.sample(items, args.sample)
        print(f"  Sampled {len(items):,} annotations")

    # Build dictionary
    print("\nBuilding term -> concept dictionary ...")
    term_map = build_term_to_concepts(TERMINOLOGY_CSV, MRCONSO_PATH)

    # First: dictionary-only results using raw span
    print(f"\n{'='*60}")
    print("STEP 1: Dictionary lookup using raw span text")
    print(f"{'='*60}")
    n_total = len(items)
    span_dict_hits = 0
    span_dict_gold = 0
    for item in items:
        span_lower = item["span"].strip().lower()
        gold_cid = item["gold_concept_id"]
        matched = term_map.get(span_lower, set())
        if matched:
            span_dict_hits += 1
            if gold_cid in matched:
                span_dict_gold += 1
    print(f"  Total: {n_total:,}")
    print(f"  Span in dict: {span_dict_hits:,} ({100*span_dict_hits/n_total:.1f}%)")
    print(f"  Span gold hit: {span_dict_gold:,} ({100*span_dict_gold/n_total:.1f}%)")

    # Load rules
    rules = []
    rules_label = "no rules"
    if str(args.rules_file).lower() != "none" and args.rules_file.exists():
        rules = load_rules(args.rules_file)
        rules_label = f"{len(rules)} rules from {args.rules_file.name}"
    print(f"  Rules: {rules_label}")

    # Generate LLM search terms
    print(f"\n{'='*60}")
    print("STEP 2: Generate LLM search terms via vLLM")
    print(f"{'='*60}")

    # Build annotation dicts with subsumption-based rule injection
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
            "gold_start": item["start"],
            "gold_end": item["end"],
        })

    # Count how many annotations got rules injected
    n_with_rules = sum(1 for a in annotations if "ANNOTATION RULES" in a["search_rules_text"])
    print(f"  Annotations with injected rules: {n_with_rules:,}/{n_total:,} "
          f"({100*n_with_rules/n_total:.1f}%)")

    orig_search_system = rt.SEARCH_SYSTEM
    t0 = time.time()
    try:
        rt.SEARCH_SYSTEM = GENERAL_SEARCH_SYSTEM
        search_terms_list, _, _, timings = rt.vllm_pipeline(
            annotations,
            base_url=args.vllm_url,
            model=args.model,
            max_concurrent=args.max_concurrent,
            reasoning_effort=args.reasoning_effort,
            search_terms_only=True,
        )
    finally:
        rt.SEARCH_SYSTEM = orig_search_system
    elapsed = time.time() - t0
    print(f"  LLM search terms generated in {elapsed:.1f}s")

    # Analyze LLM terms
    total_terms = sum(len(t) for t in search_terms_list)
    print(f"  Total terms generated: {total_terms:,} "
          f"(avg {total_terms/n_total:.1f} per annotation)")

    # Check how often the span text is actually included
    span_included = 0
    for item, terms in zip(items, search_terms_list):
        span_lower = item["span"].strip().lower()
        if any(t.strip().lower() == span_lower for t in terms):
            span_included += 1
    print(f"  Span text included in LLM terms: {span_included:,}/{n_total:,} "
          f"({100*span_included/n_total:.1f}%)")

    # Check dictionary with LLM terms
    print(f"\n{'='*60}")
    print("STEP 3: Dictionary lookup using LLM-generated terms")
    print(f"{'='*60}")

    llm_dict_gold = 0       # gold hit using any LLM term
    llm_only_gold = 0       # gold hit from LLM terms but NOT from span
    span_only_gold = 0      # gold hit from span but NOT from LLM terms
    both_gold = 0            # gold hit from both

    # Track which LLM terms rescue misses
    rescue_examples = []

    for item, terms in zip(items, search_terms_list):
        span_lower = item["span"].strip().lower()
        gold_cid = item["gold_concept_id"]

        # Span-only dict lookup
        span_matched = term_map.get(span_lower, set())
        span_hit = gold_cid in span_matched

        # LLM terms dict lookup (check ALL terms)
        llm_hit = False
        winning_term = None
        for t in terms:
            t_lower = t.strip().lower()
            t_matched = term_map.get(t_lower, set())
            if gold_cid in t_matched:
                llm_hit = True
                if t_lower != span_lower:
                    winning_term = t
                break

        if llm_hit:
            llm_dict_gold += 1
        if llm_hit and not span_hit:
            llm_only_gold += 1
            if len(rescue_examples) < 30:
                rescue_examples.append({
                    "span": item["span"],
                    "section": item["section_header"],
                    "gold_cid": gold_cid,
                    "winning_term": winning_term or item["span"],
                    "all_terms": terms,
                })
        if span_hit and not llm_hit:
            span_only_gold += 1
        if span_hit and llm_hit:
            both_gold += 1

    combined_gold = both_gold + llm_only_gold + span_only_gold

    print(f"  Span-only gold hits:     {span_dict_gold:,} ({100*span_dict_gold/n_total:.1f}%)")
    print(f"  LLM-terms gold hits:     {llm_dict_gold:,} ({100*llm_dict_gold/n_total:.1f}%)")
    print(f"  Both:                    {both_gold:,}")
    print(f"  LLM-only (rescued):      {llm_only_gold:,} ({100*llm_only_gold/n_total:.1f}%)")
    print(f"  Span-only (LLM lost):    {span_only_gold:,} ({100*span_only_gold/n_total:.1f}%)")
    print(f"  Combined oracle:         {combined_gold:,} ({100*combined_gold/n_total:.1f}%)")
    print(f"  Neither:                 {n_total - combined_gold:,} ({100*(n_total - combined_gold)/n_total:.1f}%)")

    # Show rescue examples
    if rescue_examples:
        print(f"\n  Sample LLM-rescued annotations ({len(rescue_examples)}):")
        for ex in rescue_examples[:20]:
            print(f"    span={ex['span']!r:30s}  winner={ex['winning_term']!r:40s}  "
                  f"section={ex['section']}")
            print(f"      all terms: {ex['all_terms']}")

    # Also check: how often does the LLM produce a term that's in the dict
    # but matches the WRONG concept?
    print(f"\n{'='*60}")
    print("STEP 4: Ambiguity analysis")
    print(f"{'='*60}")
    n_any_dict_match = 0
    n_gold_in_any = 0
    for item, terms in zip(items, search_terms_list):
        gold_cid = item["gold_concept_id"]
        any_match = False
        gold_found = False
        for t in terms:
            t_lower = t.strip().lower()
            matched = term_map.get(t_lower, set())
            if matched:
                any_match = True
                if gold_cid in matched:
                    gold_found = True
        if any_match:
            n_any_dict_match += 1
        if gold_found:
            n_gold_in_any += 1

    print(f"  Any LLM term in dict:    {n_any_dict_match:,} ({100*n_any_dict_match/n_total:.1f}%)")
    print(f"  Gold in any match:       {n_gold_in_any:,} ({100*n_gold_in_any/n_total:.1f}%)")
    print(f"  In dict but wrong:       {n_any_dict_match - n_gold_in_any:,} "
          f"({100*(n_any_dict_match - n_gold_in_any)/n_total:.1f}%)")


if __name__ == "__main__":
    main()
