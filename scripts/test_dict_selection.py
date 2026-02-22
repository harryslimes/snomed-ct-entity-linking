#!/usr/bin/env python3
"""Test the selection agent using dictionary candidates.

For each abbreviation:
1. Look up span in the raw (untrained) dictionary → candidate concept IDs
2. Present candidates to the LLM select phase
3. Score: did it pick the gold concept?

No search phase needed — the dictionary IS the retrieval.
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "super-dictionary"))
sys.path.insert(0, str(REPO / "rulebook"))

from build_abbrev_dictionary_llm import extract_abbreviation_items
import rule_testing as rt

# ── Config ──
SPLIT_DIR = REPO / "data" / "old-challenge-split"
RAW_DICT_PATH = REPO / "data" / "interim" / "super_dictionary_full.tsv"
VLLM_URL = "http://localhost:8000"
INDEX_URL = "http://127.0.0.1:8421"
MODEL = str(REPO / "models" / "Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")
SAMPLE_SIZE = 0  # 0 = all
SELECT_RULES_PATH = REPO / "super-dictionary" / "select_rules.json"

# Spans with too many dict candidates to be useful — fall back to index for these
EXCLUDE_SPANS = {"PAIN", "RIGHT"}


def load_select_rules() -> list[dict]:
    """Load selection rules from JSON."""
    if not SELECT_RULES_PATH.exists():
        return []
    import json
    data = json.loads(SELECT_RULES_PATH.read_text())
    return data.get("rules", [])


def format_select_rules(rules: list[dict], section: str) -> str:
    """Format applicable selection rules into text for the prompt."""
    if not rules:
        return ""
    sec_lower = section.strip().lower()
    applicable = []
    for r in rules:
        sections = r.get("applies_when", {}).get("sections", [])
        if sections and sec_lower not in [s.lower() for s in sections]:
            continue
        applicable.append(r)
    if not applicable:
        return ""
    # Sort by priority
    applicable.sort(key=lambda r: r.get("priority", 2))
    lines = ["=== DISAMBIGUATION RULES (SELECTION) ===",
             "Priority: [P1] override (apply first) → [P2] standard → [P3] tiebreaker."]
    for r in applicable:
        p = r.get("priority", 2)
        lines.append(f"[P{p}] {r['id']}: {r['rule']}")
    return "\n".join(lines)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--ctx-before", type=int, default=100,
                        help="Characters of context before the span")
    parser.add_argument("--ctx-after", type=int, default=50,
                        help="Characters of context after the span")
    args = parser.parse_args()

    rt._index_server_url = INDEX_URL

    # Load data
    print(f"Loading data (split={args.split}) ...")
    ann_df = pd.read_csv(SPLIT_DIR / f"{args.split}_annotations.csv")
    notes_df = pd.read_csv(SPLIT_DIR / f"{args.split}_notes.csv")
    for col in ("start", "end", "concept_id", "annotation_id"):
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce").fillna(0).astype(int)

    all_items = extract_abbreviation_items(ann_df, notes_df, args.ctx_before, args.ctx_after)
    print(f"  {len(all_items):,} abbreviation instances (ctx={args.ctx_before}/{args.ctx_after})")

    # Group by (span, section) — pick best context per pair
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in all_items:
        groups[(item["span"], item["section_header"])].append(item)

    all_pairs = sorted(groups.keys(), key=lambda k: len(groups[k]), reverse=True)
    print(f"  {len(all_pairs):,} unique (span, section) pairs")

    # Load dictionary + concept names + selection rules
    raw = load_raw_dict()
    concept_names = rt.load_concept_names()
    select_rules = load_select_rules()
    print(f"  Loaded {len(select_rules)} selection rules")

    # Build annotations with precomputed candidates from dictionary
    annotations = []
    precomputed_search_terms = []
    precomputed_candidates = []
    skipped_no_dict = 0
    skipped_single = 0

    if SAMPLE_SIZE > 0:
        import random
        random.seed(42)
        pairs = random.sample(all_pairs, min(SAMPLE_SIZE, len(all_pairs)))
    else:
        pairs = all_pairs

    for span, section in pairs:
        group = groups[(span, section)]
        best = max(group, key=lambda x: len(x["before"]) + len(x["after"]))
        gold_ids = {g["gold_concept_id"] for g in group}

        # Dictionary lookup
        if span in EXCLUDE_SPANS:
            skipped_no_dict += 1
            continue
        mention = span.strip().lower()
        dict_cids = raw.get(mention, set())

        if not dict_cids:
            skipped_no_dict += 1
            continue

        if len(dict_cids) == 1:
            # Only 1 candidate — no selection needed, auto-correct
            skipped_single += 1
            cid = next(iter(dict_cids))
            # Still track for scoring
            candidates = []
            for c in dict_cids:
                name, hier = concept_names.get(c, ("Unknown", "unknown"))
                candidates.append({
                    "concept_id": c,
                    "concept_name": name,
                    "hierarchy": hier,
                })

            annotations.append({
                "before": best["before"],
                "span": span,
                "after": best["after"],
                "section_header": section,
                "gold_start": best["start"],
                "gold_end": best["end"],
                "gold_concept_ids_all": sorted(gold_ids),
                "select_rules_text": format_select_rules(select_rules, section),
                "search_rules_text": "",
                "instance_count": len(group),
                "_auto_select": True,  # Mark as auto-resolved
            })
            precomputed_search_terms.append([span])
            precomputed_candidates.append(candidates)
            continue

        # Multiple candidates — needs LLM selection
        candidates = []
        for c in sorted(dict_cids):
            name, hier = concept_names.get(c, ("Unknown", "unknown"))
            candidates.append({
                "concept_id": c,
                "concept_name": name,
                "hierarchy": hier,
            })

        annotations.append({
            "before": best["before"],
            "span": span,
            "after": best["after"],
            "section_header": section,
            "gold_start": best["start"],
            "gold_end": best["end"],
            "gold_concept_ids_all": sorted(gold_ids),
            "select_rules_text": format_select_rules(select_rules, section),
            "search_rules_text": "",
            "instance_count": len(group),
            "_auto_select": False,
        })
        precomputed_search_terms.append([span])
        precomputed_candidates.append(candidates)

    n_need_select = sum(1 for a in annotations if not a.get("_auto_select"))
    n_auto = sum(1 for a in annotations if a.get("_auto_select"))

    print(f"\n  Skipped (no dict match):     {skipped_no_dict}")
    print(f"  Auto-resolved (1 candidate): {n_auto}")
    print(f"  Need LLM selection:          {n_need_select}")
    print(f"  Total with candidates:       {len(annotations)}")

    # Run the pipeline in select-only mode
    print(f"\nRunning select-only pipeline on {len(annotations)} annotations ...")
    _search_terms, _candidates, selections, timings = rt.vllm_pipeline(
        annotations,
        base_url=VLLM_URL,
        model=MODEL,
        max_concurrent=32,
        reasoning_effort="none",
        precomputed_search_terms=precomputed_search_terms,
        precomputed_candidates=precomputed_candidates,
    )

    # Score
    correct = 0
    correct_auto = 0
    correct_select = 0
    total_auto = 0
    total_select = 0
    wrong_examples = []
    failure_details = []

    for ann, sel, cands in zip(annotations, selections, precomputed_candidates):
        gold_ids = set(ann["gold_concept_ids_all"])
        is_auto = ann.get("_auto_select", False)

        if is_auto:
            total_auto += 1
            # Auto-select: the single candidate
            if cands[0]["concept_id"] in gold_ids:
                correct_auto += 1
                correct += 1
        else:
            total_select += 1
            if sel is not None and sel[0] in gold_ids:
                correct_select += 1
                correct += 1
            elif sel is not None:
                # Wrong selection — log it
                selected_name = concept_names.get(sel[0], ("?", "?"))[0]
                gold_name = concept_names.get(list(gold_ids)[0], ("?", "?"))[0]
                wrong_examples.append(
                    f"  {ann['span']!r:20s} | {ann['section_header']:25s} | "
                    f"n_cands={len(cands)} | chose: {selected_name} | gold: {gold_name}"
                )
                failure_details.append({
                    "span": ann["span"],
                    "section": ann["section_header"],
                    "context_before": ann["before"][-100:],
                    "context_after": ann["after"][:100],
                    "candidates": [
                        {"concept_id": c["concept_id"], "name": c["concept_name"], "hierarchy": c["hierarchy"]}
                        for c in cands
                    ],
                    "chosen_id": sel[0],
                    "chosen_name": selected_name,
                    "gold_id": list(gold_ids)[0],
                    "gold_name": gold_name,
                })

    total = len(annotations)
    print(f"\n{'=' * 70}")
    print(f"RESULTS on {total} (span, section) pairs with dict candidates:")
    print(f"{'=' * 70}")
    print(f"  Overall:    {correct}/{total}  = {100*correct/total:.1f}%")
    print(f"  Auto (1):   {correct_auto}/{total_auto}  = {100*correct_auto/total_auto:.1f}%"
          if total_auto else "  Auto (1):   N/A")
    print(f"  LLM select: {correct_select}/{total_select}  = {100*correct_select/total_select:.1f}%"
          if total_select else "  LLM select: N/A")
    print()

    if wrong_examples:
        wrong_examples.sort()
        print(f"WRONG selections ({len(wrong_examples)} cases):")
        for ex in wrong_examples:
            print(ex)

    # Dump failures to JSON for rule generation
    import json
    failures_path = Path("/tmp/dict_selection_failures.json")
    failures_path.write_text(json.dumps(failure_details, indent=2))
    print(f"\nFailure details written to {failures_path}")


if __name__ == "__main__":
    main()
