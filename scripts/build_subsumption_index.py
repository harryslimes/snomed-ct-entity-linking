#!/usr/bin/env python3
"""Build a SNOMED CT subsumption index from Athena CONCEPT_RELATIONSHIP data.

Produces snomed_index/subsumption.pkl containing:
  - ancestors: Dict[int, Set[int]]  — SNOMED concept_id → all ancestor concept_ids
  - concept_names: Dict[int, str]   — SNOMED concept_id → concept_name (for display)

The index uses SNOMED concept codes (not OMOP IDs) via the CONCEPT.csv mapping.

Usage:
    python scripts/build_subsumption_index.py
"""
from __future__ import annotations

import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ATHENA_DIR = REPO_ROOT / "data" / "athena"
TERMINOLOGY_CSV = REPO_ROOT / "1st Place" / "data" / "interim" / "flattened_terminology.csv"
OUTPUT_PATH = REPO_ROOT / "snomed_index" / "subsumption.pkl"


def build_omop_snomed_maps() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build OMOP concept_id <-> SNOMED concept_code maps from CONCEPT.csv.

    Returns (snomed_to_omop, omop_to_snomed, omop_to_name).
    """
    snomed_to_omop: dict[str, str] = {}
    omop_to_snomed: dict[str, str] = {}
    omop_to_name: dict[str, str] = {}

    with open(ATHENA_DIR / "CONCEPT.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["vocabulary_id"] == "SNOMED":
                snomed_to_omop[row["concept_code"]] = row["concept_id"]
                omop_to_snomed[row["concept_id"]] = row["concept_code"]
                omop_to_name[row["concept_id"]] = row["concept_name"]

    return snomed_to_omop, omop_to_snomed, omop_to_name


def build_parent_map(omop_to_snomed: dict[str, str]) -> dict[int, set[int]]:
    """Build child → parent map from CONCEPT_RELATIONSHIP.csv 'Is a' rows.

    Filters to only include concepts that have SNOMED codes.
    Returns map using SNOMED concept codes (as ints).
    """
    parents: dict[int, set[int]] = defaultdict(set)
    n_rels = 0

    with open(ATHENA_DIR / "CONCEPT_RELATIONSHIP.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["relationship_id"] != "Is a":
                continue
            if row.get("invalid_reason"):
                continue

            child_omop = row["concept_id_1"]
            parent_omop = row["concept_id_2"]

            # Only include if both are SNOMED concepts
            child_snomed = omop_to_snomed.get(child_omop)
            parent_snomed = omop_to_snomed.get(parent_omop)
            if child_snomed and parent_snomed:
                parents[int(child_snomed)].add(int(parent_snomed))
                n_rels += 1

    print(f"  Loaded {n_rels} SNOMED is-a relationships for {len(parents)} concepts")
    return parents


def compute_ancestors(parents: dict[int, set[int]]) -> dict[int, set[int]]:
    """Compute transitive closure of the is-a hierarchy.

    For each concept, returns the full set of all ancestors (including
    indirect ones). Uses iterative BFS per concept with memoisation.
    """
    ancestors: dict[int, set[int]] = {}

    def get_ancestors(cid: int) -> set[int]:
        if cid in ancestors:
            return ancestors[cid]

        # Use iterative approach to avoid stack overflow
        stack = [cid]
        visiting: set[int] = set()

        while stack:
            current = stack[-1]

            if current in ancestors:
                stack.pop()
                continue

            if current in visiting:
                # All parents already resolved
                result: set[int] = set()
                for p in parents.get(current, set()):
                    result.add(p)
                    result.update(ancestors.get(p, set()))
                ancestors[current] = result
                visiting.discard(current)
                stack.pop()
                continue

            visiting.add(current)

            # Push unresolved parents
            unresolved = [
                p for p in parents.get(current, set())
                if p not in ancestors
            ]
            if unresolved:
                stack.extend(unresolved)
            else:
                # All parents resolved, compute now
                result = set()
                for p in parents.get(current, set()):
                    result.add(p)
                    result.update(ancestors.get(p, set()))
                ancestors[current] = result
                visiting.discard(current)
                stack.pop()

    # Compute for all concepts in our terminology
    all_concepts: set[int] = set()
    with open(TERMINOLOGY_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_concepts.add(int(row["concept_id"]))

    print(f"  Computing ancestors for {len(all_concepts)} terminology concepts...")

    # Also include all concepts reachable through parents
    all_in_hierarchy = set(parents.keys())
    for parent_set in parents.values():
        all_in_hierarchy.update(parent_set)

    for cid in all_in_hierarchy:
        get_ancestors(cid)

    # Filter to only our terminology concepts (but keep full ancestor info)
    n_with_ancestors = sum(1 for c in all_concepts if c in ancestors and ancestors[c])
    print(f"  {n_with_ancestors}/{len(all_concepts)} terminology concepts have ancestors")

    return ancestors


def build_concept_names() -> dict[int, str]:
    """Build concept_id → concept_name map from our terminology."""
    names: dict[int, str] = {}
    with open(TERMINOLOGY_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names[int(row["concept_id"])] = row["concept_name"]
    return names


def main() -> None:
    print("Building SNOMED subsumption index...")

    print("Step 1: Loading OMOP↔SNOMED mappings from CONCEPT.csv...")
    snomed_to_omop, omop_to_snomed, omop_to_name = build_omop_snomed_maps()
    print(f"  {len(snomed_to_omop)} SNOMED concepts mapped")

    print("Step 2: Loading is-a relationships from CONCEPT_RELATIONSHIP.csv...")
    parents = build_parent_map(omop_to_snomed)

    print("Step 3: Computing transitive ancestor closure...")
    ancestors = compute_ancestors(parents)
    print(f"  Total concepts with ancestors: {len(ancestors)}")

    # Quick sanity check
    # Hypertension (38341003) should have ancestors
    hyp_ancestors = ancestors.get(38341003, set())
    print(f"  Sanity check — hypertension (38341003) has {len(hyp_ancestors)} ancestors")
    if hyp_ancestors:
        names = build_concept_names()
        for a in sorted(hyp_ancestors)[:5]:
            print(f"    ancestor: {a} ({names.get(a, omop_to_name.get(snomed_to_omop.get(str(a), ''), '?'))})")

    print("Step 4: Building concept names lookup...")
    concept_names = build_concept_names()

    print("Step 5: Saving index...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({
            "ancestors": ancestors,
            "parents": dict(parents),
            "concept_names": concept_names,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  Saved to {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
