#!/usr/bin/env python3
"""Greedy diversity sampler for training annotations.

Selects a representative subset of annotations that covers a target percentage
of unique SNOMED concepts while maximising within-concept diversity (different
span texts, different note sections).

Algorithm:
  1. Greedy set-cover on notes: repeatedly pick the note that adds the most
     uncovered concepts, until the target concept coverage is reached.
  2. Within the selected notes, greedily sample annotations per concept:
     - Rare concepts (≤ budget): keep all annotations
     - Common concepts: farthest-first traversal on (normalised_span, section)
       to pick the most diverse subset up to the per-concept budget.

Output: a JSON file with the selected annotations and coverage statistics.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from engine import segment_sections  # noqa: E402

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
TERMINOLOGY_CSV = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, tuple[str, str]]]:
    """Load notes, annotations, and terminology."""
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    ann_df = pd.read_csv(SPLIT_DIR / "train_annotations.csv")
    ann_df["start"] = ann_df["start"].astype(int)
    ann_df["end"] = ann_df["end"].astype(int)
    ann_df["concept_id"] = ann_df["concept_id"].astype(int)
    ann_df["annotation_id"] = ann_df["annotation_id"].astype(int)

    ft = pd.read_csv(TERMINOLOGY_CSV)
    concept_info = {
        int(row.concept_id): (row.concept_name, row.hierarchy)
        for row in ft.itertuples()
    }
    return notes_df, ann_df, concept_info


# ---------------------------------------------------------------------------
# Step 1: Greedy set-cover on notes
# ---------------------------------------------------------------------------

def greedy_note_selection(
    ann_df: pd.DataFrame,
    target_coverage: float = 0.90,
) -> list[str]:
    """Select notes greedily to cover target_coverage of unique concepts."""
    all_concepts = set(ann_df["concept_id"].unique())
    target_count = int(len(all_concepts) * target_coverage)

    # Pre-compute concepts per note
    note_concepts: dict[str, set[int]] = {}
    for note_id, group in ann_df.groupby("note_id"):
        note_concepts[note_id] = set(group["concept_id"].unique())

    covered: set[int] = set()
    selected_notes: list[str] = []

    while len(covered) < target_count and note_concepts:
        # Pick the note that adds the most new concepts
        best_note = max(note_concepts, key=lambda nid: len(note_concepts[nid] - covered))
        new_concepts = note_concepts[best_note] - covered
        if not new_concepts:
            break

        selected_notes.append(best_note)
        covered.update(new_concepts)
        del note_concepts[best_note]

        if len(selected_notes) % 10 == 0:
            print(f"  {len(selected_notes)} notes selected, "
                  f"{len(covered)}/{len(all_concepts)} concepts covered "
                  f"({100*len(covered)/len(all_concepts):.1f}%)")

    print(f"  Final: {len(selected_notes)} notes, "
          f"{len(covered)}/{len(all_concepts)} concepts covered "
          f"({100*len(covered)/len(all_concepts):.1f}%)")
    return selected_notes


# ---------------------------------------------------------------------------
# Step 2: Section assignment
# ---------------------------------------------------------------------------

def assign_sections(
    note_text: str,
    note_anns: pd.DataFrame,
) -> list[str]:
    """Return the section header for each annotation in note_anns (same order)."""
    sections = segment_sections(note_text)
    result = []
    for _, row in note_anns.iterrows():
        start = int(row["start"])
        header = "unknown"
        for sec in sections:
            if sec.start <= start < sec.end:
                header = sec.header
                break
        result.append(header)
    return result


# ---------------------------------------------------------------------------
# Step 3: Within-concept diversity sampling
# ---------------------------------------------------------------------------

def _normalise_span(span: str) -> str:
    """Normalise span text for comparison."""
    return span.strip().lower()


def _annotation_key(norm_span: str, section: str) -> tuple[str, str]:
    """Feature tuple for diversity comparison."""
    return (norm_span, section)


def diversity_sample(
    annotations: list[dict],
    budget: int,
) -> list[dict]:
    """Farthest-first traversal to pick diverse annotations.

    Diversity is based on (normalised_span, section) tuples.
    Two annotations are "same" if they share both span text and section.
    """
    if len(annotations) <= budget:
        return annotations

    # Build feature keys
    keys = [_annotation_key(a["norm_span"], a["section"]) for a in annotations]

    # Start with the first annotation
    selected_indices = [0]
    selected_keys = {keys[0]}

    while len(selected_indices) < budget:
        best_idx = -1
        best_score = -1

        for i, key in enumerate(keys):
            if i in set(selected_indices):
                continue
            # Score: prefer annotations with unseen (span, section) pairs
            # Then prefer unseen spans regardless of section
            # Then prefer unseen sections regardless of span
            span_match = any(key[0] == sk[0] for sk in selected_keys)
            section_match = any(key[1] == sk[1] for sk in selected_keys)
            exact_match = key in selected_keys

            if exact_match:
                score = 0
            elif not span_match and not section_match:
                score = 3  # New span AND new section
            elif not span_match:
                score = 2  # New span, seen section
            elif not section_match:
                score = 1  # Seen span, new section
            else:
                score = 0  # Both seen

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == -1 or best_score == 0:
            # All remaining are duplicates; just pick to fill budget
            remaining = [i for i in range(len(annotations)) if i not in set(selected_indices)]
            needed = budget - len(selected_indices)
            selected_indices.extend(remaining[:needed])
            break

        selected_indices.append(best_idx)
        selected_keys.add(keys[best_idx])

    return [annotations[i] for i in selected_indices]


def sample_annotations(
    selected_notes: list[str],
    notes_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    concept_info: dict[int, tuple[str, str]],
    per_concept_budget: int = 8,
) -> tuple[list[dict], dict]:
    """Sample diverse annotations from selected notes."""
    # Filter to selected notes
    note_set = set(selected_notes)
    sel_anns = ann_df[ann_df["note_id"].isin(note_set)].copy()

    # Build note text lookup
    note_texts = {
        row["note_id"]: row["text"]
        for _, row in notes_df[notes_df["note_id"].isin(note_set)].iterrows()
    }

    # Assign sections to all annotations
    print("Assigning sections to annotations ...")
    section_map: dict[int, str] = {}  # annotation_id -> section
    for note_id in selected_notes:
        note_text = note_texts[note_id]
        note_subset = sel_anns[sel_anns["note_id"] == note_id].sort_values("start")
        sections = assign_sections(note_text, note_subset)
        for ann_id, section in zip(note_subset["annotation_id"], sections):
            section_map[ann_id] = section

    # Group annotations by concept_id
    concept_annotations: dict[int, list[dict]] = defaultdict(list)
    for _, row in sel_anns.iterrows():
        ann_id = int(row["annotation_id"])
        cid = int(row["concept_id"])
        span = row["span"] if pd.notna(row["span"]) else note_texts[row["note_id"]][int(row["start"]):int(row["end"])]
        cname, hierarchy = concept_info.get(cid, ("Unknown", "unknown"))

        concept_annotations[cid].append({
            "annotation_id": ann_id,
            "note_id": row["note_id"],
            "start": int(row["start"]),
            "end": int(row["end"]),
            "span": span,
            "norm_span": _normalise_span(span),
            "concept_id": cid,
            "concept_name": cname,
            "hierarchy": hierarchy,
            "section": section_map.get(ann_id, "unknown"),
        })

    # Diversity-sample within each concept
    print(f"Diversity sampling within {len(concept_annotations)} concepts "
          f"(budget={per_concept_budget} per concept) ...")
    sampled: list[dict] = []
    stats = {
        "kept_all": 0,  # concepts where we kept all annotations
        "subsampled": 0,  # concepts where we subsampled
        "total_before": 0,
        "total_after": 0,
    }

    for cid, anns in concept_annotations.items():
        stats["total_before"] += len(anns)
        if len(anns) <= per_concept_budget:
            sampled.extend(anns)
            stats["kept_all"] += 1
            stats["total_after"] += len(anns)
        else:
            selected = diversity_sample(anns, per_concept_budget)
            sampled.extend(selected)
            stats["subsampled"] += 1
            stats["total_after"] += len(selected)

    return sampled, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Greedy annotation sampler")
    parser.add_argument("--coverage", type=float, default=0.90,
                        help="Target concept coverage (default: 0.90)")
    parser.add_argument("--budget", type=int, default=8,
                        help="Max annotations per concept (default: 8)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: scripts/sampled_annotations.json)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only print statistics, don't save output")
    args = parser.parse_args()

    print("Loading data ...")
    notes_df, ann_df, concept_info = load_all()

    all_concepts = set(ann_df["concept_id"].unique())
    all_notes = set(ann_df["note_id"].unique())
    print(f"Total: {len(all_notes)} notes, {len(ann_df)} annotations, "
          f"{len(all_concepts)} unique concepts")

    # Step 1: Greedy note selection
    print(f"\n=== Step 1: Greedy note selection (target {args.coverage*100:.0f}% concept coverage) ===")
    selected_notes = greedy_note_selection(ann_df, target_coverage=args.coverage)

    # Step 2: Diversity sampling
    print(f"\n=== Step 2: Diversity sampling (budget={args.budget}/concept) ===")
    sampled, stats = sample_annotations(
        selected_notes, notes_df, ann_df, concept_info,
        per_concept_budget=args.budget,
    )

    # Coverage stats
    sampled_concepts = set(a["concept_id"] for a in sampled)
    sampled_notes = set(a["note_id"] for a in sampled)

    # Span text stats
    unique_spans = set(a["norm_span"] for a in sampled)
    span_concept_pairs = set((a["norm_span"], a["concept_id"]) for a in sampled)

    # Section distribution
    section_counts: dict[str, int] = defaultdict(int)
    for a in sampled:
        section_counts[a["section"]] += 1

    # Hierarchy distribution
    hierarchy_counts: dict[str, int] = defaultdict(int)
    for a in sampled:
        hierarchy_counts[a["hierarchy"]] += 1

    print(f"\n=== Summary ===")
    print(f"Notes selected:     {len(sampled_notes)} / {len(all_notes)}")
    print(f"Concepts covered:   {len(sampled_concepts)} / {len(all_concepts)} "
          f"({100*len(sampled_concepts)/len(all_concepts):.1f}%)")
    print(f"Annotations:        {stats['total_before']} -> {stats['total_after']} "
          f"(from selected notes)")
    print(f"  kept all:         {stats['kept_all']} concepts")
    print(f"  subsampled:       {stats['subsampled']} concepts")
    print(f"Unique span texts:  {len(unique_spans)}")
    print(f"Span-concept pairs: {len(span_concept_pairs)}")
    print(f"\nSection distribution:")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {section:40s} {count:5d} ({100*count/len(sampled):.1f}%)")
    print(f"\nHierarchy distribution:")
    for h, count in sorted(hierarchy_counts.items(), key=lambda x: -x[1]):
        print(f"  {h:30s} {count:5d} ({100*count/len(sampled):.1f}%)")

    # Concept frequency in sample
    concept_counts = defaultdict(int)
    for a in sampled:
        concept_counts[a["concept_id"]] += 1
    freq_buckets = {1: 0, "2-5": 0, "6-8": 0}
    for count in concept_counts.values():
        if count == 1:
            freq_buckets[1] += 1
        elif count <= 5:
            freq_buckets["2-5"] += 1
        else:
            freq_buckets["6-8"] += 1
    print(f"\nSampled concept frequency distribution:")
    for bucket, n in freq_buckets.items():
        print(f"  {str(bucket):10s} {n:5d} concepts")

    if args.stats_only:
        return

    # Save output
    output_path = Path(args.output) if args.output else (
        Path(__file__).parent / "sampled_annotations.json"
    )

    # Strip norm_span from output (internal field)
    for a in sampled:
        del a["norm_span"]

    output = {
        "coverage_target": args.coverage,
        "per_concept_budget": args.budget,
        "n_notes_selected": len(sampled_notes),
        "n_notes_total": len(all_notes),
        "n_concepts_covered": len(sampled_concepts),
        "n_concepts_total": len(all_concepts),
        "concept_coverage": len(sampled_concepts) / len(all_concepts),
        "n_annotations_sampled": len(sampled),
        "n_annotations_total": len(ann_df),
        "selected_note_ids": selected_notes,
        "annotations": sampled,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(sampled)} annotations to {output_path}")


if __name__ == "__main__":
    main()
