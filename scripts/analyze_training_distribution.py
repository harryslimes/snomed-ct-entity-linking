#!/usr/bin/env python3
"""Analyze training annotation data distribution for SNOMED CT entity linking."""

import sys
import pandas as pd
import numpy as np
from collections import Counter

# ── Load data ──────────────────────────────────────────────────────────────
DATA_DIR = "/workspaces/snomed-ct-entity-linking/data/old-challenge-split"
notes = pd.read_csv(f"{DATA_DIR}/train_notes.csv")
annots = pd.read_csv(f"{DATA_DIR}/train_annotations.csv")
terminology = pd.read_csv(
    "/workspaces/snomed-ct-entity-linking/3rd Place/assets/dataflattened_terminology.csv"
)

# Build concept_id -> concept_name and concept_id -> hierarchy lookup
concept_name_map = dict(zip(terminology["concept_id"], terminology["concept_name"]))
concept_hierarchy_map = dict(zip(terminology["concept_id"], terminology["hierarchy"]))

# Build note_id -> text lookup
note_text_map = dict(zip(notes["note_id"], notes["text"]))

# Extract span text from note text + offsets (verify against 'span' column)
def extract_span(row):
    text = note_text_map.get(row["note_id"], "")
    start, end = int(row["start"]), int(row["end"])
    return text[start:end]

annots["span_extracted"] = annots.apply(extract_span, axis=1)

print("=" * 80)
print("  SNOMED CT ENTITY LINKING - TRAINING DATA DISTRIBUTION ANALYSIS")
print("=" * 80)

# ======================================================================
# 1. BASIC STATS
# ======================================================================
print("\n" + "-" * 80)
print("1. BASIC STATS")
print("-" * 80)

n_notes = notes.shape[0]
n_annots = annots.shape[0]
annots_per_note = annots.groupby("note_id").size()

print(f"  Number of notes:                {n_notes:,}")
print(f"  Total annotations:              {n_annots:,}")
print(f"  Annotations per note:")
print(f"    Min:    {annots_per_note.min()}")
print(f"    Max:    {annots_per_note.max()}")
print(f"    Mean:   {annots_per_note.mean():.1f}")
print(f"    Median: {annots_per_note.median():.1f}")
print(f"    Std:    {annots_per_note.std():.1f}")

# ======================================================================
# 2. CONCEPT FREQUENCY DISTRIBUTION
# ======================================================================
print("\n" + "-" * 80)
print("2. CONCEPT FREQUENCY DISTRIBUTION")
print("-" * 80)

concept_counts = annots["concept_id"].value_counts()
n_unique_concepts = concept_counts.shape[0]
print(f"  Unique concept_ids: {n_unique_concepts:,}")

bins = [
    ("1 time", 1, 1),
    ("2-5 times", 2, 5),
    ("6-10 times", 6, 10),
    ("11-50 times", 11, 50),
    ("51-100 times", 51, 100),
    ("100+ times", 101, float("inf")),
]
print(f"\n  Annotation count per concept_id:")
print(f"  {'Bucket':<15} {'# Concepts':>12} {'% of Concepts':>15} {'% of Annots':>13}")
for label, lo, hi in bins:
    mask = (concept_counts >= lo) & (concept_counts <= hi)
    n_concepts = mask.sum()
    n_annots_in_bucket = concept_counts[mask].sum()
    print(
        f"  {label:<15} {n_concepts:>12,} {100*n_concepts/n_unique_concepts:>14.1f}% {100*n_annots_in_bucket/n_annots:>12.1f}%"
    )

print(f"\n  Top 20 most frequent concept_ids:")
print(f"  {'Rank':<5} {'concept_id':<12} {'Count':>6} {'Name'}")
for i, (cid, count) in enumerate(concept_counts.head(20).items(), 1):
    name = concept_name_map.get(cid, "<not in terminology>")
    # Truncate long names
    if len(str(name)) > 60:
        name = str(name)[:57] + "..."
    print(f"  {i:<5} {cid:<12} {count:>6} {name}")

# ======================================================================
# 3. SPAN TEXT ANALYSIS
# ======================================================================
print("\n" + "-" * 80)
print("3. SPAN TEXT ANALYSIS")
print("-" * 80)

# Use the 'span' column directly (it's provided in the CSV)
span_col = "span"
unique_spans = annots[span_col].nunique()
print(f"  Unique span texts: {unique_spans:,}")

# Span length distribution
span_lengths = annots[span_col].str.len()
print(f"\n  Span text length (chars):")
print(f"    Min:    {span_lengths.min()}")
print(f"    Max:    {span_lengths.max()}")
print(f"    Mean:   {span_lengths.mean():.1f}")
print(f"    Median: {span_lengths.median():.1f}")
print(f"    Std:    {span_lengths.std():.1f}")

percentiles = [10, 25, 50, 75, 90, 95, 99]
pvals = np.percentile(span_lengths.dropna(), percentiles)
print(f"    Percentiles:")
for p, v in zip(percentiles, pvals):
    print(f"      P{p:<3}: {v:.0f} chars")

# Ambiguous spans: span texts mapping to multiple concepts
span_concept_map = annots.groupby(span_col)["concept_id"].nunique()
ambiguous_spans = span_concept_map[span_concept_map > 1]
print(f"\n  Span texts mapping to multiple concept_ids: {len(ambiguous_spans):,} / {unique_spans:,} ({100*len(ambiguous_spans)/unique_spans:.1f}%)")

# Distribution of ambiguity
amb_dist = ambiguous_spans.value_counts().sort_index()
print(f"  Distribution of # concepts per ambiguous span:")
for n_concepts, count in amb_dist.items():
    print(f"    Maps to {n_concepts} concepts: {count} span texts")

# Top 20 most frequent span texts
span_counts = annots[span_col].value_counts()
print(f"\n  Top 20 most frequent span texts:")
print(f"  {'Rank':<5} {'Count':>6} {'# Concepts':>11} {'Span Text'}")
for i, (span_text, count) in enumerate(span_counts.head(20).items(), 1):
    n_concepts = span_concept_map.get(span_text, 0)
    display = str(span_text)
    if len(display) > 50:
        display = display[:47] + "..."
    print(f"  {i:<5} {count:>6} {n_concepts:>11} {display}")

# Top ambiguous span texts (most concepts)
print(f"\n  Top 20 most ambiguous span texts (by # of distinct concepts):")
top_ambig = ambiguous_spans.sort_values(ascending=False).head(20)
print(f"  {'# Concepts':>11} {'Total Count':>12} {'Span Text'}")
for span_text, n_concepts in top_ambig.items():
    total_count = span_counts.get(span_text, 0)
    display = str(span_text)
    if len(display) > 50:
        display = display[:47] + "..."
    print(f"  {n_concepts:>11} {total_count:>12} {display}")

# ======================================================================
# 4. CONCEPT-TO-SPAN DIVERSITY
# ======================================================================
print("\n" + "-" * 80)
print("4. CONCEPT-TO-SPAN DIVERSITY (Top 20 concepts by frequency)")
print("-" * 80)

top20_concepts = concept_counts.head(20).index
print(f"  {'Rank':<5} {'concept_id':<12} {'# Annots':>9} {'# Spans':>8} {'Concept Name'}")
for i, cid in enumerate(top20_concepts, 1):
    subset = annots[annots["concept_id"] == cid]
    n_annots_c = len(subset)
    n_spans = subset[span_col].nunique()
    name = concept_name_map.get(cid, "<not in terminology>")
    if len(str(name)) > 45:
        name = str(name)[:42] + "..."
    print(f"  {i:<5} {cid:<12} {n_annots_c:>9} {n_spans:>8} {name}")

# Show the actual span texts for the top 5 concepts
print(f"\n  Span texts for the top 5 concepts:")
for i, cid in enumerate(top20_concepts[:5], 1):
    subset = annots[annots["concept_id"] == cid]
    name = concept_name_map.get(cid, "<not in terminology>")
    span_freq = subset[span_col].value_counts()
    print(f"\n  [{i}] {cid} - {name}")
    for s, c in span_freq.head(10).items():
        print(f"      {c:>4}x  \"{s}\"")
    if len(span_freq) > 10:
        print(f"      ... and {len(span_freq) - 10} more distinct span texts")

# ======================================================================
# 5. SNOMED HIERARCHY DISTRIBUTION
# ======================================================================
print("\n" + "-" * 80)
print("5. SNOMED HIERARCHY DISTRIBUTION")
print("-" * 80)

annots["hierarchy"] = annots["concept_id"].map(concept_hierarchy_map)
hierarchy_counts = annots["hierarchy"].value_counts()
n_with_hierarchy = annots["hierarchy"].notna().sum()
n_without = annots["hierarchy"].isna().sum()

print(f"  Annotations with known hierarchy:  {n_with_hierarchy:,}")
print(f"  Annotations with unknown hierarchy: {n_without:,}")
print(f"\n  {'Hierarchy':<45} {'Count':>8} {'%':>7}")
print(f"  {'-'*45} {'-'*8} {'-'*7}")
for hier, count in hierarchy_counts.items():
    pct = 100 * count / n_annots
    label = str(hier) if pd.notna(hier) else "<unknown>"
    if len(label) > 44:
        label = label[:41] + "..."
    print(f"  {label:<45} {count:>8} {pct:>6.1f}%")

# ======================================================================
# 6. SECTION DISTRIBUTION (sample of 20 notes)
# ======================================================================
print("\n" + "-" * 80)
print("6. SECTION DISTRIBUTION (sampled 20 notes)")
print("-" * 80)

sys.path.insert(0, "/workspaces/snomed-ct-entity-linking/super-dictionary")
from engine import segment_sections, get_section_for_pos

# Sample 20 notes (deterministic)
rng = np.random.RandomState(42)
sample_note_ids = rng.choice(notes["note_id"].values, size=min(20, len(notes)), replace=False)

section_counter = Counter()
total_sampled_annots = 0
no_section_count = 0

for nid in sample_note_ids:
    text = note_text_map.get(nid, "")
    sections = segment_sections(text)
    note_annots = annots[annots["note_id"] == nid]
    total_sampled_annots += len(note_annots)

    for _, row in note_annots.iterrows():
        start = int(row["start"])
        sec = get_section_for_pos(start, sections)
        if sec is not None:
            section_counter[sec.header] += 1
        else:
            no_section_count += 1

print(f"  Notes sampled: {len(sample_note_ids)}")
print(f"  Annotations in sample: {total_sampled_annots}")
print(f"  Annotations with no detected section: {no_section_count}")

print(f"\n  {'Section Header':<45} {'Count':>8} {'%':>7}")
print(f"  {'-'*45} {'-'*8} {'-'*7}")
for header, count in sorted(section_counter.items(), key=lambda x: -x[1]):
    pct = 100 * count / total_sampled_annots if total_sampled_annots else 0
    print(f"  {header:<45} {count:>8} {pct:>6.1f}%")
if no_section_count:
    pct = 100 * no_section_count / total_sampled_annots
    print(f"  {'<no section / preamble>':<45} {no_section_count:>8} {pct:>6.1f}%")

# ======================================================================
# Summary insights
# ======================================================================
print("\n" + "=" * 80)
print("  SUMMARY INSIGHTS FOR SAMPLING STRATEGY")
print("=" * 80)

singleton_concepts = (concept_counts == 1).sum()
rare_concepts = (concept_counts <= 5).sum()
hier_top3 = ', '.join(hierarchy_counts.head(3).index.tolist())
print(f"""
  - {n_annots:,} annotations across {n_notes:,} notes (~{annots_per_note.mean():.0f} per note)
  - {n_unique_concepts:,} unique concepts, of which:
      {singleton_concepts:,} ({100*singleton_concepts/n_unique_concepts:.1f}%) appear only once (singletons)
      {rare_concepts:,} ({100*rare_concepts/n_unique_concepts:.1f}%) appear 5 or fewer times (rare)
  - {len(ambiguous_spans):,} span texts are ambiguous (map to 2+ concepts)
  - The long tail is significant: rare concepts dominate the concept space
    but frequent concepts dominate the annotation volume
  - Top hierarchy types: {hier_top3}
""")
