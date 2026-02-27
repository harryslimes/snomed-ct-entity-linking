#!/usr/bin/env python3
"""Analyze structured lab result tables in training notes and quantify
how many annotations (especially procedures) fall within them."""

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Load SNOMED tag mapping
# ---------------------------------------------------------------------------
sys.path.insert(0, "/workspaces/gliner2-training/scripts")
from prepare_gliner2_data import TAG_TO_CLASS, load_sctid_to_tag

sctid_to_tag = load_sctid_to_tag(Path("/workspaces/snomed-ct-entity-linking"))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_DIR = Path("/workspaces/gliner2-training/data/old-challenge-split")
notes_df = pd.read_csv(DATA_DIR / "train_notes.csv")
annot_df = pd.read_csv(DATA_DIR / "train_annotations.csv")

print(f"Loaded {len(notes_df)} notes, {len(annot_df)} annotations")

# ---------------------------------------------------------------------------
# Classify annotations by entity class
# ---------------------------------------------------------------------------
def get_class(concept_id):
    try:
        cid = int(float(concept_id))
    except (ValueError, TypeError):
        return "medical finding, symptom, or disease"
    tag = sctid_to_tag.get(cid, "")
    return TAG_TO_CLASS.get(tag, "medical finding, symptom, or disease")

annot_df["entity_class"] = annot_df["concept_id"].apply(get_class)

print(f"\nAnnotation class distribution:")
for cls, cnt in annot_df["entity_class"].value_counts().items():
    print(f"  {cls}: {cnt}")

# ---------------------------------------------------------------------------
# Detect structured lab result lines
# ---------------------------------------------------------------------------
# Pattern: uppercase test name (2-6 chars) followed by hyphen and a number
# e.g., WBC-8.4, RBC-4.06*, Na-139, Hgb-12.8*
LAB_VALUE_PATTERN = re.compile(r"[A-Z][A-Za-z0-9]{1,5}-\d+\.?\d*\*?")

# A line is a "lab table line" if it contains 2+ lab value patterns
MIN_LAB_VALUES_PER_LINE = 2


def find_lab_regions(text: str):
    """Return list of (start, end) character ranges for lab table lines,
    plus a list of all test names found."""
    regions = []
    test_names = []

    for m in re.finditer(r"[^\n]*\n?", text):
        line = m.group()
        lab_matches = list(LAB_VALUE_PATTERN.finditer(line))
        if len(lab_matches) >= MIN_LAB_VALUES_PER_LINE:
            line_start = m.start()
            line_end = m.end()
            regions.append((line_start, line_end))
            for lm in lab_matches:
                # Extract test name (part before hyphen)
                token = lm.group()
                name = token.split("-")[0]
                test_names.append(name)

    return regions, test_names


# ---------------------------------------------------------------------------
# Process each note
# ---------------------------------------------------------------------------
all_test_names = Counter()
notes_with_labs = 0
total_lab_lines = 0
note_lab_regions = {}  # note_id -> list of (start, end)

for _, row in notes_df.iterrows():
    note_id = row["note_id"]
    text = row["text"]

    regions, test_names = find_lab_regions(text)
    if regions:
        notes_with_labs += 1
        total_lab_lines += len(regions)
        note_lab_regions[note_id] = regions
        all_test_names.update(test_names)

print(f"\n{'='*60}")
print(f"LAB TABLE DETECTION SUMMARY")
print(f"{'='*60}")
print(f"Notes with lab tables: {notes_with_labs} / {len(notes_df)} "
      f"({100*notes_with_labs/len(notes_df):.1f}%)")
print(f"Total lab table lines: {total_lab_lines}")
if notes_with_labs > 0:
    print(f"Average lab table lines per note (among those with labs): "
          f"{total_lab_lines/notes_with_labs:.1f}")
print(f"Average lab table lines per note (overall): "
      f"{total_lab_lines/len(notes_df):.1f}")

# ---------------------------------------------------------------------------
# Top test names
# ---------------------------------------------------------------------------
print(f"\nTop 20 most common test names in lab tables:")
for name, cnt in all_test_names.most_common(20):
    print(f"  {name}: {cnt}")

# ---------------------------------------------------------------------------
# Cross-reference annotations with lab regions
# ---------------------------------------------------------------------------
def annotation_in_lab_region(note_id, start, end, lab_regions_map):
    """Check if annotation overlaps with any lab table region."""
    regions = lab_regions_map.get(note_id, [])
    for r_start, r_end in regions:
        # Overlap check: annotation [start, end) overlaps region [r_start, r_end)
        if start < r_end and end > r_start:
            return True
    return False


annot_df["in_lab_table"] = annot_df.apply(
    lambda r: annotation_in_lab_region(
        r["note_id"], float(r["start"]), float(r["end"]), note_lab_regions
    ),
    axis=1,
)

print(f"\n{'='*60}")
print(f"ANNOTATIONS IN LAB TABLE REGIONS")
print(f"{'='*60}")

total_in_lab = annot_df["in_lab_table"].sum()
print(f"Total annotations in lab tables: {total_in_lab} / {len(annot_df)} "
      f"({100*total_in_lab/len(annot_df):.1f}%)")

print(f"\nBreakdown by entity class:")
for cls in sorted(annot_df["entity_class"].unique()):
    subset = annot_df[annot_df["entity_class"] == cls]
    in_lab = subset["in_lab_table"].sum()
    print(f"  {cls}:")
    print(f"    In lab tables: {in_lab} / {len(subset)} "
          f"({100*in_lab/len(subset):.1f}%)")

# Focus on procedures
print(f"\n{'='*60}")
print(f"PROCEDURE ANNOTATIONS IN LAB TABLES (DETAIL)")
print(f"{'='*60}")
proc_in_lab = annot_df[
    (annot_df["entity_class"] == "procedure") & (annot_df["in_lab_table"])
]
print(f"Procedure annotations in lab tables: {len(proc_in_lab)}")
if len(proc_in_lab) > 0:
    print(f"\nTop 20 most common procedure spans in lab tables:")
    for span, cnt in proc_in_lab["span"].value_counts().head(20).items():
        concept_ids = proc_in_lab[proc_in_lab["span"] == span]["concept_id"].unique()
        print(f"  '{span}' (concept_ids: {list(concept_ids)}): {cnt}")

    # Show some examples with context
    print(f"\nExamples of procedure annotations inside lab tables:")
    shown = 0
    for _, r in proc_in_lab.head(10).iterrows():
        note_text = notes_df[notes_df["note_id"] == r["note_id"]]["text"].iloc[0]
        start = int(float(r["start"]))
        end = int(float(r["end"]))
        ctx_start = max(0, start - 60)
        ctx_end = min(len(note_text), end + 60)
        context = note_text[ctx_start:ctx_end].replace("\n", "\\n")
        span_text = note_text[start:end]
        print(f"  Note {r['note_id']}, chars [{start}:{end}], "
              f"span='{span_text}', concept={r['concept_id']}")
        print(f"    Context: ...{context}...")
        shown += 1

# ---------------------------------------------------------------------------
# Also show all-class annotations in lab tables for review
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"ALL ANNOTATIONS IN LAB TABLES - TOP SPANS")
print(f"{'='*60}")
all_in_lab = annot_df[annot_df["in_lab_table"]]
if len(all_in_lab) > 0:
    print(f"\nTop 30 most common annotation spans found in lab tables:")
    for span, cnt in all_in_lab["span"].value_counts().head(30).items():
        classes = all_in_lab[all_in_lab["span"] == span]["entity_class"].unique()
        print(f"  '{span}' [{', '.join(classes)}]: {cnt}")
