#!/usr/bin/env python3
"""Test lab table detection and annotation filtering.

Shows exactly which annotations would be removed/kept, with context,
so we can verify the filter isn't too aggressive or too lenient.

v2 improvements:
- Tighter region: spans from first lab match to last lab match (not full line)
- Require 2+ alpha chars in test name to exclude spinal levels (L1-2, C4-5)
- Small buffer after last match to catch trailing Plt ___, etc.
"""

import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from prepare_gliner2_data import TAG_TO_CLASS, load_sctid_to_tag

# ---------------------------------------------------------------------------
# Lab table detection (v2 — tighter regions)
# ---------------------------------------------------------------------------

# Pattern: test name (starts with 2+ alpha chars, may end with digits like HCO3)
# followed by hyphen and a numeric value, optionally with decimal and/or asterisk.
# Requires 2+ alpha chars at start to exclude spinal levels (L1-2, C4-5, T2-3).
LAB_VALUE_PATTERN = re.compile(
    r"[A-Z][A-Za-z][A-Za-z0-9]{0,4}-\d+\.?\d*\*?"
)

# Minimum number of TESTNAME-VALUE patterns on a single line to call it a lab line
MIN_LAB_VALUES_PER_LINE = 2

# Characters of buffer after the last lab match to include in the region
# (catches trailing Plt ___, spaces, asterisks)
POST_MATCH_BUFFER = 15


def find_lab_regions(text: str) -> list[tuple[int, int]]:
    """Return list of (start, end) character ranges for lab value clusters.

    v2: region spans from first match start to last match end + buffer,
    NOT the full line. This avoids catching narrative text that continues
    on the same line after lab values (e.g., 'Mg-2.___ year old man with...').
    """
    regions = []
    for line_match in re.finditer(r"[^\n]*\n?", text):
        line = line_match.group()
        line_offset = line_match.start()

        lab_matches = list(LAB_VALUE_PATTERN.finditer(line))
        if len(lab_matches) >= MIN_LAB_VALUES_PER_LINE:
            # Region = first match start to last match end + buffer
            first_start = lab_matches[0].start() + line_offset
            last_end = lab_matches[-1].end() + line_offset
            region_end = min(last_end + POST_MATCH_BUFFER, line_match.end())
            regions.append((first_start, region_end))

    return regions


def merge_adjacent_regions(
    regions: list[tuple[int, int]], gap: int = 5
) -> list[tuple[int, int]]:
    """Merge lab regions that are within `gap` characters of each other."""
    if not regions:
        return []
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        if start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def annotation_in_lab_region(
    start: int, end: int, regions: list[tuple[int, int]]
) -> bool:
    """Check if annotation is fully contained within any lab region."""
    for r_start, r_end in regions:
        if start >= r_start and end <= r_end:
            return True
    return False


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path("/workspaces/snomed-ct-entity-linking")
DATA_DIR = Path("/workspaces/gliner2-training/data/old-challenge-split")

print("Loading SNOMED tag mapping...")
sctid_to_tag = load_sctid_to_tag(PROJECT_ROOT)


def get_class(concept_id):
    try:
        cid = int(float(concept_id))
    except (ValueError, TypeError):
        return "medical finding, symptom, or disease"
    tag = sctid_to_tag.get(cid, "")
    return TAG_TO_CLASS.get(tag, "medical finding, symptom, or disease")


# Load both train and test
for split_name in ["train", "test"]:
    print(f"\n{'='*70}")
    print(f"  {split_name.upper()} SPLIT")
    print(f"{'='*70}")

    notes_df = pd.read_csv(DATA_DIR / f"{split_name}_notes.csv")
    annot_df = pd.read_csv(DATA_DIR / f"{split_name}_annotations.csv")
    annot_df["start"] = annot_df["start"].astype(int)
    annot_df["end"] = annot_df["end"].astype(int)
    annot_df["cls"] = annot_df["concept_id"].apply(get_class)

    print(f"Notes: {len(notes_df)}, Annotations: {len(annot_df)}")

    # Detect lab regions per note
    note_lab_regions = {}
    for _, row in notes_df.iterrows():
        regions = find_lab_regions(row["text"])
        regions = merge_adjacent_regions(regions)
        if regions:
            note_lab_regions[row["note_id"]] = regions

    # Tag each annotation
    annot_df["in_lab"] = annot_df.apply(
        lambda r: annotation_in_lab_region(
            r["start"], r["end"],
            note_lab_regions.get(r["note_id"], [])
        ),
        axis=1,
    )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    removed = annot_df[annot_df["in_lab"]]
    kept = annot_df[~annot_df["in_lab"]]

    print(f"\nAnnotations REMOVED (in lab tables): {len(removed):,}")
    print(f"Annotations KEPT:                    {len(kept):,}")
    print(f"Removal rate:                        {100*len(removed)/len(annot_df):.1f}%")

    print(f"\nBy class:")
    for cls in sorted(annot_df["cls"].unique()):
        total = len(annot_df[annot_df["cls"] == cls])
        in_lab = len(removed[removed["cls"] == cls])
        print(f"  {cls:45s}  {in_lab:5d} / {total:5d} removed ({100*in_lab/total:.1f}%)")

    # ------------------------------------------------------------------
    # Check for false positives: non-procedure annotations being removed
    # ------------------------------------------------------------------
    fp_findings = removed[removed["cls"] == "medical finding, symptom, or disease"]
    fp_body = removed[removed["cls"] == "anatomical body part"]

    print(f"\n--- FALSE POSITIVE CHECK (non-procedure removals) ---")
    print(f"Findings removed:  {len(fp_findings)}")
    print(f"Body parts removed: {len(fp_body)}")

    note_texts = dict(zip(notes_df["note_id"], notes_df["text"]))

    if len(fp_findings) > 0:
        print(f"\n  Top finding spans being removed:")
        for span, cnt in fp_findings["span"].value_counts().head(15).items():
            print(f"    '{span}' x{cnt}")
        print(f"\n  Finding removal examples (with context):")
        for i, (_, r) in enumerate(fp_findings.head(10).iterrows()):
            text = note_texts[r["note_id"]]
            s, e = r["start"], r["end"]
            ctx_s = max(0, s - 80)
            ctx_e = min(len(text), e + 80)
            ctx = text[ctx_s:ctx_e].replace("\n", "\\n")
            span_text = text[s:e]
            print(f"    [{i+1}] span='{span_text}', concept={r['concept_id']}")
            print(f"        ...{ctx}...")

    if len(fp_body) > 0:
        print(f"\n  Top body-part spans being removed:")
        for span, cnt in fp_body["span"].value_counts().head(10).items():
            print(f"    '{span}' x{cnt}")
        print(f"\n  Body-part removal examples (with context):")
        for i, (_, r) in enumerate(fp_body.head(5).iterrows()):
            text = note_texts[r["note_id"]]
            s, e = r["start"], r["end"]
            ctx_s = max(0, s - 80)
            ctx_e = min(len(text), e + 80)
            ctx = text[ctx_s:ctx_e].replace("\n", "\\n")
            print(f"    [{i+1}] span='{text[s:e]}', concept={r['concept_id']}")
            print(f"        ...{ctx}...")

    # ------------------------------------------------------------------
    # Check for false negatives: lab-like procedure annotations NOT removed
    # ------------------------------------------------------------------
    LAB_ABBREVS = {
        "WBC", "RBC", "Hgb", "Hct", "MCV", "MCH", "MCHC", "RDW", "Plt",
        "Na", "K", "Cl", "HCO3", "BUN", "Creat", "Glucose", "Ca", "Mg",
        "Phos", "ALT", "AST", "AlkPhos", "TotBili", "Albumin",
        "UreaN", "AnGap", "Lipase", "Fibrinogen",
    }

    kept_procs = kept[kept["cls"] == "procedure"]
    fn_lab_abbrevs = kept_procs[kept_procs["span"].isin(LAB_ABBREVS)]
    print(f"\n--- FALSE NEGATIVE CHECK (lab abbrevs NOT removed) ---")
    print(f"Known lab abbreviation procedure spans still kept: {len(fn_lab_abbrevs)}")
    if len(fn_lab_abbrevs) > 0:
        print(f"  Top kept lab-like procedure spans:")
        for span, cnt in fn_lab_abbrevs["span"].value_counts().head(15).items():
            print(f"    '{span}' x{cnt}")
        print(f"\n  Examples of lab abbrevs NOT removed (appear outside lab tables):")
        for i, (_, r) in enumerate(fn_lab_abbrevs.head(10).iterrows()):
            text = note_texts[r["note_id"]]
            s, e = r["start"], r["end"]
            ctx_s = max(0, s - 80)
            ctx_e = min(len(text), e + 80)
            ctx = text[ctx_s:ctx_e].replace("\n", "\\n")
            print(f"    [{i+1}] span='{r['span']}'")
            print(f"        ...{ctx}...")

    # ------------------------------------------------------------------
    # Spot-check: show 20 random procedure annotations being removed
    # ------------------------------------------------------------------
    removed_procs = removed[removed["cls"] == "procedure"]
    print(f"\n--- SPOT CHECK: 20 random procedure annotations being REMOVED ---")
    sample = removed_procs.sample(min(20, len(removed_procs)), random_state=42)
    for i, (_, r) in enumerate(sample.iterrows()):
        text = note_texts[r["note_id"]]
        s, e = r["start"], r["end"]
        ctx_s = max(0, s - 60)
        ctx_e = min(len(text), e + 60)
        ctx = text[ctx_s:ctx_e].replace("\n", "\\n")
        span_text = text[s:e]
        print(f"  [{i+1:2d}] '{span_text}' (concept {r['concept_id']})")
        print(f"       ...{ctx}...")

    # ------------------------------------------------------------------
    # Spot-check: show 20 random procedure annotations being KEPT
    # ------------------------------------------------------------------
    print(f"\n--- SPOT CHECK: 20 random procedure annotations being KEPT ---")
    kept_proc_sample = kept_procs.sample(min(20, len(kept_procs)), random_state=42)
    for i, (_, r) in enumerate(kept_proc_sample.iterrows()):
        text = note_texts[r["note_id"]]
        s, e = r["start"], r["end"]
        ctx_s = max(0, s - 60)
        ctx_e = min(len(text), e + 60)
        ctx = text[ctx_s:ctx_e].replace("\n", "\\n")
        span_text = text[s:e]
        print(f"  [{i+1:2d}] '{span_text}' (concept {r['concept_id']})")
        print(f"       ...{ctx}...")

    # ------------------------------------------------------------------
    # Impact on class balance
    # ------------------------------------------------------------------
    print(f"\n--- CLASS BALANCE AFTER FILTERING ---")
    for cls in sorted(annot_df["cls"].unique()):
        before = len(annot_df[annot_df["cls"] == cls])
        after = len(kept[kept["cls"] == cls])
        print(f"  {cls:45s}  {before:5d} -> {after:5d} ({100*after/before:.1f}% retained)")

print("\nDone.")
