#!/usr/bin/env python3
"""
Analyze the newline character issue in super dictionary matching.

The issue: When clinical text contains newlines within a medical term,
the dictionary matching fails because:
1. Dictionary terms are normalized (newlines → spaces)
2. The regex pattern expects exact spacing
3. A term like "biliary\npancreatitis" won't match the pattern "biliary pancreatitis"
"""

import csv
import re
from collections import defaultdict
from pathlib import Path


def load_notes(notes_csv: Path) -> dict[str, str]:
    """Load notes into a dict mapping note_id -> text."""
    notes = {}
    with notes_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            note_id = row["note_id"]
            text = row["text"]
            notes[note_id] = text
    return notes


def load_annotations(annotations_csv: Path):
    """Load annotations from CSV."""
    annotations = []
    with annotations_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            annotations.append({
                "annotation_id": row["annotation_id"],
                "note_id": row["note_id"],
                "start": int(float(row["start"])),
                "end": int(float(row["end"])),
                "span": row["span"],
                "concept_id": row["concept_id"],
            })
    return annotations


def normalize_term(term: str) -> str:
    """Normalize whitespace like the super dictionary does."""
    return " ".join(term.strip().split())


def main():
    data_dir = Path("data/old-challenge-split")

    # Analyze both train and test
    for split in ["train", "test"]:
        print(f"\n{'='*80}")
        print(f"Analyzing {split.upper()} split")
        print(f"{'='*80}\n")

        notes_csv = data_dir / f"{split}_notes.csv"
        annotations_csv = data_dir / f"{split}_annotations.csv"

        notes = load_notes(notes_csv)
        annotations = load_annotations(annotations_csv)

        print(f"Loaded {len(notes)} notes and {len(annotations)} annotations\n")

        # Track various issues
        mismatches = []
        newline_in_gold_span = []
        newline_in_text_span = []
        both_have_newlines = []
        normalization_helps = []

        for ann in annotations:
            note_id = ann["note_id"]
            start = ann["start"]
            end = ann["end"]
            gold_span = ann["span"]
            concept_id = ann["concept_id"]

            if note_id not in notes:
                continue

            text = notes[note_id]
            actual_text_span = text[start:end]

            # Check if spans match exactly
            if actual_text_span != gold_span:
                mismatches.append({
                    "annotation_id": ann["annotation_id"],
                    "note_id": note_id,
                    "gold_span": gold_span,
                    "actual_span": actual_text_span,
                    "concept_id": concept_id,
                })

            # Check for newlines in gold span
            if "\n" in gold_span or "\r" in gold_span:
                newline_in_gold_span.append(ann)

            # Check for newlines in actual text span
            if "\n" in actual_text_span or "\r" in actual_text_span:
                newline_in_text_span.append(ann)

            # Check if both have newlines
            if ("\n" in gold_span or "\r" in gold_span) and ("\n" in actual_text_span or "\r" in actual_text_span):
                both_have_newlines.append(ann)

            # Check if normalization would help matching
            normalized_gold = normalize_term(gold_span)
            normalized_actual = normalize_term(actual_text_span)

            if actual_text_span != gold_span and normalized_actual == normalized_gold:
                normalization_helps.append({
                    "annotation_id": ann["annotation_id"],
                    "note_id": note_id,
                    "gold_span": repr(gold_span),
                    "actual_span": repr(actual_text_span),
                    "normalized": normalized_gold,
                    "concept_id": concept_id,
                })

        # Report findings
        print(f"Total annotations: {len(annotations)}")
        print(f"Annotations where gold_span != actual text at [start:end]: {len(mismatches)}")
        print(f"Annotations with newlines in gold span: {len(newline_in_gold_span)}")
        print(f"Annotations with newlines in actual text span: {len(newline_in_text_span)}")
        print(f"Annotations with newlines in BOTH: {len(both_have_newlines)}")
        print(f"Cases where normalization fixes the mismatch: {len(normalization_helps)}")

        # Show examples of normalization helping
        if normalization_helps:
            print(f"\n{'─'*80}")
            print("Examples where normalizing whitespace would fix the match:")
            print(f"{'─'*80}")
            for i, case in enumerate(normalization_helps[:10], 1):
                print(f"\n{i}. Annotation ID: {case['annotation_id']}")
                print(f"   Concept ID: {case['concept_id']}")
                print(f"   Gold span:   {case['gold_span']}")
                print(f"   Actual span: {case['actual_span']}")
                print(f"   Normalized:  {case['normalized']!r}")

        # Now check: for terms with newlines, would dictionary matching fail?
        print(f"\n{'─'*80}")
        print("DICTIONARY MATCHING IMPACT:")
        print(f"{'─'*80}\n")

        # For each annotation with newlines in the actual text
        would_fail_to_match = []
        for ann in newline_in_text_span:
            note_id = ann["note_id"]
            start = ann["start"]
            end = ann["end"]
            text = notes[note_id]
            actual_span = text[start:end]

            # The dictionary would have the normalized term
            dict_term = normalize_term(ann["span"])

            # Try to match it using regex like the super dictionary does
            pattern = re.compile(rf"\b{re.escape(dict_term)}\b", re.IGNORECASE)
            match = pattern.search(actual_span)

            # Also try matching in a small window around the annotation
            window_start = max(0, start - 10)
            window_end = min(len(text), end + 10)
            window_text = text[window_start:window_end]
            window_match = pattern.search(window_text)

            if not match:
                would_fail_to_match.append({
                    "annotation_id": ann["annotation_id"],
                    "note_id": note_id,
                    "span_in_text": repr(actual_span),
                    "dict_term": dict_term,
                    "concept_id": ann["concept_id"],
                    "window_has_match": window_match is not None,
                })

        print(f"Annotations with newlines where dictionary matching would FAIL: {len(would_fail_to_match)}")
        print(f"  (out of {len(newline_in_text_span)} annotations with newlines)\n")

        if would_fail_to_match:
            print("Examples of failed matches:")
            for i, case in enumerate(would_fail_to_match[:10], 1):
                print(f"\n{i}. Annotation ID: {case['annotation_id']}")
                print(f"   Concept ID: {case['concept_id']}")
                print(f"   Text span:      {case['span_in_text']}")
                print(f"   Dict term:      {case['dict_term']!r}")
                print(f"   Window match:   {case['window_has_match']}")

        # Calculate recall impact
        if len(annotations) > 0:
            recall_loss = len(would_fail_to_match) / len(annotations) * 100
            print(f"\n{'─'*80}")
            print(f"RECALL IMPACT: {recall_loss:.2f}% of annotations would be missed due to newlines")
            print(f"  ({len(would_fail_to_match)} out of {len(annotations)} annotations)")
            print(f"{'─'*80}")


if __name__ == "__main__":
    main()
