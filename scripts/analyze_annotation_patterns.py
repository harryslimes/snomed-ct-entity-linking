#!/usr/bin/env python3
"""Analyze annotation patterns from the inline-annotated note."""
import re
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load the original note for section identification
notes_df = pd.read_csv(REPO_ROOT / "data" / "old-challenge-split" / "train_notes.csv")
note_text = notes_df.iloc[0]["text"]

# Define sections with regex patterns
SECTION_PATTERNS = [
    ("Allergies", r"Allergies:.*?(?=Attending:|$)", re.DOTALL),
    ("Chief Complaint", r"Chief Complaint:.*?(?=Major Surgical|$)", re.DOTALL),
    ("Major Surgical/Invasive Procedure", r"Major Surgical.*?:.*?(?=History of Present|$)", re.DOTALL),
    ("History of Present Illness", r"History of Present Illness:.*?(?=Past Medical|$)", re.DOTALL),
    ("Past Medical History", r"Past Medical History:.*?(?=Social History|$)", re.DOTALL),
    ("Social History", r"Social History:.*?(?=Family History|$)", re.DOTALL),
    ("Family History", r"Family History:.*?(?=Physical Exam|$)", re.DOTALL),
    ("Physical Exam", r"Physical Exam:.*?(?=Pertinent Results|$)", re.DOTALL),
    ("Pertinent Results", r"Pertinent Results:.*?(?=Brief Hospital|$)", re.DOTALL),
    ("Brief Hospital Course", r"Brief Hospital Course:.*?(?=Medications on|$)", re.DOTALL),
    ("Medications on Admission", r"Medications on Admission:.*?(?=Discharge Medications|$)", re.DOTALL),
    ("Discharge Medications", r"Discharge Medications:.*?(?=Discharge Disposition|$)", re.DOTALL),
    ("Discharge Disposition", r"Discharge Disposition:.*?(?=Discharge Diagnosis|$)", re.DOTALL),
    ("Discharge Diagnosis", r"Discharge Diagnosis:.*?(?=Discharge Condition|$)", re.DOTALL),
    ("Discharge Condition", r"Discharge Condition:.*?(?=Discharge Instructions|$)", re.DOTALL),
    ("Discharge Instructions", r"Discharge Instructions:.*?(?=Followup Instructions|$)", re.DOTALL),
    ("Followup Instructions", r"Followup Instructions:.*$", re.DOTALL),
]

def find_section(char_offset, note_text):
    """Find which section a character offset belongs to."""
    for section_name, pattern, flags in SECTION_PATTERNS:
        match = re.search(pattern, note_text, flags)
        if match and match.start() <= char_offset < match.end():
            return section_name
    return "Unknown"

# Read inline-annotated note
with open(REPO_ROOT / "inline_annotated_note.txt", "r") as f:
    annotated_text = f.read()

# Parse annotations from the inline format
annotation_pattern = r'\[([^\|]+?) \| ([^\]]+?)\]\{id=(\d+)\}'
annotations = []

# Find all annotations and their positions
for match in re.finditer(annotation_pattern, annotated_text):
    span_text = match.group(1)
    concept_name = match.group(2)
    ann_id = match.group(3)

    # Find the original position in the un-annotated text
    # This is approximate - we'll need the original annotations CSV for exact positions
    annotations.append({
        'annotation_id': ann_id,
        'span_text': span_text,
        'concept_name': concept_name,
        'match_start': match.start(),
        'match_end': match.end()
    })

# Load original annotations to get exact positions
ann_df = pd.read_csv(REPO_ROOT / "data" / "old-challenge-split" / "train_annotations.csv")
first_note_anns = ann_df[ann_df['note_id'] == '10060142-DS-9'].copy()

# Create detailed analysis dataframe
analysis_data = []

for _, ann in first_note_anns.iterrows():
    ann_id = str(ann['annotation_id'])
    start = int(ann['start'])
    end = int(ann['end'])
    span_text = note_text[start:end]

    # Find section
    section = find_section(start, note_text)

    # Get concept info from inline annotations
    concept_info = next((a for a in annotations if a['annotation_id'] == ann_id), None)
    if concept_info:
        concept_name = concept_info['concept_name']
    else:
        concept_name = "Unknown"

    # Extract hierarchy from concept name
    hierarchy_match = re.search(r'\((.*?)\)$', concept_name)
    hierarchy = hierarchy_match.group(1) if hierarchy_match else "unknown"

    # Determine pattern categories
    is_abbreviation = len(span_text) <= 4 and span_text.isupper()
    is_negated = any(neg in note_text[max(0, start-50):start].lower() for neg in ['no ', 'without', 'afebrile', 'denies'])
    is_multi_word = ' ' in span_text.strip()

    # Get context
    context_start = max(0, start - 100)
    context_end = min(len(note_text), end + 50)
    context = note_text[context_start:start] + "[" + span_text + "]" + note_text[end:context_end]
    context = context.replace('\n', ' ').strip()

    # Analyze span-to-concept mapping pattern
    span_lower = span_text.lower()
    concept_lower = concept_name.lower()

    if span_lower in concept_lower:
        mapping_type = "Direct"
    elif any(word in concept_lower for word in span_lower.split()):
        mapping_type = "Partial"
    elif is_abbreviation:
        mapping_type = "Abbreviation Expansion"
    elif is_negated:
        mapping_type = "Negation Handling"
    else:
        mapping_type = "Semantic/Implicit"

    analysis_data.append({
        'annotation_id': ann_id,
        'section': section,
        'span_text': span_text,
        'concept_name': concept_name.replace(f' ({hierarchy})', ''),
        'hierarchy': hierarchy,
        'span_length': len(span_text),
        'is_abbreviation': is_abbreviation,
        'is_negated': is_negated,
        'is_multi_word': is_multi_word,
        'mapping_type': mapping_type,
        'context': context[:200] + '...' if len(context) > 200 else context
    })

# Create DataFrame
analysis_df = pd.DataFrame(analysis_data)

# Save detailed analysis
analysis_df.to_csv('annotation_patterns_detailed.csv', index=False)

# Create summary statistics
print("=== ANNOTATION PATTERNS ANALYSIS ===\n")

print(f"Total annotations: {len(analysis_df)}\n")

print("Annotations by section:")
section_counts = analysis_df['section'].value_counts()
for section, count in section_counts.items():
    print(f"  {section}: {count}")

print("\nAnnotations by hierarchy type:")
hierarchy_counts = analysis_df['hierarchy'].value_counts()
for hierarchy, count in hierarchy_counts.items():
    print(f"  {hierarchy}: {count}")

print("\nMapping type distribution:")
mapping_counts = analysis_df['mapping_type'].value_counts()
for mapping, count in mapping_counts.items():
    print(f"  {mapping}: {count}")

print("\nAbbreviation annotations:")
abbrev_df = analysis_df[analysis_df['is_abbreviation']]
print(f"  Total: {len(abbrev_df)}")
print("  Examples:")
for _, row in abbrev_df.head(5).iterrows():
    print(f"    {row['span_text']} -> {row['concept_name']}")

print("\nNegated annotations:")
negated_df = analysis_df[analysis_df['is_negated']]
print(f"  Total: {len(negated_df)}")
print("  Examples:")
for _, row in negated_df.head(5).iterrows():
    print(f"    {row['span_text']} -> {row['concept_name']}")

print("\nMulti-word annotations:")
multi_df = analysis_df[analysis_df['is_multi_word']]
print(f"  Total: {len(multi_df)}")
print("  Examples:")
for _, row in multi_df.head(5).iterrows():
    print(f"    {row['span_text']} -> {row['concept_name']}")

# Save summary by section
section_summary = []
for section in section_counts.index:
    section_df = analysis_df[analysis_df['section'] == section]
    section_summary.append({
        'section': section,
        'total_annotations': len(section_df),
        'abbreviations': len(section_df[section_df['is_abbreviation']]),
        'negations': len(section_df[section_df['is_negated']]),
        'multi_word': len(section_df[section_df['is_multi_word']]),
        'top_hierarchies': section_df['hierarchy'].value_counts().head(3).to_dict()
    })

section_summary_df = pd.DataFrame(section_summary)
section_summary_df.to_csv('annotation_patterns_by_section.csv', index=False)

print("\n\nDetailed analysis saved to:")
print("  - annotation_patterns_detailed.csv")
print("  - annotation_patterns_by_section.csv")