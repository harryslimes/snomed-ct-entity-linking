#!/usr/bin/env python3
"""Comprehensive pattern analysis for SNOMED CT entity linking."""
import re
import pandas as pd
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load all necessary data
notes_df = pd.read_csv(REPO_ROOT / "data" / "old-challenge-split" / "train_notes.csv")
note_text = notes_df.iloc[0]["text"]
ann_df = pd.read_csv(REPO_ROOT / "data" / "old-challenge-split" / "train_annotations.csv")
first_note_anns = ann_df[ann_df['note_id'] == '10060142-DS-9'].copy()

# Load terminology for concept info
terminology_df = pd.read_csv(REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv")
concept_lookup = {
    int(row.concept_id): (row.concept_name, row.hierarchy)
    for row in terminology_df.itertuples()
}

# Define sections
SECTION_PATTERNS = [
    ("Allergies", r"Allergies:.*?(?=Attending:|$)", re.DOTALL),
    ("Chief Complaint", r"Chief Complaint:.*?(?=Major Surgical|$)", re.DOTALL),
    ("Major Surgical/Invasive Procedure", r"Major Surgical.*?:.*?(?=History of Present|$)", re.DOTALL),
    ("History of Present Illness", r"History of Present Illness:.*?(?=Past Medical|$)", re.DOTALL),
    ("Past Medical History", r"Past Medical History:.*?(?=Social History|$)", re.DOTALL),
    ("Family History", r"Family History:.*?(?=Physical Exam|$)", re.DOTALL),
    ("Physical Exam", r"Physical Exam:.*?(?=Pertinent Results|$)", re.DOTALL),
    ("Brief Hospital Course", r"Brief Hospital Course:.*?(?=Medications on|$)", re.DOTALL),
    ("Discharge Condition", r"Discharge Condition:.*?(?=Discharge Instructions|$)", re.DOTALL),
    ("Discharge Instructions", r"Discharge Instructions:.*?(?=Followup Instructions|$)", re.DOTALL),
    ("Discharge Diagnosis", r"Discharge Diagnosis:.*?(?=Discharge Condition|$)", re.DOTALL),
]

def find_section(char_offset, note_text):
    """Find which section a character offset belongs to."""
    for section_name, pattern, flags in SECTION_PATTERNS:
        match = re.search(pattern, note_text, flags)
        if match and match.start() <= char_offset < match.end():
            return section_name
    return "Unknown"

# Create comprehensive analysis
comprehensive_patterns = []

for _, ann in first_note_anns.iterrows():
    ann_id = int(ann['annotation_id'])
    start = int(ann['start'])
    end = int(ann['end'])
    concept_id = int(ann['concept_id'])

    span_text = note_text[start:end]
    section = find_section(start, note_text)

    # Get concept info
    concept_name, hierarchy = concept_lookup.get(concept_id, ("Unknown", "unknown"))

    # Get context
    context_start = max(0, start - 300)
    context_end = min(len(note_text), end + 100)
    pre_context = note_text[context_start:start]
    post_context = note_text[end:context_end]
    full_context = pre_context + "[" + span_text + "]" + post_context

    # 1. Span Identification Patterns
    span_patterns = []

    # Check if it's an abbreviation
    if re.match(r'^[A-Z]{1,5}$', span_text):
        span_patterns.append("single_capital_abbreviation")
    elif re.match(r'^[A-Z][a-z]+[A-Z][a-z]+', span_text):
        span_patterns.append("camelcase_abbreviation")

    # Check if multi-word clinical phrase
    if ' ' in span_text:
        if any(term in span_text.lower() for term in ['biliary', 'pancreatic', 'gallstone', 'metastatic']):
            span_patterns.append("compound_clinical_condition")
        elif any(term in span_text.lower() for term in ['increased', 'decreased', 'well controlled']):
            span_patterns.append("modified_clinical_state")

    # Check if it's a standalone medical term
    if span_text.lower() in ['cholecystectomy', 'pancreatitis', 'heparin', 'ambulate']:
        span_patterns.append("standalone_medical_term")

    # Check position in sentence
    if pre_context.strip().endswith(':'):
        span_patterns.append("follows_colon")
    elif pre_context.strip().endswith('.'):
        span_patterns.append("sentence_start")

    # 2. Search Term Generation Strategies
    search_strategies = []

    # Abbreviation expansion needed
    if span_text in ['VS', 'CV', 'GEN', 'ABD', 'PULM', 'EXTR', 'NAD', 'RRR', 'CTAB', 'PACU', 'PCP', 'CVA', 'RA']:
        search_strategies.append("expand_abbreviation")
        if span_text == 'VS':
            search_strategies.append("expand_to: vital signs")
        elif span_text == 'CV':
            search_strategies.append("expand_to: cardiovascular")
        elif span_text == 'NAD':
            search_strategies.append("expand_to: no abnormality detected")

    # Normalize negations
    if 'afebrile' in span_text.lower():
        search_strategies.append("normalize_negation")
        search_strategies.append("search_for: fever")
    elif 'without' in pre_context.lower()[-20:]:
        search_strategies.append("handle_negation_context")

    # Multi-word handling
    if ' ' in span_text and 'pancreatitis' in span_text.lower():
        search_strategies.append("preserve_compound_term")
        search_strategies.append("search_as: gallstone pancreatitis")

    # 3. Selection Logic Patterns
    selection_logic = []

    # Section-specific preferences
    if section == "Physical Exam":
        selection_logic.append("prefer_examination_procedures")
        selection_logic.append(f"expected_hierarchy: procedure")
    elif section == "History of Present Illness":
        selection_logic.append("prefer_disorders_and_findings")
        selection_logic.append(f"expected_hierarchy: disorder or finding")
    elif section == "Discharge Instructions":
        selection_logic.append("prefer_education_procedures")
        selection_logic.append(f"expected_hierarchy: procedure")

    # Hierarchy preferences based on span
    if any(action in span_text.lower() for action in ['drained', 'discussion', 'teaching', 'resection']):
        selection_logic.append("action_word_prefers_procedure")
    elif any(state in span_text.lower() for state in ['stable', 'improved', 'controlled', 'alive']):
        selection_logic.append("state_word_prefers_finding")

    # 4. Section-Specific Rules
    section_rules = []

    if section == "Physical Exam":
        section_rules.append("abbreviations_map_to_examination_procedures")
        section_rules.append("findings_map_to_normal_or_abnormal_states")
    elif section == "Allergies":
        section_rules.append("drug_names_map_to_allergy_findings")
    elif section == "Discharge Instructions":
        section_rules.append("activity_instructions_map_to_education_procedures")
        section_rules.append("medication_instructions_map_to_medication_education")
    elif section == "Brief Hospital Course":
        section_rules.append("clinical_actions_map_to_procedures")
        section_rules.append("patient_states_map_to_findings")

    # 5. Special Patterns
    special_patterns = []

    # Abbreviation handling
    if len(span_text) <= 5 and span_text.isupper():
        special_patterns.append(f"abbreviation: {span_text} -> {concept_name.split(' (')[0]}")

    # Negation handling
    if any(neg in pre_context.lower()[-50:] for neg in ['no ', 'without', 'denies']):
        special_patterns.append(f"negation_context: maps to positive concept")
    elif span_text.lower() == 'afebrile':
        special_patterns.append(f"inherent_negation: afebrile -> Fever")

    # Implicit mappings
    span_lower = span_text.lower()
    concept_lower = concept_name.lower()
    if span_lower not in concept_lower and not any(word in concept_lower for word in span_lower.split()):
        special_patterns.append(f"implicit_mapping: '{span_text}' -> '{concept_name.split(' (')[0]}'")

    # c/c/e pattern
    if span_text in ['c', 'e'] and '/c/' in full_context:
        special_patterns.append("part_of_abbreviation_sequence: c/c/e")

    comprehensive_patterns.append({
        'annotation_id': ann_id,
        'section': section,
        'span_text': span_text,
        'concept_name': concept_name,
        'hierarchy': hierarchy,
        'span_identification_patterns': '; '.join(span_patterns) if span_patterns else 'standard_medical_term',
        'search_term_strategies': '; '.join(search_strategies) if search_strategies else 'use_exact_span',
        'selection_logic': '; '.join(selection_logic) if selection_logic else 'standard_selection',
        'section_specific_rules': '; '.join(section_rules) if section_rules else 'none',
        'special_patterns': '; '.join(special_patterns) if special_patterns else 'none',
        'context_snippet': full_context.replace('\n', ' ').strip()[:200] + '...'
    })

# Create DataFrame
patterns_df = pd.DataFrame(comprehensive_patterns)

# Save detailed analysis
patterns_df.to_csv('comprehensive_pattern_analysis.csv', index=False)

# Generate pattern summary report
print("=== COMPREHENSIVE PATTERN ANALYSIS REPORT ===\n")

print("1. SPAN IDENTIFICATION PATTERNS:")
span_id_counts = {}
for _, row in patterns_df.iterrows():
    patterns = row['span_identification_patterns'].split('; ')
    for p in patterns:
        if p:
            span_id_counts[p] = span_id_counts.get(p, 0) + 1

for pattern, count in sorted(span_id_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {pattern}: {count}")

print("\n2. SEARCH TERM GENERATION STRATEGIES:")
search_counts = {}
for _, row in patterns_df.iterrows():
    strategies = row['search_term_strategies'].split('; ')
    for s in strategies:
        if s and not s.startswith('expand_to:') and not s.startswith('search_'):
            search_counts[s] = search_counts.get(s, 0) + 1

for strategy, count in sorted(search_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {strategy}: {count}")

print("\n3. SELECTION LOGIC PATTERNS:")
selection_counts = {}
for _, row in patterns_df.iterrows():
    logic = row['selection_logic'].split('; ')
    for l in logic:
        if l and not l.startswith('expected_'):
            selection_counts[l] = selection_counts.get(l, 0) + 1

for logic, count in sorted(selection_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {logic}: {count}")

print("\n4. SECTION-SPECIFIC PATTERNS:")
section_pattern_counts = {}
for _, row in patterns_df.iterrows():
    rules = row['section_specific_rules'].split('; ')
    section = row['section']
    for rule in rules:
        if rule and rule != 'none':
            key = f"{section}: {rule}"
            section_pattern_counts[key] = section_pattern_counts.get(key, 0) + 1

for pattern, count in sorted(section_pattern_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {pattern}: {count}")

print("\n5. SPECIAL PATTERNS:")
# Abbreviations
abbrev_df = patterns_df[patterns_df['special_patterns'].str.contains('abbreviation:', na=False)]
print(f"\nAbbreviations ({len(abbrev_df)} total):")
for _, row in abbrev_df.head(10).iterrows():
    pattern = [p for p in row['special_patterns'].split('; ') if 'abbreviation:' in p][0]
    print(f"  {pattern}")

# Negations
negation_df = patterns_df[patterns_df['special_patterns'].str.contains('negation', na=False)]
print(f"\nNegation patterns ({len(negation_df)} total):")
for _, row in negation_df.iterrows():
    patterns = [p for p in row['special_patterns'].split('; ') if 'negation' in p]
    for p in patterns:
        print(f"  {row['span_text']}: {p}")

# Implicit mappings
implicit_df = patterns_df[patterns_df['special_patterns'].str.contains('implicit_mapping:', na=False)]
print(f"\nImplicit mappings ({len(implicit_df)} total examples):")
for _, row in implicit_df.head(10).iterrows():
    pattern = [p for p in row['special_patterns'].split('; ') if 'implicit_mapping:' in p][0]
    print(f"  {pattern}")

# Generate actionable rules JSON
rules = {
    "span_identification_rules": [
        {
            "pattern": "single_capital_abbreviation",
            "description": "Single capital letter abbreviations (VS, CV, GEN, etc.) in Physical Exam section",
            "regex": r"^[A-Z]{1,5}$",
            "sections": ["Physical Exam"]
        },
        {
            "pattern": "compound_clinical_condition",
            "description": "Multi-word clinical conditions that form a single concept",
            "keywords": ["biliary pancreatitis", "gallstone pancreatitis", "pancreatic necrosis"],
            "sections": ["any"]
        }
    ],
    "search_generation_rules": [
        {
            "pattern": "expand_abbreviation",
            "mappings": {
                "VS": "vital signs",
                "CV": "cardiovascular",
                "NAD": "no abnormality detected",
                "RRR": "regular rate rhythm",
                "CTAB": "clear to auscultation bilaterally",
                "PACU": "post anesthesia care unit",
                "PCP": "primary care provider",
                "CVA": "cerebrovascular accident"
            }
        },
        {
            "pattern": "normalize_negation",
            "examples": {
                "afebrile": "fever",
                "voided without problem": "normal micturition"
            }
        }
    ],
    "selection_rules": [
        {
            "section": "Physical Exam",
            "prefer_hierarchy": "procedure",
            "rationale": "Physical exam abbreviations map to examination procedures"
        },
        {
            "section": "Discharge Instructions",
            "prefer_hierarchy": "procedure",
            "rationale": "Instructions map to education/training procedures"
        },
        {
            "pattern": "action_words",
            "keywords": ["drained", "discussion", "teaching", "resection"],
            "prefer_hierarchy": "procedure"
        },
        {
            "pattern": "state_words",
            "keywords": ["stable", "improved", "controlled", "alive"],
            "prefer_hierarchy": "finding"
        }
    ]
}

with open('actionable_snomed_rules.json', 'w') as f:
    json.dump(rules, f, indent=2)

print("\n\nFiles generated:")
print("  - comprehensive_pattern_analysis.csv (detailed per-annotation analysis)")
print("  - actionable_snomed_rules.json (structured rules for implementation)")
print(f"\nTotal annotations analyzed: {len(patterns_df)}")