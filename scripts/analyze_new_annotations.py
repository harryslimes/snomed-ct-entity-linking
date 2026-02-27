#!/usr/bin/env python3
"""Analyze new SNOMED CT annotations to identify patterns and rule extensions"""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Load existing rules
with open('/workspaces/snomed-ct-entity-linking/scripts/rules_output.json', 'r') as f:
    existing_rules = json.load(f)

# Load annotations
annotations_df = pd.read_csv('/workspaces/snomed-ct-entity-linking/first_note_annotations.csv')

# Load inline annotated note to get context
with open('/workspaces/snomed-ct-entity-linking/inline_annotated_note.txt', 'r') as f:
    note_text = f.read()

# Extract section headers and their positions
sections = {
    'Allergies': (158, 169),
    'Chief Complaint': (212, 227),
    'Major Surgical or Invasive Procedure': (253, 291),
    'History of Present Illness': (352, 379),
    'Past Medical History': (980, 1000),
    'Social History': (1072, 1086),
    'Family History': (1090, 1104),
    'Physical Exam': (1238, 1251),
    'Pertinent Results': (1464, 1481),
    'Brief Hospital Course': (1509, 1530),
    'Medications on Admission': (2265, 2289),
    'Discharge Medications': (2324, 2345),
    'Discharge Disposition': (2471, 2492),
    'Discharge Diagnosis': (2503, 2522),
    'Discharge Condition': (2543, 2562),
    'Discharge Instructions': (2675, 2697),
    'Followup Instructions': (4079, 4099),
}

def get_section(start_pos):
    """Determine which section a span belongs to"""
    for section, (sec_start, sec_end) in sections.items():
        if start_pos >= sec_start:
            next_sections = [s for s, (s_start, _) in sections.items() if s_start > sec_start]
            if next_sections:
                next_start = min([sections[s][0] for s in next_sections])
                if start_pos < next_start:
                    return section
            else:
                return section
    return 'Header'

# Analyze each annotation
analysis_results = []
uncovered_patterns = []
rule_refinements = []
new_abbreviations = {}

# Helper function to check if a pattern is covered by existing rules
def is_covered_by_existing_rules(annotation_id, existing_rules):
    return str(annotation_id) in existing_rules['annotation_rule_map']

# Analyze each annotation
for idx, row in annotations_df.iterrows():
    annotation = {
        'id': row['annotation_id'],
        'span': row['span'],
        'concept_id': row['concept_id'],
        'start': int(row['start']),
        'end': int(row['end']),
        'section': get_section(int(row['start']))
    }

    # Check if this annotation has existing rule mapping
    if not is_covered_by_existing_rules(annotation['id'], existing_rules):
        uncovered_patterns.append(annotation)

    analysis_results.append(annotation)

# Identify new patterns
pattern_categories = {
    'medication_administration': [],
    'temporal_qualifiers': [],
    'anatomical_laterality': [],
    'clinical_context_modifiers': [],
    'compound_medical_terms': [],
    'measurement_units': [],
    'procedure_modifiers': [],
    'symptom_severity': [],
    'care_transitions': [],
    'clinical_relationships': []
}

# Analyze uncovered patterns
for pattern in uncovered_patterns:
    span_text = pattern['span'].lower()

    # Check for medication administration patterns
    if any(term in span_text for term in ['mg', 'ml', 'po', 'iv', 'prn', 'bid', 'tid', 'qid']):
        pattern_categories['medication_administration'].append(pattern)

    # Check for temporal qualifiers
    elif any(term in span_text for term in ['prior', 'after', 'before', 'during', 'post', 'pre']):
        pattern_categories['temporal_qualifiers'].append(pattern)

    # Check for anatomical laterality
    elif any(term in span_text for term in ['left', 'right', 'bilateral', 'unilateral']):
        pattern_categories['anatomical_laterality'].append(pattern)

    # Check for clinical context modifiers
    elif any(term in span_text for term in ['severe', 'mild', 'moderate', 'acute', 'chronic']):
        pattern_categories['clinical_context_modifiers'].append(pattern)

    # Check for compound medical terms with line breaks
    elif '\n' in span_text:
        pattern_categories['compound_medical_terms'].append(pattern)

# Identify new abbreviations not in existing mappings
for annotation in analysis_results:
    span = annotation['span'].strip()
    # Check if it's an abbreviation (all caps, short)
    if span.isupper() and len(span) <= 4 and span not in existing_rules['mappings']:
        # Try to find the expansion from context or known patterns
        if span == 'c' and annotation['section'] == 'Physical Exam':
            new_abbreviations['c'] = 'cyanosis'
        elif span == 'e' and annotation['section'] == 'Physical Exam':
            new_abbreviations['e'] = 'edema'

# Identify rules that need refinement based on new evidence
refinement_candidates = {
    'R10': [],  # Implicit Clinical Concept
    'R3': [],   # Clinical Procedure
    'R6': [],   # Diagnostic Finding
}

for annotation in analysis_results:
    # Check for implicit concepts that need better handling
    if annotation['span'] == 'Laparoscopic' and annotation['concept_id'] == '51316009':
        refinement_candidates['R10'].append({
            'annotation': annotation,
            'issue': 'Single adjective mapped to procedure concept',
            'recommendation': 'Add handling for procedure modifiers/adjectives'
        })

    if annotation['span'] == 'pancreatic rest' and annotation['concept_id'] == '19387007':
        refinement_candidates['R10'].append({
            'annotation': annotation,
            'issue': 'Clinical intent differs from literal mapping',
            'recommendation': 'Clarify that "pancreatic rest" means therapeutic rest, not ectopic pancreas'
        })

# Create structured analysis report
report = {
    'total_annotations': len(annotations_df),
    'covered_annotations': len(annotations_df) - len(uncovered_patterns),
    'uncovered_annotations': len(uncovered_patterns),
    'new_patterns_by_category': {k: len(v) for k, v in pattern_categories.items() if v},
    'new_abbreviations': new_abbreviations,
    'rules_needing_refinement': {k: len(v) for k, v in refinement_candidates.items() if v},
    'specific_recommendations': []
}

# Add specific recommendations
recommendations = [
    {
        'type': 'new_rule',
        'rule_id': 'R13',
        'concept_type': 'Procedural_Modifier',
        'description': 'Adjectives or modifiers that describe characteristics of procedures (e.g., "laparoscopic", "open", "minimally invasive")',
        'stage_2_search': 'Search for the modifier term itself or related technique/approach concepts',
        'stage_3_select': 'Select qualifier value or technique concepts over procedure concepts'
    },
    {
        'type': 'new_rule',
        'rule_id': 'R14',
        'concept_type': 'Clinical_State_Modifier',
        'description': 'Single letters or abbreviations in physical exam representing clinical states (e.g., "c" for cyanosis, "e" for edema)',
        'stage_2_search': 'Expand single letters to their medical meaning based on exam context',
        'stage_3_select': 'Select finding concepts for clinical states'
    },
    {
        'type': 'mapping_addition',
        'mappings': {
            'c': 'cyanosis',
            'e': 'edema',
            'pp': 'peripheral pulses present',
            'AAO': 'awake, alert, and oriented',
            'c/d/i': 'clean, dry, intact'
        }
    },
    {
        'type': 'rule_refinement',
        'rule_id': 'R10',
        'change': 'Add explicit examples: "pancreatic rest" -> dietary restriction for pancreatic recovery, not ectopic pancreas'
    },
    {
        'type': 'new_rule',
        'rule_id': 'R15',
        'concept_type': 'Multiline_Clinical_Term',
        'description': 'Clinical terms split across line breaks that should be treated as single concepts',
        'stage_2_search': 'Remove line breaks and search for the complete term',
        'stage_3_select': 'Apply standard hierarchy preferences based on the complete term context'
    }
]

report['specific_recommendations'] = recommendations

# Save the analysis report
with open('/workspaces/snomed-ct-entity-linking/scripts/new_annotation_analysis.json', 'w') as f:
    json.dump(report, f, indent=2)

# Print summary
print(f"Analysis Complete:")
print(f"- Total annotations analyzed: {report['total_annotations']}")
print(f"- Currently covered: {report['covered_annotations']}")
print(f"- Uncovered patterns: {report['uncovered_annotations']}")
print(f"- New pattern categories identified: {len([k for k, v in report['new_patterns_by_category'].items() if v > 0])}")
print(f"- New abbreviations found: {len(report['new_abbreviations'])}")
print(f"- Rules needing refinement: {sum(report['rules_needing_refinement'].values())}")
print(f"\nDetailed report saved to: new_annotation_analysis.json")

# Export uncovered patterns for manual review
uncovered_df = pd.DataFrame(uncovered_patterns)
if not uncovered_df.empty:
    uncovered_df.to_csv('/workspaces/snomed-ct-entity-linking/scripts/uncovered_patterns.csv', index=False)
    print(f"Uncovered patterns exported to: uncovered_patterns.csv")