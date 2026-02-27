#!/usr/bin/env python3
"""Deep analysis of SNOMED CT annotations to identify subtle patterns and improvements"""

import json
import pandas as pd
from collections import defaultdict

# Load existing rules
with open('/workspaces/snomed-ct-entity-linking/scripts/rules_output.json', 'r') as f:
    existing_rules = json.load(f)

# Load annotations
annotations_df = pd.read_csv('/workspaces/snomed-ct-entity-linking/first_note_annotations.csv')

# Create a detailed analysis of each annotation
detailed_analysis = []

# Key insights to look for:
insights = {
    'single_letter_expansions': [],
    'implicit_mappings': [],
    'multiline_spans': [],
    'abbreviation_gaps': [],
    'section_specific_patterns': [],
    'procedure_modifiers': [],
    'finding_vs_procedure': [],
    'compound_concepts': []
}

# Analyze each annotation
for idx, row in annotations_df.iterrows():
    ann_id = str(row['annotation_id'])
    span = row['span']
    concept_id = row['concept_id']

    # Check current rule mappings
    current_rules = existing_rules['annotation_rule_map'].get(ann_id, [])

    analysis = {
        'id': ann_id,
        'span': span,
        'concept_id': concept_id,
        'current_rules': current_rules,
        'insights': []
    }

    # Check for single letter patterns
    if len(span.strip()) == 1 and span.strip().isalpha():
        insights['single_letter_expansions'].append({
            'id': ann_id,
            'span': span,
            'concept': concept_id,
            'note': 'Single letter abbreviation'
        })
        analysis['insights'].append('single_letter')

    # Check for multiline spans
    if '\n' in span:
        insights['multiline_spans'].append({
            'id': ann_id,
            'span': span.replace('\n', '\\n'),
            'concept': concept_id,
            'note': 'Contains line break'
        })
        analysis['insights'].append('multiline')

    # Check for implicit mappings
    implicit_cases = {
        '23241': ('pancreatic rest', '19387007', 'Maps to "Ectopic pancreas" but means dietary restriction'),
        '23247': ('clinic', '737492002', 'Maps to "Outpatient care management" from just "clinic"'),
        '23269': ('Laparoscopic', '51316009', 'Adjective alone mapped to technique'),
        '23273': ('c', '3415004', 'Single letter "c" to "Cyanosis"'),
        '23274': ('e', '424372002', 'Single letter "e" to "Edema of extremity"'),
        '23287': ('afebrile', '386661006', 'Negation prefix mapped to "Fever" finding')
    }

    if ann_id in implicit_cases:
        expected_span, expected_concept, note = implicit_cases[ann_id]
        if span == expected_span:
            insights['implicit_mappings'].append({
                'id': ann_id,
                'span': span,
                'concept': concept_id,
                'note': note
            })
            analysis['insights'].append('implicit')

    # Check for abbreviations not in current mappings
    if span.isupper() and len(span) <= 6:
        if span not in existing_rules['mappings']:
            insights['abbreviation_gaps'].append({
                'id': ann_id,
                'span': span,
                'concept': concept_id,
                'note': 'Uppercase abbreviation not in mappings'
            })
            analysis['insights'].append('missing_abbreviation')

    # Check for procedure modifiers
    procedure_modifiers = ['Laparoscopic', 'surgical', 'post operative']
    if any(mod in span for mod in procedure_modifiers):
        insights['procedure_modifiers'].append({
            'id': ann_id,
            'span': span,
            'concept': concept_id
        })
        analysis['insights'].append('procedure_modifier')

    detailed_analysis.append(analysis)

# Identify patterns that could benefit from new rules or refinements
new_rule_suggestions = []

# Suggest rule for single-letter physical exam abbreviations
if insights['single_letter_expansions']:
    new_rule_suggestions.append({
        'rule_id': 'R13',
        'concept_type': 'Physical_Exam_Single_Letter',
        'stage_1_span': {
            'label': 'Physical_Exam_Single_Letter',
            'description': 'Single letters in physical exam sections representing clinical findings (c=cyanosis, e=edema)'
        },
        'stage_2_search': {
            'filtering_logic': None,
            'intent_translation': 'Expand single letters to full medical terms based on physical exam context patterns'
        },
        'stage_3_select': {
            'disambiguation_logic': 'Select finding concepts for clinical states or conditions',
            'preferred_hierarchy': 'finding',
            'reject_hierarchies': []
        }
    })

# Suggest rule for procedure technique modifiers
if any('Laparoscopic' in item['span'] for item in insights['procedure_modifiers']):
    new_rule_suggestions.append({
        'rule_id': 'R14',
        'concept_type': 'Procedure_Technique_Modifier',
        'stage_1_span': {
            'label': 'Procedure_Technique_Modifier',
            'description': 'Standalone adjectives describing surgical or procedural techniques (laparoscopic, open, percutaneous)'
        },
        'stage_2_search': {
            'filtering_logic': None,
            'intent_translation': 'Search for technique or approach concepts rather than full procedures'
        },
        'stage_3_select': {
            'disambiguation_logic': 'Select qualifier value or technique concepts when modifier appears alone',
            'preferred_hierarchy': 'qualifier value',
            'reject_hierarchies': ['procedure']
        }
    })

# Suggest new abbreviation mappings
suggested_mappings = {
    'c': 'cyanosis',
    'e': 'edema',
    'pp': 'peripheral pulses',
    'AAO': 'awake, alert and oriented',
    'c/d/i': 'clean dry intact'
}

# Check for complex patterns that need special handling
complex_patterns = []

# Patterns with multiline spans
for item in insights['multiline_spans']:
    complex_patterns.append({
        'pattern': 'multiline_term',
        'example': item,
        'recommendation': 'Normalize line breaks in search terms'
    })

# Create comprehensive report
comprehensive_report = {
    'summary': {
        'total_annotations': len(annotations_df),
        'single_letter_patterns': len(insights['single_letter_expansions']),
        'implicit_mappings': len(insights['implicit_mappings']),
        'multiline_spans': len(insights['multiline_spans']),
        'missing_abbreviations': len(insights['abbreviation_gaps']),
        'procedure_modifiers': len(insights['procedure_modifiers'])
    },
    'new_rule_suggestions': new_rule_suggestions,
    'new_abbreviation_mappings': suggested_mappings,
    'complex_patterns': complex_patterns,
    'specific_insights': insights,
    'refinement_suggestions': [
        {
            'rule': 'G6',
            'current': 'For implicit or non-literal clinical concepts, search for the medically implied meaning',
            'suggested': 'For implicit or non-literal clinical concepts, search for the medically implied meaning. Examples: "pancreatic rest" → dietary restriction, "clinic" → outpatient care, single letters in exam → clinical findings'
        },
        {
            'rule': 'R10',
            'enhancement': 'Add specific examples of implicit mappings to guide implementation'
        }
    ]
}

# Generate updated annotation-rule mappings for new patterns
updated_mappings = {}
for ann in detailed_analysis:
    if 'single_letter' in ann['insights'] and ann['id'] not in ['23273', '23274']:
        # These are already mapped, but we can suggest R13 for future similar cases
        pass
    elif 'procedure_modifier' in ann['insights'] and ann['span'] == 'Laparoscopic':
        # Suggest adding R14 to annotation 23269
        updated_mappings[ann['id']] = ann['current_rules'] + ['R14'] if 'R14' not in ann['current_rules'] else ann['current_rules']

comprehensive_report['suggested_mapping_updates'] = updated_mappings

# Save comprehensive report
with open('/workspaces/snomed-ct-entity-linking/scripts/comprehensive_annotation_analysis.json', 'w') as f:
    json.dump(comprehensive_report, f, indent=2)

print("\n=== Comprehensive Analysis Results ===")
print(f"\nKey Findings:")
print(f"- Single letter abbreviations found: {comprehensive_report['summary']['single_letter_patterns']}")
print(f"- Implicit/non-literal mappings: {comprehensive_report['summary']['implicit_mappings']}")
print(f"- Multiline spans: {comprehensive_report['summary']['multiline_spans']}")
print(f"- Missing abbreviations: {comprehensive_report['summary']['missing_abbreviations']}")
print(f"- Procedure modifiers: {comprehensive_report['summary']['procedure_modifiers']}")
print(f"\nNew rule suggestions: {len(new_rule_suggestions)}")
print(f"New abbreviation mappings: {len(suggested_mappings)}")
print(f"\nReport saved to: comprehensive_annotation_analysis.json")

# Export specific examples for manual review
examples_df = pd.DataFrame([
    {'annotation_id': item['id'], 'span': item['span'], 'concept': item['concept'], 'issue': item['note']}
    for item in insights['implicit_mappings'] + insights['single_letter_expansions']
])
if not examples_df.empty:
    examples_df.to_csv('/workspaces/snomed-ct-entity-linking/scripts/special_cases_to_review.csv', index=False)
    print("Special cases exported to: special_cases_to_review.csv")