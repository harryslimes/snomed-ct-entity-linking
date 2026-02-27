#!/usr/bin/env python3
"""Enhanced pattern analysis with SNOMED subsumption for rule generation."""

import csv
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# Add paths
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from engine import segment_sections, SectionSpan
from snomed_subsumption import SubsumptionIndex

TERMINOLOGY_CSV = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"


def load_concept_info() -> Dict[int, Tuple[str, str]]:
    """Load concept info mapping concept_id -> (concept_name, hierarchy)."""
    concept_info = {}
    with open(TERMINOLOGY_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            concept_info[int(row['concept_id'])] = (row['concept_name'], row['hierarchy'])
    return concept_info


def load_note_and_annotations() -> Tuple[str, List[Dict], List[SectionSpan]]:
    """Load the note text and annotations."""
    # Load note text
    with open(REPO_ROOT / "first_note.txt", 'r') as f:
        note_text = f.read()

    # Load annotations
    annotations = []
    with open(REPO_ROOT / "first_note_annotations.csv", 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            annotations.append({
                'annotation_id': int(row['annotation_id']),
                'note_id': row['note_id'],
                'start': int(float(row['start'])),
                'end': int(float(row['end'])),
                'span': row['span'],
                'concept_id': int(row['concept_id']),
                'annotation_type': row['annotation_type']
            })

    # Detect sections
    sections = segment_sections(note_text)

    return note_text, annotations, sections


def get_section_for_annotation(annotation: Dict, sections: List[SectionSpan]) -> str:
    """Determine which section an annotation belongs to."""
    start = annotation['start']
    for section in sections:
        if section.start <= start < section.end:
            return section.header
    return "Unknown"


def analyze_section_specific_patterns(note_text: str, annotations: List[Dict],
                                     sections: List[SectionSpan], concept_info: Dict) -> Dict:
    """Analyze section-specific interpretation patterns."""
    section_analysis = {}

    # Group annotations by section
    by_section = defaultdict(list)
    for ann in annotations:
        section = get_section_for_annotation(ann, sections)
        by_section[section].append(ann)

    # Analyze each section
    for section_name, section_anns in by_section.items():
        section_info = {
            'annotation_count': len(section_anns),
            'hierarchy_distribution': Counter(),
            'patterns': []
        }

        for ann in section_anns:
            concept_id = ann['concept_id']
            concept_name, hierarchy = concept_info.get(concept_id, ("Unknown", "unknown"))
            section_info['hierarchy_distribution'][hierarchy] += 1

            # Check for specific patterns
            pattern_info = {
                'span': ann['span'],
                'concept_name': concept_name,
                'hierarchy': hierarchy,
                'concept_id': concept_id,
                'annotation_id': ann['annotation_id']
            }

            # Analyze Allergies section specifically
            if section_name == "Allergies":
                if 'allergy' in concept_name.lower():
                    # This is a medication in allergy section -> should interpret as allergy
                    pattern_info['interpretation_pattern'] = 'medication_as_allergy'
                    pattern_info['original_medication'] = ann['span']
                    pattern_info['allergy_concept'] = concept_name

            # Analyze Physical Exam abbreviations
            elif section_name == "Physical Exam":
                if len(ann['span']) <= 5 and ann['span'].isupper():
                    pattern_info['interpretation_pattern'] = 'physical_exam_abbreviation'

            # Analyze medication sections
            elif 'Medication' in section_name:
                pattern_info['interpretation_pattern'] = 'medication_context'

            section_info['patterns'].append(pattern_info)

        section_analysis[section_name] = section_info

    return section_analysis


def analyze_with_subsumption(annotations: List[Dict], concept_info: Dict) -> Dict:
    """Analyze patterns using SNOMED subsumption relationships."""
    print("Loading SNOMED subsumption index...")
    try:
        subsumption_idx = SubsumptionIndex.load()
    except Exception as e:
        print(f"Could not load subsumption index: {e}")
        return {}

    # Analyze hierarchy families and common ancestors
    hierarchy_families = defaultdict(list)
    concept_ancestors = {}

    for ann in annotations:
        concept_id = ann['concept_id']
        concept_name, hierarchy = concept_info.get(concept_id, ("Unknown", "unknown"))

        hierarchy_families[hierarchy].append({
            'concept_id': concept_id,
            'concept_name': concept_name,
            'span': ann['span'],
            'annotation_id': ann['annotation_id']
        })

        # Get ancestors for this concept
        try:
            parents = subsumption_idx.get_parents(concept_id)
            concept_ancestors[concept_id] = {
                'immediate_parents': list(parents),
                'concept_name': concept_name,
                'hierarchy': hierarchy
            }
        except:
            concept_ancestors[concept_id] = {
                'immediate_parents': [],
                'concept_name': concept_name,
                'hierarchy': hierarchy
            }

    # Find common ancestors for each hierarchy
    hierarchy_ancestor_analysis = {}
    for hierarchy, concepts in hierarchy_families.items():
        concept_ids = [c['concept_id'] for c in concepts]

        if len(concept_ids) > 1:
            try:
                # Find common ancestors
                lca_options = subsumption_idx.find_lca_with_depth(concept_ids, min_depth=1)
                hierarchy_ancestor_analysis[hierarchy] = {
                    'concept_count': len(concept_ids),
                    'sample_concepts': concepts[:5],
                    'common_ancestors': lca_options[:5] if lca_options else [],
                    'recommended_ancestor_id': lca_options[0][0] if lca_options else None
                }
            except:
                hierarchy_ancestor_analysis[hierarchy] = {
                    'concept_count': len(concept_ids),
                    'sample_concepts': concepts[:5],
                    'common_ancestors': [],
                    'recommended_ancestor_id': None
                }
        else:
            # Single concept - use its immediate parents
            concept_id = concept_ids[0]
            parents = concept_ancestors.get(concept_id, {}).get('immediate_parents', [])
            hierarchy_ancestor_analysis[hierarchy] = {
                'concept_count': 1,
                'sample_concepts': concepts,
                'common_ancestors': [],
                'recommended_ancestor_id': parents[0] if parents else concept_id
            }

    return {
        'hierarchy_families': dict(hierarchy_families),
        'concept_ancestors': concept_ancestors,
        'hierarchy_ancestor_analysis': hierarchy_ancestor_analysis
    }


def generate_structured_rules(section_analysis: Dict, subsumption_analysis: Dict,
                            abbreviations: Dict, implicit_patterns: List[Dict],
                            negation_patterns: List[Dict]) -> Dict:
    """Generate structured rules based on the analysis."""

    mappings = {}
    g_rules = []
    structured_rules = []
    rule_counter = 1

    # Generate mappings for abbreviations
    for abbrev, info in abbreviations.items():
        concept_name = info['concept_name']
        if 'allergy' in concept_name.lower():
            # For allergy concepts, map to search intent
            mappings[abbrev] = f"allergy to {abbrev.lower()}"
        else:
            # Standard abbreviation expansion
            mappings[abbrev] = generate_search_term(concept_name)

    # Add key clinical abbreviations that are deterministic
    standard_abbreviations = {
        'CVA': 'cerebrovascular accident',
        'VS': 'vital signs',
        'NAD': 'no abnormality detected',
        'RRR': 'regular rate and rhythm',
        'CTAB': 'clear to auscultation bilaterally',
        'PCA': 'patient controlled analgesia',
        'PACU': 'post anesthesia care unit',
        'PCP': 'primary care provider',
        'IV': 'intravenous'
    }
    mappings.update(standard_abbreviations)

    # Generate G-rules (universal rules)
    g_rules = [
        {
            "id": "G1",
            "rule": "For abbreviations in clinical notes, expand to their standard medical meaning. Use context to disambiguate when abbreviations have multiple meanings.",
            "stages": ["stage_2_search"]
        },
        {
            "id": "G2",
            "rule": "When a span appears in a negated context, search for the positive form of the concept. Do not search for 'absence of' or 'no' variants.",
            "stages": ["stage_2_search", "stage_3_select"]
        },
        {
            "id": "G3",
            "rule": "For procedure spans, distinguish between the procedure being performed versus documentation/discussion of the procedure based on surrounding context.",
            "stages": ["stage_3_select"]
        },
        {
            "id": "G4",
            "rule": "Prefer more specific SNOMED concepts over general ones when multiple candidates match the clinical context.",
            "stages": ["stage_3_select"]
        }
    ]

    # Generate structured rules based on hierarchy analysis
    hierarchy_rules = generate_hierarchy_rules(subsumption_analysis)
    structured_rules.extend(hierarchy_rules)

    # Generate section-specific rules
    section_rules = generate_section_rules(section_analysis, subsumption_analysis)
    structured_rules.extend(section_rules)

    # Generate implicit pattern rules
    implicit_rules = generate_implicit_rules(implicit_patterns, subsumption_analysis)
    structured_rules.extend(implicit_rules)

    return {
        'version': '4.0',
        'mappings': mappings,
        'g_rules': g_rules,
        'structured_rules': structured_rules
    }


def generate_search_term(concept_name: str) -> str:
    """Generate a search-friendly term from a concept name."""
    # Remove parenthetical hierarchy info
    clean_name = re.sub(r'\s*\([^)]+\)$', '', concept_name)
    return clean_name.lower()


def generate_hierarchy_rules(subsumption_analysis: Dict) -> List[Dict]:
    """Generate rules based on hierarchy patterns."""
    rules = []
    rule_id = 1

    hierarchy_analysis = subsumption_analysis.get('hierarchy_ancestor_analysis', {})

    # Finding rule
    if 'finding' in hierarchy_analysis:
        finding_info = hierarchy_analysis['finding']
        ancestor_id = finding_info.get('recommended_ancestor_id', 404684003)  # Clinical finding

        rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Clinical_Finding",
            "applies_to": {
                "ancestor_concept_ids": [ancestor_id] if ancestor_id else [404684003],
                "sections": None,
                "span_pattern": None
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Search for the clinical observation, symptom, or finding described. Use medical terminology for the observation."
            },
            "stage_3_select": {
                "disambiguation_logic": "Select finding concepts for observations and symptoms. Prefer specific findings over general findings when context supports specificity.",
                "preferred_hierarchy": "finding",
                "reject_hierarchies": ["situation", "qualifier value"]
            }
        })
        rule_id += 1

    # Procedure rule
    if 'procedure' in hierarchy_analysis:
        procedure_info = hierarchy_analysis['procedure']
        ancestor_id = procedure_info.get('recommended_ancestor_id', 71388002)  # Procedure

        rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Clinical_Procedure",
            "applies_to": {
                "ancestor_concept_ids": [ancestor_id] if ancestor_id else [71388002],
                "sections": None,
                "span_pattern": None
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Search for the procedure, intervention, or action being performed. Include both the procedure name and any qualifiers."
            },
            "stage_3_select": {
                "disambiguation_logic": "Select procedure concepts for actions and interventions. Consider context to distinguish between procedure performance vs procedure discussion.",
                "preferred_hierarchy": "procedure",
                "reject_hierarchies": ["finding", "situation"]
            }
        })
        rule_id += 1

    # Disorder rule
    if 'disorder' in hierarchy_analysis:
        disorder_info = hierarchy_analysis['disorder']
        ancestor_id = disorder_info.get('recommended_ancestor_id', 64572001)  # Disease

        rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Clinical_Disorder",
            "applies_to": {
                "ancestor_concept_ids": [ancestor_id] if ancestor_id else [64572001],
                "sections": None,
                "span_pattern": None
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Search for the disease, disorder, or pathological condition. Include synonyms and alternative medical terms."
            },
            "stage_3_select": {
                "disambiguation_logic": "Select disorder concepts for diseases and pathological conditions. Prefer specific disorders over general categories.",
                "preferred_hierarchy": "disorder",
                "reject_hierarchies": ["finding"]
            }
        })
        rule_id += 1

    return rules


def generate_section_rules(section_analysis: Dict, subsumption_analysis: Dict) -> List[Dict]:
    """Generate section-specific rules."""
    rules = []
    rule_id = 10  # Start after hierarchy rules

    # Allergy section rule
    if 'Allergies' in section_analysis:
        allergy_patterns = section_analysis['Allergies']['patterns']
        allergy_concepts = [p['concept_id'] for p in allergy_patterns if 'allergy' in p['concept_name'].lower()]

        if allergy_concepts:
            rules.append({
                "rule_id": f"R{rule_id}",
                "concept_type": "Allergy_Finding",
                "applies_to": {
                    "ancestor_concept_ids": [420134006],  # Propensity to adverse reactions
                    "sections": ["Allergies", "Adverse Drug"],
                    "span_pattern": None
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "For medication names in allergy sections, search for 'allergy to [medication name]' or 'allergic reaction to [medication name]'."
                },
                "stage_3_select": {
                    "disambiguation_logic": "In allergy sections, select allergy or adverse reaction concepts rather than the medication substance itself.",
                    "preferred_hierarchy": "finding",
                    "reject_hierarchies": ["substance", "product"]
                }
            })
            rule_id += 1

    # Physical exam abbreviations
    if 'Physical Exam' in section_analysis:
        rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Physical_Exam_Abbreviation",
            "applies_to": {
                "ancestor_concept_ids": [5880005],  # Physical examination procedure
                "sections": ["Physical Exam", "Examination"],
                "span_pattern": "abbreviation"
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Expand physical examination abbreviations to their full medical meaning (e.g., CV -> cardiovascular examination, RRR -> regular rate and rhythm)."
            },
            "stage_3_select": {
                "disambiguation_logic": "For physical exam abbreviations, prefer procedure concepts for examination actions and finding concepts for normal/abnormal results.",
                "preferred_hierarchy": None,
                "reject_hierarchies": ["substance"]
            }
        })
        rule_id += 1

    return rules


def generate_implicit_rules(implicit_patterns: List[Dict], subsumption_analysis: Dict) -> List[Dict]:
    """Generate rules for implicit/semantic concept patterns."""
    rules = []
    rule_id = 20  # Start after section rules

    # Group implicit patterns by semantic relationship
    semantic_groups = defaultdict(list)
    for pattern in implicit_patterns:
        rel_type = pattern.get('semantic_relationship', 'other_semantic')
        semantic_groups[rel_type].append(pattern)

    # Anatomical reference rule
    if 'anatomical_reference' in semantic_groups:
        rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Anatomical_Reference",
            "applies_to": {
                "ancestor_concept_ids": [91723000],  # Anatomical structure
                "sections": None,
                "span_pattern": None
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "For anatomical references, search using the specific body part or structure name. Expand colloquial terms to medical anatomy terms."
            },
            "stage_3_select": {
                "disambiguation_logic": "Select the most specific anatomical structure concept. Prefer body structure hierarchy over morphologic abnormality for normal anatomy references.",
                "preferred_hierarchy": "body structure",
                "reject_hierarchies": ["morphologic abnormality", "finding"]
            }
        })
        rule_id += 1

    # Clinical state rule
    if 'clinical_state' in semantic_groups:
        rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Clinical_State_Description",
            "applies_to": {
                "ancestor_concept_ids": [404684003],  # Clinical finding
                "sections": None,
                "span_pattern": None
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "For clinical state descriptions (improved, stable, controlled), search for the specific clinical finding or assessment being described."
            },
            "stage_3_select": {
                "disambiguation_logic": "Select finding concepts that describe patient status or clinical assessments. Prefer findings over situations.",
                "preferred_hierarchy": "finding",
                "reject_hierarchies": ["situation", "event"]
            }
        })
        rule_id += 1

    return rules


def main():
    print("Starting enhanced pattern analysis...")

    # Load data
    concept_info = load_concept_info()
    note_text, annotations, sections = load_note_and_annotations()

    print(f"Loaded {len(annotations)} annotations across {len(sections)} sections")

    # Analyze section patterns
    print("Analyzing section-specific patterns...")
    section_analysis = analyze_section_specific_patterns(note_text, annotations, sections, concept_info)

    # Analyze with subsumption
    print("Analyzing with SNOMED subsumption...")
    subsumption_analysis = analyze_with_subsumption(annotations, concept_info)

    # Analyze abbreviations (reuse from comprehensive analysis)
    abbreviations = {}
    for ann in annotations:
        span = ann['span'].strip()
        if len(span) <= 6 and span.isupper() and span.isalpha():
            concept_id = ann['concept_id']
            concept_name, hierarchy = concept_info.get(concept_id, ("Unknown", "unknown"))
            abbreviations[span] = {
                'concept_name': concept_name,
                'hierarchy': hierarchy,
                'concept_id': concept_id
            }

    # Simple implicit pattern analysis
    implicit_patterns = []
    negation_patterns = []  # Simplified for this example

    # Generate structured rules
    print("Generating structured rules...")
    rules = generate_structured_rules(
        section_analysis, subsumption_analysis, abbreviations,
        implicit_patterns, negation_patterns
    )

    # Save analysis
    comprehensive_analysis = {
        'section_analysis': section_analysis,
        'subsumption_analysis': subsumption_analysis,
        'abbreviations': abbreviations,
        'generated_rules': rules
    }

    output_path = REPO_ROOT / "enhanced_pattern_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(comprehensive_analysis, f, indent=2, default=str)

    print(f"Saved analysis to {output_path}")

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print(f"Total sections analyzed: {len(section_analysis)}")
    print(f"Generated mappings: {len(rules['mappings'])}")
    print(f"Generated G-rules: {len(rules['g_rules'])}")
    print(f"Generated structured rules: {len(rules['structured_rules'])}")

    print("\n=== SECTION-SPECIFIC PATTERNS ===")
    for section, info in section_analysis.items():
        print(f"{section}: {info['annotation_count']} annotations")
        if info['patterns']:
            special_patterns = [p for p in info['patterns'] if 'interpretation_pattern' in p]
            if special_patterns:
                print(f"  Special patterns: {len(special_patterns)}")
                for pattern in special_patterns[:2]:
                    print(f"    - {pattern['span']} -> {pattern.get('interpretation_pattern')}")

    print(f"\n=== RECOMMENDED MAPPINGS ===")
    for abbrev, expansion in list(rules['mappings'].items())[:10]:
        print(f"{abbrev} -> {expansion}")


if __name__ == "__main__":
    main()