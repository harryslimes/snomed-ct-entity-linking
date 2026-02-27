#!/usr/bin/env python3
"""Comprehensive analysis of the clinical discharge note with 238 SNOMED CT annotations.

This script analyzes the first_note.txt and first_note_annotations.csv to identify:
1. Common abbreviation patterns for mappings
2. Section-specific interpretation patterns
3. Implicit concept patterns
4. Negation and context patterns
5. Hierarchy distribution for applies_to fields

Usage:
    python comprehensive_note_analysis.py
"""

import csv
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# Add super-dictionary to path for section detection
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from engine import segment_sections, SectionSpan

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


def is_abbreviation(text: str) -> bool:
    """Determine if a span is an abbreviation."""
    # All caps, ≤6 characters, no spaces
    return (len(text) <= 6 and
            text.isupper() and
            ' ' not in text and
            text.isalpha())


def is_multi_word(text: str) -> bool:
    """Check if span contains multiple words."""
    return len(text.split()) > 1


def analyze_abbreviations(note_text: str, annotations: List[Dict], concept_info: Dict) -> Dict:
    """Identify abbreviation patterns for mappings."""
    abbreviations = {}
    abbreviation_contexts = defaultdict(list)

    for ann in annotations:
        span = ann['span'].strip()
        concept_id = ann['concept_id']
        concept_name, hierarchy = concept_info.get(concept_id, ("Unknown", "unknown"))

        if is_abbreviation(span):
            # Get context around the abbreviation
            start, end = ann['start'], ann['end']
            context_start = max(0, start - 100)
            context_end = min(len(note_text), end + 100)
            context = note_text[context_start:context_end].replace('\n', ' ')

            abbreviations[span] = {
                'concept_name': concept_name,
                'hierarchy': hierarchy,
                'concept_id': concept_id,
                'count': 1,
                'contexts': [context]
            }
            abbreviation_contexts[span].append(context)

        # Also look for single letters that might be abbreviations
        elif len(span) == 1 and span.isalpha():
            context_start = max(0, ann['start'] - 50)
            context_end = min(len(note_text), ann['end'] + 50)
            context = note_text[context_start:context_end].replace('\n', ' ')

            abbreviations[span] = {
                'concept_name': concept_name,
                'hierarchy': hierarchy,
                'concept_id': concept_id,
                'count': 1,
                'contexts': [context],
                'single_letter': True
            }

    return abbreviations


def analyze_section_patterns(annotations: List[Dict], sections: List[SectionSpan], concept_info: Dict) -> Dict:
    """Identify section-specific interpretation patterns."""
    section_patterns = defaultdict(lambda: defaultdict(list))

    for ann in annotations:
        section = get_section_for_annotation(ann, sections)
        span = ann['span'].strip()
        concept_id = ann['concept_id']
        concept_name, hierarchy = concept_info.get(concept_id, ("Unknown", "unknown"))

        section_patterns[section][hierarchy].append({
            'span': span,
            'concept_name': concept_name,
            'concept_id': concept_id,
            'annotation_id': ann['annotation_id']
        })

    # Look for specific section interpretation patterns
    special_patterns = {}

    # Check for medication names in Allergies section
    if 'Allergies' in section_patterns:
        allergy_medications = []
        for hierarchy, items in section_patterns['Allergies'].items():
            for item in items:
                if 'allergy' in item['concept_name'].lower() or 'allergic' in item['concept_name'].lower():
                    allergy_medications.append(item)
        if allergy_medications:
            special_patterns['allergy_section_medications'] = allergy_medications

    return dict(section_patterns), special_patterns


def analyze_implicit_patterns(note_text: str, annotations: List[Dict], concept_info: Dict) -> List[Dict]:
    """Identify implicit concept patterns where spans don't literally contain concept names."""
    implicit_patterns = []

    for ann in annotations:
        span = ann['span'].strip().lower()
        concept_id = ann['concept_id']
        concept_name, hierarchy = concept_info.get(concept_id, ("Unknown", "unknown"))
        concept_name_lower = concept_name.lower()

        # Check if span text doesn't literally match concept name
        span_words = set(span.split())
        concept_words = set(concept_name_lower.split())

        # Remove common stop words and articles
        stop_words = {'a', 'an', 'the', 'of', 'and', 'or', 'to', 'in', 'on', 'at', 'by', 'for', 'with'}
        span_words -= stop_words
        concept_words -= stop_words

        # If there's little overlap, it's likely an implicit pattern
        if len(span_words & concept_words) / max(len(concept_words), 1) < 0.5:
            # Get context
            context_start = max(0, ann['start'] - 150)
            context_end = min(len(note_text), ann['end'] + 150)
            context = note_text[context_start:context_end].replace('\n', ' ')

            implicit_patterns.append({
                'span': ann['span'],
                'concept_name': concept_name,
                'hierarchy': hierarchy,
                'concept_id': concept_id,
                'annotation_id': ann['annotation_id'],
                'context': context,
                'semantic_relationship': classify_semantic_relationship(span, concept_name)
            })

    return implicit_patterns


def classify_semantic_relationship(span: str, concept_name: str) -> str:
    """Classify the type of semantic relationship between span and concept."""
    span_lower = span.lower()
    concept_lower = concept_name.lower()

    # Body structure implications
    if any(word in concept_lower for word in ['structure', 'body', 'anatomy']):
        return "anatomical_reference"

    # Procedure implications
    if any(word in concept_lower for word in ['procedure', 'surgery', 'operation', 'intervention']):
        if any(word in span_lower for word in ['clinic', 'discussion', 'teaching']):
            return "procedure_context"

    # Finding/observation implications
    if any(word in concept_lower for word in ['finding', 'observation']):
        return "clinical_finding"

    # State/condition implications
    if any(word in span_lower for word in ['improved', 'stable', 'controlled']):
        return "clinical_state"

    # Activity implications
    if any(word in span_lower for word in ['ambulating', 'voiding', 'drained']):
        return "activity_description"

    return "other_semantic"


def analyze_negation_patterns(note_text: str, annotations: List[Dict], concept_info: Dict) -> List[Dict]:
    """Identify negation and context patterns."""
    negation_patterns = []
    negation_words = ['no', 'not', 'without', 'denies', 'absent', 'negative', 'never']

    for ann in annotations:
        # Get context around annotation
        start, end = ann['start'], ann['end']
        context_start = max(0, start - 100)
        context_end = min(len(note_text), end + 100)
        context = note_text[context_start:context_end]

        # Check for negation markers before the span
        pre_context = note_text[context_start:start].lower()

        is_negated = any(neg_word in pre_context.split()[-5:] for neg_word in negation_words)

        if is_negated:
            concept_id = ann['concept_id']
            concept_name, hierarchy = concept_info.get(concept_id, ("Unknown", "unknown"))

            negation_patterns.append({
                'span': ann['span'],
                'concept_name': concept_name,
                'hierarchy': hierarchy,
                'concept_id': concept_id,
                'annotation_id': ann['annotation_id'],
                'context': context.replace('\n', ' '),
                'negation_marker': identify_negation_marker(pre_context)
            })

    return negation_patterns


def identify_negation_marker(pre_context: str) -> str:
    """Identify the specific negation marker used."""
    negation_words = ['no', 'not', 'without', 'denies', 'absent', 'negative', 'never']
    words = pre_context.split()[-10:]  # Look at last 10 words

    for word in reversed(words):
        if word in negation_words:
            return word
    return "unknown"


def analyze_hierarchy_distribution(annotations: List[Dict], concept_info: Dict) -> Dict:
    """Analyze the distribution of SNOMED hierarchies."""
    hierarchy_counts = Counter()
    hierarchy_concepts = defaultdict(set)

    for ann in annotations:
        concept_id = ann['concept_id']
        concept_name, hierarchy = concept_info.get(concept_id, ("Unknown", "unknown"))

        hierarchy_counts[hierarchy] += 1
        hierarchy_concepts[hierarchy].add(concept_id)

    # Calculate ancestor concept recommendations
    hierarchy_analysis = {}
    for hierarchy, count in hierarchy_counts.most_common():
        concepts = list(hierarchy_concepts[hierarchy])
        hierarchy_analysis[hierarchy] = {
            'count': count,
            'unique_concepts': len(concepts),
            'concept_ids': concepts[:10],  # Sample of concept IDs
            'percentage': round(count / len(annotations) * 100, 1)
        }

    return hierarchy_analysis


def generate_mapping_recommendations(abbreviations: Dict) -> Dict[str, str]:
    """Generate recommended mappings for abbreviations."""
    mappings = {}

    for abbrev, info in abbreviations.items():
        concept_name = info['concept_name']
        hierarchy = info['hierarchy']

        # For allergy-related concepts in Allergies section
        if 'allergy' in concept_name.lower():
            mappings[abbrev] = f"allergy to {abbrev.lower()}"

        # For standard medical abbreviations
        elif hierarchy in ['finding', 'procedure', 'body structure']:
            # Create a search-friendly term
            if abbrev == 'CVA':
                mappings[abbrev] = "cerebrovascular accident"
            elif abbrev == 'VS':
                mappings[abbrev] = "vital signs"
            elif abbrev == 'RA':
                mappings[abbrev] = "room air"  # Context dependent - may need rule instead
            elif abbrev == 'NAD':
                mappings[abbrev] = "no abnormality detected"
            elif abbrev == 'RRR':
                mappings[abbrev] = "regular rate and rhythm"
            elif abbrev == 'CTAB':
                mappings[abbrev] = "clear to auscultation bilaterally"
            elif abbrev == 'PCA':
                mappings[abbrev] = "patient controlled analgesia"
            elif abbrev == 'PACU':
                mappings[abbrev] = "post anesthesia care unit"
            elif abbrev == 'PCP':
                mappings[abbrev] = "primary care provider"
            else:
                # Default: use concept name but make it search-friendly
                mappings[abbrev] = concept_name.lower()

    return mappings


def main():
    print("Loading data...")
    concept_info = load_concept_info()
    note_text, annotations, sections = load_note_and_annotations()

    print(f"Loaded note with {len(annotations)} annotations and {len(sections)} sections")

    # 1. Abbreviation Analysis
    print("\n1. Analyzing abbreviation patterns...")
    abbreviations = analyze_abbreviations(note_text, annotations, concept_info)
    print(f"Found {len(abbreviations)} abbreviation patterns")

    # 2. Section-specific patterns
    print("\n2. Analyzing section-specific patterns...")
    section_patterns, special_patterns = analyze_section_patterns(annotations, sections, concept_info)
    print(f"Analyzed {len(section_patterns)} sections with {len(special_patterns)} special patterns")

    # 3. Implicit concept patterns
    print("\n3. Analyzing implicit concept patterns...")
    implicit_patterns = analyze_implicit_patterns(note_text, annotations, concept_info)
    print(f"Found {len(implicit_patterns)} implicit concept patterns")

    # 4. Negation patterns
    print("\n4. Analyzing negation patterns...")
    negation_patterns = analyze_negation_patterns(note_text, annotations, concept_info)
    print(f"Found {len(negation_patterns)} negation patterns")

    # 5. Hierarchy distribution
    print("\n5. Analyzing hierarchy distribution...")
    hierarchy_analysis = analyze_hierarchy_distribution(annotations, concept_info)
    print(f"Analyzed {len(hierarchy_analysis)} hierarchy types")

    # Generate comprehensive analysis report
    analysis_report = {
        'summary': {
            'total_annotations': len(annotations),
            'total_sections': len(sections),
            'abbreviation_patterns': len(abbreviations),
            'implicit_patterns': len(implicit_patterns),
            'negation_patterns': len(negation_patterns),
            'section_patterns': len(section_patterns)
        },
        'abbreviations': abbreviations,
        'section_patterns': section_patterns,
        'special_section_patterns': special_patterns,
        'implicit_patterns': implicit_patterns,
        'negation_patterns': negation_patterns,
        'hierarchy_distribution': hierarchy_analysis,
        'recommended_mappings': generate_mapping_recommendations(abbreviations),
        'section_headers': [s.header for s in sections]
    }

    # Save detailed analysis
    output_path = REPO_ROOT / "comprehensive_clinical_note_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(analysis_report, f, indent=2, default=str)

    print(f"\nSaved comprehensive analysis to {output_path}")

    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Total annotations: {analysis_report['summary']['total_annotations']}")
    print(f"Abbreviation patterns: {analysis_report['summary']['abbreviation_patterns']}")
    print(f"Implicit patterns: {analysis_report['summary']['implicit_patterns']}")
    print(f"Negation patterns: {analysis_report['summary']['negation_patterns']}")
    print(f"Section patterns: {analysis_report['summary']['section_patterns']}")

    print("\n=== TOP HIERARCHY TYPES ===")
    for hierarchy, info in list(hierarchy_analysis.items())[:10]:
        print(f"{hierarchy}: {info['count']} annotations ({info['percentage']}%)")

    print("\n=== RECOMMENDED MAPPINGS ===")
    for abbrev, mapping in list(analysis_report['recommended_mappings'].items())[:10]:
        print(f"{abbrev} -> {mapping}")

    print("\n=== SPECIAL SECTION PATTERNS ===")
    for pattern_type, items in special_patterns.items():
        print(f"{pattern_type}: {len(items)} items")
        for item in items[:3]:
            print(f"  - {item['span']} -> {item['concept_name']}")


if __name__ == "__main__":
    main()