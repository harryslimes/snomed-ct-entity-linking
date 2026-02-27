#!/usr/bin/env python3
"""Analyze inline SNOMED CT annotations to extract patterns and generate rules."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
import sys

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from engine import SectionSpan, segment_sections


class AnnotationAnalyzer:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.annotations = []
        self.sections = []
        self.mappings = {}
        self.g_rules = []
        self.structured_rules = []
        self.annotation_rule_map = {}

    def parse_inline_annotations(self, text: str):
        """Parse inline annotations from the text."""
        pattern = r'\[([^\]]+) \| ([^\]]+)\]\{id=(\d+)\}'

        # Track position adjustments due to annotation removal
        offset = 0
        clean_text = text

        for match in re.finditer(pattern, text):
            span_text = match.group(1)
            concept_info = match.group(2)
            ann_id = match.group(3)

            # Parse concept name and hierarchy
            concept_match = re.match(r'(.+?) \((.+?)\)', concept_info)
            if concept_match:
                concept_name = concept_match.group(1)
                hierarchy = concept_match.group(2)
            else:
                concept_name = concept_info
                hierarchy = "unknown"

            # Calculate original position in clean text
            start = match.start() - offset

            # Store annotation
            self.annotations.append({
                'id': ann_id,
                'span_text': span_text,
                'concept_name': concept_name,
                'hierarchy': hierarchy,
                'full_match': match.group(0),
                'char_position': match.start(),
                'line_number': text[:match.start()].count('\n') + 1
            })

            # Update offset for next iteration
            offset += len(match.group(0)) - len(span_text)

        # Get clean text for section analysis
        clean_text = re.sub(pattern, r'\1', text)
        return clean_text

    def analyze_sections(self, clean_text: str):
        """Segment text into sections."""
        self.sections = segment_sections(clean_text)

        # Map annotations to sections
        for ann in self.annotations:
            ann['section'] = self._find_section(ann['line_number'])

    def _find_section(self, line_num: int) -> str:
        """Find which section a line number belongs to."""
        # Read the file to get line positions
        with open(self.file_path) as f:
            lines = f.readlines()

        # Common section headers in clinical notes
        section_headers = [
            "Chief Complaint:", "History of Present Illness:", "Past Medical History:",
            "Social History:", "Family History:", "Physical Exam:", "Pertinent Results:",
            "Brief Hospital Course:", "Medications on Admission:", "Discharge Medications:",
            "Discharge Disposition:", "Discharge Diagnosis:", "Discharge Condition:",
            "Discharge Instructions:", "Followup Instructions:", "Allergies:"
        ]

        current_section = "Header"
        for i, line in enumerate(lines[:line_num]):
            line_stripped = line.strip()
            for header in section_headers:
                if line_stripped.startswith(header):
                    current_section = header.rstrip(':')
                    break

        return current_section

    def analyze_patterns(self):
        """Analyze annotation patterns to generate rules."""
        # Pattern categories
        abbreviations = defaultdict(list)
        section_specific = defaultdict(list)
        hierarchy_patterns = defaultdict(list)
        negations = []
        complex_mappings = []

        for ann in self.annotations:
            span = ann['span_text']
            concept = ann['concept_name']
            hierarchy = ann['hierarchy']
            section = ann['section']

            # Detect abbreviations (all caps or short with capitals)
            if (len(span) <= 5 and span.isupper()) or (len(span) <= 3 and any(c.isupper() for c in span)):
                abbreviations[span].append({
                    'concept': concept,
                    'section': section,
                    'hierarchy': hierarchy
                })

            # Section-specific patterns
            section_specific[section].append(ann)

            # Hierarchy patterns
            hierarchy_patterns[hierarchy].append(ann)

            # Negation patterns
            if 'no ' in span.lower() or 'without' in span.lower():
                negations.append(ann)

            # Complex mappings (span very different from concept)
            if self._is_complex_mapping(span, concept):
                complex_mappings.append(ann)

        # Generate mappings for consistent abbreviations
        for abbrev, instances in abbreviations.items():
            if len(set(inst['concept'] for inst in instances)) == 1:
                # Same concept across all sections - add to mappings
                self.mappings[abbrev] = instances[0]['concept']

        # Generate rules based on patterns
        self._generate_g_rules()
        self._generate_structured_rules(section_specific, hierarchy_patterns, abbreviations, negations, complex_mappings)
        self._generate_annotation_map()

    def _is_complex_mapping(self, span: str, concept: str) -> bool:
        """Check if span to concept mapping is non-obvious."""
        span_lower = span.lower()
        concept_lower = concept.lower()

        # Check if span words appear in concept
        span_words = set(span_lower.split())
        concept_words = set(concept_lower.split())

        # If no overlap, it's complex
        if not span_words.intersection(concept_words):
            return True

        # Special cases
        if span_lower == "improved" and "improved" in concept_lower:
            return False
        if span_lower == "alive" and "alive" in concept_lower:
            return False

        return False

    def _generate_g_rules(self):
        """Generate universal rules."""
        self.g_rules = [
            {
                "id": "G1",
                "rule": "For medical abbreviations and acronyms, expand to full clinical terms considering the document section context. Physical exam abbreviations often denote body systems or examination findings.",
                "stages": ["stage_2_search"]
            },
            {
                "id": "G2",
                "rule": "When searching for SNOMED concepts, include clinically equivalent synonyms and alternative phrasings alongside the literal span text to improve retrieval coverage.",
                "stages": ["stage_2_search"]
            },
            {
                "id": "G3",
                "rule": "Prefer concepts from hierarchies that match the clinical context - procedures for actions/interventions, findings for observations/assessments, disorders for diagnoses, body structures for anatomy.",
                "stages": ["stage_3_select"]
            },
            {
                "id": "G4",
                "rule": "For negated findings or absent conditions, search for the positive finding/condition and select the appropriate negative or absent variant from the results.",
                "stages": ["stage_2_search", "stage_3_select"]
            },
            {
                "id": "G5",
                "rule": "Section context strongly influences interpretation - the same abbreviation or term may map to different concepts in different sections of the clinical note.",
                "stages": ["stage_2_search", "stage_3_select"]
            }
        ]

    def _generate_structured_rules(self, section_specific, hierarchy_patterns, abbreviations, negations, complex_mappings):
        """Generate specific structured rules."""
        rule_id = 1

        # Physical exam abbreviations
        pe_abbrevs = []
        for abbrev, instances in abbreviations.items():
            if any(inst['section'] == 'Physical Exam' for inst in instances):
                pe_abbrevs.append(abbrev)

        if pe_abbrevs:
            self.structured_rules.append({
                "rule_id": f"R{rule_id}",
                "concept_type": "Physical_Exam_Abbreviation",
                "stage_1_span": {
                    "label": "Physical_Exam_Abbreviation",
                    "description": "Single capital letters or short uppercase abbreviations in Physical Exam sections representing body systems, examination types, or clinical findings"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Expand physical examination abbreviations to their full medical terms based on standard clinical nomenclature for body systems and examination findings"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select examination or finding concepts over procedure concepts when both are available",
                    "preferred_hierarchy": "procedure",
                    "reject_hierarchies": []
                }
            })
            rule_id += 1

        # Medication allergies
        allergy_meds = [ann for ann in self.annotations if ann['section'] == 'Allergies']
        if allergy_meds:
            self.structured_rules.append({
                "rule_id": f"R{rule_id}",
                "concept_type": "Medication_Allergy",
                "stage_1_span": {
                    "label": "Medication_Allergy",
                    "description": "Medication names appearing in the Allergies section"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Transform medication name to allergy concept by searching for 'allergy to [medication]' pattern"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select allergy findings over substance or product concepts",
                    "preferred_hierarchy": "finding",
                    "reject_hierarchies": ["substance", "product"]
                }
            })
            rule_id += 1

        # Procedure references
        procedures = [ann for ann in self.annotations if ann['hierarchy'] == 'procedure']
        if procedures:
            self.structured_rules.append({
                "rule_id": f"R{rule_id}",
                "concept_type": "Procedure_Reference",
                "stage_1_span": {
                    "label": "Procedure_Reference",
                    "description": "References to medical procedures, surgeries, or interventions performed or planned"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Search for the procedure name, including common variations and abbreviations used in clinical documentation"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Prefer specific procedure concepts over general action concepts",
                    "preferred_hierarchy": "procedure",
                    "reject_hierarchies": ["finding"]
                }
            })
            rule_id += 1

        # Negated findings
        if negations:
            self.structured_rules.append({
                "rule_id": f"R{rule_id}",
                "concept_type": "Negated_Finding",
                "stage_1_span": {
                    "label": "Negated_Finding",
                    "description": "Clinical findings that are explicitly absent or negated, often prefixed with 'no', 'without', or 'denies'"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Extract the core finding from the negated phrase and search for both positive and negative variants"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select concepts that represent absence or negation of the finding",
                    "preferred_hierarchy": "finding",
                    "reject_hierarchies": []
                }
            })
            rule_id += 1

        # Activity status
        activity_annotations = [ann for ann in self.annotations if 'ambula' in ann['span_text'].lower() or 'activity' in ann['concept_name'].lower()]
        if activity_annotations:
            self.structured_rules.append({
                "rule_id": f"R{rule_id}",
                "concept_type": "Activity_Status",
                "stage_1_span": {
                    "label": "Activity_Status",
                    "description": "Descriptions of patient mobility, ambulation status, or activity levels"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Search for functional status and mobility-related concepts corresponding to the described activity level"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Prefer finding concepts that describe patient status over procedure concepts for training or therapy",
                    "preferred_hierarchy": "finding",
                    "reject_hierarchies": []
                }
            })
            rule_id += 1

    def _generate_annotation_map(self):
        """Map each annotation to applicable rules."""
        for ann in self.annotations:
            ann_id = ann['id']
            applicable_rules = []

            # Check which G-rules apply
            applicable_rules.extend(["G1", "G2", "G3", "G5"])  # Most apply broadly

            if 'no ' in ann['span_text'].lower() or 'without' in ann['span_text'].lower():
                applicable_rules.append("G4")

            # Check which R-rules apply based on the annotation characteristics
            for rule in self.structured_rules:
                if self._rule_applies_to_annotation(rule, ann):
                    applicable_rules.append(rule['rule_id'])

            self.annotation_rule_map[ann_id] = applicable_rules

    def _rule_applies_to_annotation(self, rule, ann):
        """Check if a structured rule applies to an annotation."""
        concept_type = rule['concept_type']

        if concept_type == "Physical_Exam_Abbreviation":
            return ann['section'] == 'Physical Exam' and len(ann['span_text']) <= 5
        elif concept_type == "Medication_Allergy":
            return ann['section'] == 'Allergies'
        elif concept_type == "Procedure_Reference":
            return ann['hierarchy'] == 'procedure'
        elif concept_type == "Negated_Finding":
            return 'no ' in ann['span_text'].lower() or 'without' in ann['span_text'].lower()
        elif concept_type == "Activity_Status":
            return 'ambula' in ann['span_text'].lower() or 'activity' in ann['concept_name'].lower()

        return False

    def generate_output(self):
        """Generate the final JSON output."""
        output = {
            "version": "3.0",
            "mappings": self.mappings,
            "g_rules": self.g_rules,
            "structured_rules": self.structured_rules,
            "annotation_rule_map": self.annotation_rule_map
        }
        return output

    def analyze(self):
        """Run the complete analysis."""
        # Read file
        with open(self.file_path) as f:
            text = f.read()

        # Parse annotations
        clean_text = self.parse_inline_annotations(text)

        # Analyze sections
        self.analyze_sections(clean_text)

        # Analyze patterns
        self.analyze_patterns()

        # Generate output
        return self.generate_output()


def main():
    file_path = REPO_ROOT / "inline_annotated_note.txt"
    analyzer = AnnotationAnalyzer(file_path)
    result = analyzer.analyze()

    # Save result
    output_path = REPO_ROOT / "scripts" / "rules_output.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Analysis complete. Results saved to {output_path}")
    print(f"Found {len(analyzer.annotations)} annotations")
    print(f"Generated {len(result['mappings'])} mappings")
    print(f"Generated {len(result['g_rules'])} general rules")
    print(f"Generated {len(result['structured_rules'])} structured rules")


if __name__ == "__main__":
    main()