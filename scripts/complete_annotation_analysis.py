#!/usr/bin/env python3
"""Complete analysis of SNOMED CT annotations using both inline and CSV data."""
from __future__ import annotations

import json
import re
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
import sys

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from engine import SectionSpan, segment_sections

# Load terminology for concept lookups
TERMINOLOGY_CSV = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"


class CompleteAnnotationAnalyzer:
    def __init__(self):
        self.annotations = []
        self.mappings = {}
        self.g_rules = []
        self.structured_rules = []
        self.annotation_rule_map = {}
        # Pattern tracking
        self.section_patterns = defaultdict(list)
        self.hierarchy_patterns = defaultdict(list)
        self.abbreviation_patterns = defaultdict(list)
        self.span_to_concept_patterns = defaultdict(set)
        self.complex_mappings = []
        self.implicit_concepts = []
        # Load concept terminology
        self.concept_info = self._load_concept_info()

    def _load_concept_info(self):
        """Load SNOMED concept information."""
        ft = pd.read_csv(TERMINOLOGY_CSV)
        return {
            int(row.concept_id): (row.concept_name, row.hierarchy)
            for _, row in ft.iterrows()
        }

    def load_annotations_from_csv(self, csv_path: Path, note_text_path: Path):
        """Load annotations from CSV and correlate with note text."""
        # Load annotations
        ann_df = pd.read_csv(csv_path)

        # Load note text
        with open(note_text_path) as f:
            note_text = f.read()

        # Get clean text for section analysis
        pattern = r'\[([^\]]+) \| ([^\]]+)\]\{id=(\d+)\}'
        clean_text = re.sub(pattern, r'\1', note_text)

        # Segment into sections
        sections = self._segment_note_into_sections(note_text)

        # Process each annotation
        for _, row in ann_df.iterrows():
            ann_id = str(row['annotation_id'])
            start = int(row['start'])
            end = int(row['end'])
            span_text = row['span'].strip()
            concept_id = int(row['concept_id'])

            # Get concept info
            concept_name, hierarchy = self.concept_info.get(concept_id, ("Unknown", "unknown"))

            # Find section
            line_num = note_text[:start].count('\n') + 1
            section = self._find_section_for_line(line_num, sections)

            # Store annotation
            ann = {
                'id': ann_id,
                'span_text': span_text,
                'concept_id': concept_id,
                'concept_name': concept_name,
                'hierarchy': hierarchy,
                'section': section,
                'start': start,
                'end': end
            }
            self.annotations.append(ann)

            # Track patterns
            self.section_patterns[section].append(ann)
            self.hierarchy_patterns[hierarchy].append(ann)
            self.span_to_concept_patterns[span_text.lower()].add((concept_name, hierarchy))

    def _segment_note_into_sections(self, text):
        """Segment text into sections with line numbers."""
        sections = []
        section_headers = [
            "Allergies:", "Chief Complaint:", "Major Surgical or Invasive Procedure:",
            "History of Present Illness:", "Past Medical History:", "Social History:",
            "Family History:", "Physical Exam:", "Prior Discharge:", "Pertinent Results:",
            "Brief Hospital Course:", "Medications on Admission:", "Discharge Medications:",
            "Discharge Disposition:", "Discharge Diagnosis:", "Discharge Condition:",
            "Discharge Instructions:", "Followup Instructions:"
        ]

        lines = text.split('\n')
        current_section = "Header"
        section_start = 0

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            for header in section_headers:
                if line_stripped.startswith(header):
                    # Save previous section
                    sections.append({
                        'name': current_section,
                        'start_line': section_start,
                        'end_line': i - 1
                    })
                    current_section = header.rstrip(':')
                    section_start = i
                    break

        # Add final section
        sections.append({
            'name': current_section,
            'start_line': section_start,
            'end_line': len(lines) - 1
        })

        return sections

    def _find_section_for_line(self, line_num, sections):
        """Find which section a line number belongs to."""
        for section in sections:
            if section['start_line'] < line_num <= section['end_line'] + 1:
                return section['name']
        return sections[-1]['name'] if sections else "Unknown"

    def analyze_patterns(self):
        """Comprehensive pattern analysis."""
        print(f"\nAnalyzing {len(self.annotations)} annotations...")

        # Analyze each annotation
        for ann in self.annotations:
            span = ann['span_text']
            concept = ann['concept_name']
            hierarchy = ann['hierarchy']
            section = ann['section']

            # Detect abbreviations
            if self._is_abbreviation(span):
                self.abbreviation_patterns[span].append({
                    'concept': concept,
                    'section': section,
                    'hierarchy': hierarchy,
                    'id': ann['id']
                })

            # Detect complex mappings
            if self._is_complex_mapping(span, concept):
                self.complex_mappings.append(ann)

            # Detect implicit concepts
            if self._is_implicit_concept(span, concept):
                self.implicit_concepts.append(ann)

        # Generate rules based on patterns
        self._generate_comprehensive_rules()

    def _is_abbreviation(self, span: str) -> bool:
        """Check if span is likely an abbreviation."""
        # All uppercase and short
        if len(span) <= 6 and span.isupper() and not span.isdigit():
            return True
        # Short with capitals (but not just numbers)
        if len(span) <= 4 and any(c.isupper() for c in span) and not span.isdigit():
            return True
        # Single letters used as abbreviations
        if len(span) == 1 and span.isupper():
            return True
        return False

    def _is_complex_mapping(self, span: str, concept: str) -> bool:
        """Check if span to concept mapping is non-obvious."""
        span_lower = span.lower()
        concept_lower = concept.lower()

        # Known semantic equivalents
        semantic_equivalents = [
            ("biliary pancreatitis", "gallstone pancreatitis"),
            ("pancreatic rest", "ectopic pancreas"),
            ("afebrile", "fever"),
            ("surgical resection of your gallbladder", "cholecystectomy"),
            ("post operative", "postoperative"),
            ("voided without problem", "normal micturition"),
            ("pain was well controlled", "adequate pain control"),
            ("diet was tolerated well", "diet good"),
            ("ambulating", "fully mobile"),
            ("voiding without assistance", "continence independent"),
            ("avoid lifting weights", "functional activity education"),
            ("activity restrictions", "functional activity education"),
            ("while taking pain medications", "patient medication education"),
            ("drink adequate amounts of fluids", "fluid intake education")
        ]

        for span_pattern, concept_pattern in semantic_equivalents:
            if span_pattern in span_lower and concept_pattern in concept_lower:
                return True

        # Non-overlapping words indicates complex mapping
        span_words = set(span_lower.split())
        concept_words = set(concept_lower.split())
        if not span_words.intersection(concept_words):
            # Check for partial matches
            partial_match = any(
                any(sw in cw or cw in sw for sw in span_words if len(sw) > 2)
                for cw in concept_words if len(cw) > 2
            )
            if not partial_match:
                return True

        return False

    def _is_implicit_concept(self, span: str, concept: str) -> bool:
        """Check if the concept is implicit/inferred rather than literal."""
        implicit_mappings = [
            ("pancreatic rest", "ectopic pancreas"),
            ("c", "cyanosis"),
            ("e", "edema"),
            ("afebrile", "fever"),
            ("clinic", "outpatient care"),
            ("teaching", "patient education"),
            ("instructions", "recommendation"),
            ("shower", "functional activity"),
            ("laparoscopic", "unknown")  # When used alone without procedure
        ]

        span_lower = span.lower()
        for pattern, expected in implicit_mappings:
            if pattern == span_lower and expected in concept.lower():
                return True
        return False

    def _generate_comprehensive_rules(self):
        """Generate comprehensive rules based on all patterns found."""
        # Generate deterministic mappings
        self._generate_mappings()

        # Generate general rules
        self._generate_g_rules()

        # Generate structured rules
        self._generate_structured_rules()

        # Generate annotation mapping
        self._generate_annotation_map()

    def _generate_mappings(self):
        """Generate deterministic 1:1 mappings."""
        # Analyze abbreviations for consistent mappings
        for abbrev, instances in self.abbreviation_patterns.items():
            concepts = [inst['concept'] for inst in instances]
            sections = [inst['section'] for inst in instances]

            # If same concept across all sections, add to mappings
            if len(set(concepts)) == 1:
                concept = concepts[0]
                # Special handling for allergy medications
                if 'Allergies' in sections and 'allergy' in concept.lower():
                    self.mappings[abbrev] = concept
                else:
                    # Standard abbreviation expansion
                    self.mappings[abbrev] = self._get_search_term_for_abbreviation(abbrev, concept)

        # Add other deterministic mappings found
        additional_mappings = {
            "Penicillins": "allergy to penicillin",
            "MI": "myocardial infarction",
            "IV": "intravenous",
            "BID": "twice daily",
            "PO": "by mouth",
            "Q3H": "every 3 hours",
            "Q24H": "every 24 hours",
            "PRN": "as needed"
        }

        for k, v in additional_mappings.items():
            if k not in self.mappings:
                self.mappings[k] = v

    def _get_search_term_for_abbreviation(self, abbrev: str, concept: str) -> str:
        """Get appropriate search term for abbreviation."""
        # Known expansions
        expansions = {
            "CVA": "cerebrovascular accident",
            "VS": "vital signs",
            "RA": "breathing room air",
            "GEN": "general examination",
            "NAD": "no abnormality detected",
            "CV": "cardiovascular examination",
            "RRR": "regular rate and rhythm",
            "PULM": "pulmonary examination",
            "CTAB": "clear to auscultation bilaterally",
            "ABD": "abdominal examination",
            "EXTR": "extremity examination",
            "PACU": "post anesthesia care unit",
            "PCA": "patient controlled analgesia",
            "PCP": "primary care provider",
            "GI": "gastrointestinal",
            "HIPAA": "health insurance portability and accountability act"
        }

        return expansions.get(abbrev, concept)

    def _generate_g_rules(self):
        """Generate comprehensive general rules."""
        self.g_rules = [
            {
                "id": "G1",
                "rule": "Expand medical abbreviations to their full clinical terms using standard medical nomenclature. Consider document section context for disambiguation when the same abbreviation has multiple meanings.",
                "stages": ["stage_2_search"]
            },
            {
                "id": "G2",
                "rule": "Include clinically equivalent synonyms and alternative phrasings when searching SNOMED. For compound clinical terms, search both the exact phrase and semantically equivalent alternatives that express the same clinical concept.",
                "stages": ["stage_2_search"]
            },
            {
                "id": "G3",
                "rule": "Select SNOMED concepts from hierarchies that match the clinical context: procedures for interventions/actions, findings for observations/symptoms/signs, disorders for diagnoses/diseases, body structures for anatomy, situations for care contexts, morphologic abnormality for pathological structures.",
                "stages": ["stage_3_select"]
            },
            {
                "id": "G4",
                "rule": "For negated findings or absent conditions, search for the positive concept and select the appropriate negative/absent variant. Terms like 'no', 'without', 'denies', 'absent', or negative descriptors like 'afebrile' indicate negation.",
                "stages": ["stage_2_search", "stage_3_select"]
            },
            {
                "id": "G5",
                "rule": "Document section headers provide critical disambiguation context. The same term may map to different SNOMED concepts based on its section (e.g., abbreviations in Physical Exam vs Past Medical History sections).",
                "stages": ["stage_2_search", "stage_3_select"]
            },
            {
                "id": "G6",
                "rule": "For implicit or non-literal clinical concepts, search for the medically implied meaning rather than the literal text. Clinical context and medical knowledge determine the appropriate conceptual mapping.",
                "stages": ["stage_2_search"]
            },
            {
                "id": "G7",
                "rule": "When multiple valid SNOMED concepts exist, prefer more specific concepts over general ones. Also prefer concepts that match the grammatical role (noun vs verb) and clinical intent of the span in its sentence context.",
                "stages": ["stage_3_select"]
            },
            {
                "id": "G8",
                "rule": "For multi-word clinical phrases, search for both the complete phrase and individual clinically significant components. Compound medical terms may have specific SNOMED concepts that differ from their component words.",
                "stages": ["stage_2_search"]
            },
            {
                "id": "G9",
                "rule": "Single-letter abbreviations in clinical notes often represent standardized medical abbreviations or examination findings. These require expansion based on their section context and surrounding clinical information.",
                "stages": ["stage_2_search"]
            }
        ]

    def _generate_structured_rules(self):
        """Generate comprehensive structured rules."""
        self.structured_rules = [
            {
                "rule_id": "R1",
                "concept_type": "Physical_Exam_Abbreviation",
                "stage_1_span": {
                    "label": "Physical_Exam_Abbreviation",
                    "description": "Capital letter abbreviations or short uppercase terms in Physical Exam sections representing body systems, examination procedures, or clinical findings"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Expand examination abbreviations to full medical terminology for body systems and examination procedures"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Prefer examination procedure concepts when available, followed by finding concepts for examination results",
                    "preferred_hierarchy": "procedure",
                    "reject_hierarchies": []
                }
            },
            {
                "rule_id": "R2",
                "concept_type": "Allergy_Medication",
                "stage_1_span": {
                    "label": "Allergy_Medication",
                    "description": "Medication names, drug classes, or substances listed in the Allergies section of the clinical note"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Transform substance name to allergy finding by searching for 'allergy to [substance]' or 'hypersensitivity to [substance]' patterns"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select allergy or hypersensitivity findings over substance, product, or organism concepts",
                    "preferred_hierarchy": "finding",
                    "reject_hierarchies": ["substance", "product", "organism"]
                }
            },
            {
                "rule_id": "R3",
                "concept_type": "Clinical_Procedure",
                "stage_1_span": {
                    "label": "Clinical_Procedure",
                    "description": "Medical procedures, surgeries, interventions, therapeutic actions, or clinical services performed or planned"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Search for procedure terms including common clinical variations, abbreviations, and both formal and colloquial medical terminology"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select specific procedure concepts over general action or observation concepts",
                    "preferred_hierarchy": "procedure",
                    "reject_hierarchies": ["finding", "qualifier value"]
                }
            },
            {
                "rule_id": "R4",
                "concept_type": "Negated_Finding",
                "stage_1_span": {
                    "label": "Negated_Finding",
                    "description": "Clinical findings explicitly noted as absent, denied, or negative, including negation words or negative descriptors"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Extract the core clinical finding and search for both positive and corresponding negative/absent concept variants"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select concepts that explicitly represent absence, negation, or normal variants of the finding",
                    "preferred_hierarchy": "finding",
                    "reject_hierarchies": []
                }
            },
            {
                "rule_id": "R5",
                "concept_type": "Functional_Status",
                "stage_1_span": {
                    "label": "Functional_Status",
                    "description": "Patient functional abilities, mobility status, activity levels, or independence in daily activities"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Search for functional status, ability, and mobility concepts matching the described activity or functional level"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Prefer finding concepts describing patient functional status over procedure concepts for therapies or interventions",
                    "preferred_hierarchy": "finding",
                    "reject_hierarchies": ["regime/therapy"]
                }
            },
            {
                "rule_id": "R6",
                "concept_type": "Diagnostic_Finding",
                "stage_1_span": {
                    "label": "Diagnostic_Finding",
                    "description": "Diagnoses, pathological conditions, disorders, disease states, or abnormal clinical findings"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Search for disorder, disease, and pathological finding concepts including clinical synonyms and variations"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Prefer disorder concepts for diseases and diagnoses, morphologic abnormality for structural abnormalities",
                    "preferred_hierarchy": "disorder",
                    "reject_hierarchies": []
                }
            },
            {
                "rule_id": "R7",
                "concept_type": "Anatomical_Structure",
                "stage_1_span": {
                    "label": "Anatomical_Structure",
                    "description": "Body parts, organs, organ systems, anatomical locations, or body regions"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Search for anatomical structure concepts using both formal anatomical terms and common clinical terminology"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select body structure concepts over procedure or finding concepts when referring to anatomy",
                    "preferred_hierarchy": "body structure",
                    "reject_hierarchies": ["procedure"]
                }
            },
            {
                "rule_id": "R8",
                "concept_type": "Clinical_Observation",
                "stage_1_span": {
                    "label": "Clinical_Observation",
                    "description": "Clinical observations, assessment findings, vital signs, or measurable clinical parameters"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Search for clinical finding concepts representing observations, measurements, or assessment results"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select finding concepts that represent clinical observations or states",
                    "preferred_hierarchy": "finding",
                    "reject_hierarchies": ["procedure"]
                }
            },
            {
                "rule_id": "R9",
                "concept_type": "Care_Context",
                "stage_1_span": {
                    "label": "Care_Context",
                    "description": "Clinical care settings, care management activities, or healthcare delivery contexts"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Search for regime/therapy, situation, or environment concepts matching the care context"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select regime/therapy or situation concepts for care contexts and management activities",
                    "preferred_hierarchy": "regime/therapy",
                    "reject_hierarchies": []
                }
            },
            {
                "rule_id": "R10",
                "concept_type": "Implicit_Clinical_Concept",
                "stage_1_span": {
                    "label": "Implicit_Clinical_Concept",
                    "description": "Terms where the clinical meaning differs significantly from literal interpretation, requiring medical domain knowledge"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Search for the clinically implied concept based on medical knowledge rather than literal text interpretation"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select concepts based on clinical interpretation and medical context rather than literal text matching",
                    "preferred_hierarchy": None,
                    "reject_hierarchies": []
                }
            },
            {
                "rule_id": "R11",
                "concept_type": "Single_Letter_Abbreviation",
                "stage_1_span": {
                    "label": "Single_Letter_Abbreviation",
                    "description": "Single capital letters used as medical abbreviations in clinical documentation"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Expand single-letter abbreviations to their full medical terms based on section context and clinical usage patterns"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select concepts matching the standard medical interpretation of the abbreviation in its clinical context",
                    "preferred_hierarchy": None,
                    "reject_hierarchies": []
                }
            },
            {
                "rule_id": "R12",
                "concept_type": "Patient_Education_Activity",
                "stage_1_span": {
                    "label": "Patient_Education_Activity",
                    "description": "References to patient education, instructions, recommendations, or self-care activities"
                },
                "stage_2_search": {
                    "filtering_logic": None,
                    "intent_translation": "Search for patient education, counseling, or health education procedure concepts"
                },
                "stage_3_select": {
                    "disambiguation_logic": "Select procedure concepts related to patient education or counseling activities",
                    "preferred_hierarchy": "procedure",
                    "reject_hierarchies": ["finding"]
                }
            }
        ]

    def _generate_annotation_map(self):
        """Map each annotation to applicable rules."""
        for ann in self.annotations:
            ann_id = ann['id']
            applicable_rules = []

            # Add broadly applicable G-rules
            applicable_rules.extend(["G1", "G2", "G3", "G5", "G7"])

            # Check specific G-rule conditions
            if self._is_negated(ann['span_text']):
                applicable_rules.append("G4")
            if self._is_implicit_concept(ann['span_text'], ann['concept_name']):
                applicable_rules.append("G6")
            if len(ann['span_text'].split()) > 1:
                applicable_rules.append("G8")
            if len(ann['span_text']) == 1 and ann['span_text'].isupper():
                applicable_rules.append("G9")

            # Add applicable R-rules
            for rule in self.structured_rules:
                if self._rule_applies_to_annotation(rule, ann):
                    applicable_rules.append(rule['rule_id'])

            self.annotation_rule_map[ann_id] = applicable_rules

    def _is_negated(self, span: str) -> bool:
        """Check if span represents a negated finding."""
        negation_indicators = ['no ', 'without', 'denies', 'absent', 'negative']
        span_lower = span.lower()
        return any(neg in span_lower for neg in negation_indicators) or span_lower == 'afebrile'

    def _rule_applies_to_annotation(self, rule, ann):
        """Check if a structured rule applies to an annotation."""
        rule_id = rule['rule_id']
        span = ann['span_text']
        section = ann['section']
        hierarchy = ann['hierarchy']
        concept = ann['concept_name']

        if rule_id == "R1":  # Physical exam abbreviations
            return section == 'Physical Exam' and self._is_abbreviation(span)
        elif rule_id == "R2":  # Allergy medications
            return section == 'Allergies'
        elif rule_id == "R3":  # Clinical procedures
            return hierarchy == 'procedure' or 'procedure' in concept.lower()
        elif rule_id == "R4":  # Negated findings
            return self._is_negated(span)
        elif rule_id == "R5":  # Functional status
            func_keywords = ['ambula', 'activity', 'mobile', 'independent', 'walking', 'ambulatory']
            return any(kw in span.lower() or kw in concept.lower() for kw in func_keywords)
        elif rule_id == "R6":  # Diagnostic findings
            return hierarchy in ['disorder', 'morphologic abnormality']
        elif rule_id == "R7":  # Anatomical structures
            return hierarchy == 'body structure'
        elif rule_id == "R8":  # Clinical observations
            return hierarchy == 'finding' and section in ['Physical Exam', 'Discharge Condition', 'Brief Hospital Course']
        elif rule_id == "R9":  # Care context
            return hierarchy in ['regime/therapy', 'situation', 'environment']
        elif rule_id == "R10":  # Implicit concepts
            return self._is_implicit_concept(span, concept)
        elif rule_id == "R11":  # Single letter abbreviations
            return len(span) == 1 and span.isupper()
        elif rule_id == "R12":  # Patient education
            edu_keywords = ['teaching', 'education', 'instructions', 'counseling']
            return any(kw in span.lower() or kw in concept.lower() for kw in edu_keywords)

        return False

    def generate_output(self):
        """Generate the final JSON output."""
        return {
            "version": "3.0",
            "mappings": self.mappings,
            "g_rules": self.g_rules,
            "structured_rules": self.structured_rules,
            "annotation_rule_map": self.annotation_rule_map
        }

    def print_analysis_summary(self):
        """Print analysis summary."""
        print(f"\nAnalysis Summary:")
        print(f"  Total annotations: {len(self.annotations)}")
        print(f"  Unique spans: {len(self.span_to_concept_patterns)}")
        print(f"  Sections covered: {len(self.section_patterns)}")
        print(f"  Hierarchy types: {len(self.hierarchy_patterns)}")

        print(f"\nAnnotations by section:")
        for section, anns in sorted(self.section_patterns.items(), key=lambda x: -len(x[1])):
            print(f"  {section}: {len(anns)}")

        print(f"\nAnnotations by hierarchy:")
        for hierarchy, anns in sorted(self.hierarchy_patterns.items(), key=lambda x: -len(x[1])):
            print(f"  {hierarchy}: {len(anns)}")

        print(f"\nPattern Analysis:")
        print(f"  Abbreviations found: {len(self.abbreviation_patterns)}")
        print(f"  Complex mappings: {len(self.complex_mappings)}")
        print(f"  Implicit concepts: {len(self.implicit_concepts)}")

        # Show some complex mapping examples
        if self.complex_mappings:
            print(f"\nExample complex mappings:")
            for ann in self.complex_mappings[:5]:
                print(f"  '{ann['span_text']}' → '{ann['concept_name']}'")


def main():
    analyzer = CompleteAnnotationAnalyzer()

    # Load annotations from CSV
    csv_path = REPO_ROOT / "first_note_annotations.csv"
    note_path = REPO_ROOT / "inline_annotated_note.txt"

    analyzer.load_annotations_from_csv(csv_path, note_path)

    # Analyze patterns
    analyzer.analyze_patterns()

    # Print summary
    analyzer.print_analysis_summary()

    # Generate output
    result = analyzer.generate_output()

    # Save result
    output_path = REPO_ROOT / "scripts" / "rules_output.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nRules Generation Summary:")
    print(f"  Mappings: {len(result['mappings'])}")
    print(f"  General rules: {len(result['g_rules'])}")
    print(f"  Structured rules: {len(result['structured_rules'])}")
    print(f"  Annotations mapped: {len(result['annotation_rule_map'])}")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()