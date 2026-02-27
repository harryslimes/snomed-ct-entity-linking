#!/usr/bin/env python3
"""Comprehensive analysis of inline SNOMED CT annotations to extract patterns and generate rules."""
from __future__ import annotations

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import sys

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from engine import SectionSpan, segment_sections


class ComprehensiveAnnotationAnalyzer:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.annotations = []
        self.sections = []
        self.mappings = {}
        self.g_rules = []
        self.structured_rules = []
        self.annotation_rule_map = {}
        # Track patterns for comprehensive analysis
        self.section_patterns = defaultdict(list)
        self.hierarchy_patterns = defaultdict(list)
        self.abbreviation_patterns = defaultdict(list)
        self.complex_mappings = []
        self.implicit_concepts = []
        self.disambiguation_cases = []

    def parse_inline_annotations(self, text: str):
        """Parse ALL inline annotations from the text."""
        pattern = r'\[([^\]]+) \| ([^\]]+)\]\{id=(\d+)\}'

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

            # Store annotation with line number
            self.annotations.append({
                'id': ann_id,
                'span_text': span_text,
                'concept_name': concept_name,
                'hierarchy': hierarchy,
                'full_match': match.group(0),
                'char_position': match.start(),
                'line_number': text[:match.start()].count('\n') + 1
            })

        # Get clean text for section analysis
        clean_text = re.sub(pattern, r'\1', text)
        return clean_text

    def analyze_sections(self, clean_text: str):
        """Segment text into sections and map annotations."""
        # Common section headers in clinical notes
        section_headers = [
            "Allergies:", "Chief Complaint:", "Major Surgical or Invasive Procedure:",
            "History of Present Illness:", "Past Medical History:", "Social History:",
            "Family History:", "Physical Exam:", "Pertinent Results:",
            "Brief Hospital Course:", "Medications on Admission:", "Discharge Medications:",
            "Discharge Disposition:", "Discharge Diagnosis:", "Discharge Condition:",
            "Discharge Instructions:", "Followup Instructions:"
        ]

        # Read file and track section boundaries
        with open(self.file_path) as f:
            lines = f.readlines()

        current_section = "Header"
        section_line_map = {}

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            for header in section_headers:
                if line_stripped.startswith(header):
                    current_section = header.rstrip(':')
                    break
            section_line_map[i+1] = current_section

        # Map annotations to sections
        for ann in self.annotations:
            ann['section'] = section_line_map.get(ann['line_number'], "Unknown")
            self.section_patterns[ann['section']].append(ann)

    def analyze_patterns(self):
        """Comprehensive pattern analysis."""
        # Analyze each annotation
        for ann in self.annotations:
            span = ann['span_text']
            concept = ann['concept_name']
            hierarchy = ann['hierarchy']
            section = ann['section']

            # Track hierarchy usage
            self.hierarchy_patterns[hierarchy].append(ann)

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

            # Detect disambiguation cases
            if self._needs_disambiguation(span, section):
                self.disambiguation_cases.append(ann)

        # Generate comprehensive rules based on patterns
        self._generate_comprehensive_rules()

    def _is_abbreviation(self, span: str) -> bool:
        """Check if span is likely an abbreviation."""
        # All uppercase and short
        if len(span) <= 5 and span.isupper():
            return True
        # Short with capitals
        if len(span) <= 3 and any(c.isupper() for c in span):
            return True
        # Common medical abbreviation patterns
        if span in ["MI", "CVA", "GI", "IV", "PO", "BID", "PRN"]:
            return True
        return False

    def _is_complex_mapping(self, span: str, concept: str) -> bool:
        """Check if span to concept mapping is non-obvious."""
        span_lower = span.lower()
        concept_lower = concept.lower()

        # Special cases that are NOT complex
        simple_mappings = {
            "improved": "improved",
            "alive": "alive",
            "alert": "alert",
            "interactive": "communicat"
        }

        for s, c in simple_mappings.items():
            if s in span_lower and c in concept_lower:
                return False

        # Check for semantic equivalence
        semantic_equivalents = {
            "biliary pancreatitis": "gallstone pancreatitis",
            "pancreatic rest": "ectopic pancreas",
            "afebrile": "fever",
            "surgical resection of your gallbladder": "cholecystectomy",
            "post operative": "postoperative"
        }

        for equiv_span, equiv_concept in semantic_equivalents.items():
            if equiv_span in span_lower and equiv_concept in concept_lower:
                return True

        # If no word overlap, it's complex
        span_words = set(span_lower.split())
        concept_words = set(concept_lower.split())
        if not span_words.intersection(concept_words):
            return True

        return False

    def _is_implicit_concept(self, span: str, concept: str) -> bool:
        """Check if the concept is implicit/inferred rather than literal."""
        implicit_patterns = {
            "pancreatic rest": "ectopic pancreas",  # Clinical context != literal
            "c": "cyanosis",  # Single letter to condition
            "e": "edema",  # Single letter to condition
            "afebrile": "fever"  # Negative state to positive concept
        }

        span_lower = span.lower()
        for pattern, expected in implicit_patterns.items():
            if pattern in span_lower and expected in concept.lower():
                return True
        return False

    def _needs_disambiguation(self, span: str, section: str) -> bool:
        """Check if span needs section-based disambiguation."""
        # Known ambiguous terms
        ambiguous_terms = {
            "RA": ["room air", "rheumatoid arthritis"],
            "MS": ["multiple sclerosis", "morphine sulfate", "mental status"],
            "PT": ["physical therapy", "prothrombin time", "patient"],
            "c": ["with", "cyanosis"],
            "e": ["edema", "extremity"]
        }

        return span in ambiguous_terms

    def _generate_comprehensive_rules(self):
        """Generate comprehensive rules based on all patterns found."""
        # Generate mappings for consistent abbreviations
        for abbrev, instances in self.abbreviation_patterns.items():
            concepts = [inst['concept'] for inst in instances]
            if len(set(concepts)) == 1:
                # Consistent mapping across all sections
                self.mappings[abbrev] = self._generate_search_term(abbrev, concepts[0])

        # Generate general rules
        self._generate_g_rules()

        # Generate structured rules
        self._generate_structured_rules()

        # Generate annotation mapping
        self._generate_annotation_map()

    def _generate_search_term(self, span: str, concept: str) -> str:
        """Generate appropriate search term for mappings."""
        # Special handling for allergy section medications
        if "allergy" in concept.lower() and "to" in concept.lower():
            return concept  # Already in correct format

        # For abbreviations, return the expanded form
        abbrev_expansions = {
            "CVA": "cerebrovascular accident",
            "VS": "vital signs",
            "RA": "breathing room air",  # In vital signs context
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
            "PCP": "primary care provider"
        }

        return abbrev_expansions.get(span, concept)

    def _generate_g_rules(self):
        """Generate comprehensive universal rules."""
        self.g_rules = [
            {
                "id": "G1",
                "rule": "Expand medical abbreviations to their full clinical terms using standard medical nomenclature, considering document section context for disambiguation.",
                "stages": ["stage_2_search"]
            },
            {
                "id": "G2",
                "rule": "Include clinically equivalent synonyms and alternative phrasings when searching. For compound terms, search both the exact phrase and semantically equivalent alternatives.",
                "stages": ["stage_2_search"]
            },
            {
                "id": "G3",
                "rule": "Select concepts from SNOMED hierarchies that align with clinical context: procedures for interventions, findings for observations/symptoms, disorders for diagnoses, body structures for anatomy, situations for contexts.",
                "stages": ["stage_3_select"]
            },
            {
                "id": "G4",
                "rule": "For negated findings, search for the positive concept and select the appropriate negative/absent variant. Terms like 'afebrile', 'without', 'no' indicate negation.",
                "stages": ["stage_2_search", "stage_3_select"]
            },
            {
                "id": "G5",
                "rule": "Section headers provide critical context for interpretation. The same term may have different meanings in different sections (e.g., abbreviations in Physical Exam vs History sections).",
                "stages": ["stage_2_search", "stage_3_select"]
            },
            {
                "id": "G6",
                "rule": "For implicit or non-literal concepts, search for the clinically implied meaning rather than the literal text. Clinical context determines the appropriate conceptual mapping.",
                "stages": ["stage_2_search"]
            },
            {
                "id": "G7",
                "rule": "When multiple valid concepts exist, prefer more specific over general concepts, and prefer concepts that match the grammatical role (noun vs verb) of the span in context.",
                "stages": ["stage_3_select"]
            }
        ]

    def _generate_structured_rules(self):
        """Generate comprehensive structured rules."""
        rule_id = 1

        # Physical Exam Abbreviations
        self.structured_rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Physical_Exam_Abbreviation",
            "stage_1_span": {
                "label": "Physical_Exam_Abbreviation",
                "description": "Capital letter abbreviations or short uppercase terms in Physical Exam sections denoting body systems, examination procedures, or clinical findings"
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Expand examination abbreviations to full medical terminology for body systems and examination procedures"
            },
            "stage_3_select": {
                "disambiguation_logic": "Prefer examination procedure concepts over general finding concepts",
                "preferred_hierarchy": "procedure",
                "reject_hierarchies": []
            }
        })
        rule_id += 1

        # Allergy Section Medications
        self.structured_rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Allergy_Medication",
            "stage_1_span": {
                "label": "Allergy_Medication",
                "description": "Medication names, drug classes, or substances listed in the Allergies section"
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Transform substance name to allergy finding by searching for 'allergy to [substance]' pattern"
            },
            "stage_3_select": {
                "disambiguation_logic": "Select allergy/hypersensitivity findings over substance or product concepts",
                "preferred_hierarchy": "finding",
                "reject_hierarchies": ["substance", "product", "physical object"]
            }
        })
        rule_id += 1

        # Clinical Procedures
        self.structured_rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Clinical_Procedure",
            "stage_1_span": {
                "label": "Clinical_Procedure",
                "description": "Medical procedures, surgeries, interventions, or therapeutic actions performed or planned"
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Search for procedure terms including common clinical variations and abbreviations"
            },
            "stage_3_select": {
                "disambiguation_logic": "Select specific procedure concepts over general action or finding concepts",
                "preferred_hierarchy": "procedure",
                "reject_hierarchies": ["finding", "qualifier value"]
            }
        })
        rule_id += 1

        # Negated Findings
        self.structured_rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Negated_Finding",
            "stage_1_span": {
                "label": "Negated_Finding",
                "description": "Clinical findings explicitly noted as absent, including 'no', 'without', 'denies', or negative descriptors like 'afebrile'"
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Extract core finding and search for both positive and negative/absent variants"
            },
            "stage_3_select": {
                "disambiguation_logic": "Select concepts explicitly representing absence, negation, or normal variants of the finding",
                "preferred_hierarchy": "finding",
                "reject_hierarchies": []
            }
        })
        rule_id += 1

        # Functional Status
        self.structured_rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Functional_Status",
            "stage_1_span": {
                "label": "Functional_Status",
                "description": "Patient functional abilities, activity levels, mobility status, or independence in activities"
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Search for functional status and ability concepts matching the described activity level"
            },
            "stage_3_select": {
                "disambiguation_logic": "Prefer finding concepts describing patient status over procedure concepts for interventions",
                "preferred_hierarchy": "finding",
                "reject_hierarchies": ["regime/therapy"]
            }
        })
        rule_id += 1

        # Diagnostic Findings
        self.structured_rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Diagnostic_Finding",
            "stage_1_span": {
                "label": "Diagnostic_Finding",
                "description": "Diagnoses, pathological conditions, disorders, or disease states"
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Search for disorder and disease concepts, including synonymous clinical terms"
            },
            "stage_3_select": {
                "disambiguation_logic": "Prefer disorder concepts over finding or procedure concepts for diagnosed conditions",
                "preferred_hierarchy": "disorder",
                "reject_hierarchies": []
            }
        })
        rule_id += 1

        # Anatomical References
        self.structured_rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Anatomical_Structure",
            "stage_1_span": {
                "label": "Anatomical_Structure",
                "description": "Body parts, organs, anatomical locations, or body systems"
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Search for anatomical structure concepts including common clinical terminology"
            },
            "stage_3_select": {
                "disambiguation_logic": "Select body structure concepts over procedure or finding concepts when referring to anatomy",
                "preferred_hierarchy": "body structure",
                "reject_hierarchies": []
            }
        })
        rule_id += 1

        # Clinical Context Terms
        self.structured_rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Clinical_Context",
            "stage_1_span": {
                "label": "Clinical_Context",
                "description": "Clinical settings, care contexts, or situational descriptors like admission status or care location"
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Search for situation, context, or regime/therapy concepts matching the clinical context"
            },
            "stage_3_select": {
                "disambiguation_logic": "Select situation or regime/therapy concepts for care contexts",
                "preferred_hierarchy": "situation",
                "reject_hierarchies": []
            }
        })
        rule_id += 1

        # Implicit Concept Mapping
        self.structured_rules.append({
            "rule_id": f"R{rule_id}",
            "concept_type": "Implicit_Concept",
            "stage_1_span": {
                "label": "Implicit_Concept",
                "description": "Terms where the clinical meaning differs from literal interpretation, requiring domain knowledge for correct mapping"
            },
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": "Search for the clinically implied concept rather than the literal text meaning"
            },
            "stage_3_select": {
                "disambiguation_logic": "Select concepts based on clinical interpretation rather than literal text matching",
                "preferred_hierarchy": None,
                "reject_hierarchies": []
            }
        })
        rule_id += 1

    def _generate_annotation_map(self):
        """Map each annotation to applicable rules."""
        for ann in self.annotations:
            ann_id = ann['id']
            applicable_rules = []

            # Add applicable G-rules
            applicable_rules.extend(["G1", "G2", "G3", "G5", "G7"])  # Base rules

            # Check for specific G-rule conditions
            if self._is_negated(ann['span_text']):
                applicable_rules.append("G4")
            if self._is_implicit_concept(ann['span_text'], ann['concept_name']):
                applicable_rules.append("G6")

            # Add applicable R-rules based on annotation characteristics
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
        concept_type = rule['concept_type']
        span = ann['span_text']
        section = ann['section']
        hierarchy = ann['hierarchy']

        if concept_type == "Physical_Exam_Abbreviation":
            return section == 'Physical Exam' and self._is_abbreviation(span)
        elif concept_type == "Allergy_Medication":
            return section == 'Allergies'
        elif concept_type == "Clinical_Procedure":
            return hierarchy == 'procedure' or 'procedure' in ann['concept_name'].lower()
        elif concept_type == "Negated_Finding":
            return self._is_negated(span)
        elif concept_type == "Functional_Status":
            func_keywords = ['ambula', 'activity', 'mobile', 'independent', 'walking']
            return any(kw in span.lower() or kw in ann['concept_name'].lower() for kw in func_keywords)
        elif concept_type == "Diagnostic_Finding":
            return hierarchy == 'disorder'
        elif concept_type == "Anatomical_Structure":
            return hierarchy == 'body structure'
        elif concept_type == "Clinical_Context":
            return hierarchy in ['situation', 'regime/therapy', 'environment']
        elif concept_type == "Implicit_Concept":
            return self._is_implicit_concept(span, ann['concept_name'])

        return False

    def generate_output(self):
        """Generate the final comprehensive JSON output."""
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

        print(f"Analyzing {self.file_path}...")

        # Parse annotations
        clean_text = self.parse_inline_annotations(text)
        print(f"Found {len(self.annotations)} annotations")

        # Analyze sections
        self.analyze_sections(clean_text)

        # Count annotations by section
        section_counts = Counter(ann['section'] for ann in self.annotations)
        print("\nAnnotations by section:")
        for section, count in sorted(section_counts.items()):
            print(f"  {section}: {count}")

        # Count by hierarchy
        hierarchy_counts = Counter(ann['hierarchy'] for ann in self.annotations)
        print("\nAnnotations by hierarchy:")
        for hierarchy, count in sorted(hierarchy_counts.items(), key=lambda x: -x[1]):
            print(f"  {hierarchy}: {count}")

        # Analyze patterns
        self.analyze_patterns()

        # Generate output
        return self.generate_output()


def main():
    file_path = REPO_ROOT / "inline_annotated_note.txt"
    analyzer = ComprehensiveAnnotationAnalyzer(file_path)
    result = analyzer.analyze()

    # Save result
    output_path = REPO_ROOT / "scripts" / "rules_output.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nAnalysis complete. Results saved to {output_path}")
    print(f"\nSummary:")
    print(f"  Total annotations: {len(analyzer.annotations)}")
    print(f"  Mappings generated: {len(result['mappings'])}")
    print(f"  General rules: {len(result['g_rules'])}")
    print(f"  Structured rules: {len(result['structured_rules'])}")
    print(f"  Complex mappings found: {len(analyzer.complex_mappings)}")
    print(f"  Implicit concepts found: {len(analyzer.implicit_concepts)}")


if __name__ == "__main__":
    main()