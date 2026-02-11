import csv
import unittest
from collections import Counter
from pathlib import Path

from scripts.super_dictionary.engine import (
    COMMON_HEADERS,
    count_active_snomed_concepts,
    expand_coordination,
    generate_lookup_keys,
    get_spacy_model,
    iter_snomed_synonym_rows,
    segment_sections,
    find_term_matches,
)


ROOT = Path(__file__).resolve().parents[1]
ATHENA_DIR = ROOT / "data" / "athena"
CONCEPT_PATH = ATHENA_DIR / "CONCEPT.csv"
SYNONYM_PATH = ATHENA_DIR / "CONCEPT_SYNONYM.csv"
TRAIN_ANN_PATH = ROOT / "data" / "train_annotations.csv"


class TestAthenaIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CONCEPT_PATH.exists() or not SYNONYM_PATH.exists():
            raise unittest.SkipTest("Athena CONCEPT/CONCEPT_SYNONYM not found.")

    def test_active_snomed_count(self):
        count = count_active_snomed_concepts(CONCEPT_PATH)
        self.assertGreaterEqual(count, 300_000)
        self.assertLessEqual(
            count,
            450_000,
            "Active SNOMED count exceeds expected range; check source release/version.",
        )

    def test_gold_standard_synonyms(self):
        if not TRAIN_ANN_PATH.exists():
            raise unittest.SkipTest("train_annotations.csv not found.")

        counter = Counter()
        with TRAIN_ANN_PATH.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                concept_id = (row.get("concept_id") or "").strip()
                if concept_id:
                    counter[concept_id] += 1

        top_codes = [code for code, _ in counter.most_common(200)]
        target_codes = set(top_codes)
        syn_counts = {code: set() for code in target_codes}

        for row in iter_snomed_synonym_rows(
            CONCEPT_PATH,
            SYNONYM_PATH,
            language_concept_id="4180186",
            include_concept_name=True,
        ):
            if row.snomed_concept_id in target_codes:
                syn_counts[row.snomed_concept_id].add(row.term)

        eligible = [code for code in top_codes if len(syn_counts[code]) >= 3]
        self.assertGreaterEqual(
            len(eligible),
            50,
            "Fewer than 50 common SNOMED concepts have >=3 English synonyms. "
            "Consider adding UMLS MRCONSO for CHV coverage or expanding sources.",
        )
        for code in eligible[:50]:
            self.assertGreaterEqual(len(syn_counts[code]), 3)


class TestLinguisticRules(unittest.TestCase):
    def test_coordination_expansion(self):
        expansions = expand_coordination("Fracture of tibia and fibula")
        self.assertIn("Fracture of tibia", expansions)
        self.assertIn("Fracture of fibula", expansions)

    def test_coordination_with_left_right(self):
        expansions = expand_coordination("fracture of the left and right femur")
        self.assertIn("fracture of left femur", [e.lower() for e in expansions])
        self.assertIn("fracture of right femur", [e.lower() for e in expansions])

    def test_dependency_coordination(self):
        if get_spacy_model() is None:
            raise unittest.SkipTest("spaCy model not available")
        expansions = expand_coordination("fracture of tibia and fibula")
        self.assertIn("fracture of tibia", [e.lower() for e in expansions])
        self.assertIn("fracture of fibula", [e.lower() for e in expansions])

    def test_abbreviation_expansion(self):
        keys = generate_lookup_keys("pt c/o L leg pain")
        self.assertIn("patient complains of Left leg pain", keys)

    def test_abbreviation_permutations(self):
        keys = generate_lookup_keys("L femur fx")
        self.assertIn("Left femur fracture", keys)
        self.assertIn("fracture of Left femur", keys)
        self.assertIn("fx of L femur", keys)

    def test_stopword_transparent_matching(self):
        text = "The patient complains of fracture of the femur."
        terms = {"fracture femur": "C002"}
        matches = find_term_matches(
            text,
            terms,
            headers=COMMON_HEADERS,
            stopword_transparent=True,
        )
        self.assertEqual(len(matches), 1)


class TestSectionHeaderScoping(unittest.TestCase):
    def test_exclusion_by_header(self):
        text = (
            "FAMILY HISTORY:\n"
            "Diabetes.\n\n"
            "HISTORY OF PRESENT ILLNESS:\n"
            "Diabetes.\n"
        )
        terms = {"Diabetes": "C001"}
        matches = find_term_matches(
            text,
            terms,
            headers=COMMON_HEADERS,
            excluded_headers={"family history"},
        )
        self.assertEqual(len(matches), 1)
        self.assertIn("history of present illness", matches[0][3])

    def test_boundary_between_sections(self):
        text = (
            "PAST MEDICAL HISTORY:\n"
            "Hypertension.\n\n"
            "HISTORY OF PRESENT ILLNESS:\n"
            "Headache.\n"
        )
        sections = segment_sections(text, headers=COMMON_HEADERS)
        family = next(s for s in sections if s.header == "past medical history")
        hpi = next(s for s in sections if s.header == "history of present illness")
        hpi_header_start = text.lower().find("history of present illness")
        self.assertEqual(family.end, hpi_header_start)
        self.assertGreater(hpi.start, hpi_header_start)
