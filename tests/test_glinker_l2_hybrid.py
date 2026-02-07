from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.l2_dictionary import ExactDictionaryMatcher, L2Candidate
from scripts.glinker.l2_hybrid import HybridL2Config, HybridL2Retriever


class _FakeFuzzy:
    def __init__(self, results: list[L2Candidate]):
        self.results = results
        self.calls = 0

    def retrieve(
        self,
        mention: str,
        *,
        top_k: int = 50,
        l1_type: str | None = None,
        fuzziness: str = "AUTO",
    ) -> list[L2Candidate]:
        self.calls += 1
        rows = list(self.results)
        if l1_type is not None:
            rows = [r for r in rows if r.l1_type == l1_type]
        if top_k > 0:
            return rows[:top_k]
        return rows


def _make_exact(path: Path, rows: list[list[str]]) -> ExactDictionaryMatcher:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["alias", "concept_id", "l1_type", "priority"])
        writer.writerows(rows)
    return ExactDictionaryMatcher.from_tsv(path)


class TestHybridL2Retriever(unittest.TestCase):
    def test_unique_exact_short_circuit(self):
        with tempfile.TemporaryDirectory() as td:
            exact = _make_exact(
                Path(td) / "exact.tsv",
                [["hypertension", "111", "finding", "0"]],
            )
            fake = _FakeFuzzy(
                [
                    L2Candidate("900", "finding", "hpertension", 5.0, "l2_es_fuzzy"),
                ]
            )
            r = HybridL2Retriever(
                exact_matcher=exact,
                fuzzy_retriever=fake,
                config=HybridL2Config(),
            ).retrieve("hypertension", l1_type="finding")

            self.assertEqual(r.route, "exact_short_circuit")
            self.assertEqual(fake.calls, 0)
            self.assertEqual([c.concept_id for c in r.final_candidates], ["111"])

    def test_ambiguous_exact_falls_back(self):
        with tempfile.TemporaryDirectory() as td:
            exact = _make_exact(
                Path(td) / "exact.tsv",
                [
                    ["dm", "111", "finding", "0"],
                    ["dm", "112", "finding", "1"],
                ],
            )
            fake = _FakeFuzzy(
                [
                    L2Candidate("112", "finding", "dm2", 9.0, "l2_es_fuzzy"),
                    L2Candidate("113", "finding", "diabetes", 8.0, "l2_es_fuzzy"),
                ]
            )
            r = HybridL2Retriever(
                exact_matcher=exact,
                fuzzy_retriever=fake,
                config=HybridL2Config(exact_ambiguity_max_candidates=1),
            ).retrieve("dm", l1_type="finding")

            self.assertEqual(r.route, "exact_plus_fuzzy")
            self.assertEqual(fake.calls, 1)
            self.assertEqual([c.concept_id for c in r.final_candidates], ["111", "112", "113"])

    def test_no_exact_fuzzy_only(self):
        with tempfile.TemporaryDirectory() as td:
            exact = _make_exact(Path(td) / "exact.tsv", [["htn", "111", "finding", "0"]])
            fake = _FakeFuzzy([L2Candidate("211", "procedure", "cabg", 7.0, "l2_es_fuzzy")])
            r = HybridL2Retriever(
                exact_matcher=exact,
                fuzzy_retriever=fake,
                config=HybridL2Config(),
            ).retrieve("cabg", l1_type="procedure")

            self.assertEqual(r.route, "fuzzy_only")
            self.assertEqual(fake.calls, 1)
            self.assertEqual([c.concept_id for c in r.final_candidates], ["211"])

    def test_ambiguous_exact_no_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            exact = _make_exact(
                Path(td) / "exact.tsv",
                [
                    ["dm", "111", "finding", "0"],
                    ["dm", "112", "finding", "1"],
                ],
            )
            fake = _FakeFuzzy([L2Candidate("113", "finding", "diabetes", 8.0, "l2_es_fuzzy")])
            r = HybridL2Retriever(
                exact_matcher=exact,
                fuzzy_retriever=fake,
                config=HybridL2Config(fallback_on_ambiguous_exact=False),
            ).retrieve("dm", l1_type="finding")

            self.assertEqual(r.route, "exact_ambiguous_no_fallback")
            self.assertEqual(fake.calls, 0)
            self.assertEqual([c.concept_id for c in r.final_candidates], ["111", "112"])


if __name__ == "__main__":
    unittest.main()

