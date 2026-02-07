from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.l2_dictionary import ExactDictionaryMatcher


class TestExactDictionaryMatcher(unittest.TestCase):
    def test_lookup_and_sort(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "l2_exact.tsv"
            with path.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp, delimiter="\t")
                writer.writerow(["alias", "concept_id", "l1_type", "priority"])
                writer.writerow(["hypertension", "111", "finding", "0"])
                writer.writerow(["hypertension", "112", "finding", "2"])
                writer.writerow(["hypertension", "211", "procedure", "1"])

            matcher = ExactDictionaryMatcher.from_tsv(path)
            cands = matcher.lookup("  Hypertension ", top_k=10)
            self.assertEqual([c.concept_id for c in cands], ["111", "211", "112"])

            finding = matcher.lookup("hypertension", l1_type="finding")
            self.assertEqual([c.concept_id for c in finding], ["111", "112"])
            self.assertTrue(all(c.method == "l2_exact" for c in finding))


if __name__ == "__main__":
    unittest.main()

