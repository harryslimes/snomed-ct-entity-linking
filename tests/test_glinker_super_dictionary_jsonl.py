from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.build_super_dictionary_jsonl import export_super_dictionary_jsonl


class TestSuperDictionaryJsonl(unittest.TestCase):
    def test_scope_and_dedup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            super_tsv = root / "super.tsv"
            with super_tsv.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp, delimiter="\t")
                writer.writerow(["snomed_concept_id", "term", "source", "source_detail"])
                writer.writerow(["111", "Hypertension", "athena", "concept_synonym"])
                writer.writerow(["111", "hypertension", "train_span", "annotation_span"])
                writer.writerow(["111", "Hypertensive disorder", "snomed_rf2", "FSN"])
                writer.writerow(["211", "CABG", "train_span", "annotation_span"])
                writer.writerow(["999", "Out of scope", "athena", "concept_synonym"])

            out_jsonl = root / "super.jsonl"
            allowed_map = {"111": "finding", "211": "procedure"}
            stats = export_super_dictionary_jsonl(
                super_dict_tsv=super_tsv,
                allowed_map=allowed_map,
                out_jsonl=out_jsonl,
                max_aliases_per_concept=2,
            )

            self.assertEqual(stats["concept_count"], 2)
            records = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
            by_id = {r["id"]: r for r in records}
            self.assertEqual(set(by_id.keys()), {"111", "211"})
            self.assertEqual(by_id["111"]["l1_type"], "finding")
            self.assertIn("hypertension", by_id["111"]["names"])
            self.assertLessEqual(len(by_id["111"]["names"]), 2)


if __name__ == "__main__":
    unittest.main()

