from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.build_es_index import iter_alias_entries, write_exact_dictionary_tsv


class TestBuildEsIndex(unittest.TestCase):
    def test_exact_dictionary_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            super_jsonl = root / "super.jsonl"
            rows = [
                {
                    "id": "111",
                    "l1_type": "finding",
                    "names": ["Hypertension", "  hypertension  ", "HTN"],
                    "sources": ["athena", "train_span"],
                },
                {
                    "id": "211",
                    "l1_type": "procedure",
                    "names": ["CABG", "Coronary artery bypass graft"],
                    "sources": ["snomed_rf2"],
                },
            ]
            super_jsonl.write_text(
                "".join(json.dumps(r) + "\n" for r in rows),
                encoding="utf-8",
            )
            entries = list(iter_alias_entries(super_jsonl))
            self.assertTrue(any(e.alias == "hypertension" for e in entries))
            self.assertTrue(any(e.alias == "cabg" for e in entries))

            out_tsv = root / "exact.tsv"
            stats = write_exact_dictionary_tsv(entries, out_tsv, max_candidates_per_alias=1)
            self.assertGreater(stats["unique_aliases"], 0)

            with out_tsv.open("r", encoding="utf-8", newline="") as fp:
                reader = csv.DictReader(fp, delimiter="\t")
                out_rows = list(reader)
            aliases = [r["alias"] for r in out_rows]
            self.assertIn("hypertension", aliases)
            self.assertIn("cabg", aliases)


if __name__ == "__main__":
    unittest.main()

