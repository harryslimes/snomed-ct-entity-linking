from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.resolve_l2_links import ResolverConfig, resolve_record, main


class TestResolveL2Links(unittest.TestCase):
    def test_resolve_record_accept_exact(self):
        rec = {
            "mention_id": "m1",
            "note_id": "n1",
            "start_char": 0,
            "end_char": 5,
            "l1_type": "finding",
            "final_candidates": [
                {"concept_id": "111", "l1_type": "finding", "score": 1.0, "method": "l2_exact"}
            ],
        }
        ok, reason, row = resolve_record(rec, cfg=ResolverConfig())
        self.assertTrue(ok)
        self.assertEqual(reason, "accepted")
        self.assertEqual(row["concept_id"], "111")

    def test_resolve_record_reject_low_fuzzy_score(self):
        rec = {
            "mention_id": "m1",
            "note_id": "n1",
            "start_char": 0,
            "end_char": 5,
            "l1_type": "finding",
            "final_candidates": [
                {"concept_id": "111", "l1_type": "finding", "score": 2.0, "method": "l2_es_fuzzy"}
            ],
        }
        ok, reason, _ = resolve_record(
            rec,
            cfg=ResolverConfig(min_top1_score_fuzzy=6.0),
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "low_score")

    def test_cli_writes_resolved_csv(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            in_jsonl = root / "cands.jsonl"
            out_csv = root / "resolved.csv"
            out_decisions = root / "decisions.csv"
            records = [
                {
                    "mention_id": "m1",
                    "note_id": "n1",
                    "start_char": 0,
                    "end_char": 5,
                    "l1_type": "finding",
                    "final_candidates": [
                        {"concept_id": "111", "l1_type": "finding", "score": 1.0, "method": "l2_exact"}
                    ],
                },
                {
                    "mention_id": "m2",
                    "note_id": "n1",
                    "start_char": 10,
                    "end_char": 20,
                    "l1_type": "finding",
                    "final_candidates": [
                        {"concept_id": "222", "l1_type": "finding", "score": 3.0, "method": "l2_es_fuzzy"}
                    ],
                },
            ]
            in_jsonl.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

            rc = main(
                [
                    "--candidates-jsonl",
                    str(in_jsonl),
                    "--out-resolved-csv",
                    str(out_csv),
                    "--out-decisions-csv",
                    str(out_decisions),
                    "--min-top1-score-fuzzy",
                    "6.0",
                ]
            )
            self.assertEqual(rc, 0)
            with out_csv.open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["concept_id"], "111")

            with out_decisions.open("r", encoding="utf-8", newline="") as fp:
                drows = list(csv.DictReader(fp))
            self.assertEqual(len(drows), 2)


if __name__ == "__main__":
    unittest.main()

