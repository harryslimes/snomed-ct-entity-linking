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
        ok, reason, row, _ = resolve_record(rec, cfg=ResolverConfig())
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
        ok, reason, _, _ = resolve_record(
            rec,
            cfg=ResolverConfig(min_top1_score_fuzzy=6.0),
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "low_score")

    def test_resolve_record_reject_low_l3_score(self):
        rec = {
            "mention_id": "m1",
            "note_id": "n1",
            "start_char": 0,
            "end_char": 5,
            "l1_type": "finding",
            "final_candidates": [
                {"concept_id": "111", "l1_type": "finding", "score": 0.1, "method": "l3_biencoder"}
            ],
        }
        ok, reason, _, _ = resolve_record(
            rec,
            cfg=ResolverConfig(min_top1_score_l3=0.2),
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "low_score")

    def test_resolve_record_route_specific_score_override(self):
        rec = {
            "mention_id": "m1",
            "note_id": "n1",
            "start_char": 0,
            "end_char": 5,
            "l1_type": "finding",
            "route": "none+l3+l4",
            "final_candidates": [
                {"concept_id": "111", "l1_type": "finding", "score": 3.0, "method": "l2_es_fuzzy"}
            ],
        }
        ok, reason, row, _ = resolve_record(
            rec,
            cfg=ResolverConfig(
                min_top1_score_fuzzy=6.0,
                route_min_top1_score={"none+l3+l4": 2.5},
            ),
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "accepted")
        assert row is not None
        self.assertEqual(row["route"], "none+l3+l4")

    def test_resolve_record_route_specific_margin_override(self):
        rec = {
            "mention_id": "m1",
            "note_id": "n1",
            "start_char": 0,
            "end_char": 5,
            "l1_type": "finding",
            "route": "exact_plus_fuzzy+l4",
            "final_candidates": [
                {"concept_id": "111", "l1_type": "finding", "score": 7.0, "method": "l2_es_fuzzy"},
                {"concept_id": "222", "l1_type": "finding", "score": 6.8, "method": "l2_es_fuzzy"},
            ],
        }
        ok, reason, _, _ = resolve_record(
            rec,
            cfg=ResolverConfig(
                min_top1_score_fuzzy=6.0,
                min_score_margin=0.0,
                route_min_score_margin={"exact_plus_fuzzy+l4": 0.5},
            ),
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "low_margin")

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
                    "route": "exact_short_circuit",
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
                    "route": "none+l3+l4",
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
                    "--route-min-top1-score",
                    "none+l3+l4=2.0",
                ]
            )
            self.assertEqual(rc, 0)
            with out_csv.open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["concept_id"], "111")
            self.assertEqual(rows[1]["concept_id"], "222")

            with out_decisions.open("r", encoding="utf-8", newline="") as fp:
                drows = list(csv.DictReader(fp))
            self.assertEqual(len(drows), 2)
            self.assertIn("route", drows[0])


if __name__ == "__main__":
    unittest.main()
