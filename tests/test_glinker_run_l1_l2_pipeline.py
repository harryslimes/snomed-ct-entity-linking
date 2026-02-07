from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.run_l1_l2_pipeline import main


class TestRunL1L2Pipeline(unittest.TestCase):
    def test_cli_no_es_with_explicit_mention(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spans = root / "spans.csv"
            exact = root / "exact.tsv"
            out_jsonl = root / "out.jsonl"
            out_flat = root / "out.csv"

            with spans.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["mention_id", "note_id", "start_char", "end_char", "mention", "l1_type"])
                writer.writerow(["m1", "n1", "0", "12", "hypertension", "finding"])
                writer.writerow(["m2", "n1", "15", "20", "unknown", "finding"])

            with exact.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp, delimiter="\t")
                writer.writerow(["alias", "concept_id", "l1_type", "priority"])
                writer.writerow(["hypertension", "111", "finding", "0"])

            rc = main(
                [
                    "--l1-spans-csv",
                    str(spans),
                    "--exact-dict-tsv",
                    str(exact),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--out-flat-csv",
                    str(out_flat),
                    "--no-es",
                ]
            )
            self.assertEqual(rc, 0)
            recs = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(recs), 2)
            by_id = {r["mention_id"]: r for r in recs}
            self.assertEqual(by_id["m1"]["route"], "exact_short_circuit")
            self.assertEqual(by_id["m1"]["final_candidates"][0]["concept_id"], "111")
            self.assertEqual(by_id["m2"]["route"], "none")

    def test_cli_extracts_mention_from_notes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            spans = root / "spans.csv"
            notes = root / "notes.csv"
            exact = root / "exact.tsv"
            out_jsonl = root / "out.jsonl"

            note_text = "Patient has hypertension and diabetes."
            # hypertension span in note_text
            start = note_text.index("hypertension")
            end = start + len("hypertension")

            with spans.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["mention_id", "note_id", "start_char", "end_char", "l1_type"])
                writer.writerow(["m1", "n1", str(start), str(end), "finding"])

            with notes.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["note_id", "text"])
                writer.writerow(["n1", note_text])

            with exact.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp, delimiter="\t")
                writer.writerow(["alias", "concept_id", "l1_type", "priority"])
                writer.writerow(["hypertension", "111", "finding", "0"])

            rc = main(
                [
                    "--l1-spans-csv",
                    str(spans),
                    "--notes-csv",
                    str(notes),
                    "--exact-dict-tsv",
                    str(exact),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--no-es",
                ]
            )
            self.assertEqual(rc, 0)
            rec = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
            self.assertEqual(rec["mention"], "hypertension")
            self.assertEqual(rec["route"], "exact_short_circuit")
            self.assertEqual(rec["final_candidates"][0]["concept_id"], "111")


if __name__ == "__main__":
    unittest.main()

