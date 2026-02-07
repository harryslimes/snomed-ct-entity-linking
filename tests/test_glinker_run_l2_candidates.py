from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.run_l2_candidates import main


class TestRunL2Candidates(unittest.TestCase):
    def test_cli_no_es(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mentions = root / "mentions.csv"
            exact = root / "exact.tsv"
            out_jsonl = root / "out.jsonl"
            out_flat = root / "out.csv"

            with mentions.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["mention_id", "mention", "l1_type"])
                writer.writerow(["m1", "hypertension", "finding"])
                writer.writerow(["m2", "unknown term", "finding"])

            with exact.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp, delimiter="\t")
                writer.writerow(["alias", "concept_id", "l1_type", "priority"])
                writer.writerow(["hypertension", "111", "finding", "0"])

            rc = main(
                [
                    "--mentions-csv",
                    str(mentions),
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

            lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            recs = [json.loads(line) for line in lines]
            by_id = {r["mention_id"]: r for r in recs}
            self.assertEqual(by_id["m1"]["route"], "exact_short_circuit")
            self.assertEqual(by_id["m2"]["route"], "none")
            self.assertEqual(len(by_id["m1"]["final_candidates"]), 1)


if __name__ == "__main__":
    unittest.main()

