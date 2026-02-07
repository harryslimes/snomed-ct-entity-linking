from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.eval_cv import main


class TestEvalCV(unittest.TestCase):
    def test_eval_cv_gold_no_es(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            folds_dir = root / "folds"
            fold0 = folds_dir / "fold_00"
            fold0.mkdir(parents=True, exist_ok=True)

            # One-note validation fold.
            note_text = "Patient has hypertension."
            start = note_text.index("hypertension")
            end = start + len("hypertension")

            with (fold0 / "val_notes.csv").open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["note_id", "text"])
                writer.writerow(["n1", note_text])

            with (fold0 / "val_annotations.csv").open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["note_id", "start", "end", "concept_id", "span"])
                writer.writerow(["n1", start, end, 111, "hypertension"])

            exact_tsv = root / "l2_exact.tsv"
            with exact_tsv.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp, delimiter="\t")
                writer.writerow(["alias", "concept_id", "l1_type", "priority"])
                writer.writerow(["hypertension", "111", "finding", "0"])

            out_dir = root / "eval_out"
            rc = main(
                [
                    "--folds-dir",
                    str(folds_dir),
                    "--output-dir",
                    str(out_dir),
                    "--l1-source",
                    "gold",
                    "--exact-dict-tsv",
                    str(exact_tsv),
                    "--no-es",
                    "--resolver-min-top1-score-exact",
                    "0.0",
                    "--resolver-min-top1-score-fuzzy",
                    "0.0",
                ]
            )
            self.assertEqual(rc, 0)
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_folds"], 1)
            self.assertGreaterEqual(summary["macro_char_iou_mean"], 0.99)


if __name__ == "__main__":
    unittest.main()

