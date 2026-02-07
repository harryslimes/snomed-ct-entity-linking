from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.run_l1_inference import (
    extract_l1_spans_for_note,
    normalize_l1_type,
    run_inference,
)


class _FakeModel:
    def __init__(self, out):
        self.out = out

    def predict_entities(self, text, labels=None, threshold=None):
        return list(self.out)


class TestRunL1Inference(unittest.TestCase):
    def test_normalize_l1_type(self):
        self.assertEqual(normalize_l1_type("finding"), "finding")
        self.assertEqual(normalize_l1_type("body structure"), "body_structure")
        self.assertEqual(normalize_l1_type("morphologic abnormality"), "body_structure")
        self.assertEqual(normalize_l1_type("unknown"), None)

    def test_extract_spans(self):
        text = "Patient has hypertension and diabetes."
        out = [
            {
                "start": text.index("hypertension"),
                "end": text.index("hypertension") + len("hypertension"),
                "text": "hypertension",
                "label": "finding",
                "score": 0.9,
            }
        ]
        spans = extract_l1_spans_for_note(
            model=_FakeModel(out),
            note_id="n1",
            text=text,
            labels=["finding", "procedure", "body_structure"],
            threshold=0.4,
            window_chars=0,
            window_overlap_chars=256,
            strict_label_filter=True,
        )
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].mention, "hypertension")
        self.assertEqual(spans[0].l1_type, "finding")

    def test_run_inference_with_fake_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            notes_csv = root / "notes.csv"
            out_csv = root / "spans.csv"

            with notes_csv.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["note_id", "text"])
                writer.writerow(["n1", "Patient has hypertension."])

            text = "Patient has hypertension."
            fake_out = [
                {
                    "start": text.index("hypertension"),
                    "end": text.index("hypertension") + len("hypertension"),
                    "text": "hypertension",
                    "label": "finding",
                    "score": 0.85,
                }
            ]
            stats = run_inference(
                notes_csv=notes_csv,
                model_path="unused",
                out_spans_csv=out_csv,
                model=_FakeModel(fake_out),
            )
            self.assertEqual(stats["notes"], 1)
            self.assertEqual(stats["spans"], 1)

            with out_csv.open("r", encoding="utf-8", newline="") as fp:
                reader = csv.DictReader(fp)
                rows = list(reader)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["l1_type"], "finding")
            self.assertEqual(rows[0]["mention"], "hypertension")


if __name__ == "__main__":
    unittest.main()

