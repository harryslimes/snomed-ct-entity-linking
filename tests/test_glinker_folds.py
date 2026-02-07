from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.glinker.make_folds import build_folds


class TestGlinkerFolds(unittest.TestCase):
    def test_fold_no_overlap_and_train_span_leakage(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            notes_path = root / "notes.csv"
            anns_path = root / "anns.csv"
            out_dir = root / "folds"

            notes = pd.DataFrame(
                {
                    "note_id": [f"n{i}" for i in range(10)],
                    "text": [f"note text {i}" for i in range(10)],
                }
            )
            anns = []
            for i in range(10):
                anns.append(
                    {
                        "note_id": f"n{i}",
                        "start": 0,
                        "end": 4,
                        "concept_id": 1000 + i,
                        "span": f"unique_span_{i}",
                    }
                )
            ann_df = pd.DataFrame(anns)
            notes.to_csv(notes_path, index=False)
            ann_df.to_csv(anns_path, index=False)

            manifest = build_folds(
                notes_csv=notes_path,
                annotations_csv=anns_path,
                out_dir=out_dir,
                n_folds=5,
                seed=42,
                write_full_train=True,
            )
            self.assertEqual(manifest["n_folds"], 5)

            for fold_idx in range(5):
                fold_dir = out_dir / f"fold_{fold_idx:02d}"
                train_notes = pd.read_csv(fold_dir / "train_notes.csv")
                val_notes = pd.read_csv(fold_dir / "val_notes.csv")
                train_ids = set(train_notes["note_id"].astype(str).tolist())
                val_ids = set(val_notes["note_id"].astype(str).tolist())
                self.assertFalse(train_ids & val_ids)

                val_ann = pd.read_csv(fold_dir / "val_annotations.csv")
                train_span_aliases = pd.read_csv(fold_dir / "train_span_aliases.tsv", sep="\t")
                val_spans = {s.lower() for s in val_ann["span"].astype(str)}
                train_spans = set(train_span_aliases["alias"].astype(str).tolist())
                self.assertFalse(val_spans & train_spans)

            self.assertTrue((out_dir / "full_train" / "train_span_aliases.tsv").exists())


if __name__ == "__main__":
    unittest.main()

