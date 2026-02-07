from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.build_allowed_concepts import build_allowed_concepts_df


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


class TestAllowedConcepts(unittest.TestCase):
    def test_descendants_and_train_override(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snomed_dir = root / "rf2"
            term_dir = snomed_dir / "Snapshot" / "Terminology"

            _write_tsv(
                term_dir / "sct2_Concept_Snapshot_INT_20260101.txt",
                ["id", "effectiveTime", "active", "moduleId", "definitionStatusId"],
                [
                    ["404684003", "20260101", "1", "m", "d"],  # finding root
                    ["71388002", "20260101", "1", "m", "d"],  # procedure root
                    ["123037004", "20260101", "1", "m", "d"],  # body root
                    ["111", "20260101", "1", "m", "d"],  # finding child
                    ["112", "20260101", "1", "m", "d"],  # finding grandchild
                    ["211", "20260101", "1", "m", "d"],  # procedure child
                    ["311", "20260101", "1", "m", "d"],  # body child
                    ["999", "20260101", "1", "m", "d"],  # unrelated, train override
                ],
            )
            _write_tsv(
                term_dir / "sct2_Relationship_Snapshot_INT_20260101.txt",
                [
                    "id",
                    "effectiveTime",
                    "active",
                    "moduleId",
                    "sourceId",
                    "destinationId",
                    "relationshipGroup",
                    "typeId",
                    "characteristicTypeId",
                    "modifierId",
                ],
                [
                    ["r1", "20260101", "1", "m", "111", "404684003", "0", "116680003", "x", "x"],
                    ["r2", "20260101", "1", "m", "112", "111", "0", "116680003", "x", "x"],
                    ["r3", "20260101", "1", "m", "211", "71388002", "0", "116680003", "x", "x"],
                    ["r4", "20260101", "1", "m", "311", "123037004", "0", "116680003", "x", "x"],
                ],
            )

            train_ann = root / "train_annotations.csv"
            with train_ann.open("w", encoding="utf-8", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerow(["note_id", "start", "end", "concept_id", "span"])
                writer.writerow(["n1", "0", "4", "999", "mystery"])

            df = build_allowed_concepts_df(snomed_dir=snomed_dir, train_annotations=train_ann)
            by_id = {str(r["concept_id"]): r for _, r in df.iterrows()}

            self.assertIn("112", by_id)
            self.assertEqual(by_id["112"]["l1_type"], "finding")
            self.assertFalse(bool(by_id["112"]["train_override"]))

            self.assertIn("211", by_id)
            self.assertEqual(by_id["211"]["l1_type"], "procedure")

            self.assertIn("311", by_id)
            self.assertEqual(by_id["311"]["l1_type"], "body_structure")

            self.assertIn("999", by_id)
            self.assertTrue(bool(by_id["999"]["train_override"]))
            self.assertFalse(bool(by_id["999"]["is_descendant"]))


if __name__ == "__main__":
    unittest.main()

