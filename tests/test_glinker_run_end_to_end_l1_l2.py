from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.glinker.run_end_to_end_l1_l2 import main


class TestRunEndToEndL1L2(unittest.TestCase):
    def test_orchestrates_l1_then_l2(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            notes_csv = root / "notes.csv"
            notes_csv.write_text("note_id,text\nn1,Patient has hypertension.\n", encoding="utf-8")
            out_jsonl = root / "out.jsonl"

            called = {"l1": 0, "l2": 0}

            def _fake_l1(argv):
                called["l1"] += 1
                self.assertIn("--notes-csv", argv)
                out_idx = argv.index("--out-spans-csv") + 1
                spans_path = Path(argv[out_idx])
                spans_path.parent.mkdir(parents=True, exist_ok=True)
                spans_path.write_text(
                    "mention_id,note_id,start_char,end_char,mention,l1_type\n"
                    "m1,n1,12,24,hypertension,finding\n",
                    encoding="utf-8",
                )
                return 0

            def _fake_l2(argv):
                called["l2"] += 1
                self.assertIn("--l1-spans-csv", argv)
                out_idx = argv.index("--out-jsonl") + 1
                Path(argv[out_idx]).write_text(
                    '{"mention_id":"m1","final_candidates":[{"concept_id":"111"}]}\n',
                    encoding="utf-8",
                )
                return 0

            with patch("scripts.glinker.run_end_to_end_l1_l2.run_l1_inference.main", side_effect=_fake_l1), patch(
                "scripts.glinker.run_end_to_end_l1_l2.run_l1_l2_pipeline.main", side_effect=_fake_l2
            ):
                rc = main(
                    [
                        "--notes-csv",
                        str(notes_csv),
                        "--l1-model-path",
                        "dummy-model",
                        "--out-jsonl",
                        str(out_jsonl),
                        "--no-es",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(called["l1"], 1)
            self.assertEqual(called["l2"], 1)
            self.assertTrue(out_jsonl.exists())

    def test_orchestrates_resolver_when_requested(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            notes_csv = root / "notes.csv"
            notes_csv.write_text("note_id,text\nn1,Patient has hypertension.\n", encoding="utf-8")
            out_jsonl = root / "out.jsonl"
            out_resolved = root / "resolved.csv"

            called = {"l1": 0, "l2": 0, "resolve": 0}

            def _fake_l1(argv):
                called["l1"] += 1
                out_idx = argv.index("--out-spans-csv") + 1
                spans_path = Path(argv[out_idx])
                spans_path.parent.mkdir(parents=True, exist_ok=True)
                spans_path.write_text(
                    "mention_id,note_id,start_char,end_char,mention,l1_type\n"
                    "m1,n1,12,24,hypertension,finding\n",
                    encoding="utf-8",
                )
                return 0

            def _fake_l2(argv):
                called["l2"] += 1
                out_idx = argv.index("--out-jsonl") + 1
                Path(argv[out_idx]).write_text(
                    '{"mention_id":"m1","note_id":"n1","start_char":12,"end_char":24,'
                    '"l1_type":"finding","final_candidates":[{"concept_id":"111","method":"l2_exact","score":1.0}]}\n',
                    encoding="utf-8",
                )
                return 0

            def _fake_resolve(argv):
                called["resolve"] += 1
                out_idx = argv.index("--out-resolved-csv") + 1
                Path(argv[out_idx]).write_text(
                    "note_id,start_char,end_char,concept_id\nn1,12,24,111\n",
                    encoding="utf-8",
                )
                return 0

            with patch("scripts.glinker.run_end_to_end_l1_l2.run_l1_inference.main", side_effect=_fake_l1), patch(
                "scripts.glinker.run_end_to_end_l1_l2.run_l1_l2_pipeline.main", side_effect=_fake_l2
            ), patch(
                "scripts.glinker.run_end_to_end_l1_l2.resolve_l2_links.main", side_effect=_fake_resolve
            ):
                rc = main(
                    [
                        "--notes-csv",
                        str(notes_csv),
                        "--l1-model-path",
                        "dummy-model",
                        "--out-jsonl",
                        str(out_jsonl),
                        "--out-resolved-csv",
                        str(out_resolved),
                        "--no-es",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(called["l1"], 1)
            self.assertEqual(called["l2"], 1)
            self.assertEqual(called["resolve"], 1)
            self.assertTrue(out_resolved.exists())


if __name__ == "__main__":
    unittest.main()
