from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.l3_biencoder import L3BiEncoderRetriever, build_l3_index


class TestL3BiEncoder(unittest.TestCase):
    def test_build_and_retrieve_hash_backend(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            super_jsonl = root / "super.jsonl"
            out_npz = root / "l3_index.npz"
            records = [
                {"id": "111", "l1_type": "finding", "names": ["hypertension", "high blood pressure"]},
                {"id": "222", "l1_type": "finding", "names": ["diabetes mellitus"]},
            ]
            super_jsonl.write_text(
                "".join(json.dumps(r) + "\n" for r in records),
                encoding="utf-8",
            )

            stats = build_l3_index(
                super_dict_jsonl=super_jsonl,
                out_index_npz=out_npz,
                backend="hash",
                model_path="hash",
            )
            self.assertEqual(stats["rows"], 3)
            self.assertTrue(out_npz.exists())

            retriever = L3BiEncoderRetriever(
                index_npz=out_npz,
                model_path="hash",
                backend="hash",
            )
            cands = retriever.retrieve("hypertension", l1_type="finding", top_k=5)
            self.assertTrue(cands)
            self.assertEqual(cands[0].concept_id, "111")


if __name__ == "__main__":
    unittest.main()
