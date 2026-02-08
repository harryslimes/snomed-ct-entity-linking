from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.glinker.l3_biencoder import L3BiEncoderRetriever, build_l3_index
from scripts.glinker.l3_l4_enrich import L3Config, L3L4Config, L4Config, enrich_record
from scripts.glinker.l4_reranker import CrossEncoderReranker, L4RerankConfig


class TestL3L4Enrich(unittest.TestCase):
    def test_enrich_with_l3_and_l4_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            super_jsonl = root / "super.jsonl"
            idx_npz = root / "l3.npz"
            records = [
                {"id": "111", "l1_type": "finding", "names": ["hypertension"]},
                {"id": "222", "l1_type": "finding", "names": ["diabetes mellitus"]},
            ]
            super_jsonl.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            build_l3_index(
                super_dict_jsonl=super_jsonl,
                out_index_npz=idx_npz,
                backend="hash",
                model_path="hash",
            )
            l3 = L3BiEncoderRetriever(index_npz=idx_npz, model_path="hash", backend="hash")
            l4 = CrossEncoderReranker(
                model_path="hash",
                backend="hash",
                config=L4RerankConfig(top_n=1),
            )
            rec = {
                "mention_id": "m1",
                "note_id": "n1",
                "start_char": 0,
                "end_char": 12,
                "mention": "hypertension",
                "l1_type": "finding",
                "route": "none",
                "n_exact": 0,
                "n_fuzzy": 0,
                "final_candidates": [],
            }
            cfg = L3L4Config(
                l3=L3Config(enabled=True, trigger_mode="no_exact", top_k=5, max_merge_k=5),
                l4=L4Config(enabled=True, top_n=1, max_pool_k=5, trigger_mode="always", min_candidates=1),
            )
            out = enrich_record(rec, cfg=cfg, l3_retriever=l3, l4_reranker=l4)
            self.assertTrue(out["final_candidates"])
            self.assertEqual(out["final_candidates"][0]["concept_id"], "111")
            self.assertIn("+l3", out["route"])
            self.assertIn("+l4", out["route"])

    def test_l4_gating_skips_single_candidate_when_ambiguous_mode(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            super_jsonl = root / "super.jsonl"
            idx_npz = root / "l3.npz"
            records = [{"id": "111", "l1_type": "finding", "names": ["hypertension"]}]
            super_jsonl.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            build_l3_index(
                super_dict_jsonl=super_jsonl,
                out_index_npz=idx_npz,
                backend="hash",
                model_path="hash",
            )
            l3 = L3BiEncoderRetriever(index_npz=idx_npz, model_path="hash", backend="hash")
            l4 = CrossEncoderReranker(model_path="hash", backend="hash")
            rec = {
                "mention_id": "m1",
                "note_id": "n1",
                "start_char": 0,
                "end_char": 12,
                "mention": "hypertension",
                "l1_type": "finding",
                "route": "none",
                "n_exact": 0,
                "n_fuzzy": 0,
                "final_candidates": [],
            }
            cfg = L3L4Config(
                l3=L3Config(enabled=True, trigger_mode="no_exact", top_k=5, max_merge_k=5),
                l4=L4Config(enabled=True, top_n=1, max_pool_k=5, trigger_mode="ambiguous", min_candidates=2),
            )
            out = enrich_record(rec, cfg=cfg, l3_retriever=l3, l4_reranker=l4)
            self.assertTrue(out["final_candidates"])
            self.assertEqual(out["final_candidates"][0]["method"], "l3_biencoder")
            self.assertNotIn("+l4", out["route"])


if __name__ == "__main__":
    unittest.main()
