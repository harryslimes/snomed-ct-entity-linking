from __future__ import annotations

import unittest

from scripts.glinker.l2_elasticsearch import build_alias_query, parse_search_hits


class TestL2Elasticsearch(unittest.TestCase):
    def test_build_query_with_filter(self):
        q = build_alias_query("History of diabetes", top_k=7, l1_type="finding")
        self.assertEqual(q["size"], 7)
        bool_q = q["query"]["bool"]
        self.assertIn("filter", bool_q)
        self.assertEqual(bool_q["filter"][0]["term"]["l1_type"], "finding")
        self.assertIn("should", bool_q)

    def test_parse_hits(self):
        payload = {
            "hits": {
                "hits": [
                    {
                        "_score": 12.3,
                        "_source": {
                            "concept_id": "111",
                            "l1_type": "finding",
                            "alias": "diabetes mellitus",
                        },
                    },
                    {
                        "_score": 10.2,
                        "_source": {
                            "concept_id": "211",
                            "l1_type": "procedure",
                            "alias": "cabg",
                        },
                    },
                ]
            }
        }
        cands = parse_search_hits(payload)
        self.assertEqual(len(cands), 2)
        self.assertEqual(cands[0].concept_id, "111")
        self.assertEqual(cands[0].method, "l2_es_fuzzy")


if __name__ == "__main__":
    unittest.main()

