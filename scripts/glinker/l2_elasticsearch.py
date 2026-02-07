#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from scripts.glinker.io import normalize_alias
from scripts.glinker.l2_dictionary import L2Candidate


def build_alias_query(
    mention: str,
    *,
    top_k: int = 50,
    l1_type: str | None = None,
    fuzziness: str = "AUTO",
) -> dict[str, Any]:
    mention_norm = normalize_alias(mention)
    bool_query: dict[str, Any] = {
        "must": [
            {
                "match": {
                    "alias": {
                        "query": mention_norm,
                        "operator": "and",
                        "fuzziness": fuzziness,
                    }
                }
            }
        ],
        "should": [
            {"term": {"alias_exact": {"value": mention_norm, "boost": 12.0}}},
            {"match_phrase": {"alias": {"query": mention_norm, "boost": 4.0}}},
        ],
        "minimum_should_match": 0,
    }
    if l1_type is not None:
        bool_query["filter"] = [{"term": {"l1_type": l1_type}}]

    return {
        "size": int(top_k),
        "_source": ["concept_id", "l1_type", "alias", "source_count"],
        "query": {"bool": bool_query},
    }


def parse_search_hits(search_json: dict[str, Any]) -> list[L2Candidate]:
    hits = search_json.get("hits", {}).get("hits", [])
    out: list[L2Candidate] = []
    for hit in hits:
        src = hit.get("_source", {})
        concept_id = str(src.get("concept_id") or "").strip()
        if not concept_id:
            continue
        out.append(
            L2Candidate(
                concept_id=concept_id,
                l1_type=str(src.get("l1_type") or "").strip(),
                matched_alias=str(src.get("alias") or "").strip(),
                score=float(hit.get("_score") or 0.0),
                method="l2_es_fuzzy",
            )
        )
    return out


@dataclass(frozen=True)
class ElasticsearchConfig:
    base_url: str = "http://127.0.0.1:9200"
    index_name: str = "snomed_super_dict_v1"
    timeout_s: float = 10.0
    api_key: str | None = None


class ElasticsearchAliasRetriever:
    def __init__(self, config: ElasticsearchConfig):
        self.config = config
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        if config.api_key:
            self._session.headers.update({"Authorization": f"ApiKey {config.api_key}"})

    def close(self) -> None:
        self._session.close()

    def health_check(self) -> bool:
        url = f"{self.config.base_url.rstrip('/')}/_cluster/health"
        r = self._session.get(url, timeout=self.config.timeout_s)
        return r.status_code == 200

    def retrieve(
        self,
        mention: str,
        *,
        top_k: int = 50,
        l1_type: str | None = None,
        fuzziness: str = "AUTO",
    ) -> list[L2Candidate]:
        body = build_alias_query(
            mention,
            top_k=top_k,
            l1_type=l1_type,
            fuzziness=fuzziness,
        )
        url = (
            f"{self.config.base_url.rstrip('/')}/"
            f"{self.config.index_name}/_search"
        )
        r = self._session.post(url, json=body, timeout=self.config.timeout_s)
        r.raise_for_status()
        return parse_search_hits(r.json())

