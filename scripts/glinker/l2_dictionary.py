#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from scripts.glinker.io import normalize_alias


@dataclass(frozen=True)
class L2Candidate:
    concept_id: str
    l1_type: str
    matched_alias: str
    score: float
    method: str


class ExactDictionaryMatcher:
    def __init__(self, alias_to_candidates: dict[str, list[L2Candidate]]):
        self.alias_to_candidates = alias_to_candidates

    @classmethod
    def from_tsv(
        cls,
        path: Path,
        *,
        max_candidates_per_alias: int = 0,
    ) -> "ExactDictionaryMatcher":
        alias_to_rows: dict[str, list[tuple[int, str, str]]] = {}
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp, delimiter="\t")
            for row in reader:
                alias = normalize_alias(row.get("alias") or "")
                concept_id = str(row.get("concept_id") or "").strip()
                l1_type = str(row.get("l1_type") or "").strip()
                if not alias or not concept_id:
                    continue
                try:
                    priority = int(str(row.get("priority") or "999"))
                except Exception:
                    priority = 999
                alias_to_rows.setdefault(alias, []).append((priority, concept_id, l1_type))

        alias_to_candidates: dict[str, list[L2Candidate]] = {}
        for alias, rows in alias_to_rows.items():
            rows_sorted = sorted(rows, key=lambda x: (x[0], x[1]))
            if max_candidates_per_alias > 0:
                rows_sorted = rows_sorted[: max_candidates_per_alias]
            cands = [
                L2Candidate(
                    concept_id=concept_id,
                    l1_type=l1_type,
                    matched_alias=alias,
                    score=1.0 / (1.0 + float(priority)),
                    method="l2_exact",
                )
                for priority, concept_id, l1_type in rows_sorted
            ]
            alias_to_candidates[alias] = cands

        return cls(alias_to_candidates=alias_to_candidates)

    def lookup(self, mention: str, *, top_k: int = 50, l1_type: str | None = None) -> list[L2Candidate]:
        alias = normalize_alias(mention)
        if not alias:
            return []
        cands = self.alias_to_candidates.get(alias, [])
        if l1_type is not None:
            cands = [c for c in cands if c.l1_type == l1_type]
        if top_k > 0:
            return cands[:top_k]
        return list(cands)

