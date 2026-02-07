#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from scripts.glinker.l2_dictionary import ExactDictionaryMatcher, L2Candidate


class FuzzyRetriever(Protocol):
    def retrieve(
        self,
        mention: str,
        *,
        top_k: int = 50,
        l1_type: str | None = None,
        fuzziness: str = "AUTO",
    ) -> list[L2Candidate]:
        ...


@dataclass(frozen=True)
class HybridL2Config:
    top_k_exact: int = 50
    top_k_fuzzy: int = 50
    top_k_final: int = 50
    exact_short_circuit_unique: bool = True
    exact_decisive_min_score: float = 0.0
    exact_decisive_min_margin: float = 0.0
    exact_ambiguity_max_candidates: int = 1
    fallback_on_no_exact: bool = True
    fallback_on_ambiguous_exact: bool = True
    fuzziness: str = "AUTO"


@dataclass(frozen=True)
class HybridL2Result:
    mention: str
    l1_type: str | None
    route: str
    used_fallback: bool
    exact_candidates: tuple[L2Candidate, ...]
    fuzzy_candidates: tuple[L2Candidate, ...]
    final_candidates: tuple[L2Candidate, ...]


class HybridL2Retriever:
    def __init__(
        self,
        *,
        exact_matcher: ExactDictionaryMatcher,
        fuzzy_retriever: FuzzyRetriever | None,
        config: HybridL2Config | None = None,
    ):
        self.exact_matcher = exact_matcher
        self.fuzzy_retriever = fuzzy_retriever
        self.config = config or HybridL2Config()

    def _is_exact_decisive(self, exact: list[L2Candidate]) -> bool:
        if not exact:
            return False
        top1 = exact[0]
        if top1.score < self.config.exact_decisive_min_score:
            return False
        if len(exact) == 1:
            return True
        if len(exact) > int(self.config.exact_ambiguity_max_candidates):
            return False
        top2 = exact[1]
        margin = top1.score - top2.score
        return margin >= self.config.exact_decisive_min_margin

    @staticmethod
    def _merge_candidates(
        exact: list[L2Candidate],
        fuzzy: list[L2Candidate],
        *,
        top_k_final: int,
    ) -> list[L2Candidate]:
        out: list[L2Candidate] = []
        seen_concepts: set[str] = set()
        for c in exact:
            if c.concept_id in seen_concepts:
                continue
            out.append(c)
            seen_concepts.add(c.concept_id)
            if 0 < top_k_final <= len(out):
                return out
        for c in fuzzy:
            if c.concept_id in seen_concepts:
                continue
            out.append(c)
            seen_concepts.add(c.concept_id)
            if 0 < top_k_final <= len(out):
                return out
        return out

    def retrieve(self, mention: str, *, l1_type: str | None = None) -> HybridL2Result:
        exact = self.exact_matcher.lookup(
            mention,
            top_k=self.config.top_k_exact,
            l1_type=l1_type,
        )
        decisive = self._is_exact_decisive(exact)
        if decisive and self.config.exact_short_circuit_unique:
            final = exact[: self.config.top_k_final] if self.config.top_k_final > 0 else list(exact)
            return HybridL2Result(
                mention=mention,
                l1_type=l1_type,
                route="exact_short_circuit",
                used_fallback=False,
                exact_candidates=tuple(exact),
                fuzzy_candidates=tuple(),
                final_candidates=tuple(final),
            )

        need_fallback = False
        route = "exact_only"
        if not exact and self.config.fallback_on_no_exact:
            need_fallback = True
            route = "fuzzy_only"
        elif exact and not decisive and self.config.fallback_on_ambiguous_exact:
            need_fallback = True
            route = "exact_plus_fuzzy"
        elif exact and not decisive:
            route = "exact_ambiguous_no_fallback"

        fuzzy: list[L2Candidate] = []
        if need_fallback and self.fuzzy_retriever is not None:
            fuzzy = self.fuzzy_retriever.retrieve(
                mention,
                top_k=self.config.top_k_fuzzy,
                l1_type=l1_type,
                fuzziness=self.config.fuzziness,
            )

        final = self._merge_candidates(exact, fuzzy, top_k_final=self.config.top_k_final)
        if not final:
            route = "none"
        return HybridL2Result(
            mention=mention,
            l1_type=l1_type,
            route=route,
            used_fallback=bool(need_fallback and self.fuzzy_retriever is not None),
            exact_candidates=tuple(exact),
            fuzzy_candidates=tuple(fuzzy),
            final_candidates=tuple(final),
        )

