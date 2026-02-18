"""SNOMED CT subsumption utilities.

Provides ancestor lookup, descendant testing, and LCA computation
using the pre-built subsumption index (snomed_index/subsumption.pkl).

Usage:
    from snomed_subsumption import SubsumptionIndex
    idx = SubsumptionIndex.load()
    idx.is_descendant(233604007, 106048009)  # pneumonia under respiratory finding?
    idx.find_lca([29857009, 21522001])  # LCA of chest pain + abdominal pain
"""
from __future__ import annotations

import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = REPO_ROOT / "snomed_index" / "subsumption.pkl"


class SubsumptionIndex:
    """SNOMED CT subsumption index for ancestor/descendant queries."""

    def __init__(
        self,
        ancestors: dict[int, set[int]],
        parents: dict[int, set[int]],
        concept_names: dict[int, str],
    ) -> None:
        self.ancestors = ancestors
        self.parents = parents
        self.concept_names = concept_names

    @classmethod
    def load(cls, path: Path | str = DEFAULT_INDEX_PATH) -> SubsumptionIndex:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            ancestors=data["ancestors"],
            parents=data["parents"],
            concept_names=data["concept_names"],
        )

    def is_descendant(self, concept_id: int, ancestor_id: int) -> bool:
        """Check if concept_id is a descendant of ancestor_id."""
        if concept_id == ancestor_id:
            return True
        return ancestor_id in self.ancestors.get(concept_id, set())

    def is_descendant_of_any(self, concept_id: int, ancestor_ids: list[int]) -> bool:
        """Check if concept_id is a descendant of any of the given ancestors."""
        if concept_id in ancestor_ids:
            return True
        anc = self.ancestors.get(concept_id, set())
        return bool(anc & set(ancestor_ids))

    def get_ancestors(self, concept_id: int) -> set[int]:
        """Get all ancestors of a concept (transitive closure of is-a)."""
        return self.ancestors.get(concept_id, set())

    def get_parents(self, concept_id: int) -> set[int]:
        """Get direct parents (one hop) of a concept."""
        return self.parents.get(concept_id, set())

    def get_name(self, concept_id: int) -> str:
        """Get concept name, or '?' if unknown."""
        return self.concept_names.get(concept_id, "?")

    def find_lca(self, concept_ids: list[int]) -> int | None:
        """Find the least common ancestor (most specific) of a set of concepts.

        Returns the common ancestor with the most ancestors itself (deepest
        in the hierarchy tree). Returns None if no common ancestor exists.
        """
        if not concept_ids:
            return None
        if len(concept_ids) == 1:
            return concept_ids[0]

        # Intersect: each concept's ancestors + itself
        common = self.ancestors.get(concept_ids[0], set()).copy()
        common.add(concept_ids[0])
        for cid in concept_ids[1:]:
            anc = self.ancestors.get(cid, set()).copy()
            anc.add(cid)
            common &= anc

        if not common:
            return None

        # Most specific = most ancestors (deepest in tree)
        return max(common, key=lambda c: len(self.ancestors.get(c, set())))

    def find_lca_with_depth(
        self, concept_ids: list[int], min_depth: int = 3
    ) -> list[tuple[int, str, int]]:
        """Find common ancestors sorted by depth (most specific first).

        Returns list of (concept_id, concept_name, depth) tuples.
        Useful for the rule agent to pick an appropriate level of specificity.
        """
        if not concept_ids:
            return []

        common = self.ancestors.get(concept_ids[0], set()).copy()
        common.add(concept_ids[0])
        for cid in concept_ids[1:]:
            anc = self.ancestors.get(cid, set()).copy()
            anc.add(cid)
            common &= anc

        results = []
        for c in common:
            depth = len(self.ancestors.get(c, set()))
            if depth >= min_depth:
                results.append((c, self.get_name(c), depth))

        results.sort(key=lambda x: -x[2])  # most specific first
        return results

    def match_rule_applies_to(
        self,
        concept_id: int,
        section: str | None,
        rule_applies_to: dict,
    ) -> bool:
        """Check if an annotation matches a rule's applies_to criteria.

        Args:
            concept_id: The annotation's gold SNOMED concept ID.
            section: The section header the annotation appears in (or None).
            rule_applies_to: The rule's applies_to dict with fields:
                - ancestor_concept_ids: list[int] (OR logic)
                - sections: list[str] or null (case-insensitive substring match)
                - span_pattern: str or null (not evaluated here)

        Returns True if the annotation matches ALL non-null criteria.
        """
        # Check ancestor subsumption (required)
        ancestor_ids = rule_applies_to.get("ancestor_concept_ids", [])
        if ancestor_ids:
            if not self.is_descendant_of_any(concept_id, ancestor_ids):
                return False

        # Check section filter (optional)
        rule_sections = rule_applies_to.get("sections")
        if rule_sections and section:
            section_lower = section.lower()
            if not any(s.lower() in section_lower for s in rule_sections):
                return False
        elif rule_sections and not section:
            # Rule requires specific sections but annotation has no section
            return False

        return True
