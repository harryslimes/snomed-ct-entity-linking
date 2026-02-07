#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path


IS_A_TYPE_ID = "116680003"

ROOTS = {
    "404684003": "finding",
    "71388002": "procedure",
    "123037004": "body_structure",
}

ROOT_PRECEDENCE = ["404684003", "71388002", "123037004"]


def _find_snapshot_file(snomed_dir: Path, stem: str) -> Path:
    term_dir = snomed_dir / "Snapshot" / "Terminology"
    matches = sorted(term_dir.glob(f"{stem}_Snapshot*_*.txt"))
    if not matches:
        raise FileNotFoundError(f"Could not find {stem} snapshot under {term_dir}")
    return matches[0]


def concept_snapshot_path(snomed_dir: Path) -> Path:
    return _find_snapshot_file(snomed_dir, "sct2_Concept")


def relationship_snapshot_path(snomed_dir: Path) -> Path:
    return _find_snapshot_file(snomed_dir, "sct2_Relationship")


def load_active_concept_ids(concept_path: Path) -> set[str]:
    active: set[str] = set()
    with concept_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            if (row.get("active") or "").strip() != "1":
                continue
            cid = (row.get("id") or "").strip()
            if cid:
                active.add(cid)
    return active


def load_active_is_a_children_by_parent(
    relationship_path: Path,
    active_concepts: set[str],
) -> dict[str, set[str]]:
    children_by_parent: dict[str, set[str]] = defaultdict(set)
    with relationship_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            if (row.get("active") or "").strip() != "1":
                continue
            if (row.get("typeId") or "").strip() != IS_A_TYPE_ID:
                continue
            child = (row.get("sourceId") or "").strip()
            parent = (row.get("destinationId") or "").strip()
            if not child or not parent:
                continue
            if child not in active_concepts or parent not in active_concepts:
                continue
            children_by_parent[parent].add(child)
    return children_by_parent


def descendants_by_root(snomed_dir: Path, roots: dict[str, str] | None = None) -> dict[str, set[str]]:
    roots = roots or ROOTS
    concept_path = concept_snapshot_path(snomed_dir)
    relationship_path = relationship_snapshot_path(snomed_dir)
    active = load_active_concept_ids(concept_path)
    children_by_parent = load_active_is_a_children_by_parent(relationship_path, active)

    out: dict[str, set[str]] = {}
    for root in roots:
        seen: set[str] = set()
        q = deque([root])
        while q:
            current = q.popleft()
            if current in seen:
                continue
            seen.add(current)
            for child in children_by_parent.get(current, set()):
                if child not in seen:
                    q.append(child)
        out[root] = seen
    return out


def assign_l1_type(concept_id: str, descendants_map: dict[str, set[str]], roots: dict[str, str] | None = None) -> str | None:
    roots = roots or ROOTS
    for root in ROOT_PRECEDENCE:
        if concept_id in descendants_map.get(root, set()):
            return roots[root]
    for root, l1_type in roots.items():
        if concept_id in descendants_map.get(root, set()):
            return l1_type
    return None

