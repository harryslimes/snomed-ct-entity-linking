#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.io import has_alnum, normalize_alias


_SRC_PRIORITY = {
    ("athena", "concept_synonym"): 10,
    ("athena", "concept_name"): 20,
    ("snomed_rf2", "SYN"): 30,
    ("snomed_rf2", "FSN"): 40,
}


def _load_allowed_map(path: Path) -> dict[str, str]:
    if not path.exists() and path.suffix.lower() == ".parquet":
        csv_fallback = path.with_suffix(".csv")
        if csv_fallback.exists():
            path = csv_fallback
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    out: dict[str, str] = {}
    for cid, l1 in df[["concept_id", "l1_type"]].itertuples(index=False, name=None):
        c = str(cid).strip()
        t = str(l1).strip()
        if c and t:
            out[c] = t
    return out


def build_fold_exact_dictionary(
    *,
    super_dict_tsv: Path,
    allowed_concepts: Path,
    train_span_aliases_tsv: Path,
    out_tsv: Path,
) -> dict[str, int]:
    l1_map = _load_allowed_map(allowed_concepts)
    rows_by_key: dict[tuple[str, str], tuple[str, int]] = {}

    # Keep only static terminology aliases (no global train_span rows).
    with super_dict_tsv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            concept_id = str(row.get("snomed_concept_id") or "").strip()
            if concept_id not in l1_map:
                continue
            source = str(row.get("source") or "").strip()
            source_detail = str(row.get("source_detail") or "").strip()
            if source == "train_span":
                continue
            pr = _SRC_PRIORITY.get((source, source_detail), 99)
            alias = normalize_alias(row.get("term") or "")
            if not alias or not has_alnum(alias):
                continue
            key = (alias, concept_id)
            prev = rows_by_key.get(key)
            if prev is None or pr < prev[1]:
                rows_by_key[key] = (l1_map[concept_id], pr)

    # Add leakage-safe fold-specific train spans at top priority.
    with train_span_aliases_tsv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            concept_id = str(row.get("concept_id") or "").strip()
            if concept_id not in l1_map:
                continue
            alias = normalize_alias(row.get("alias") or "")
            if not alias or not has_alnum(alias):
                continue
            key = (alias, concept_id)
            prev = rows_by_key.get(key)
            if prev is None or 0 < prev[1]:
                rows_by_key[key] = (l1_map[concept_id], 0)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    aliases: set[str] = set()
    concepts: set[str] = set()
    with out_tsv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["alias", "concept_id", "l1_type", "priority"])
        items = sorted(
            rows_by_key.items(),
            key=lambda kv: (kv[0][0], kv[1][1], kv[0][1]),
        )
        for (alias, concept_id), (l1_type, pr) in items:
            writer.writerow([alias, concept_id, l1_type, pr])
            n_rows += 1
            aliases.add(alias)
            concepts.add(concept_id)
    return {
        "rows_written": n_rows,
        "unique_aliases": len(aliases),
        "unique_concepts": len(concepts),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build leakage-safe exact dictionary for one fold by combining "
            "static terminology aliases + fold-specific train span aliases."
        )
    )
    ap.add_argument("--super-dict-tsv", default="data/interim/super_dictionary_full.tsv")
    ap.add_argument("--allowed-concepts", default="data/interim/glinker/allowed_concepts.csv")
    ap.add_argument("--train-span-aliases-tsv", required=True)
    ap.add_argument("--out-tsv", required=True)
    args = ap.parse_args(argv)

    stats = build_fold_exact_dictionary(
        super_dict_tsv=Path(args.super_dict_tsv),
        allowed_concepts=Path(args.allowed_concepts),
        train_span_aliases_tsv=Path(args.train_span_aliases_tsv),
        out_tsv=Path(args.out_tsv),
    )
    print(f"rows_written: {stats['rows_written']:,}")
    print(f"unique_aliases: {stats['unique_aliases']:,}")
    print(f"unique_concepts: {stats['unique_concepts']:,}")
    print(f"out_tsv: {args.out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
