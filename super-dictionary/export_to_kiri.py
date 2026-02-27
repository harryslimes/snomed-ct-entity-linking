#!/usr/bin/env python3
"""Convert super_dictionary_full.tsv into KIRI's flattened_terminology_syn_* format.

Maps SNOMED concept codes to concept IDs via hierarchy lookup and outputs a CSV
compatible with the KIRI dictionary matcher.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _detect_delimiter(path: Path) -> str:
    with path.open("rb") as fp:
        sample = fp.read(4096)
    tab_count = sample.count(b"\t")
    comma_count = sample.count(b",")
    return "\t" if tab_count >= comma_count else ","


def load_hierarchy_map(flattened_path: Path) -> dict[int, str]:
    hierarchy = {}
    delim = _detect_delimiter(flattened_path)
    with flattened_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter=delim)
        for row in reader:
            try:
                cid = int(row["concept_id"])
            except Exception:
                continue
            h = (row.get("hierarchy") or "").strip()
            if h:
                hierarchy[cid] = h
    return hierarchy


def export_kiri_synonyms(super_path: Path, flattened_path: Path, out_path: Path) -> int:
    hierarchy_map = load_hierarchy_map(flattened_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[int, str]] = set()
    rows_written = 0

    with super_path.open("r", encoding="utf-8", newline="") as fp_in, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as fp_out:
        reader = csv.DictReader(fp_in, delimiter="\t")
        writer = csv.writer(fp_out)
        writer.writerow(["concept_id", "concept_name", "hierarchy", "type"])

        for row in reader:
            concept_code = (row.get("snomed_concept_id") or "").strip()
            term = (row.get("term") or "").strip()
            if not concept_code or not term:
                continue
            try:
                concept_id = int(concept_code)
            except Exception:
                continue

            hierarchy = hierarchy_map.get(concept_id)
            if not hierarchy:
                continue

            key = (concept_id, term)
            if key in seen:
                continue
            seen.add(key)

            source_detail = (row.get("source_detail") or "").strip().upper()
            term_type = "FSN" if source_detail == "FSN" else "SYN"
            writer.writerow([concept_id, term, hierarchy, term_type])
            rows_written += 1

    return rows_written


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert super_dictionary_full.tsv into KIRI's flattened_terminology_syn_* format."
        )
    )
    parser.add_argument(
        "--super-dict",
        default="data/interim/super_dictionary_full.tsv",
        help="Path to super_dictionary_full.tsv",
    )
    parser.add_argument(
        "--flattened-terminology",
        default="1st Place/data/interim/flattened_terminology.csv",
        help="KIRI flattened_terminology.csv path for hierarchy mapping",
    )
    parser.add_argument(
        "--out",
        default="1st Place/data/interim/flattened_terminology_syn_super.csv",
        help="Output KIRI synonym file",
    )
    args = parser.parse_args(argv)

    super_path = Path(args.super_dict)
    flattened_path = Path(args.flattened_terminology)
    out_path = Path(args.out)

    if not super_path.exists():
        print(f"Missing super dictionary: {super_path}", file=sys.stderr)
        return 2
    if not flattened_path.exists():
        print(f"Missing flattened terminology: {flattened_path}", file=sys.stderr)
        return 2

    rows_written = export_kiri_synonyms(super_path, flattened_path, out_path)
    print(f"Wrote {rows_written:,} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
