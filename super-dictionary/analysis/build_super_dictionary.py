#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.super_dictionary.engine import (  # noqa: E402
    SynonymRow,
    iter_snomed_synonym_rows,
    load_active_snomed_concepts,
)


@dataclass(frozen=True)
class SuperRow:
    snomed_concept_id: str
    term: str
    source: str
    source_detail: str


def _norm_term(term: str) -> str:
    return " ".join(term.strip().split())


def iter_snomed_rf2_rows(
    snomed_dir: Path,
    allowed_snomed_codes: set[str],
) -> Iterator[SuperRow]:
    # Snapshot/Terminology/sct2_Description_Snapshot-en_INT_YYYYMMDD.txt
    term_files = list((snomed_dir / "Snapshot" / "Terminology").glob("sct2_Description_Snapshot-en_INT_*.txt"))
    if not term_files:
        raise FileNotFoundError(
            f"No SNOMED description file found under {snomed_dir / 'Snapshot' / 'Terminology'}"
        )
    term_path = term_files[0]

    try:
        csv.field_size_limit(min(10**7, sys.maxsize))
    except OverflowError:
        csv.field_size_limit(sys.maxsize)

    with term_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            if row.get("active") != "1":
                continue
            concept_id = (row.get("conceptId") or "").strip()
            if concept_id not in allowed_snomed_codes:
                continue
            term = _norm_term(row.get("term") or "")
            if not term:
                continue
            type_id = row.get("typeId") or ""
            if type_id == "900000000000003001":
                source_detail = "FSN"
            elif type_id == "900000000000013009":
                source_detail = "SYN"
            else:
                source_detail = "OTHER"
            yield SuperRow(
                snomed_concept_id=concept_id,
                term=term,
                source="snomed_rf2",
                source_detail=source_detail,
            )


def iter_train_span_rows(train_annotations_path: Path) -> Iterator[SuperRow]:
    with train_annotations_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            concept_id = (row.get("concept_id") or "").strip()
            if not concept_id:
                continue
            span = _norm_term(row.get("span") or "")
            if not span:
                continue
            yield SuperRow(
                snomed_concept_id=concept_id,
                term=span,
                source="train_span",
                source_detail="annotation_span",
            )


def _dedupe_rows(rows: Iterable[SuperRow]) -> Iterator[SuperRow]:
    seen: dict[str, set[str]] = {}
    for row in rows:
        norm = row.term.casefold()
        seen_terms = seen.setdefault(row.snomed_concept_id, set())
        if norm in seen_terms:
            continue
        seen_terms.add(norm)
        yield row


def write_super_dictionary(path: Path, rows: Iterable[SuperRow]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fp:
        fp.write("snomed_concept_id\tterm\tsource\tsource_detail\n")
        for row in rows:
            fp.write(f"{row.snomed_concept_id}\t{row.term}\t{row.source}\t{row.source_detail}\n")
            count += 1
    return count


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a consolidated Super-Dictionary by merging Athena synonyms, SNOMED RF2 descriptions, "
            "and training annotation spans."
        )
    )
    parser.add_argument("--athena-dir", default="data/athena")
    parser.add_argument(
        "--snomed-dir",
        default="data/SnomedCT_InternationalRF2_PRODUCTION_20260101T120000Z",
    )
    parser.add_argument("--train-annotations", default="data/train_annotations.csv")
    parser.add_argument("--out", default="data/interim/super_dictionary_full.tsv")
    parser.add_argument("--language-concept-id", default="4180186")
    parser.add_argument("--include-concept-name", action="store_true")
    parser.add_argument("--no-athena", action="store_true")
    parser.add_argument("--no-snomed-rf2", action="store_true")
    parser.add_argument("--no-train-spans", action="store_true")
    args = parser.parse_args(argv)

    athena_dir = Path(args.athena_dir)
    concept_path = athena_dir / "CONCEPT.csv"
    synonym_path = athena_dir / "CONCEPT_SYNONYM.csv"

    if not concept_path.exists():
        print(f"Missing required file: {concept_path}", file=sys.stderr)
        return 2

    snomed_code_map, _ = load_active_snomed_concepts(
        concept_path, include_concept_name=False
    )
    allowed_snomed_codes = set(snomed_code_map.values())

    rows: list[SuperRow] = []

    if not args.no_athena:
        if not synonym_path.exists():
            print(
                f"Missing required file: {synonym_path}\n"
                "Re-download your Athena bundle with CONCEPT_SYNONYM.csv included.",
                file=sys.stderr,
            )
            return 2
        for row in iter_snomed_synonym_rows(
            concept_path,
            synonym_path,
            language_concept_id=args.language_concept_id,
            include_concept_name=args.include_concept_name,
        ):
            rows.append(
                SuperRow(
                    snomed_concept_id=row.snomed_concept_id,
                    term=row.term,
                    source="athena",
                    source_detail=row.source_kind,
                )
            )

    if not args.no_snomed_rf2:
        snomed_dir = Path(args.snomed_dir)
        rows.extend(iter_snomed_rf2_rows(snomed_dir, allowed_snomed_codes))

    if not args.no_train_spans:
        train_path = Path(args.train_annotations)
        if not train_path.exists():
            print(f"Missing train annotations: {train_path}", file=sys.stderr)
            return 2
        rows.extend(iter_train_span_rows(train_path))

    deduped = _dedupe_rows(rows)
    out_path = Path(args.out)
    count = write_super_dictionary(out_path, deduped)
    print(f"Wrote {count:,} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
