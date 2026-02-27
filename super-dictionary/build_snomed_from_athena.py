#!/usr/bin/env python3
"""Build a SNOMED synonym table from Athena OMOP files.

Filters CONCEPT.csv to vocabulary_id='SNOMED' and active (invalid_reason IS NULL),
then joins CONCEPT_SYNONYM.csv on concept_id.
Optionally filters synonyms to English (language_concept_id=4180186).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from engine import (
    iter_snomed_synonym_rows,
    write_snomed_synonym_tsv,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a SNOMED synonym table from Athena OMOP files. "
            "Filters CONCEPT to vocabulary_id='SNOMED' and active (invalid_reason IS NULL), "
            "then joins CONCEPT_SYNONYM on concept_id. "
            "Optionally filters synonyms to English (language_concept_id=4180186)."
        )
    )
    parser.add_argument(
        "--athena-dir",
        default="data/athena",
        help="Directory containing Athena OMOP CSVs (default: data/athena).",
    )
    parser.add_argument(
        "--out",
        default="data/interim/super_dictionary_snomed_athena.tsv",
        help="Output TSV path (default: data/interim/super_dictionary_snomed_athena.tsv).",
    )
    parser.add_argument(
        "--language-concept-id",
        default="4180186",
        help="Filter synonyms to this language_concept_id (default: 4180186 for English). "
        "Use 'any' to disable language filtering.",
    )
    parser.add_argument(
        "--include-concept-name",
        action="store_true",
        help="Include CONCEPT.concept_name as a synonym row (preferred term).",
    )
    parser.add_argument(
        "--max-terms-per-concept",
        type=int,
        default=0,
        help="If >0, cap unique terms per SNOMED concept (after normalization).",
    )
    args = parser.parse_args(argv)

    athena_dir = Path(args.athena_dir)
    concept_path = athena_dir / "CONCEPT.csv"
    synonym_path = athena_dir / "CONCEPT_SYNONYM.csv"

    if not concept_path.exists():
        print(f"Missing required file: {concept_path}", file=sys.stderr)
        return 2
    if not synonym_path.exists():
        print(
            f"Missing required file: {synonym_path}\n"
            "Re-download your Athena bundle with CONCEPT_SYNONYM.csv included.",
            file=sys.stderr,
        )
        return 2

    lang_filter = args.language_concept_id

    rows_iter = iter_snomed_synonym_rows(
        concept_path,
        synonym_path,
        language_concept_id=lang_filter,
        include_concept_name=args.include_concept_name,
    )
    if args.max_terms_per_concept > 0:
        seen_by_concept: dict[str, set[str]] = {}

        def _limited_rows():
            for row in rows_iter:
                seen = seen_by_concept.setdefault(row.snomed_concept_id, set())
                if len(seen) >= args.max_terms_per_concept:
                    continue
                norm = row.term.casefold()
                if norm in seen:
                    continue
                seen.add(norm)
                yield row

        rows_iter = _limited_rows()

    rows_written = write_snomed_synonym_tsv(Path(args.out), rows_iter)
    print(f"Wrote {rows_written:,} rows -> {args.out}")

    print(
        "Note: This Athena-only build approximates CHV by including all English synonyms "
        "for SNOMED concepts, but it does not include UMLS CHV term-type tags (TTY).",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
