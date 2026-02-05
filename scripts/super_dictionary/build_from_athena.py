#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path


def _detect_delimiter(path: Path) -> str:
    with path.open("rb") as fp:
        sample = fp.read(4096)
    # Athena exports are typically tab-delimited but be tolerant.
    tab_count = sample.count(b"\t")
    comma_count = sample.count(b",")
    return "\t" if tab_count >= comma_count else ","


def _open_csv(path: Path):
    delim = _detect_delimiter(path)
    fp = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(fp, delimiter=delim)
    return fp, reader


def _norm_term(term: str) -> str:
    return " ".join(term.strip().split())


def _iter_concepts(concept_path: Path, needed_vocab_ids: set[str]):
    fp, reader = _open_csv(concept_path)
    try:
        for row in reader:
            yield row
    finally:
        fp.close()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CHV-focused SNOMED synonym table from OHDSI Athena OMOP vocabulary files. "
            "Uses CONCEPT.csv + CONCEPT_RELATIONSHIP.csv (and CONCEPT_SYNONYM.csv if present)."
        )
    )
    parser.add_argument(
        "--athena-dir",
        default="data/athena",
        help="Directory containing Athena OMOP CSVs (default: data/athena).",
    )
    parser.add_argument(
        "--out",
        default="data/interim/super_dictionary_chv_athena.tsv",
        help="Output TSV path (default: data/interim/super_dictionary_chv_athena.tsv).",
    )
    parser.add_argument(
        "--list-vocabs",
        action="store_true",
        help="Scan CONCEPT.csv and print vocabulary_id counts, then exit.",
    )
    parser.add_argument(
        "--list-relationships",
        action="store_true",
        help="Scan CONCEPT_RELATIONSHIP.csv and print relationship_id counts, then exit.",
    )
    parser.add_argument(
        "--relationship-id",
        default="Maps to",
        help="Relationship to use for mapping to SNOMED (default: 'Maps to').",
    )
    parser.add_argument(
        "--source-vocab",
        default="CHV",
        help="Source vocabulary_id to pull terms from (default: CHV).",
    )
    parser.add_argument(
        "--target-vocab",
        default="SNOMED",
        help="Target vocabulary_id to map to (default: SNOMED).",
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
    rel_path = athena_dir / "CONCEPT_RELATIONSHIP.csv"
    syn_path = athena_dir / "CONCEPT_SYNONYM.csv"

    if not concept_path.exists():
        print(f"Missing required file: {concept_path}", file=sys.stderr)
        return 2
    if not rel_path.exists():
        print(f"Missing required file: {rel_path}", file=sys.stderr)
        return 2

    source_vocab = args.source_vocab
    target_vocab = args.target_vocab

    print(f"Athena dir:          {athena_dir}")
    print(f"Source vocab_id:     {source_vocab}")
    print(f"Target vocab_id:     {target_vocab}")
    print(f"Relationship_id:     {args.relationship_id}")
    print(f"CONCEPT_SYNONYM.csv: {'found' if syn_path.exists() else 'not found'}")

    if args.list_vocabs:
        vocab_counts: dict[str, int] = defaultdict(int)
        fp, reader = _open_csv(concept_path)
        rows = 0
        try:
            for row in reader:
                rows += 1
                vocab = row.get("vocabulary_id") or ""
                if vocab:
                    vocab_counts[vocab] += 1
                if rows % 2_000_000 == 0:
                    print(f"Scanned {rows:,} CONCEPT rows...", file=sys.stderr)
        finally:
            fp.close()

        print(f"Scanned {rows:,} rows. Top vocabulary_id values:")
        for vocab, count in sorted(vocab_counts.items(), key=lambda kv: kv[1], reverse=True)[:50]:
            print(f"{vocab}\t{count}")
        return 0

    if args.list_relationships:
        rel_counts: dict[str, int] = defaultdict(int)
        fp, reader = _open_csv(rel_path)
        rows = 0
        try:
            for row in reader:
                rows += 1
                rel = row.get("relationship_id") or ""
                if rel:
                    rel_counts[rel] += 1
                if rows % 5_000_000 == 0:
                    print(f"Scanned {rows:,} REL rows...", file=sys.stderr)
        finally:
            fp.close()

        print(f"Scanned {rows:,} rows. Top relationship_id values:")
        for rel, count in sorted(rel_counts.items(), key=lambda kv: kv[1], reverse=True)[:50]:
            print(f"{rel}\t{count}")
        return 0

    # Pass 1: collect needed concept metadata.
    source_id_to_name: dict[int, str] = {}
    target_omop_id_to_code: dict[int, str] = {}

    concept_rows = 0
    for row in _iter_concepts(concept_path, {source_vocab, target_vocab}):
        concept_rows += 1
        try:
            vocab_id = row["vocabulary_id"]
        except KeyError:
            print(f"{concept_path} missing expected column vocabulary_id", file=sys.stderr)
            return 2
        if vocab_id != source_vocab and vocab_id != target_vocab:
            continue

        try:
            concept_id = int(row["concept_id"])
        except Exception:
            continue

        if vocab_id == source_vocab:
            name = row.get("concept_name") or ""
            name = _norm_term(name)
            if name:
                source_id_to_name[concept_id] = name
        else:
            code = row.get("concept_code") or ""
            code = code.strip()
            if code:
                target_omop_id_to_code[concept_id] = code

        if concept_rows % 2_000_000 == 0:
            print(
                f"Read {concept_rows:,} CONCEPT rows | "
                f"{len(source_id_to_name):,} {source_vocab} concepts | "
                f"{len(target_omop_id_to_code):,} {target_vocab} concepts",
                file=sys.stderr,
            )

    if not source_id_to_name:
        print(
            f"No {source_vocab} rows found in {concept_path}. "
            f"Did you include the {source_vocab} vocabulary in your Athena download? "
            f"(Tip: run with --list-vocabs to see what’s present.)",
            file=sys.stderr,
        )
        return 2
    if not target_omop_id_to_code:
        print(
            f"No {target_vocab} rows found in {concept_path}. "
            f"Did you include the {target_vocab} vocabulary in your Athena download?",
            file=sys.stderr,
        )
        return 2

    # Pass 2: map source OMOP IDs -> target SNOMED codes.
    source_ids = set(source_id_to_name.keys())
    mapped_source_ids: set[int] = set()
    target_code_to_source_ids: dict[str, set[int]] = defaultdict(set)

    fp, reader = _open_csv(rel_path)
    rel_rows = 0
    try:
        for row in reader:
            rel_rows += 1
            if row.get("relationship_id") != args.relationship_id:
                continue

            try:
                source_id = int(row["concept_id_1"])
                target_id = int(row["concept_id_2"])
            except Exception:
                continue

            if source_id not in source_ids:
                continue

            target_code = target_omop_id_to_code.get(target_id)
            if not target_code:
                continue

            target_code_to_source_ids[target_code].add(source_id)
            mapped_source_ids.add(source_id)

            if rel_rows % 5_000_000 == 0:
                print(
                    f"Read {rel_rows:,} REL rows | "
                    f"{len(mapped_source_ids):,} mapped {source_vocab} concepts | "
                    f"{len(target_code_to_source_ids):,} mapped {target_vocab} codes",
                    file=sys.stderr,
                )
    finally:
        fp.close()

    if not target_code_to_source_ids:
        print(
            f"No mappings found with relationship_id='{args.relationship_id}' from {source_vocab} to {target_vocab}.",
            file=sys.stderr,
        )
        return 2

    # Pass 3 (optional): load source synonyms if the file is available.
    source_id_to_synonyms: dict[int, list[str]] = defaultdict(list)
    if syn_path.exists():
        fp, reader = _open_csv(syn_path)
        syn_rows = 0
        try:
            for row in reader:
                syn_rows += 1
                try:
                    concept_id = int(row["concept_id"])
                except Exception:
                    continue
                if concept_id not in mapped_source_ids:
                    continue
                term = row.get("concept_synonym_name") or ""
                term = _norm_term(term)
                if term:
                    source_id_to_synonyms[concept_id].append(term)
                if syn_rows % 2_000_000 == 0:
                    print(
                        f"Read {syn_rows:,} CONCEPT_SYNONYM rows | "
                        f"{len(source_id_to_synonyms):,} {source_vocab} concepts with synonyms",
                        file=sys.stderr,
                    )
        finally:
            fp.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_terms = args.max_terms_per_concept

    # Output format: target_concept_code \t term \t source_vocab \t source_kind
    rows_written = 0
    with out_path.open("w", encoding="utf-8", newline="") as fp_out:
        fp_out.write("snomed_concept_id\tterm\tsource_vocab\tsource_kind\n")
        for snomed_code, source_ids_for_code in sorted(target_code_to_source_ids.items()):
            seen_norm: set[str] = set()
            unique_terms: list[tuple[str, str]] = []

            for source_id in source_ids_for_code:
                name = source_id_to_name.get(source_id)
                if name:
                    n = name.casefold()
                    if n not in seen_norm:
                        seen_norm.add(n)
                        unique_terms.append((name, "concept_name"))

                for syn in source_id_to_synonyms.get(source_id, []):
                    n = syn.casefold()
                    if n not in seen_norm:
                        seen_norm.add(n)
                        unique_terms.append((syn, "concept_synonym"))

            if max_terms > 0 and len(unique_terms) > max_terms:
                unique_terms = unique_terms[:max_terms]

            for term, kind in unique_terms:
                fp_out.write(f"{snomed_code}\t{term}\t{source_vocab}\t{kind}\n")
                rows_written += 1

    print(f"Wrote {rows_written:,} rows -> {out_path}")

    if not syn_path.exists():
        print(
            "Note: CONCEPT_SYNONYM.csv was not present, so only CHV concept_name values were used. "
            "Re-download your Athena bundle including CONCEPT_SYNONYM.csv to expand term coverage.",
            file=sys.stderr,
        )

    # Important nuance: Athena won't preserve UMLS-specific CHV term-type fields like MRCONSO.TTY.
    print(
        "Note: Athena CHV does not expose UMLS MRCONSO columns (e.g., TTY), so you cannot exactly "
        "filter to UMLS 'clinician-entered' term-types until you ingest MRCONSO.RRF.",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
