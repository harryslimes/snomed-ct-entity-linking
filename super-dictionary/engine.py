#!/usr/bin/env python3
"""Core engine for the Super Dictionary: CSV utilities, SNOMED concept loading,
linguistic rules (abbreviation expansion, coordination splitting, stopword-transparent
matching, fracture permutations), and section segmentation."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Optional


# ---------------------------------------------------------------------------
# Common clinical note section headers
# ---------------------------------------------------------------------------
COMMON_HEADERS = [
    "Allergies",
    "History of Present Illness",
    "Family History",
    "Name",
    "Major Surgical or Invasive Procedure",
    "Admission Date",
    "Discharge Disposition",
    "Past Medical History",
    "Attending",
    "Service",
    "Date of Birth",
    "Discharge Instructions",
    "Discharge Condition",
    "Chief Complaint",
    "Physical Exam",
    "Pertinent Results",
    "Discharge Medications",
    "Social History",
    "Followup Instructions",
    "Medications on Admission",
    "Discharge Diagnosis",
]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _detect_delimiter(path: Path) -> str:
    with path.open("rb") as fp:
        sample = fp.read(4096)
    tab_count = sample.count(b"\t")
    comma_count = sample.count(b",")
    return "\t" if tab_count >= comma_count else ","


def _open_csv(path: Path):
    delim = _detect_delimiter(path)
    fp = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(fp, delimiter=delim)
    return fp, reader


def _is_active_invalid_reason(value: str | None) -> bool:
    if value is None:
        return True
    value = value.strip()
    return value == "" or value.upper() == "NULL"


def _norm_term(term: str) -> str:
    return " ".join(term.strip().split())


# ---------------------------------------------------------------------------
# SNOMED concept loading from Athena CONCEPT.csv
# ---------------------------------------------------------------------------
def count_active_snomed_concepts(concept_path: Path) -> int:
    fp, reader = _open_csv(concept_path)
    count = 0
    try:
        for row in reader:
            if row.get("vocabulary_id") != "SNOMED":
                continue
            if not _is_active_invalid_reason(row.get("invalid_reason")):
                continue
            concept_code = (row.get("concept_code") or "").strip()
            if not concept_code:
                continue
            count += 1
    finally:
        fp.close()
    return count


def load_active_snomed_concepts(
    concept_path: Path, *, include_concept_name: bool = False
) -> tuple[dict[int, str], dict[int, str]]:
    snomed_concept_id_to_code: dict[int, str] = {}
    snomed_concept_id_to_name: dict[int, str] = {}

    fp, reader = _open_csv(concept_path)
    try:
        for row in reader:
            if row.get("vocabulary_id") != "SNOMED":
                continue
            if not _is_active_invalid_reason(row.get("invalid_reason")):
                continue

            try:
                concept_id = int(row["concept_id"])
            except Exception:
                continue

            concept_code = (row.get("concept_code") or "").strip()
            if not concept_code:
                continue

            snomed_concept_id_to_code[concept_id] = concept_code
            if include_concept_name:
                name = _norm_term(row.get("concept_name") or "")
                if name:
                    snomed_concept_id_to_name[concept_id] = name
    finally:
        fp.close()

    return snomed_concept_id_to_code, snomed_concept_id_to_name


# ---------------------------------------------------------------------------
# Synonym row data model and iterators
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SynonymRow:
    snomed_concept_id: str
    omop_concept_id: int
    term: str
    source_kind: str
    language_concept_id: str


def iter_snomed_synonym_rows(
    concept_path: Path,
    synonym_path: Path,
    *,
    language_concept_id: str = "4180186",
    include_concept_name: bool = False,
) -> Iterator[SynonymRow]:
    snomed_concept_id_to_code, snomed_concept_id_to_name = load_active_snomed_concepts(
        concept_path, include_concept_name=include_concept_name
    )

    if include_concept_name:
        for concept_id, term in snomed_concept_id_to_name.items():
            yield SynonymRow(
                snomed_concept_id=snomed_concept_id_to_code[concept_id],
                omop_concept_id=concept_id,
                term=term,
                source_kind="concept_name",
                language_concept_id="",
            )

    fp_syn, reader_syn = _open_csv(synonym_path)
    try:
        for row in reader_syn:
            try:
                concept_id = int(row["concept_id"])
            except Exception:
                continue

            snomed_code = snomed_concept_id_to_code.get(concept_id)
            if not snomed_code:
                continue

            if language_concept_id.lower() != "any":
                if (row.get("language_concept_id") or "").strip() != language_concept_id:
                    continue

            term = _norm_term(row.get("concept_synonym_name") or "")
            if not term:
                continue

            yield SynonymRow(
                snomed_concept_id=snomed_code,
                omop_concept_id=concept_id,
                term=term,
                source_kind="concept_synonym",
                language_concept_id=row.get("language_concept_id") or "",
            )
    finally:
        fp_syn.close()


def build_snomed_synonym_index(
    concept_path: Path,
    synonym_path: Path,
    *,
    language_concept_id: str = "4180186",
    include_concept_name: bool = False,
    max_terms_per_concept: int = 0,
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}

    for row in iter_snomed_synonym_rows(
        concept_path,
        synonym_path,
        language_concept_id=language_concept_id,
        include_concept_name=include_concept_name,
    ):
        terms = index.setdefault(row.snomed_concept_id, set())
        if max_terms_per_concept > 0 and len(terms) >= max_terms_per_concept:
            continue
        terms.add(row.term)

    return index


def write_snomed_synonym_tsv(
    output_path: Path,
    rows: Iterable[SynonymRow],
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        fp.write("snomed_concept_id\tomop_concept_id\tterm\tsource_kind\tlanguage_concept_id\n")
        for row in rows:
            fp.write(
                f"{row.snomed_concept_id}\t{row.omop_concept_id}\t{row.term}"
                f"\t{row.source_kind}\t{row.language_concept_id}\n"
            )
            rows_written += 1
    return rows_written


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------
DEFAULT_ABBREVIATIONS = {
    "pt": "patient",
    "c/o": "complains of",
    "l": "left",
    "r": "right",
    "fx": "fracture",
}

CASE_SENSITIVE_ABBREVIATIONS = {
    "L": "Left",
    "R": "Right",
}


def expand_abbreviations(
    text: str,
    abbreviations: Optional[dict[str, str]] = None,
    case_sensitive: Optional[dict[str, str]] = None,
) -> str:
    abbreviations = abbreviations or DEFAULT_ABBREVIATIONS
    case_sensitive = case_sensitive or CASE_SENSITIVE_ABBREVIATIONS

    tokens = text.split()
    expanded = []
    for token in tokens:
        if token in case_sensitive:
            expanded.append(case_sensitive[token])
            continue
        lower = token.lower()
        if lower in abbreviations:
            expanded.append(abbreviations[lower])
        else:
            expanded.append(token)
    return " ".join(expanded)


# ---------------------------------------------------------------------------
# Coordination splitting
# ---------------------------------------------------------------------------
def expand_coordination(text: str) -> list[str]:
    spacy_expansions = expand_coordination_dependency(text)

    if " and " not in text or " of " not in text:
        return spacy_expansions

    prefix, rest = text.split(" of ", 1)
    coord_match = re.match(
        r"(?:(?:the)\s+)?(left|right|bilateral)\s+and\s+(left|right|bilateral)\s+(.+)",
        rest,
        flags=re.IGNORECASE,
    )
    if coord_match:
        first = coord_match.group(1).strip()
        second = coord_match.group(2).strip()
        noun = coord_match.group(3).strip()
        if first and second and noun:
            return list(
                {
                    *spacy_expansions,
                    f"{prefix.strip()} of {first} {noun}",
                    f"{prefix.strip()} of {second} {noun}",
                }
            )

    if " and " not in rest:
        return spacy_expansions
    left, right = rest.split(" and ", 1)
    left = left.strip()
    right = right.strip()
    prefix = prefix.strip()
    if not left or not right or not prefix:
        return spacy_expansions
    return list(
        {
            *spacy_expansions,
            f"{prefix} of {left}",
            f"{prefix} of {right}",
        }
    )


@lru_cache(maxsize=1)
def _get_spacy_model():
    try:
        import spacy  # type: ignore
    except Exception:
        return None

    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except Exception:
        return None


def get_spacy_model():
    return _get_spacy_model()


def expand_coordination_dependency(text: str) -> list[str]:
    nlp = _get_spacy_model()
    if nlp is None:
        return []

    doc = nlp(text)
    expansions: set[str] = set()

    for token in doc:
        if token.dep_ != "conj":
            continue
        head = token.head
        group = [head] + [t for t in head.conjuncts]
        group_indices = sorted({t.i for t in group})
        if not group_indices:
            continue

        span_start = min(group_indices)
        span_end = max(group_indices)
        start_char = doc[span_start].idx
        end_char = doc[span_end].idx + len(doc[span_end])

        before = text[:start_char]
        after = text[end_char:]
        for alt in group:
            alt_text = alt.text
            candidate = f"{before}{alt_text}{after}".strip()
            if candidate and candidate != text:
                expansions.add(candidate)

    return list(expansions)


# ---------------------------------------------------------------------------
# Abbreviation variant generation
# ---------------------------------------------------------------------------
def expand_abbreviation_variants(
    text: str,
    abbreviations: Optional[dict[str, str]] = None,
    case_sensitive: Optional[dict[str, str]] = None,
    max_variants: int = 32,
) -> list[str]:
    abbreviations = abbreviations or DEFAULT_ABBREVIATIONS
    case_sensitive = case_sensitive or CASE_SENSITIVE_ABBREVIATIONS

    tokens = text.split()
    variants = {""}
    for token in tokens:
        options = [token]
        if token in case_sensitive:
            options.append(case_sensitive[token])
        lower = token.lower()
        if lower in abbreviations:
            options.append(abbreviations[lower])

        next_variants = set()
        for v in variants:
            for opt in options:
                combined = f"{v} {opt}".strip()
                next_variants.add(combined)
                if len(next_variants) >= max_variants:
                    break
            if len(next_variants) >= max_variants:
                break
        variants = next_variants
        if len(variants) >= max_variants:
            break

    return list(variants)


# ---------------------------------------------------------------------------
# Fracture phrase permutations
# ---------------------------------------------------------------------------
def permute_fracture_phrases(text: str) -> list[str]:
    permutations = []
    if "fracture" in text.lower():
        lower = text.lower()
        if lower.startswith("fracture of "):
            rest = text[len("fracture of ") :].strip()
            if rest:
                permutations.append(f"fx of {rest}")
        if lower.endswith(" fracture"):
            rest = text[: -len(" fracture")].strip()
            if rest:
                permutations.append(f"fracture of {rest}")
                permutations.append(f"fx of {rest}")
        if lower.startswith("fracture ") and " of " not in lower:
            rest = text[len("fracture ") :].strip()
            if rest:
                permutations.append(f"fracture of {rest}")
                permutations.append(f"fx of {rest}")
        if " fracture " in lower:
            parts = text.split(" fracture ", 1)
            if len(parts) == 2 and parts[0].strip():
                permutations.append(f"fracture of {parts[0].strip()} {parts[1].strip()}".strip())
    return permutations


# ---------------------------------------------------------------------------
# Lookup key generation (combines all transformations)
# ---------------------------------------------------------------------------
def generate_lookup_keys(text: str) -> list[str]:
    variants = expand_abbreviation_variants(text)
    keys: list[str] = []
    for variant in variants:
        keys.append(variant)
        keys.extend(expand_coordination(variant))
        keys.extend(permute_fracture_phrases(variant))
    # de-dup preserve order
    seen = set()
    ordered = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered


# ---------------------------------------------------------------------------
# Section segmentation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SectionSpan:
    header: str
    start: int
    end: int


def _compile_header_regex(headers: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(h) for h in headers]
    return re.compile(rf"(?m)^(?P<header>{'|'.join(escaped)})\s*:?\s*$", re.IGNORECASE)


def segment_sections(
    text: str,
    headers: Optional[list[str]] = None,
) -> list[SectionSpan]:
    headers = headers or COMMON_HEADERS
    pattern = _compile_header_regex(headers)

    matches = []
    for match in pattern.finditer(text):
        header = match.group("header")
        line_start = match.start()
        line_end = match.end()
        matches.append((line_start, line_end, header))

    matches.sort(key=lambda x: x[0])
    sections: list[SectionSpan] = []
    for i, (start, end, header) in enumerate(matches):
        section_start = end
        section_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        sections.append(
            SectionSpan(header=header.strip().lower(), start=section_start, end=section_end)
        )
    return sections


def get_section_for_pos(pos: int, sections: list[SectionSpan]) -> Optional[SectionSpan]:
    for section in sections:
        if section.start <= pos < section.end:
            return section
    return None


# ---------------------------------------------------------------------------
# Term matching in clinical text
# ---------------------------------------------------------------------------
def find_term_matches(
    text: str,
    terms: dict[str, str],
    *,
    headers: Optional[list[str]] = None,
    excluded_headers: Optional[set[str]] = None,
    case_insensitive: bool = True,
    stopword_transparent: bool = False,
    stopwords: Optional[set[str]] = None,
) -> list[tuple[int, int, str, str]]:
    sections = segment_sections(text, headers=headers)
    excluded_headers = {h.lower() for h in (excluded_headers or set())}
    stopwords = {s.lower() for s in (stopwords or {"patient", "complains", "of", "the", "a", "an"})}

    matches: list[tuple[int, int, str, str]] = []
    for term, concept_id in terms.items():
        if not term:
            continue
        flags = re.IGNORECASE if case_insensitive else 0
        if stopword_transparent:
            tokens = [t for t in term.split() if t]
            filtered = [t for t in tokens if t.lower() not in stopwords]
            if not filtered:
                filtered = tokens
            stopword_alt = "|".join(re.escape(s) for s in stopwords) or ""
            between = rf"(?:[\s,;:/-]+(?:{stopword_alt})\b)*[\s,;:/-]+"
            pattern_str = rf"\b{re.escape(filtered[0])}\b"
            for tok in filtered[1:]:
                pattern_str += between + rf"\b{re.escape(tok)}\b"
            pattern = re.compile(pattern_str, flags)
        else:
            pattern = re.compile(rf"\b{re.escape(term)}\b", flags)
        for match in pattern.finditer(text):
            section = get_section_for_pos(match.start(), sections)
            if section and section.header in excluded_headers:
                continue
            matches.append((match.start(), match.end(), concept_id, section.header if section else ""))
    return matches
