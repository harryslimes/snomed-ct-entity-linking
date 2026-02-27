#!/usr/bin/env python3
"""Convert old-challenge-split CSV data to GLiNER2 InputExample format.

Handles:
- SNOMED concept_id → 3-class mapping (finding/procedure/body) via semantic tags
- Chunking long clinical notes into ~400-token overlapping windows
- Annotation alignment to chunks
- Negative example strategy (entity type negatives, hard negative chunks)

Usage:
    python scripts/prepare_gliner2_data.py [--data-dir data/old-challenge-split]
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Semantic tag → 3-class mapping
# ---------------------------------------------------------------------------

TAG_TO_CLASS = {
    "finding": "medical finding, symptom, or disease",
    "disorder": "medical finding, symptom, or disease",
    "procedure": "procedure",
    "regime/therapy": "procedure",
    "body structure": "anatomical body part",
    "morphologic abnormality": "anatomical body part",
    "cell structure": "anatomical body part",
}

# ---------------------------------------------------------------------------
# 3-class entity descriptions (matching SNOMED semantic tag categories)
# ---------------------------------------------------------------------------

ENTITY_DESCRIPTIONS = {
    "medical finding, symptom, or disease": "Clinical findings, diagnoses, disorders, symptoms, and signs",
    "procedure": "Medical procedures, surgeries, therapies, and treatments",
    "anatomical body part": "Body structures, anatomical parts, and morphologic features",
}

ALL_ENTITY_TYPES = list(ENTITY_DESCRIPTIONS.keys())

# ---------------------------------------------------------------------------
# Lab table detection (v2, from test_lab_table_filter.py)
# ---------------------------------------------------------------------------

LAB_VALUE_PATTERN = re.compile(
    r"[A-Z][A-Za-z][A-Za-z0-9]{0,4}-\d+\.?\d*\*?"
)
MIN_LAB_VALUES_PER_LINE = 2
POST_MATCH_BUFFER = 15


def find_lab_regions(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character ranges for lab value clusters."""
    regions = []
    for line_match in re.finditer(r"[^\n]*\n?", text):
        line = line_match.group()
        line_offset = line_match.start()
        lab_matches = list(LAB_VALUE_PATTERN.finditer(line))
        if len(lab_matches) >= MIN_LAB_VALUES_PER_LINE:
            first_start = lab_matches[0].start() + line_offset
            last_end = lab_matches[-1].end() + line_offset
            region_end = min(last_end + POST_MATCH_BUFFER, line_match.end())
            regions.append((first_start, region_end))
    return regions


def merge_adjacent_regions(
    regions: list[tuple[int, int]], gap: int = 5,
) -> list[tuple[int, int]]:
    """Merge lab regions within `gap` characters of each other."""
    if not regions:
        return []
    merged = [list(regions[0])]
    for start, end in regions[1:]:
        if start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def annotation_in_lab_region(
    start: int, end: int, regions: list[tuple[int, int]],
) -> bool:
    """Check if annotation is fully contained within any lab region."""
    for r_start, r_end in regions:
        if start >= r_start and end <= r_end:
            return True
    return False

# ---------------------------------------------------------------------------
# Section detection (matches super-dictionary/engine.py COMMON_HEADERS)
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
    "Brief Hospital Course",
    "Facility",
    "Impression",
    "Sex",
]

# Sections that are known hard negatives (high medical vocabulary, few annotations)
HARD_NEGATIVE_SECTIONS = {
    "discharge medications",
    "medications on admission",
    "discharge instructions",
    "followup instructions",
    "discharge condition",
    "discharge disposition",
}



def _compile_header_regex(headers: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(h) for h in headers]
    return re.compile(rf"(?m)^(?P<header>{'|'.join(escaped)})\s*:?\s*$", re.IGNORECASE)


def segment_sections(
    text: str,
    headers: list[str] | None = None,
) -> list[tuple[str, int, int]]:
    """Segment a note into (header, section_start, section_end) tuples.

    section_start is the char position after the header line.
    section_end is the char position where the next header starts (or end of text).
    The header is lowercased.

    A leading "preamble" section (before the first header) gets header="preamble".
    """
    headers = headers or COMMON_HEADERS
    pattern = _compile_header_regex(headers)

    matches = []
    for match in pattern.finditer(text):
        header = match.group("header")
        matches.append((match.start(), match.end(), header.strip().lower()))

    matches.sort(key=lambda x: x[0])

    sections = []
    # Preamble before first header
    if matches:
        if matches[0][0] > 0:
            sections.append(("preamble", 0, matches[0][0]))
    else:
        # No headers found — entire note is one section
        return [("preamble", 0, len(text))]

    for i, (hdr_start, hdr_end, header) in enumerate(matches):
        section_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        sections.append((header, hdr_end, section_end))

    return sections


# ---------------------------------------------------------------------------
# SNOMED concept → class mapping
# ---------------------------------------------------------------------------

def load_sctid_to_tag(
    project_root: Path,
) -> dict[int, str]:
    """Load sctid_to_tag from SNOMED RF2 files using the recall_analysis loader.

    Returns dict mapping SNOMED concept ID → semantic tag string.
    """
    sys.path.insert(0, str(project_root))
    from snomed_ct_entity_linking.recall_analysis.config import Config
    from snomed_ct_entity_linking.recall_analysis.snomed_loader import (
        FSN_TYPE_ID,
        parse_semantic_tag,
    )

    cfg = Config(project_root=project_root)

    # Load descriptions and extract semantic tags from FSNs
    desc = pd.read_csv(
        cfg.description_file, sep="\t",
        dtype={"id": int, "active": int, "conceptId": int, "typeId": int},
        usecols=["id", "active", "conceptId", "typeId", "term"],
    )
    desc = desc[(desc["active"] == 1) & (desc["typeId"] == FSN_TYPE_ID)]
    desc["semantic_tag"] = desc["term"].apply(parse_semantic_tag)

    sctid_to_tag: dict[int, str] = {}
    for _, row in desc.iterrows():
        cid = int(row["conceptId"])
        if row["semantic_tag"]:
            sctid_to_tag[cid] = row["semantic_tag"]

    # Also load legacy descriptions for retired concepts
    if cfg.legacy_description_file.exists():
        legacy_desc = pd.read_csv(
            cfg.legacy_description_file, sep="\t",
            dtype={"id": int, "active": int, "conceptId": int, "typeId": int},
            usecols=["id", "active", "conceptId", "typeId", "term"],
        )
        legacy_desc = legacy_desc[
            (legacy_desc["active"] == 1) & (legacy_desc["typeId"] == FSN_TYPE_ID)
        ]
        legacy_desc["semantic_tag"] = legacy_desc["term"].apply(parse_semantic_tag)
        for _, row in legacy_desc.iterrows():
            cid = int(row["conceptId"])
            if cid not in sctid_to_tag and row["semantic_tag"]:
                sctid_to_tag[cid] = row["semantic_tag"]

    print(f"Loaded {len(sctid_to_tag):,} concept → semantic tag mappings")
    return sctid_to_tag


def map_concept_to_class(
    concept_ids: pd.Series,
    sctid_to_tag: dict[int, str],
) -> pd.Series:
    """Map concept_ids to 3-class labels via SNOMED semantic tags."""
    default_class = "medical finding, symptom, or disease"
    classes = []
    unmapped = set()
    for cid in concept_ids:
        cid = int(cid)
        tag = sctid_to_tag.get(cid)
        if tag and tag in TAG_TO_CLASS:
            classes.append(TAG_TO_CLASS[tag])
        else:
            classes.append(default_class)
            unmapped.add(cid)
    if unmapped:
        print(f"  WARNING: {len(unmapped)} concept IDs had no matching semantic tag, "
              f"defaulted to '{default_class}'")
    return pd.Series(classes, index=concept_ids.index)


def map_concept_to_fine_class(
    concept_id: int,
    sctid_to_tag: dict[int, str],
) -> str:
    """Map a single concept ID to a 3-class label via SNOMED semantic tag."""
    tag = sctid_to_tag.get(concept_id, "")
    return TAG_TO_CLASS.get(tag, "medical finding, symptom, or disease")


# ---------------------------------------------------------------------------
# Section-aware chunking
# ---------------------------------------------------------------------------

def chunk_span(
    text: str,
    span_start: int,
    span_end: int,
    window_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[tuple[int, int]]:
    """Chunk a character span [span_start, span_end) of text into overlapping windows.

    Returns list of (char_start, char_end) tuples in note-level coordinates.
    """
    section_text = text[span_start:span_end]
    tokens = []
    for m in re.finditer(r"\S+", section_text):
        # Store in note-level coordinates
        tokens.append((m.start() + span_start, m.end() + span_start))

    if not tokens:
        return [(span_start, span_end)]

    chunks = []
    stride = max(1, window_tokens - overlap_tokens)
    i = 0
    while i < len(tokens):
        end_idx = min(i + window_tokens, len(tokens))
        char_start = tokens[i][0]
        char_end = tokens[end_idx - 1][1]
        chunks.append((char_start, char_end))

        if end_idx >= len(tokens):
            break
        i += stride

    return chunks


def chunk_note_by_section(
    text: str,
    window_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[tuple[str, int, int]]:
    """Chunk a note into overlapping windows, respecting section boundaries.

    Returns list of (section_header, char_start, char_end) tuples.
    Chunks never cross section boundaries. Each chunk carries its section header.
    """
    sections = segment_sections(text)
    result = []

    for header, sec_start, sec_end in sections:
        chunks = chunk_span(text, sec_start, sec_end, window_tokens, overlap_tokens)
        for char_start, char_end in chunks:
            result.append((header, char_start, char_end))

    return result


def align_annotations_to_chunk(
    annotations: pd.DataFrame,
    chunk_start: int,
    chunk_end: int,
) -> pd.DataFrame:
    """Return annotations whose spans fall within the chunk boundaries."""
    mask = (annotations["start"] >= chunk_start) & (annotations["end"] <= chunk_end)
    return annotations[mask].copy()


# ---------------------------------------------------------------------------
# Build GLiNER2 examples
# ---------------------------------------------------------------------------

# Section header display names for prepending to chunk text
HEADER_DISPLAY = {
    "preamble": "",  # No prefix for the preamble
    "history of present illness": "History of Present Illness",
    "past medical history": "Past Medical History",
    "chief complaint": "Chief Complaint",
    "major surgical or invasive procedure": "Major Surgical or Invasive Procedure",
    "physical exam": "Physical Exam",
    "pertinent results": "Pertinent Results",
    "brief hospital course": "Brief Hospital Course",
    "allergies": "Allergies",
    "family history": "Family History",
    "social history": "Social History",
    "discharge diagnosis": "Discharge Diagnosis",
    "discharge medications": "Discharge Medications",
    "medications on admission": "Medications on Admission",
    "discharge instructions": "Discharge Instructions",
    "followup instructions": "Followup Instructions",
    "discharge condition": "Discharge Condition",
    "discharge disposition": "Discharge Disposition",
    "impression": "Impression",
    "facility": "Facility",
    "attending": "Attending",
    "service": "Service",
    "name": "Name",
    "admission date": "Admission Date",
    "date of birth": "Date of Birth",
    "sex": "Sex",
}


def _truncate_to_token_limit(
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
) -> str:
    """Truncate text so it stays under max_tokens subword tokens."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    # Decode the truncated token ids back to text
    truncated = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
    return truncated


def build_examples(
    notes_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    sctid_to_tag: dict[int, str],
    window_tokens: int = 400,
    overlap_tokens: int = 50,
    max_subword_tokens: int = 350,
    filter_lab_tables: bool = False,
) -> list[dict]:
    """Convert notes + annotations to GLiNER2 InputExample dicts.

    Section-aware: chunks respect section boundaries and get header prepended.
    Chunks are truncated to max_subword_tokens to prevent OOM from DeBERTa's
    O(n^2) attention (GLiNER2 adds ~50 schema tokens on top).
    Returns list of dicts with keys: text, entities, entity_descriptions, section
    """
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    # Add class labels to annotations
    ann_df = ann_df.copy()
    ann_df["start"] = ann_df["start"].astype(int)
    ann_df["end"] = ann_df["end"].astype(int)
    ann_df["cls"] = map_concept_to_class(ann_df["concept_id"], sctid_to_tag)

    note_texts = dict(zip(notes_df["note_id"], notes_df["text"]))

    # Lab table filtering: remove annotations inside structured lab regions
    if filter_lab_tables:
        n_before = len(ann_df)
        keep_mask = pd.Series(True, index=ann_df.index)
        for note_id, text in note_texts.items():
            regions = merge_adjacent_regions(find_lab_regions(text))
            if not regions:
                continue
            note_mask = ann_df["note_id"] == note_id
            for idx in ann_df[note_mask].index:
                row = ann_df.loc[idx]
                if annotation_in_lab_region(row["start"], row["end"], regions):
                    keep_mask[idx] = False
        ann_df = ann_df[keep_mask]
        n_removed = n_before - len(ann_df)
        print(f"  Lab table filter: removed {n_removed:,} / {n_before:,} annotations "
              f"({100*n_removed/n_before:.1f}%)")
    examples = []
    stats = {
        "total_chunks": 0,
        "empty_chunks": 0,
        "hard_negative_chunks": 0,
        "truncated_chunks": 0,
        "total_annotations": 0,
        "sections_seen": set(),
    }

    for note_id, text in note_texts.items():
        note_ann = ann_df[ann_df["note_id"] == note_id]
        chunks = chunk_note_by_section(text, window_tokens, overlap_tokens)

        for section_header, chunk_start, chunk_end in chunks:
            chunk_ann = align_annotations_to_chunk(note_ann, chunk_start, chunk_end)
            raw_chunk_text = text[chunk_start:chunk_end].strip()

            if not raw_chunk_text:
                continue

            stats["sections_seen"].add(section_header)

            # Prepend section header for context
            display_header = HEADER_DISPLAY.get(section_header, section_header.title())
            if display_header:
                chunk_text = f"[{display_header}] {raw_chunk_text}"
            else:
                chunk_text = raw_chunk_text

            # Hard truncation to stay under DeBERTa's safe token limit
            # (GLiNER2 adds ~50 schema tokens for entity types on top of this)
            original_len = len(tokenizer.encode(chunk_text, add_special_tokens=False))
            if original_len > max_subword_tokens:
                chunk_text = _truncate_to_token_limit(chunk_text, tokenizer, max_subword_tokens)
                stats["truncated_chunks"] += 1

            # Build entity dict with all types (empty list = entity type negative)
            entities: dict[str, list[str]] = {t: [] for t in ALL_ENTITY_TYPES}
            for _, row in chunk_ann.iterrows():
                # Extract span text from the original note
                span_text = text[int(row["start"]):int(row["end"])]
                # Normalize whitespace in spans (newlines → spaces)
                span_text = re.sub(r"\s+", " ", span_text).strip()
                if not span_text:
                    continue
                entities[row["cls"]].append(span_text)

            # Deduplicate spans within each entity type
            for k in entities:
                entities[k] = list(dict.fromkeys(entities[k]))

            is_empty = all(len(v) == 0 for v in entities.values())
            is_hard_neg_section = section_header in HARD_NEGATIVE_SECTIONS

            stats["total_chunks"] += 1
            stats["total_annotations"] += sum(len(v) for v in entities.values())
            if is_empty:
                stats["empty_chunks"] += 1
            if is_hard_neg_section:
                stats["hard_negative_chunks"] += 1

            examples.append({
                "text": chunk_text,
                "entities": entities,
                "entity_descriptions": ENTITY_DESCRIPTIONS.copy(),
                "section": section_header,
            })

    print(f"  Chunks: {stats['total_chunks']:,} "
          f"({stats['empty_chunks']:,} empty, "
          f"{stats['hard_negative_chunks']:,} from hard-negative sections, "
          f"{stats['truncated_chunks']:,} truncated to {max_subword_tokens} tokens)")
    print(f"  Annotations aligned: {stats['total_annotations']:,}")
    print(f"  Sections seen: {sorted(stats['sections_seen'])}")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare GLiNER2 training data")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("data/old-challenge-split"),
        help="Directory containing train/test notes and annotations CSVs",
    )
    parser.add_argument(
        "--project-root", type=Path,
        default=Path("/workspaces/snomed-ct-entity-linking"),
        help="Root of the main project checkout (for SNOMED RF2 files)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/gliner2"),
        help="Output directory for prepared JSONL files",
    )
    parser.add_argument("--window-tokens", type=int, default=200)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--max-subword-tokens", type=int, default=350,
                        help="Hard cap on subword tokens per chunk (DeBERTa limit, "
                             "GLiNER adds ~50 schema tokens on top)")
    parser.add_argument("--filter-lab-tables", action="store_true",
                        help="Remove annotations inside structured lab result tables")
    args = parser.parse_args()

    # Load SNOMED tag mapping
    print("Loading SNOMED semantic tag mapping...")
    sctid_to_tag = load_sctid_to_tag(args.project_root)

    # Load data
    print("\nLoading training data...")
    train_notes = pd.read_csv(args.data_dir / "train_notes.csv")
    train_ann = pd.read_csv(args.data_dir / "train_annotations.csv",
                            dtype={"concept_id": int})
    print(f"  Notes: {len(train_notes)}, Annotations: {len(train_ann)}")

    print("\nLoading test data...")
    test_notes = pd.read_csv(args.data_dir / "test_notes.csv")
    test_ann = pd.read_csv(args.data_dir / "test_annotations.csv",
                           dtype={"concept_id": int})
    print(f"  Notes: {len(test_notes)}, Annotations: {len(test_ann)}")

    # Build examples
    print(f"\nBuilding training examples... (filter_lab_tables={args.filter_lab_tables})")
    train_examples = build_examples(
        train_notes, train_ann, sctid_to_tag,
        args.window_tokens, args.overlap_tokens, args.max_subword_tokens,
        filter_lab_tables=args.filter_lab_tables,
    )

    print(f"\nBuilding test examples... (filter_lab_tables={args.filter_lab_tables})")
    test_examples = build_examples(
        test_notes, test_ann, sctid_to_tag,
        args.window_tokens, args.overlap_tokens, args.max_subword_tokens,
        filter_lab_tables=args.filter_lab_tables,
    )

    # Split training into train/val (85/15 by note, but we work at chunk level)
    # Use a deterministic split: first 85% of examples for train, rest for val
    split_idx = int(len(train_examples) * 0.85)
    val_examples = train_examples[split_idx:]
    train_examples = train_examples[:split_idx]

    print(f"\nFinal splits:")
    print(f"  Train: {len(train_examples):,} examples")
    print(f"  Val:   {len(val_examples):,} examples")
    print(f"  Test:  {len(test_examples):,} examples")

    # Class distribution
    for name, examples in [("train", train_examples), ("val", val_examples),
                           ("test", test_examples)]:
        counts = {t: 0 for t in ALL_ENTITY_TYPES}
        for ex in examples:
            for t in ALL_ENTITY_TYPES:
                counts[t] += len(ex["entities"][t])
        print(f"  {name} class counts: {counts}")

    # Save as JSONL
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, examples in [("train", train_examples), ("val", val_examples),
                           ("test", test_examples)]:
        out_path = args.output_dir / f"{name}.jsonl"
        with open(out_path, "w") as f:
            for ex in examples:
                record = {
                    "input": ex["text"],
                    "output": {
                        "entities": ex["entities"],
                        "entity_descriptions": ex["entity_descriptions"],
                    },
                    "section": ex.get("section", ""),
                }
                f.write(json.dumps(record) + "\n")
        print(f"  Wrote {out_path} ({len(examples):,} examples)")

    # Also save as pickle for direct InputExample loading
    for name, examples in [("train", train_examples), ("val", val_examples),
                           ("test", test_examples)]:
        out_path = args.output_dir / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(examples, f)

    print("\nDone.")


if __name__ == "__main__":
    main()
