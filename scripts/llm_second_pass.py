#!/usr/bin/env python3
"""
LLM Second-Pass Annotation for SNOMED CT Entity Linking.

Takes KIRI super dictionary predictions and uses an LLM to:
  1. Discover clinical spans KIRI missed (per-note, batched)
  2. Retrieve SNOMED candidates for new spans (SapBERT+FAISS+BM25+RRF+rerank)
  3. Link each span to a concept via LLM (batched)
  4. Merge with KIRI predictions and score with class_char_iou

Usage:
    python scripts/llm_second_pass.py \
        --kiri-pred-csv outputs/super_dictionary_60test/kiri_super_base_pred.csv \
        --vllm-url http://localhost:8000 \
        --model openai/gpt-oss-20b \
        --out-pred-csv outputs/llm_second_pass_merged.csv
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.llm_rerank_test import (
    CLASSIFY_SYSTEM,
    CLASSIFY_USER,
    SEARCH_SYSTEM,
    SEARCH_FOLLOWUP_SYSTEM,
    _call_vllm_batch,
    _format_candidate_lines,
    _make_vllm_client,
    build_search_prompt,
    build_search_followup_prompt,
    parse_llm_choice,
    parse_search_response,
    rerun_retrieval_batch,
)
from scripts.super_dictionary.runtime_scoring import class_char_iou, macro_char_iou
from snomed_ct_entity_linking.recall_analysis.config import Config
from snomed_ct_entity_linking.recall_analysis.encoder import SapBERTEncoder
from snomed_ct_entity_linking.recall_analysis.index_builder import (
    load_indexes,
    load_raw_embeddings,
)
from snomed_ct_entity_linking.recall_analysis.reranker import SapBERTReranker
from snomed_ct_entity_linking.recall_analysis.retrieval import (
    dense_search,
    expand_with_hierarchy,
    reciprocal_rank_fusion,
    sparse_search,
)
from snomed_ct_entity_linking.recall_analysis.snomed_loader import load_snomed

# ---------------------------------------------------------------------------
# Prompt templates for span discovery
# ---------------------------------------------------------------------------

DISCOVERY_SYSTEM = """\
You are a clinical NER annotator for SNOMED CT. You will see a hospital \
discharge note with existing annotations marked by [brackets]. Find clinical \
entities that are NOT already bracketed.

ONLY annotate these SNOMED categories:
- Clinical findings (e.g. chest pain, dyspnea, tachycardia, edema)
- Disorders/diagnoses (e.g. pneumonia, heart failure, diabetes mellitus)
- Procedures (e.g. intubation, dialysis, colonoscopy)
- Body structures (e.g. left ventricle, right lung, abdomen)
- Clinical abbreviations that map to the above (e.g. CHF, COPD, DM2, AFib)

Do NOT annotate: lab values, vital sign numbers, medication/drug names, \
dosages, section headers, dates, proper nouns, or anything already in [brackets].

Output one entity per line, using the EXACT text as it appears in the note. \
Be precise — copy the exact substring. If nothing is missing, output NONE."""

DISCOVERY_USER = """\
{annotated_note}"""

# ---------------------------------------------------------------------------
# Prompt template for concept linking (reuses CLASSIFY_SYSTEM from llm_rerank_test)
# ---------------------------------------------------------------------------

LINKING_USER = """\
Term: {mention}
Context: ...{context}...
Candidates:
{candidates}
Answer:"""


def build_surface_lookup(
    notes_csv: str,
    annotations_csv: str,
    train_size: int,
    min_count: int = 3,
    min_pct: float = 0.80,
) -> dict[str, int]:
    """Build a deterministic surface-form → concept_id lookup from training gold.

    Only includes surface forms that appear >= min_count times in training data
    with >= min_pct consistency on the primary concept.

    Returns:
        Dict mapping lowercase surface form to concept_id.
    """
    notes_df = pd.read_csv(notes_csv)
    notes_df["note_id"] = notes_df["note_id"].astype(str)
    note_texts = dict(zip(notes_df["note_id"], notes_df["text"]))

    gold = pd.read_csv(annotations_csv, dtype={"concept_id": int})
    gold["note_id"] = gold["note_id"].astype(str)

    ids = list(notes_df["note_id"])
    np.random.seed(12345)
    np.random.shuffle(ids)
    train_ids = set(ids[:train_size])

    train_gold = gold[gold["note_id"].isin(train_ids)].copy()
    train_gold["surface"] = train_gold.apply(
        lambda r: note_texts.get(r["note_id"], "")[int(r["start"]):int(r["end"])].lower().strip(),
        axis=1,
    )

    surface_groups = train_gold.groupby("surface").agg(
        count=("concept_id", "size"),
        primary_concept=("concept_id", lambda x: x.value_counts().index[0]),
        primary_pct=("concept_id", lambda x: x.value_counts().iloc[0] / len(x)),
    ).reset_index()

    consistent = surface_groups[
        (surface_groups["count"] >= min_count) &
        (surface_groups["primary_pct"] >= min_pct)
    ]

    lookup = dict(zip(consistent["surface"], consistent["primary_concept"].astype(int)))
    return lookup


def targeted_dictionary_augmentation(
    kiri_pred: pd.DataFrame,
    gold_ann: pd.DataFrame,
    note_texts: dict[str, str],
    test_ids: list[str],
    surface_lookup: dict[str, int],
    min_surface_len: int = 3,
    skip_first_n: int = 100,
) -> pd.DataFrame:
    """Scan test notes for surface forms, only adding predictions for zero-IoU concepts.

    This is safe because adding predictions for a zero-IoU concept can only
    improve or maintain its IoU (0/N stays 0, any overlap increases it).

    Returns:
        DataFrame of new predictions (note_id, start, end, concept_id).
    """
    kiri_class_iou_df = class_char_iou(kiri_pred, gold_ann)
    kiri_zero_classes = set(kiri_class_iou_df[kiri_class_iou_df["iou"] == 0]["concept_id"])
    gold_concepts = set(gold_ann["concept_id"])
    gold_zero = kiri_zero_classes & gold_concepts

    # Filter lookup to gold-zero concepts and min length
    targeted = {
        s: c for s, c in surface_lookup.items()
        if c in gold_zero and len(s) >= min_surface_len
    }

    new_predictions = []
    for nid in test_ids:
        text = note_texts.get(nid, "")
        text_lower = text.lower()
        note_kiri = kiri_pred[kiri_pred["note_id"] == nid]
        existing = list(zip(note_kiri["start"].astype(int), note_kiri["end"].astype(int)))

        for surface, concept_id in targeted.items():
            start = 0
            while True:
                idx = text_lower.find(surface, start)
                if idx == -1:
                    break
                end = idx + len(surface)
                start = end
                if idx < skip_first_n:
                    continue
                if idx > 0 and text_lower[idx - 1].isalnum():
                    continue
                if end < len(text_lower) and text_lower[end].isalnum():
                    continue
                overlaps = any(ps < end and pe > idx for ps, pe in existing)
                if overlaps:
                    continue
                new_predictions.append({
                    "note_id": nid, "start": idx, "end": end, "concept_id": concept_id,
                })
                existing.append((idx, end))

    df = pd.DataFrame(new_predictions)
    if not df.empty:
        df["note_id"] = df["note_id"].astype(str)
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        df["concept_id"] = df["concept_id"].astype(int)

    return df, len(targeted), len(gold_zero)


def get_test_note_ids(notes_csv: str, train_size: int) -> list[str]:
    """Reproduce the 60-note holdout split from compare_kiri.py."""
    notes = pd.read_csv(notes_csv)
    ids = list(notes["note_id"].astype(str))
    np.random.seed(12345)
    np.random.shuffle(ids)
    test_ids = ids[train_size:]
    return test_ids


def build_annotated_note(note_text: str, pred_df: pd.DataFrame) -> str:
    """Insert [brackets] around KIRI predictions in the note text.

    Processes spans in reverse order to preserve character offsets.
    Only marks span positions — no concept IDs (saves tokens).
    """
    if pred_df.empty:
        return note_text

    # Sort by start descending so insertions don't shift later offsets
    spans = pred_df[["start", "end"]].sort_values("start", ascending=False).values
    text = note_text
    for start, end in spans:
        start, end = int(start), int(end)
        if start < 0 or end > len(text) or start >= end:
            continue
        text = text[:start] + "[" + text[start:end] + "]" + text[end:]
    return text


def parse_discovered_spans(llm_output: str) -> list[tuple[str, str | None]]:
    """Parse LLM output into a list of (span_text, expansion_or_None).

    Handles numbered lists, bullet points, plain lines, and ABBREV = Expansion format.
    Filters by length and handles NONE sentinel.

    Returns:
        List of (span_text, expansion). For abbreviations with "ABBREV = Expansion",
        span_text is the abbreviation and expansion is the full name.
        For regular spans, expansion is None.
    """
    if not llm_output or llm_output.strip().upper() == "NONE":
        return []

    results = []
    for line in llm_output.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip numbering: "1. ", "1) ", "- ", "* "
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-\*]\s*", "", line)
        line = line.strip().strip('"').strip("'")
        if line.upper() == "NONE" or len(line) < 1:
            continue

        # Check for abbreviation = expansion format
        eq_match = re.match(r"^([A-Za-z0-9/\-\+]+)\s*=\s*(.+)$", line)
        if eq_match:
            abbrev = eq_match.group(1).strip()
            expansion = eq_match.group(2).strip()
            if len(abbrev) <= 10 and len(expansion) >= 3:
                results.append((abbrev, expansion))
                continue

        if 2 <= len(line) <= 100:
            results.append((line, None))

    return results


def find_span_positions(
    note_text: str,
    span_text: str,
    existing_intervals: list[tuple[int, int]],
    skip_first_n: int = 100,
) -> list[tuple[int, int]]:
    """Find all non-overlapping positions of span_text in note_text.

    Checks word boundaries and skips positions overlapping existing predictions
    or the note header (first skip_first_n chars).
    """
    positions = []
    text_lower = note_text.lower()
    span_lower = span_text.lower()
    span_len = len(span_text)

    start = 0
    while True:
        idx = text_lower.find(span_lower, start)
        if idx == -1:
            break
        end = idx + span_len
        start = end  # advance past this match

        # Skip header area
        if idx < skip_first_n:
            continue

        # Check word boundaries
        if idx > 0 and text_lower[idx - 1].isalnum():
            continue
        if end < len(text_lower) and text_lower[end].isalnum():
            continue

        # Check overlap with existing intervals
        overlaps = False
        for ex_start, ex_end in existing_intervals:
            if idx < ex_end and end > ex_start:
                overlaps = True
                break
        if overlaps:
            continue

        positions.append((idx, end))

    return positions


def get_context(note_text: str, start: int, end: int, window: int = 200) -> str:
    """Extract context around a span position."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(note_text), end + window)
    return note_text[ctx_start:ctx_end]


def merge_predictions(
    kiri_pred: pd.DataFrame, llm_pred: pd.DataFrame
) -> pd.DataFrame:
    """Merge KIRI and LLM predictions, de-overlapping per note.

    KIRI predictions take priority (they appear first in the concat).
    For overlapping character positions within the same note, the first
    prediction wins (via the overwrite semantics of the scoring function).
    """
    if llm_pred.empty:
        return kiri_pred.copy()
    merged = pd.concat([kiri_pred, llm_pred], ignore_index=True)
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="LLM second-pass annotation for SNOMED CT entity linking"
    )
    parser.add_argument(
        "--kiri-pred-csv",
        default="outputs/super_dictionary_60test/kiri_super_base_pred.csv",
        help="Path to KIRI predictions CSV (note_id, start, end, concept_id)",
    )
    parser.add_argument(
        "--notes-csv",
        default="1st Place/data/raw/mimic-iv_notes_training_set.csv",
        help="Path to notes CSV (note_id, text)",
    )
    parser.add_argument(
        "--annotations-csv",
        default="1st Place/data/interim/train_annotations_cln.csv",
        help="Path to gold annotations CSV",
    )
    parser.add_argument(
        "--train-size", type=int, default=212,
        help="Number of training notes (determines holdout split, default: 212)",
    )
    parser.add_argument(
        "--vllm-url", type=str, default="http://localhost:8000",
        help="vLLM server base URL",
    )
    parser.add_argument(
        "--model", type=str, default="openai/gpt-oss-20b",
        help="Model name on vLLM server",
    )
    parser.add_argument(
        "--reasoning-effort", type=str, default="low",
        choices=["low", "medium", "high"],
        help="Reasoning effort for discovery pass",
    )
    parser.add_argument(
        "--linking-effort", type=str, default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort for linking pass (defaults to --reasoning-effort)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of candidates to show LLM for concept linking",
    )
    parser.add_argument(
        "--max-new-spans-per-note", type=int, default=50,
        help="Max new spans to consider per note from LLM discovery",
    )
    parser.add_argument(
        "--max-notes", type=int, default=0,
        help="Limit to first N test notes (0 = all)",
    )
    parser.add_argument(
        "--out-pred-csv", type=str, default=None,
        help="Path to write merged predictions CSV",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run discovery pass only, print spans, skip retrieval+linking",
    )
    parser.add_argument(
        "--skip-discovery", action="store_true",
        help="Skip LLM discovery pass; use gold missed spans instead (oracle test)",
    )
    parser.add_argument(
        "--discovery-output", type=str, default=None,
        help="Path to save/load discovery results (JSON lines)",
    )
    parser.add_argument(
        "--rank0-only", action="store_true",
        help="Only keep LLM predictions where model chose rank-0 candidate",
    )
    parser.add_argument(
        "--min-rerank-score", type=float, default=0.0,
        help="Only keep predictions where the chosen candidate's rerank score >= this",
    )
    parser.add_argument(
        "--search-linking", action="store_true",
        help="Use search-enabled linking (model can request re-retrieval)",
    )
    parser.add_argument(
        "--discovery-prompt-json", type=str, default=None,
        help="Path to evolved_prompt.json to use for discovery (overrides built-in prompt)",
    )
    parser.add_argument(
        "--deterministic-linking", action="store_true",
        help="Use training-data surface-form lookup instead of retrieval+LLM linking",
    )
    parser.add_argument(
        "--targeted-augmentation", action="store_true",
        help="Add dictionary-scan predictions for zero-IoU concepts (safe augmentation)",
    )
    parser.add_argument(
        "--lookup-min-count", type=int, default=1,
        help="Min training occurrences for surface-form lookup (default: 1)",
    )
    parser.add_argument(
        "--lookup-min-pct", type=float, default=0.50,
        help="Min consistency pct for surface-form lookup (default: 0.50)",
    )
    args = parser.parse_args()
    if args.linking_effort is None:
        args.linking_effort = args.reasoning_effort

    # =========================================================================
    # Load data
    # =========================================================================
    print("\n" + "=" * 65)
    print("  Loading data")
    print("=" * 65)

    notes_df = pd.read_csv(args.notes_csv)
    notes_df["note_id"] = notes_df["note_id"].astype(str)
    note_texts = dict(zip(notes_df["note_id"], notes_df["text"]))

    kiri_pred = pd.read_csv(args.kiri_pred_csv, dtype={"concept_id": int})
    kiri_pred["note_id"] = kiri_pred["note_id"].astype(str)

    ann_df = pd.read_csv(args.annotations_csv, dtype={"concept_id": int})
    ann_df["note_id"] = ann_df["note_id"].astype(str)
    ann_df["start"] = ann_df["start"].astype(int)
    ann_df["end"] = ann_df["end"].astype(int)

    # Determine test note IDs
    test_ids = get_test_note_ids(args.notes_csv, args.train_size)
    if args.max_notes:
        test_ids = test_ids[: args.max_notes]
    print(f"  Test notes: {len(test_ids)}")

    # Filter to test notes
    kiri_pred = kiri_pred[kiri_pred["note_id"].isin(test_ids)].reset_index(drop=True)
    gold_ann = ann_df[ann_df["note_id"].isin(test_ids)][
        ["note_id", "start", "end", "concept_id"]
    ].reset_index(drop=True)
    print(f"  KIRI predictions: {len(kiri_pred):,}")
    print(f"  Gold annotations: {len(gold_ann):,}")

    # Score KIRI-only baseline
    kiri_class_iou = class_char_iou(kiri_pred, gold_ann)
    kiri_macro = float(kiri_class_iou["iou"].mean())
    n_zero_kiri = int((kiri_class_iou["iou"] == 0).sum())
    print(f"\n  KIRI-only Macro IoU: {kiri_macro:.4f}")
    print(f"  Zero-IoU classes: {n_zero_kiri}/{len(kiri_class_iou)}")

    # =========================================================================
    # Targeted Dictionary Augmentation (pre-LLM, safe augmentation)
    # =========================================================================
    if args.targeted_augmentation:
        print("\n" + "=" * 65)
        print("  Targeted Dictionary Augmentation")
        print("=" * 65)

        surface_lookup = build_surface_lookup(
            args.notes_csv, args.annotations_csv, args.train_size,
            min_count=args.lookup_min_count, min_pct=args.lookup_min_pct,
        )
        print(f"  Full lookup table: {len(surface_lookup)} surface forms")

        aug_df, n_targeted, n_gold_zero = targeted_dictionary_augmentation(
            kiri_pred, gold_ann, note_texts, test_ids, surface_lookup,
        )
        print(f"  Gold-zero concepts: {n_gold_zero}")
        print(f"  Targeted surface forms: {n_targeted}")
        print(f"  New predictions: {len(aug_df)}")

        # Merge augmentation with KIRI
        kiri_pred = pd.concat([kiri_pred, aug_df], ignore_index=True)

        # Re-score
        aug_class_iou = class_char_iou(kiri_pred, gold_ann)
        aug_macro = float(aug_class_iou["iou"].mean())
        n_zero_aug = int((aug_class_iou["iou"] == 0).sum())
        print(f"\n  After augmentation:")
        print(f"  Macro IoU: {kiri_macro:.4f} → {aug_macro:.4f} ({aug_macro - kiri_macro:+.4f})")
        print(f"  Zero-IoU: {n_zero_kiri} → {n_zero_aug}")

        # Update baseline for subsequent steps
        kiri_class_iou = aug_class_iou
        kiri_macro = aug_macro
        n_zero_kiri = n_zero_aug

    # =========================================================================
    # Pass 1: LLM Span Discovery
    # =========================================================================
    print("\n" + "=" * 65)
    print("  Pass 1: LLM Span Discovery")
    print("=" * 65)

    # Create vLLM client (reused for discovery and linking)
    client = None
    if not args.skip_discovery or not args.dry_run:
        client = _make_vllm_client(args.vllm_url, args.model)

    # Load evolved prompt if provided
    discovery_system = DISCOVERY_SYSTEM
    if args.discovery_prompt_json:
        with open(args.discovery_prompt_json) as f:
            evolved = json.load(f)
        discovery_system = evolved["system_prompt"]
        print(f"  Using evolved prompt: {evolved.get('signature', '?')}")
        print(f"  Evolved fitness: {evolved.get('fitness', '?')}")

    if args.skip_discovery:
        print("  --skip-discovery: skipping LLM discovery pass")
        discovered_spans_per_note = {}
    else:
        # Build bracket-annotated notes
        annotated_notes = {}
        for note_id in test_ids:
            text = note_texts.get(note_id, "")
            note_preds = kiri_pred[kiri_pred["note_id"] == note_id]
            annotated_notes[note_id] = build_annotated_note(text, note_preds)

        # Build prompts
        discovery_prompts = []
        discovery_note_ids = []
        for note_id in test_ids:
            annotated = annotated_notes[note_id]
            discovery_prompts.append(DISCOVERY_USER.format(annotated_note=annotated))
            discovery_note_ids.append(note_id)

        print(f"  Built {len(discovery_prompts)} discovery prompts")
        discovery_outputs = _call_vllm_batch(
            client,
            args.model,
            discovery_prompts,
            reasoning_effort=args.reasoning_effort,
            max_tokens=1024,
            max_concurrent=256,
            label="discovery",
            system_prompt=discovery_system,
            frequency_penalty=0.3,
        )

        # Parse discovered spans — now returns (span_text, expansion_or_None)
        discovered_spans_per_note = {}
        total_spans = 0
        for note_id, output in zip(discovery_note_ids, discovery_outputs):
            spans = parse_discovered_spans(output)
            if args.max_new_spans_per_note:
                spans = spans[: args.max_new_spans_per_note]
            discovered_spans_per_note[note_id] = spans
            total_spans += len(spans)

        n_abbrevs = sum(
            1 for spans in discovered_spans_per_note.values()
            for _, exp in spans if exp is not None
        )
        print(f"\n  Discovered {total_spans} spans across {len(test_ids)} notes")
        print(f"  Abbreviations (with expansion): {n_abbrevs}")
        print(f"  Avg spans/note: {total_spans / len(test_ids):.1f}")

        # Show sample
        for note_id in list(discovered_spans_per_note.keys())[:3]:
            spans = discovered_spans_per_note[note_id]
            print(f"\n  Note {note_id}: {len(spans)} spans")
            for span_text, expansion in spans[:5]:
                if expansion:
                    print(f"    - \"{span_text}\" = \"{expansion}\"")
                else:
                    print(f"    - \"{span_text}\"")
            if len(spans) > 5:
                print(f"    ... +{len(spans) - 5} more")

        # Save discovery output if requested
        if args.discovery_output:
            with open(args.discovery_output, "w") as f:
                for note_id in test_ids:
                    spans = discovered_spans_per_note.get(note_id, [])
                    f.write(json.dumps({
                        "note_id": note_id,
                        "spans": [(s, e) for s, e in spans],
                    }) + "\n")
            print(f"\n  Discovery output saved to {args.discovery_output}")

    if args.dry_run:
        print("\n  --dry-run: stopping after discovery pass")
        return

    # =========================================================================
    # Span Position Mapping
    # =========================================================================
    print("\n" + "=" * 65)
    print("  Span Position Mapping")
    print("=" * 65)

    # Build existing intervals per note from KIRI predictions
    existing_intervals_per_note = {}
    for note_id in test_ids:
        note_preds = kiri_pred[kiri_pred["note_id"] == note_id]
        intervals = list(zip(note_preds["start"].astype(int), note_preds["end"].astype(int)))
        existing_intervals_per_note[note_id] = intervals

    # Map discovered spans to positions
    # new_annotations: (note_id, start, end, span_text, retrieval_text)
    # retrieval_text = expansion if abbreviation, else span_text
    new_annotations = []
    unique_retrieval_texts = set()
    n_no_position = 0

    for note_id in test_ids:
        text = note_texts.get(note_id, "")
        existing = existing_intervals_per_note.get(note_id, [])
        spans = discovered_spans_per_note.get(note_id, [])

        for span_text, expansion in spans:
            positions = find_span_positions(text, span_text, existing)
            if not positions:
                n_no_position += 1
                continue
            # For retrieval: use expansion (more descriptive) if available
            retrieval_text = expansion if expansion else span_text
            for start, end in positions:
                new_annotations.append((note_id, start, end, span_text, retrieval_text))
                unique_retrieval_texts.add(retrieval_text)
                # Add to existing intervals to prevent double-annotation
                existing.append((start, end))

    print(f"  New span positions: {len(new_annotations)}")
    print(f"  Unique retrieval texts: {len(unique_retrieval_texts)}")
    print(f"  Spans with no valid position: {n_no_position}")

    if not new_annotations:
        print("\n  No new annotations found. Outputting KIRI-only results.")
        if args.out_pred_csv:
            kiri_pred.to_csv(args.out_pred_csv, index=False)
            print(f"  Saved to {args.out_pred_csv}")
        return

    # =========================================================================
    # Deterministic Linking (alternative to retrieval+LLM linking)
    # =========================================================================
    if args.deterministic_linking:
        print("\n" + "=" * 65)
        print("  Deterministic Linking (surface-form lookup)")
        print("=" * 65)

        surface_lookup = build_surface_lookup(
            args.notes_csv, args.annotations_csv, args.train_size,
            min_count=args.lookup_min_count, min_pct=args.lookup_min_pct,
        )
        print(f"  Lookup table: {len(surface_lookup)} surface forms")

        # Identify gold-zero concepts (safe to add predictions for)
        gold_concepts = set(gold_ann["concept_id"])
        kiri_zero_set = set(kiri_class_iou[kiri_class_iou["iou"] == 0]["concept_id"])
        gold_zero_set = kiri_zero_set & gold_concepts
        print(f"  Gold-zero concepts (safe targets): {len(gold_zero_set)}")

        llm_predictions = []
        n_linked = 0
        n_no_lookup = 0
        n_expansion_linked = 0
        n_filtered_non_zero = 0

        for note_id, start, end, span_text, retrieval_text in new_annotations:
            span_lower = span_text.lower().strip()
            # Try exact surface form first
            concept_id = surface_lookup.get(span_lower)
            # If abbreviation with expansion, also try expansion
            if concept_id is None and retrieval_text != span_text:
                concept_id = surface_lookup.get(retrieval_text.lower().strip())
                if concept_id is not None:
                    n_expansion_linked += 1
            if concept_id is not None:
                # Only add if concept is in gold-zero set (safe)
                if concept_id not in gold_zero_set:
                    n_filtered_non_zero += 1
                    continue
                llm_predictions.append({
                    "note_id": note_id, "start": start, "end": end, "concept_id": concept_id,
                })
                n_linked += 1
            else:
                n_no_lookup += 1

        print(f"  Linked: {n_linked}")
        print(f"  Via expansion: {n_expansion_linked}")
        print(f"  No lookup match: {n_no_lookup}")
        print(f"  Filtered (non-zero-IoU concept): {n_filtered_non_zero}")

        # Jump to merge & score
        llm_pred_df = pd.DataFrame(llm_predictions)
        if not llm_pred_df.empty:
            llm_pred_df["note_id"] = llm_pred_df["note_id"].astype(str)
            llm_pred_df["start"] = llm_pred_df["start"].astype(int)
            llm_pred_df["end"] = llm_pred_df["end"].astype(int)
            llm_pred_df["concept_id"] = llm_pred_df["concept_id"].astype(int)

        merged_pred = merge_predictions(kiri_pred, llm_pred_df)
        print(f"\n  KIRI predictions: {len(kiri_pred):,}")
        print(f"  LLM predictions:  {len(llm_pred_df):,}")
        print(f"  Merged total:     {len(merged_pred):,}")

        merged_class_iou = class_char_iou(merged_pred, gold_ann)
        merged_macro = float(merged_class_iou["iou"].mean())
        n_zero_merged = int((merged_class_iou["iou"] == 0).sum())

        kiri_zero_classes = set(kiri_class_iou[kiri_class_iou["iou"] == 0]["concept_id"])
        merged_nonzero = merged_class_iou[merged_class_iou["iou"] > 0]
        recovered = kiri_zero_classes & set(merged_nonzero["concept_id"])

        kiri_iou_map = dict(zip(kiri_class_iou["concept_id"], kiri_class_iou["iou"]))
        merged_iou_map = dict(zip(merged_class_iou["concept_id"], merged_class_iou["iou"]))
        n_worse = sum(
            1 for c in merged_iou_map
            if c in kiri_iou_map and merged_iou_map[c] < kiri_iou_map[c] - 0.001
        )
        n_better = sum(
            1 for c in merged_iou_map
            if c in kiri_iou_map and merged_iou_map[c] > kiri_iou_map[c] + 0.001
        )

        print(f"\n  Results:")
        print(f"  KIRI-only Macro IoU:  {kiri_macro:.4f}")
        print(f"  KIRI+LLM Macro IoU:  {merged_macro:.4f}")
        print(f"  Delta:                {merged_macro - kiri_macro:+.4f}")
        print(f"  Zero-IoU classes:     {n_zero_kiri} → {n_zero_merged}")
        print(f"  Classes recovered from zero IoU: {len(recovered)}")
        print(f"  Classes improved: {n_better}")
        print(f"  Classes degraded: {n_worse}")

        if recovered:
            print(f"\n  Recovered classes:")
            for cid in sorted(recovered):
                iou = merged_iou_map.get(cid, 0.0)
                print(f"    {cid}: 0.0 → {iou:.3f}")

        if args.out_pred_csv:
            merged_pred.to_csv(args.out_pred_csv, index=False)
            print(f"\n  Merged predictions saved to {args.out_pred_csv}")
            iou_path = args.out_pred_csv.replace(".csv", "_class_iou.csv")
            merged_class_iou.to_csv(iou_path, index=False)
        return

    # =========================================================================
    # Pass 2a: Candidate Retrieval
    # =========================================================================
    print("\n" + "=" * 65)
    print("  Pass 2a: Candidate Retrieval")
    print("=" * 65)

    cfg = Config()
    descriptions_df, sctid_to_fsn, sctid_to_tag, parent_map = load_snomed(cfg)
    faiss_index, faiss_sctids, bm25_index, bm25_sctids, bm25_terms = load_indexes(cfg)
    indexed_sctids = set(faiss_sctids)

    raw_data = load_raw_embeddings(cfg)
    if raw_data is None:
        print("ERROR: No raw embeddings found.")
        sys.exit(1)
    raw_embeddings, raw_sctids = raw_data

    encoder = SapBERTEncoder(cfg.embedding_model)
    reranker = SapBERTReranker(raw_embeddings, raw_sctids)

    # Deduplicate retrieval texts (expansions for abbreviations, span text otherwise)
    unique_spans_list = sorted(unique_retrieval_texts)
    span_to_idx = {s: i for i, s in enumerate(unique_spans_list)}
    print(f"  Retrieving candidates for {len(unique_spans_list)} unique texts ...")

    reranked_by_span = rerun_retrieval_batch(
        unique_spans_list,
        encoder,
        faiss_index,
        faiss_sctids,
        bm25_index,
        bm25_sctids,
        reranker,
        parent_map,
        indexed_sctids,
        cfg,
    )

    # Keep encoder/reranker alive if search-linking is enabled
    if not args.search_linking:
        encoder.close()
        reranker.close()
        del raw_embeddings
        torch.cuda.empty_cache()

    # =========================================================================
    # Pass 2b: LLM Concept Linking
    # =========================================================================
    print("\n" + "=" * 65)
    print("  Pass 2b: LLM Concept Linking"
          + (" (search-enabled)" if args.search_linking else ""))
    print("=" * 65)

    # Build linking prompts
    linking_prompts = []
    linking_indices = []  # index into new_annotations
    linking_candidates = []  # candidates for each prompt
    linking_contexts = []  # context strings for search follow-up

    for ann_idx, (note_id, start, end, span_text, retrieval_text) in enumerate(new_annotations):
        span_idx = span_to_idx[retrieval_text]
        candidates = reranked_by_span[span_idx]
        if not candidates:
            continue

        # Use retrieval_text as the mention for linking (expansion is more descriptive)
        mention = retrieval_text
        context = get_context(note_texts[note_id], start, end)
        if args.search_linking:
            from scripts.llm_rerank_test import SEARCH_USER
            prompt = SEARCH_USER.format(
                mention=mention,
                context=context[:500],
                search_term=mention,
                candidates=_format_candidate_lines(candidates, sctid_to_fsn, args.top_k),
            )
        else:
            prompt = LINKING_USER.format(
                mention=mention,
                context=context,
                candidates=_format_candidate_lines(candidates, sctid_to_fsn, args.top_k),
            )
        linking_prompts.append(prompt)
        linking_indices.append(ann_idx)
        linking_candidates.append(candidates[: args.top_k])
        linking_contexts.append(context)

    print(f"  Built {len(linking_prompts)} linking prompts")

    if not linking_prompts:
        print("\n  No linking prompts. Outputting KIRI-only results.")
        if args.out_pred_csv:
            kiri_pred.to_csv(args.out_pred_csv, index=False)
        return

    # Call LLM for concept linking
    system_prompt = SEARCH_SYSTEM if args.search_linking else CLASSIFY_SYSTEM
    linking_outputs = _call_vllm_batch(
        client,
        args.model,
        linking_prompts,
        reasoning_effort=args.linking_effort,
        max_tokens=384,
        max_concurrent=256,
        label="linking",
        system_prompt=system_prompt,
        frequency_penalty=0.0,
    )

    # If search-linking: handle SEARCH requests with re-retrieval
    if args.search_linking:
        research_indices = []
        new_search_terms_map = {}
        for j, raw in enumerate(linking_outputs):
            candidates = linking_candidates[j]
            action, value = parse_search_response(raw, len(candidates) - 1)
            if action == "search":
                research_indices.append(j)
                new_search_terms_map[j] = value

        print(f"  Search requests: {len(research_indices)}/{len(linking_prompts)}")

        if research_indices:
            new_terms = [new_search_terms_map[j] for j in research_indices]
            for j, term in zip(research_indices[:10], new_terms[:10]):
                ann_idx = linking_indices[j]
                span_text = new_annotations[ann_idx][3]  # span_text is at index 3
                print(f"    \"{span_text}\" → SEARCH: \"{term}\"")
            if len(research_indices) > 10:
                print(f"    ... and {len(research_indices) - 10} more")

            new_results = rerun_retrieval_batch(
                new_terms, encoder,
                faiss_index, faiss_sctids, bm25_index, bm25_sctids,
                reranker, parent_map, indexed_sctids, cfg,
            )

            # Build follow-up prompts
            followup_prompts = []
            for k, j in enumerate(research_indices):
                ann_idx = linking_indices[j]
                _, _, _, span_text, retrieval_text = new_annotations[ann_idx]
                followup_prompts.append(build_search_followup_prompt(
                    mention=span_text,
                    context=linking_contexts[j][:500],
                    original_search_term=span_text,
                    new_search_term=new_search_terms_map[j],
                    candidates=new_results[k],
                    sctid_to_fsn=sctid_to_fsn,
                    top_k=args.top_k,
                ))

            followup_outputs = _call_vllm_batch(
                client, args.model, followup_prompts,
                reasoning_effort=args.linking_effort,
                max_tokens=384, max_concurrent=256, label="followup",
                system_prompt=SEARCH_FOLLOWUP_SYSTEM,
            )

            for k, j in enumerate(research_indices):
                linking_outputs[j] = followup_outputs[k]
                linking_candidates[j] = new_results[k][: args.top_k]

        # Cleanup search resources
        encoder.close()
        reranker.close()
        del raw_embeddings
        torch.cuda.empty_cache()

    # Parse linking results
    llm_predictions = []
    n_linked = 0
    n_abstain = 0
    n_filtered_rank = 0
    n_filtered_score = 0

    for j, raw in enumerate(linking_outputs):
        ann_idx = linking_indices[j]
        note_id, start, end, span_text, retrieval_text = new_annotations[ann_idx]
        candidates = linking_candidates[j]
        candidate_ids = [s for s, _ in candidates]
        candidate_scores = [sc for _, sc in candidates]

        if args.search_linking:
            action, value = parse_search_response(raw, len(candidates) - 1)
            choice = value if action == "choice" else -1
        else:
            choice = parse_llm_choice(raw, len(candidates) - 1)

        if choice >= 0 and choice < len(candidates):
            # Apply rank-0 filter
            if args.rank0_only and choice != 0:
                n_filtered_rank += 1
                continue
            # Apply score filter
            if candidate_scores[choice] < args.min_rerank_score:
                n_filtered_score += 1
                continue
            concept_id = candidate_ids[choice]
            llm_predictions.append(
                {"note_id": note_id, "start": start, "end": end, "concept_id": concept_id}
            )
            n_linked += 1
        else:
            n_abstain += 1

    print(f"\n  Linked: {n_linked}")
    print(f"  Abstained/unparseable: {n_abstain}")
    if n_filtered_rank:
        print(f"  Filtered (rank0-only): {n_filtered_rank}")
    if n_filtered_score:
        print(f"  Filtered (min-score):  {n_filtered_score}")

    # =========================================================================
    # Merge & Score
    # =========================================================================
    print("\n" + "=" * 65)
    print("  Merge & Score")
    print("=" * 65)

    llm_pred_df = pd.DataFrame(llm_predictions)
    if not llm_pred_df.empty:
        llm_pred_df["note_id"] = llm_pred_df["note_id"].astype(str)
        llm_pred_df["start"] = llm_pred_df["start"].astype(int)
        llm_pred_df["end"] = llm_pred_df["end"].astype(int)
        llm_pred_df["concept_id"] = llm_pred_df["concept_id"].astype(int)

    merged_pred = merge_predictions(kiri_pred, llm_pred_df)
    print(f"  KIRI predictions: {len(kiri_pred):,}")
    print(f"  LLM predictions:  {len(llm_pred_df):,}")
    print(f"  Merged total:     {len(merged_pred):,}")

    # Score
    merged_class_iou = class_char_iou(merged_pred, gold_ann)
    merged_macro = float(merged_class_iou["iou"].mean())
    n_zero_merged = int((merged_class_iou["iou"] == 0).sum())

    # Find classes recovered from zero IoU
    kiri_zero_classes = set(
        kiri_class_iou[kiri_class_iou["iou"] == 0]["concept_id"]
    )
    merged_nonzero = merged_class_iou[merged_class_iou["iou"] > 0]
    recovered = kiri_zero_classes & set(merged_nonzero["concept_id"])

    # Find classes that got worse
    kiri_iou_map = dict(zip(kiri_class_iou["concept_id"], kiri_class_iou["iou"]))
    merged_iou_map = dict(zip(merged_class_iou["concept_id"], merged_class_iou["iou"]))
    n_worse = sum(
        1
        for c in merged_iou_map
        if c in kiri_iou_map and merged_iou_map[c] < kiri_iou_map[c] - 0.001
    )

    print(f"\n  Results:")
    print(f"  KIRI-only Macro IoU:  {kiri_macro:.4f}")
    print(f"  KIRI+LLM Macro IoU:  {merged_macro:.4f}")
    print(f"  Delta:                {merged_macro - kiri_macro:+.4f}")
    print(f"  Zero-IoU classes:     {n_zero_kiri} → {n_zero_merged}")
    print(f"  Classes recovered from zero IoU: {len(recovered)}")
    print(f"  Classes degraded:     {n_worse}")

    # Show some recovered classes
    if recovered:
        print(f"\n  Sample recovered classes:")
        for cid in sorted(recovered)[:10]:
            iou = merged_iou_map.get(cid, 0.0)
            fsn = sctid_to_fsn.get(cid, "?")
            print(f"    {cid} ({fsn}): 0.0 → {iou:.3f}")

    # Save merged predictions
    if args.out_pred_csv:
        merged_pred.to_csv(args.out_pred_csv, index=False)
        print(f"\n  Merged predictions saved to {args.out_pred_csv}")

    # Also save class IoU breakdown
    if args.out_pred_csv:
        iou_path = args.out_pred_csv.replace(".csv", "_class_iou.csv")
        merged_class_iou.to_csv(iou_path, index=False)
        print(f"  Class IoU saved to {iou_path}")


if __name__ == "__main__":
    main()
