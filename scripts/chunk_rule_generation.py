#!/usr/bin/env python3
"""Chunk-level extraction rule generation.

Automated rule generation that operates at the ~134-char chunk level — the
exact unit the extraction LLM sees. Uses contrastive examples (same span text
annotated in one context but not another) to teach decision boundaries.

Pipeline:
  Phase 1 (extract): Select priority notes via greedy set cover, run chunked
                      extraction on each, build per-chunk classification index.
  Phase 2 (group):   Build contrastive groups: same span text across chunks,
                      consistent-FP, consistent-miss, concept groups.
  Phase 3 (generate): For each group, generate candidate rules via Claude Sonnet,
                      test each via delta re-extraction on affected chunks.
  Phase 4 (all):     Full loop: extract → group → generate → iterate.

Usage:
  python scripts/chunk_rule_generation.py --phase extract --max-notes 10
  python scripts/chunk_rule_generation.py --phase group
  python scripts/chunk_rule_generation.py --phase all --max-notes 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))
sys.path.insert(0, str(REPO_ROOT))

from scripts.whole_note_extraction import (
    split_note_into_windows,
    detect_sections,
    get_section_for_offset,
    find_span_offsets,
    EXTRACTION_SYSTEM,
)
from rulebook.general_rule_loop import (
    load_rules,
    save_rules,
    _run_sonnet,
    _parse_sonnet_rules,
    _format_rules_block,
)

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
DEFAULT_MODEL = "/workspaces/snomed-ct-entity-linking/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
DEFAULT_RULES_FILE = REPO_ROOT / "rulebook" / "chunk_extraction_rules.json"
DEFAULT_RUNS_DIR = REPO_ROOT / "scripts" / "chunk_rule_runs"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GoldAnnotation:
    start: int
    end: int
    span: str
    concept_id: int


@dataclass
class ExtractedSpan:
    text: str
    start: int  # absolute offset in note
    end: int


@dataclass
class ChunkInfo:
    note_id: str
    chunk_idx: int
    chunk_text: str
    chunk_offset: int
    chunk_end: int
    section_header: str | None
    leading_context: str
    trailing_context: str
    gold_annotations: list[dict]
    extracted_spans: list[dict]
    tp_spans: list[dict] = field(default_factory=list)
    fp_spans: list[dict] = field(default_factory=list)
    fn_annotations: list[dict] = field(default_factory=list)


@dataclass
class ContrastiveGroup:
    group_type: str  # "contrastive", "consistent_fp", "consistent_miss", "concept"
    span_text: str
    concept_ids: list[int]
    positive_examples: list[dict]  # chunks where span IS gold
    negative_examples: list[dict]  # chunks where span is NOT gold (or FP)
    priority_score: float = 0.0
    affected_chunk_keys: list[tuple[str, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Note selection: greedy set cover for concept coverage
# ---------------------------------------------------------------------------

def select_priority_notes(
    ann_df: pd.DataFrame,
    max_notes: int = 30,
    focus_rare: bool = False,
    rare_threshold: int = 5,
) -> list[str]:
    """Select notes via greedy set cover to maximise concept coverage.

    If focus_rare=True, weights selection to prefer notes containing
    concepts appearing ≤ rare_threshold times in the corpus.
    """
    # Build note → concepts mapping
    note_concepts: dict[str, set[int]] = defaultdict(set)
    concept_counts: dict[int, int] = defaultdict(int)
    for _, row in ann_df.iterrows():
        nid = str(row["note_id"])
        cid = int(row["concept_id"])
        note_concepts[nid].add(cid)
        concept_counts[cid] += 1

    all_concepts = set(concept_counts.keys())
    rare_concepts = {c for c, n in concept_counts.items() if n <= rare_threshold}

    print(f"  Total concepts: {len(all_concepts)}")
    print(f"  Rare concepts (≤{rare_threshold} occurrences): {len(rare_concepts)} "
          f"({100 * len(rare_concepts) / len(all_concepts):.0f}%)")

    selected: list[str] = []
    covered: set[int] = set()
    remaining_notes = set(note_concepts.keys())

    for i in range(min(max_notes, len(remaining_notes))):
        best_note = None
        best_score = -1

        for nid in remaining_notes:
            new_concepts = note_concepts[nid] - covered
            if focus_rare:
                # Weight rare concepts higher
                score = sum(
                    3.0 if c in rare_concepts else 1.0
                    for c in new_concepts
                )
            else:
                score = len(new_concepts)

            if score > best_score:
                best_score = score
                best_note = nid

        if best_note is None or best_score <= 0:
            break

        selected.append(best_note)
        covered.update(note_concepts[best_note])
        remaining_notes.discard(best_note)

    coverage = len(covered) / len(all_concepts) if all_concepts else 0
    rare_coverage = len(covered & rare_concepts) / len(rare_concepts) if rare_concepts else 0
    print(f"  Selected {len(selected)} notes covering {len(covered)}/{len(all_concepts)} "
          f"concepts ({100 * coverage:.1f}%)")
    print(f"  Rare concept coverage: {len(covered & rare_concepts)}/{len(rare_concepts)} "
          f"({100 * rare_coverage:.1f}%)")

    return selected


# ---------------------------------------------------------------------------
# Word-boundary context snapping
# ---------------------------------------------------------------------------

def get_word_boundary_context(
    note_text: str,
    chunk_offset: int,
    chunk_end: int,
    context_chars: int = 50,
) -> tuple[str, str]:
    """Get leading/trailing context snapped to word boundaries."""
    # Leading context
    ctx_start = max(0, chunk_offset - context_chars)
    while ctx_start > 0 and not note_text[ctx_start - 1].isspace():
        ctx_start -= 1
    leading = note_text[ctx_start:chunk_offset]

    # Trailing context
    ctx_end = min(len(note_text), chunk_end + context_chars)
    while ctx_end < len(note_text) and not note_text[ctx_end].isspace():
        ctx_end += 1
    trailing = note_text[chunk_end:ctx_end]

    return leading, trailing


# ---------------------------------------------------------------------------
# Sandwich prompt for extraction
# ---------------------------------------------------------------------------

def build_sandwich_prompt(chunk_text: str, rules_block: str) -> str:
    """Build sandwich prompt: rules → note → rules reminder → output format."""
    rules_section = f"""\
=== ANNOTATION RULES ===
EXTRACT these SNOMED CT concept spans:
- Diagnoses and diseases (e.g., 'atrial fibrillation', 'pneumonia')
- Procedures (e.g., 'Cardiac catheterization', 'intubation', 'discussion')
- Clinical findings and exam terms (e.g., 'RRR', 'CTAB', 'NAD', 'tenderness')
- Body structures (e.g., 'left ventricle', 'abdomen')
- Lab tests and abbreviations (e.g., 'WBC', 'Hgb', 'Plt', 'creatinine')
- Medications in therapeutic context (e.g., 'started on heparin drip')
- Devices (e.g., 'pacemaker', 'stent', 'ventilator')

DO NOT extract:
- Section headers or field labels ('Admission Date:', 'Discharge Diagnosis:', 'Allergies:')
- Narrative verbs ('admitted', 'presented', 'noted', 'treated', 'tolerated')
- Demographics ('man', 'woman', 'year old', 'male', 'female')
- Temporal words ('initially', 'prior', 'subsequently', 'scheduled')
- Admin/disposition ('Home', 'Rehab', 'Extended Care Facility')
- Drug names that are just inventory items in a medication list (not therapeutic references)
- Dosing/route/frequency ('PO', 'BID', 'Q3H', 'tablet', 'Sig:', 'Disp:', 'Refills:')
- Isolated severity qualifiers ('severe', 'mild', 'moderate')
- Consent/risk/workflow words ('risks', 'benefits', 'pending', 'ordered')
{rules_block}=== END RULES ==="""

    return f"""\
{rules_section}

=== NOTE TEXT ===
{chunk_text}
=== END NOTE TEXT ===

=== REMINDER ===
{rules_section}

Copy each span EXACTLY as it appears — character-for-character, including capitalisation.
Respond with ONLY: {{"spans": ["exact span 1", "exact span 2", ...]}}"""


# ---------------------------------------------------------------------------
# Phase 1: Extraction index
# ---------------------------------------------------------------------------

async def _extract_chunk(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    chunk_text: str,
    rules_block: str,
    chunk_idx: int,
    semaphore: asyncio.Semaphore,
    max_context: int = 8192,
) -> list[str]:
    """Send a single chunk to vLLM and parse extracted spans."""
    prompt = build_sandwich_prompt(chunk_text, rules_block)
    # Fixed max_tokens: 1024 is plenty for extracting spans from a ~134-char
    # chunk. With ~6.5K input tokens and 1K output, total ~7.5K fits in 8K.
    max_tokens = 1024
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "chat_template_kwargs": {"enable_thinking": False},  # llama.cpp compat
    }

    async with semaphore:
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if "error" in data:
                    return []
                content = data["choices"][0]["message"].get("content") or ""
                if not content:
                    return []

                # Parse JSON
                match = re.search(r'\{[^{}]*"spans"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
                if match:
                    result = json.loads(match.group())
                    return result.get("spans", [])
                result = json.loads(content)
                return result.get("spans", [])
        except (json.JSONDecodeError, KeyError, aiohttp.ClientError):
            # Try to salvage truncated JSON
            try:
                array_match = re.search(r'"spans"\s*:\s*\[', content)
                if array_match:
                    truncated = content[array_match.end():]
                    return re.findall(r'"([^"]+)"', truncated)
            except Exception:
                pass
            return []


def _classify_chunk(
    chunk_text: str,
    chunk_offset: int,
    chunk_end: int,
    gold_annotations: list[dict],
    extracted_spans: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Classify extracted spans as TP, FP, and gold annotations as FN.

    Uses IoU >= 0.5 for matching.
    """
    def iou(a_start, a_end, b_start, b_end):
        inter = max(0, min(a_end, b_end) - max(a_start, b_start))
        union = (a_end - a_start) + (b_end - b_start) - inter
        return inter / union if union > 0 else 0

    gold_matched = set()
    pred_matched = set()

    # Match extracted → gold
    for pi, ext in enumerate(extracted_spans):
        for gi, gold in enumerate(gold_annotations):
            if iou(ext["start"], ext["end"], gold["start"], gold["end"]) >= 0.5:
                gold_matched.add(gi)
                pred_matched.add(pi)
                break

    tp = [extracted_spans[i] for i in sorted(pred_matched)]
    fp = [extracted_spans[i] for i in range(len(extracted_spans)) if i not in pred_matched]
    fn = [gold_annotations[i] for i in range(len(gold_annotations)) if i not in gold_matched]

    return tp, fp, fn


async def build_extraction_index(
    notes_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    note_ids: list[str],
    rules: list[dict],
    *,
    vllm_url: str,
    model: str,
    window_chars: int = 134,
    context_chars: int = 50,
    max_concurrent: int = 32,
    checkpoint_dir: Path | None = None,
    max_context: int = 8192,
) -> list[ChunkInfo]:
    """Run chunked extraction on selected notes and build per-chunk index.

    Returns a list of ChunkInfo records with TP/FP/FN classification.
    """
    # Build rules block
    universal = [r for r in rules if not r.get("applies_to", {}).get("ancestor_concept_ids")]
    rules_block = ""
    if universal:
        rules_block = "\n\n" + _format_rules_block(universal)

    url = f"{vllm_url}/v1/chat/completions"
    semaphore = asyncio.Semaphore(max_concurrent)
    all_chunks: list[ChunkInfo] = []

    # Check for existing checkpoint
    if checkpoint_dir:
        ckpt_path = checkpoint_dir / "extraction_index.json"
        if ckpt_path.exists():
            print(f"  Loading checkpoint from {ckpt_path}")
            saved = json.loads(ckpt_path.read_text())
            completed_notes = set(saved.get("completed_notes", []))
            for ci in saved.get("chunks", []):
                all_chunks.append(ChunkInfo(**ci))
            remaining = [nid for nid in note_ids if nid not in completed_notes]
            print(f"  Loaded {len(all_chunks)} chunks from {len(completed_notes)} notes, "
                  f"{len(remaining)} remaining")
            note_ids = remaining

    total_chunks = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    async with aiohttp.ClientSession() as session:
        for note_i, note_id in enumerate(note_ids):
            note_row = notes_df[notes_df["note_id"] == note_id]
            if note_row.empty:
                continue
            note_text = note_row.iloc[0]["text"]
            note_anns = ann_df[ann_df["note_id"] == note_id].to_dict("records")

            # Split into windows
            n_windows = max(1, len(note_text) // window_chars)
            windows = split_note_into_windows(note_text, n_windows)
            sections = detect_sections(note_text)

            # Extract from all chunks in parallel
            tasks = []
            chunk_metas = []
            for chunk_idx, (chunk_text, chunk_offset) in enumerate(windows):
                chunk_end = chunk_offset + len(chunk_text)
                section_header = get_section_for_offset(sections, chunk_offset)
                leading, trailing = get_word_boundary_context(
                    note_text, chunk_offset, chunk_end, context_chars,
                )

                # Find gold annotations overlapping this chunk
                chunk_gold = []
                for ann in note_anns:
                    ann_start = int(ann["start"])
                    ann_end = int(ann["end"])
                    # Annotation overlaps chunk if ranges intersect
                    if ann_start < chunk_end and ann_end > chunk_offset:
                        chunk_gold.append({
                            "start": ann_start,
                            "end": ann_end,
                            "span": ann["span"],
                            "concept_id": int(ann["concept_id"]),
                        })

                chunk_metas.append({
                    "note_id": note_id,
                    "chunk_idx": chunk_idx,
                    "chunk_text": chunk_text,
                    "chunk_offset": chunk_offset,
                    "chunk_end": chunk_end,
                    "section_header": section_header,
                    "leading_context": leading,
                    "trailing_context": trailing,
                    "gold_annotations": chunk_gold,
                })

                tasks.append(
                    _extract_chunk(
                        session, url, model, chunk_text, rules_block,
                        chunk_idx, semaphore, max_context=max_context,
                    )
                )

            # Run all extractions for this note
            results = await asyncio.gather(*tasks)

            # Process results
            note_chunks = []
            for meta, span_texts in zip(chunk_metas, results):
                # Find offsets of extracted spans within the chunk
                extracted = []
                for span_text in span_texts:
                    matches = find_span_offsets(
                        meta["chunk_text"], span_text, meta["chunk_offset"],
                    )
                    if matches:
                        for s_in_chunk, e_in_chunk in matches:
                            extracted.append({
                                "text": span_text,
                                "start": meta["chunk_offset"] + s_in_chunk,
                                "end": meta["chunk_offset"] + e_in_chunk,
                            })
                    else:
                        # Try in whole note
                        matches = find_span_offsets(note_text, span_text, 0)
                        for s, e in matches:
                            if s >= meta["chunk_offset"] and e <= meta["chunk_end"]:
                                extracted.append({"text": span_text, "start": s, "end": e})
                                break

                # Deduplicate by (start, end)
                seen = set()
                unique_extracted = []
                for ext in extracted:
                    key = (ext["start"], ext["end"])
                    if key not in seen:
                        seen.add(key)
                        unique_extracted.append(ext)

                # Classify
                tp, fp, fn = _classify_chunk(
                    meta["chunk_text"],
                    meta["chunk_offset"],
                    meta["chunk_end"],
                    meta["gold_annotations"],
                    unique_extracted,
                )

                chunk = ChunkInfo(
                    note_id=meta["note_id"],
                    chunk_idx=meta["chunk_idx"],
                    chunk_text=meta["chunk_text"],
                    chunk_offset=meta["chunk_offset"],
                    chunk_end=meta["chunk_end"],
                    section_header=meta["section_header"],
                    leading_context=meta["leading_context"],
                    trailing_context=meta["trailing_context"],
                    gold_annotations=meta["gold_annotations"],
                    extracted_spans=unique_extracted,
                    tp_spans=tp,
                    fp_spans=fp,
                    fn_annotations=fn,
                )
                note_chunks.append(chunk)
                total_chunks += 1
                total_tp += len(tp)
                total_fp += len(fp)
                total_fn += len(fn)

            all_chunks.extend(note_chunks)

            n_gold = sum(len(c.gold_annotations) for c in note_chunks)
            n_tp = sum(len(c.tp_spans) for c in note_chunks)
            n_fp = sum(len(c.fp_spans) for c in note_chunks)
            n_fn = sum(len(c.fn_annotations) for c in note_chunks)
            recall = n_tp / n_gold * 100 if n_gold else 0
            prec = n_tp / (n_tp + n_fp) * 100 if (n_tp + n_fp) else 0
            print(f"  [{note_i + 1}/{len(note_ids)}] {note_id}: "
                  f"{len(note_chunks)} chunks, {n_gold} gold, "
                  f"TP={n_tp} FP={n_fp} FN={n_fn}  "
                  f"recall={recall:.1f}% prec={prec:.1f}%")

            # Checkpoint every 5 notes
            if checkpoint_dir and (note_i + 1) % 5 == 0:
                _save_extraction_checkpoint(checkpoint_dir, all_chunks)

    # Final checkpoint
    if checkpoint_dir:
        _save_extraction_checkpoint(checkpoint_dir, all_chunks)

    # Print summary
    total_gold = total_tp + total_fn
    recall = total_tp / total_gold * 100 if total_gold else 0
    prec = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) else 0
    print(f"\n  Extraction index complete: {total_chunks} chunks")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"  Recall={recall:.1f}% Precision={prec:.1f}%")

    return all_chunks


def _save_extraction_checkpoint(checkpoint_dir: Path, chunks: list[ChunkInfo]):
    """Save extraction index checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed = sorted({c.note_id for c in chunks})
    data = {
        "completed_notes": completed,
        "chunks": [asdict(c) for c in chunks],
    }
    path = checkpoint_dir / "extraction_index.json"
    path.write_text(json.dumps(data))
    print(f"  Checkpoint: {len(chunks)} chunks from {len(completed)} notes → {path}")


# ---------------------------------------------------------------------------
# Phase 2: Intelligent grouping
# ---------------------------------------------------------------------------

def build_contrastive_groups(
    chunks: list[ChunkInfo],
    ann_df: pd.DataFrame,
    max_groups: int = 50,
    max_examples: int = 5,
) -> list[ContrastiveGroup]:
    """Build contrastive groups from extraction index.

    Priority order:
    1. Contrastive: same span text is gold in some chunks, not in others
    2. Consistent-FP: span texts always extracted but never gold
    3. Consistent-miss: span texts always gold but always missed
    4. Concept groups: same concept_id, different surface forms
    """
    # Build inverted index: span_text_lower → list of occurrences
    span_index: dict[str, list[dict]] = defaultdict(list)

    # Track all gold span texts across the FULL corpus (not just selected notes)
    all_gold_spans: set[str] = set()
    for _, row in ann_df.iterrows():
        all_gold_spans.add(str(row["span"]).lower())

    for chunk in chunks:
        chunk_key = (chunk.note_id, chunk.chunk_idx)
        chunk_text_lower = chunk.chunk_text.lower()

        # Gold annotations in this chunk
        gold_spans_here = set()
        for gold in chunk.gold_annotations:
            span_lower = gold["span"].lower()
            gold_spans_here.add(span_lower)
            span_index[span_lower].append({
                "chunk_key": chunk_key,
                "is_gold": True,
                "concept_id": gold["concept_id"],
                "section": chunk.section_header,
                "classification": "gold",
                "chunk": chunk,
                "original_span": gold["span"],
            })

        # Extracted spans (check if they're TP or FP)
        for ext in chunk.extracted_spans:
            span_lower = ext["text"].lower()
            is_tp = any(
                tp["text"].lower() == span_lower
                for tp in chunk.tp_spans
            )
            if not is_tp:
                span_index[span_lower].append({
                    "chunk_key": chunk_key,
                    "is_gold": False,
                    "concept_id": None,
                    "section": chunk.section_header,
                    "classification": "fp",
                    "chunk": chunk,
                    "original_span": ext["text"],
                })

        # Check for span texts that appear in the chunk text but were gold
        # and missed (FN)
        for fn in chunk.fn_annotations:
            span_lower = fn["span"].lower()
            # Only add if not already added as gold
            if span_lower not in gold_spans_here:
                span_index[span_lower].append({
                    "chunk_key": chunk_key,
                    "is_gold": True,
                    "concept_id": fn["concept_id"],
                    "section": chunk.section_header,
                    "classification": "fn",
                    "chunk": chunk,
                    "original_span": fn["span"],
                })

    groups: list[ContrastiveGroup] = []

    # --- 1. Contrastive groups ---
    for span_text, occurrences in span_index.items():
        has_gold = any(o["is_gold"] for o in occurrences)
        has_non_gold = any(not o["is_gold"] for o in occurrences)
        # Also check: does this span text appear in chunks where it IS gold
        # vs chunks where it's present but NOT gold
        if has_gold and has_non_gold:
            concept_ids = sorted({o["concept_id"] for o in occurrences if o["concept_id"]})
            positives = [o for o in occurrences if o["is_gold"]]
            negatives = [o for o in occurrences if not o["is_gold"]]

            # Priority: more concept_ids = more ambiguous
            n_concepts = len(concept_ids)
            priority = n_concepts * 10 + len(occurrences)

            groups.append(ContrastiveGroup(
                group_type="contrastive",
                span_text=span_text,
                concept_ids=concept_ids,
                positive_examples=_sample_diverse(positives, max_examples),
                negative_examples=_sample_diverse(negatives, max_examples),
                priority_score=priority,
                affected_chunk_keys=[o["chunk_key"] for o in occurrences],
            ))

    # --- 2. Consistent-FP groups ---
    for span_text, occurrences in span_index.items():
        if span_text in all_gold_spans:
            continue  # Skip if it's ever gold anywhere
        fp_occurrences = [o for o in occurrences if o["classification"] == "fp"]
        if len(fp_occurrences) >= 2:
            groups.append(ContrastiveGroup(
                group_type="consistent_fp",
                span_text=span_text,
                concept_ids=[],
                positive_examples=[],  # never gold
                negative_examples=_sample_diverse(fp_occurrences, max_examples),
                priority_score=len(fp_occurrences),
                affected_chunk_keys=[o["chunk_key"] for o in fp_occurrences],
            ))

    # --- 3. Consistent-miss groups ---
    for span_text, occurrences in span_index.items():
        gold_occ = [o for o in occurrences if o["is_gold"]]
        fn_occ = [o for o in occurrences if o["classification"] == "fn"]
        if gold_occ and fn_occ and not any(
            o["classification"] == "fp" for o in occurrences
        ):
            # All gold occurrences were missed
            all_fn = all(o["classification"] == "fn" for o in gold_occ)
            if all_fn and len(fn_occ) >= 1:
                concept_ids = sorted({o["concept_id"] for o in gold_occ if o["concept_id"]})
                # Compute rarity score from corpus-wide counts
                concept_freq = ann_df[ann_df["concept_id"].isin(concept_ids)].shape[0]
                rarity_bonus = 10 if concept_freq <= 5 else 0
                priority = len(fn_occ) + rarity_bonus

                groups.append(ContrastiveGroup(
                    group_type="consistent_miss",
                    span_text=span_text,
                    concept_ids=concept_ids,
                    positive_examples=_sample_diverse(gold_occ, max_examples),
                    negative_examples=[],  # model never extracts this
                    priority_score=priority,
                    affected_chunk_keys=[o["chunk_key"] for o in gold_occ],
                ))

    # Sort by priority (highest first) and cap
    groups.sort(key=lambda g: g.priority_score, reverse=True)

    # Print stats by type
    type_counts = defaultdict(int)
    for g in groups:
        type_counts[g.group_type] += 1
    print(f"\n  Contrastive groups built: {len(groups)} total")
    for gtype, count in sorted(type_counts.items()):
        print(f"    {gtype}: {count}")

    if len(groups) > max_groups:
        groups = groups[:max_groups]
        print(f"  Capped to top {max_groups} groups")

    return groups


def _sample_diverse(
    occurrences: list[dict],
    max_n: int,
) -> list[dict]:
    """Sample up to max_n occurrences preferring diversity across sections and notes."""
    if len(occurrences) <= max_n:
        return occurrences

    # Prefer variety of sections and notes
    by_section: dict[str | None, list[dict]] = defaultdict(list)
    for o in occurrences:
        by_section[o["section"]].append(o)

    sampled = []
    section_keys = list(by_section.keys())
    idx = 0
    while len(sampled) < max_n:
        sec = section_keys[idx % len(section_keys)]
        if by_section[sec]:
            sampled.append(by_section[sec].pop(0))
        idx += 1
        if all(not v for v in by_section.values()):
            break

    return sampled


# ---------------------------------------------------------------------------
# Phase 3: Rule generation from contrastive batches
# ---------------------------------------------------------------------------

CHUNK_RULE_GEN_SYSTEM = """\
You are a clinical NLP expert improving a span extraction pipeline.

The pipeline processes ~134-character chunks of clinical discharge notes and \
extracts annotatable SNOMED CT spans from each chunk. It sees ONLY the chunk \
text, NOT the surrounding context.

You will be shown contrastive examples: the same span text appearing in different \
chunks, where it IS annotated (extract it) in some cases and IS NOT annotated \
(skip it) in others. Your job is to write rules that help the extraction model \
distinguish when to extract vs when to skip.

The [BEFORE] and [AFTER] context is shown for YOUR understanding only — the \
extraction model does NOT see it. Write rules based on patterns visible WITHIN \
the chunk text itself.

Response format — a single JSON object:
{
  "rules": [
    {
      "id": "CE###",
      "rule": "Specific extraction rule (<100 words).",
      "priority": 2
    }
  ]
}

Rules guidance:
  - Each rule MUST be under 100 words.
  - BE MAXIMALLY SPECIFIC. Reference exact strings, field names, formatting \
    patterns, and character sequences you see in the examples. Do NOT write \
    generic guidance like "extract when in clinical context" — instead write \
    "extract when followed by '-[number]*' lab value format (e.g., 'RBC-4*', \
    'WBC-12')".
  - Quote literal text patterns from the examples. Good: "skip when chunk \
    contains 'Bi Tri Pat Ach' (DTR reflex table header)". Bad: "skip in \
    motor/sensory exam notation".
  - Name specific field formats, delimiters, keywords, and section markers \
    visible in the chunk text. The more concrete the pattern, the better.
  - Write rules the extraction model can apply using ONLY the chunk text.
  - Priority: P1 = hard override, P2 = standard, P3 = guideline.
  - Prefer P2 unless the pattern absolutely requires override.
  - Generate 1-3 rules per group of examples.\
"""


def _build_group_prompt(
    group: ContrastiveGroup,
    existing_rules: list[dict],
) -> tuple[str, int]:
    """Build the user prompt for a contrastive group. Returns (prompt, next_id)."""
    lines = [f'SPAN: "{group.span_text}"', ""]

    if group.group_type == "contrastive":
        if group.positive_examples:
            lines.append("=== ANNOTATED (extract it) ===")
            for i, ex in enumerate(group.positive_examples):
                chunk: ChunkInfo = ex["chunk"]
                concept_str = f"Gold concept: {ex['concept_id']}" if ex.get("concept_id") else ""
                lines.append(f"[Example {i + 1}]")
                lines.append(f"  Section: {chunk.section_header or 'unknown'}")
                lines.append(f"  [BEFORE]: \"{chunk.leading_context[-60:]}\"")
                lines.append(f"  [CHUNK]:  \"{chunk.chunk_text}\"")
                lines.append(f"  [AFTER]:  \"{chunk.trailing_context[:60]}\"")
                if concept_str:
                    lines.append(f"  {concept_str}")
                lines.append("")

        if group.negative_examples:
            lines.append("=== NOT ANNOTATED (skip it) ===")
            for i, ex in enumerate(group.negative_examples):
                chunk = ex["chunk"]
                lines.append(f"[Example {i + 1}]")
                lines.append(f"  Section: {chunk.section_header or 'unknown'}")
                lines.append(f"  [BEFORE]: \"{chunk.leading_context[-60:]}\"")
                lines.append(f"  [CHUNK]:  \"{chunk.chunk_text}\"")
                lines.append(f"  [AFTER]:  \"{chunk.trailing_context[:60]}\"")
                lines.append(f"  Note: extracted but not a gold annotation")
                lines.append("")

    elif group.group_type == "consistent_fp":
        lines.append("=== FALSE POSITIVES (model extracts but should NOT) ===")
        for i, ex in enumerate(group.negative_examples):
            chunk = ex["chunk"]
            lines.append(f"[Example {i + 1}]")
            lines.append(f"  Section: {chunk.section_header or 'unknown'}")
            lines.append(f"  [BEFORE]: \"{chunk.leading_context[-60:]}\"")
            lines.append(f"  [CHUNK]:  \"{chunk.chunk_text}\"")
            lines.append(f"  [AFTER]:  \"{chunk.trailing_context[:60]}\"")
            lines.append("")
        lines.append(f"This span (\"{group.span_text}\") is NEVER annotated in the corpus. "
                      "Write rules to suppress extraction of this pattern.")

    elif group.group_type == "consistent_miss":
        lines.append("=== ALWAYS MISSED (model should extract but doesn't) ===")
        for i, ex in enumerate(group.positive_examples):
            chunk = ex["chunk"]
            concept_str = f"Gold concept: {ex['concept_id']}" if ex.get("concept_id") else ""
            lines.append(f"[Example {i + 1}]")
            lines.append(f"  Section: {chunk.section_header or 'unknown'}")
            lines.append(f"  [BEFORE]: \"{chunk.leading_context[-60:]}\"")
            lines.append(f"  [CHUNK]:  \"{chunk.chunk_text}\"")
            lines.append(f"  [AFTER]:  \"{chunk.trailing_context[:60]}\"")
            if concept_str:
                lines.append(f"  {concept_str}")
            lines.append("")
        lines.append(f"This span (\"{group.span_text}\") is ALWAYS gold-annotated but the model "
                      "never extracts it. Write rules to encourage extraction.")

    # Show at most 15 existing rules to stay within context window.
    # Prefer showing the most recent rules (likely most relevant).
    shown_rules = existing_rules[-15:] if len(existing_rules) > 15 else existing_rules
    if shown_rules:
        lines.append(f"\nEXISTING RULES ({len(existing_rules)} total, "
                      f"showing last {len(shown_rules)} — do not duplicate):")
        for r in shown_rules:
            lines.append(f"  {r['id']}: {r['rule'][:120]}")

    existing_ids = [
        int(m.group())
        for r in existing_rules
        if (m := re.search(r"\d+", r.get("id", "")))
    ]
    next_id = max(existing_ids, default=0) + 1

    lines.append(f"\nGenerate 1-3 rules (start IDs from CE{next_id:03d}).")
    lines.append("Return ONLY the JSON object with 'rules' key.")

    return "\n".join(lines), next_id


async def _vllm_rule_gen(
    user_prompt: str,
    *,
    vllm_url: str,
    model: str,
    max_context: int = 4096,
    n: int = 1,
    temperature: float = 0.3,
) -> list[str]:
    """Generate rules via the local vLLM model.

    Returns a list of N completion strings (one per sample).
    Uses the OpenAI ``n`` parameter so the prompt is prefilled once
    and decoded N times — much cheaper than N separate requests.
    """
    # Rough token estimate: ~3.5 chars/token for clinical text
    est_input_tokens = (len(CHUNK_RULE_GEN_SYSTEM) + len(user_prompt)) // 3
    max_tokens = min(1024, max(256, max_context - est_input_tokens - 100))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": CHUNK_RULE_GEN_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "chat_template_kwargs": {"enable_thinking": False},  # llama.cpp compat
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{vllm_url}/v1/chat/completions", json=payload) as resp:
            data = await resp.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            return [
                c["message"].get("content") or ""
                for c in data["choices"]
            ]


def generate_rules_for_group(
    group: ContrastiveGroup,
    existing_rules: list[dict],
    *,
    backend: str = "sonnet",
    sonnet_model: str = "claude-sonnet-4-6",
    vllm_url: str = "http://localhost:8000",
    vllm_model: str = DEFAULT_MODEL,
    max_context: int = 4096,
    best_of_n: int = 1,
) -> list[list[dict]]:
    """Generate candidate rule sets for a contrastive group.

    Returns a list of rule-sets (each is a list[dict]).  When best_of_n > 1,
    multiple sets are generated in a single vLLM call (using the ``n`` param)
    with higher temperature for diversity.  The caller tests each set and
    keeps only the best-scoring one.

    backend: "sonnet" (Claude API) or "vllm" (local Qwen3 model).
    """
    user_prompt, next_id = _build_group_prompt(group, existing_rules)

    if backend == "vllm":
        temperature = 0.8 if best_of_n > 1 else 0.3
        try:
            completions = asyncio.run(
                _vllm_rule_gen(
                    user_prompt, vllm_url=vllm_url, model=vllm_model,
                    max_context=max_context, n=best_of_n,
                    temperature=temperature,
                )
            )
        except Exception as e:
            print(f"    ERROR generating rules via vLLM: {e}")
            return []
        # Parse each completion into a rule set
        rule_sets = []
        for raw in completions:
            rules, _ = _parse_sonnet_rules(raw)
            if rules:
                rule_sets.append(rules)
        return rule_sets
    else:
        # Sonnet: no n parameter, call N times sequentially
        rule_sets = []
        for i in range(best_of_n):
            try:
                raw = _run_sonnet(CHUNK_RULE_GEN_SYSTEM, user_prompt, model=sonnet_model)
            except Exception as e:
                print(f"    ERROR generating rules via Sonnet (sample {i+1}): {e}")
                continue
            rules, _ = _parse_sonnet_rules(raw)
            if rules:
                rule_sets.append(rules)
        return rule_sets


# ---------------------------------------------------------------------------
# Phase 4: Rule testing (delta computation)
# ---------------------------------------------------------------------------

async def test_candidate_rule(
    rule: dict,
    current_rules: list[dict],
    affected_chunks: list[ChunkInfo],
    *,
    vllm_url: str,
    model: str,
    max_concurrent: int = 32,
) -> tuple[int, int, list[dict], list[dict]]:
    """Test a candidate rule by re-extracting affected chunks.

    Returns (n_fixed, n_broken, fixed_details, broken_details).
    Fixed: FN→TP or FP→gone. Broken: TP→FN or new FP.
    """
    # Build rules block with the new rule injected
    test_rules = current_rules + [rule]
    universal = [r for r in test_rules if not r.get("applies_to", {}).get("ancestor_concept_ids")]
    rules_block = ""
    if universal:
        rules_block = "\n\n" + _format_rules_block(universal)

    url = f"{vllm_url}/v1/chat/completions"
    semaphore = asyncio.Semaphore(max_concurrent)

    # Re-extract each affected chunk
    async with aiohttp.ClientSession() as session:
        tasks = [
            _extract_chunk(session, url, model, chunk.chunk_text, rules_block, i, semaphore)
            for i, chunk in enumerate(affected_chunks)
        ]
        results = await asyncio.gather(*tasks)

    n_fixed = 0
    n_broken = 0
    fixed_details = []
    broken_details = []

    for chunk, new_span_texts in zip(affected_chunks, results):
        # Find offsets of newly extracted spans
        new_extracted = []
        for span_text in new_span_texts:
            matches = find_span_offsets(chunk.chunk_text, span_text, chunk.chunk_offset)
            if matches:
                for s, e in matches:
                    new_extracted.append({
                        "text": span_text,
                        "start": chunk.chunk_offset + s,
                        "end": chunk.chunk_offset + e,
                    })

        # Deduplicate
        seen = set()
        unique_new = []
        for ext in new_extracted:
            key = (ext["start"], ext["end"])
            if key not in seen:
                seen.add(key)
                unique_new.append(ext)

        # Classify new results
        new_tp, new_fp, new_fn = _classify_chunk(
            chunk.chunk_text, chunk.chunk_offset, chunk.chunk_end,
            chunk.gold_annotations, unique_new,
        )

        # Compare old vs new
        old_tp_set = {(s["start"], s["end"]) for s in chunk.tp_spans}
        old_fp_set = {(s["start"], s["end"]) for s in chunk.fp_spans}
        old_fn_set = {(a["start"], a["end"]) for a in chunk.fn_annotations}

        new_tp_set = {(s["start"], s["end"]) for s in new_tp}
        new_fp_set = {(s["start"], s["end"]) for s in new_fp}
        new_fn_set = {(a["start"], a["end"]) for a in new_fn}

        # Fixed: was FN, now TP
        for key in old_fn_set - new_fn_set:
            if key in new_tp_set:
                n_fixed += 1
                fixed_details.append({
                    "note_id": chunk.note_id,
                    "chunk_idx": chunk.chunk_idx,
                    "type": "fn_to_tp",
                    "span_range": list(key),
                })

        # Fixed: was FP, now gone
        for key in old_fp_set - new_fp_set:
            if key not in new_tp_set:
                n_fixed += 1
                fixed_details.append({
                    "note_id": chunk.note_id,
                    "chunk_idx": chunk.chunk_idx,
                    "type": "fp_removed",
                    "span_range": list(key),
                })

        # Broken: was TP, now FN
        for key in old_tp_set - new_tp_set:
            if key in new_fn_set:
                n_broken += 1
                broken_details.append({
                    "note_id": chunk.note_id,
                    "chunk_idx": chunk.chunk_idx,
                    "type": "tp_to_fn",
                    "span_range": list(key),
                })

        # Broken: new FP (wasn't FP before)
        for key in new_fp_set - old_fp_set:
            n_broken += 1
            broken_details.append({
                "note_id": chunk.note_id,
                "chunk_idx": chunk.chunk_idx,
                "type": "new_fp",
                "span_range": list(key),
            })

    return n_fixed, n_broken, fixed_details, broken_details


# ---------------------------------------------------------------------------
# Phase 5: State management
# ---------------------------------------------------------------------------

def create_run_dir(args: argparse.Namespace) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = DEFAULT_RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    return run_dir


def save_state(
    run_dir: Path,
    *,
    phase: str,
    group_idx: int = 0,
    rules: list[dict] | None = None,
    processed_groups: list[str] | None = None,
    holdout_note_ids: list[str] | None = None,
    holdout_history: list[dict] | None = None,
) -> None:
    state = {
        "phase": phase,
        "group_idx": group_idx,
        "rules": rules or [],
        "processed_groups": processed_groups or [],
        "holdout_note_ids": holdout_note_ids or [],
        "holdout_history": holdout_history or [],
    }
    path = run_dir / "state.json"
    path.write_text(json.dumps(state, indent=2))


def load_state(run_dir: Path) -> dict:
    return json.loads((run_dir / "state.json").read_text())


def log_rule_change(
    run_dir: Path,
    group_idx: int,
    rule: dict,
    action: str,
    n_fixed: int,
    n_broken: int,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "group_idx": group_idx,
        "action": action,
        "rule_id": rule.get("id", "?"),
        "rule_text": rule.get("rule", ""),
        "n_fixed": n_fixed,
        "n_broken": n_broken,
        "net": n_fixed - n_broken,
    }
    path = run_dir / "rule_changelog.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--rules-file", type=Path, default=DEFAULT_RULES_FILE)
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--window-chars", type=int, default=134)
    parser.add_argument("--context-chars", type=int, default=50)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--max-notes", type=int, default=30)
    parser.add_argument("--focus-rare", action="store_true")
    parser.add_argument("--max-groups", type=int, default=50)
    parser.add_argument("--max-rules", type=int, default=30,
                        help="Stop accepting rules after this count")
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    parser.add_argument("--holdout-seed", type=int, default=42)
    parser.add_argument("--sonnet-model", default="claude-sonnet-4-6")
    parser.add_argument(
        "--rule-gen-backend", default="vllm",
        choices=["vllm", "sonnet"],
        help="Backend for rule generation: vllm (local Qwen3) or sonnet (Claude API)",
    )
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of generate→test iterations over the groups")
    parser.add_argument("--best-of-n", type=int, default=1,
                        help="Generate N candidate rule sets per group, test all, keep best")
    parser.add_argument("--max-context", type=int, default=4096,
                        help="Max context window of the vLLM model (for token budgeting)")
    parser.add_argument(
        "--phase", default="all",
        choices=["extract", "group", "generate", "all"],
        help="Which phase to run",
    )
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Pre-flight: verify vLLM server
    # ------------------------------------------------------------------
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(f"{args.vllm_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            model_ids = [m["id"] for m in data.get("data", [])]
            print(f"vLLM server OK at {args.vllm_url}  model(s): {model_ids}")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"ERROR: Cannot reach vLLM server at {args.vllm_url}")
        print(f"  {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data from {args.split_dir} ...")
    notes_df = pd.read_csv(args.split_dir / "train_notes.csv")
    ann_df = pd.read_csv(args.split_dir / "train_annotations.csv")
    for col in ("start", "end", "concept_id", "annotation_id"):
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce").fillna(0).astype(int)
    print(f"  {len(ann_df):,} annotations, {len(notes_df):,} notes")

    # Ensure note_id is string consistently
    notes_df["note_id"] = notes_df["note_id"].astype(str)
    ann_df["note_id"] = ann_df["note_id"].astype(str)

    # ------------------------------------------------------------------
    # Load or create rules
    # ------------------------------------------------------------------
    rules = load_rules(args.rules_file)

    # ------------------------------------------------------------------
    # Holdout split (by note_id)
    # ------------------------------------------------------------------
    import random
    all_note_ids = sorted(notes_df["note_id"].unique().tolist())
    rng = random.Random(args.holdout_seed)
    shuffled = list(all_note_ids)
    rng.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * args.holdout_frac))
    holdout_ids = set(shuffled[:n_holdout])
    available_ids = [nid for nid in all_note_ids if nid not in holdout_ids]
    print(f"  Holdout: {len(holdout_ids)} notes  |  Available: {len(available_ids)} notes")

    # ------------------------------------------------------------------
    # Note selection
    # ------------------------------------------------------------------
    print(f"\nSelecting priority notes (max={args.max_notes}, focus_rare={args.focus_rare}) ...")
    avail_ann = ann_df[~ann_df["note_id"].isin(holdout_ids)]
    selected_notes = select_priority_notes(
        avail_ann, max_notes=args.max_notes, focus_rare=args.focus_rare,
    )

    # Estimate chunks
    total_chars = 0
    for nid in selected_notes:
        row = notes_df[notes_df["note_id"] == nid]
        if not row.empty:
            total_chars += len(row.iloc[0]["text"])
    est_chunks = total_chars // args.window_chars
    print(f"  Estimated chunks: {est_chunks}")

    # ------------------------------------------------------------------
    # Run directory
    # ------------------------------------------------------------------
    if args.resume:
        run_dir = args.resume.resolve()
    else:
        run_dir = create_run_dir(args)
    print(f"  Run directory: {run_dir}")

    # Save note selection metadata for provenance
    note_meta = []
    for nid in selected_notes:
        n_ann = len(ann_df[ann_df["note_id"] == nid])
        n_concepts = ann_df[ann_df["note_id"] == nid]["concept_id"].nunique()
        row = notes_df[notes_df["note_id"] == nid]
        n_chars = len(row.iloc[0]["text"]) if not row.empty else 0
        note_meta.append({
            "note_id": nid,
            "n_annotations": n_ann,
            "n_concepts": n_concepts,
            "n_chars": n_chars,
            "est_chunks": n_chars // args.window_chars,
        })
    selection_info = {
        "selected_notes": note_meta,
        "holdout_note_ids": sorted(holdout_ids),
        "total_notes_in_corpus": len(all_note_ids),
        "total_available": len(available_ids),
        "selection_method": "greedy_set_cover",
        "focus_rare": args.focus_rare,
        "max_notes": args.max_notes,
    }
    (run_dir / "note_selection.json").write_text(
        json.dumps(selection_info, indent=2)
    )
    print(f"  Note selection metadata saved to {run_dir / 'note_selection.json'}")

    # ------------------------------------------------------------------
    # Phase: Extract
    # ------------------------------------------------------------------
    if args.phase in ("extract", "all"):
        print(f"\n{'='*70}")
        print("PHASE 1: EXTRACTION INDEX")
        print(f"{'='*70}")
        t0 = time.time()

        chunks = asyncio.run(
            build_extraction_index(
                notes_df, ann_df, selected_notes, rules,
                vllm_url=args.vllm_url,
                model=args.model,
                window_chars=args.window_chars,
                context_chars=args.context_chars,
                max_concurrent=args.max_concurrent,
                checkpoint_dir=run_dir,
                max_context=args.max_context,
            )
        )

        print(f"\n  Phase 1 done in {time.time() - t0:.1f}s")

        if args.phase == "extract":
            save_state(run_dir, phase="extract", rules=rules,
                       holdout_note_ids=sorted(holdout_ids))
            print(f"\nExtraction index saved. Run --phase group to continue.")
            return
    else:
        # Load existing extraction index
        ckpt_path = run_dir / "extraction_index.json"
        if not ckpt_path.exists():
            print(f"ERROR: No extraction index found at {ckpt_path}")
            print("Run --phase extract first.")
            sys.exit(1)
        saved = json.loads(ckpt_path.read_text())
        chunks = [ChunkInfo(**ci) for ci in saved["chunks"]]
        print(f"  Loaded {len(chunks)} chunks from extraction index")

    # ------------------------------------------------------------------
    # Phase: Group (also runs when generate needs it)
    # ------------------------------------------------------------------
    groups: list[ContrastiveGroup] | None = None

    if args.phase in ("group", "generate", "all"):
        print(f"\n{'='*70}")
        print("PHASE 2: INTELLIGENT GROUPING")
        print(f"{'='*70}")

        groups = build_contrastive_groups(
            chunks, ann_df, max_groups=args.max_groups,
        )

        # Save groups
        groups_data = []
        for g in groups:
            gd = {
                "group_type": g.group_type,
                "span_text": g.span_text,
                "concept_ids": g.concept_ids,
                "priority_score": g.priority_score,
                "n_positive": len(g.positive_examples),
                "n_negative": len(g.negative_examples),
                "n_affected_chunks": len(g.affected_chunk_keys),
            }
            groups_data.append(gd)
        (run_dir / "groups.json").write_text(json.dumps(groups_data, indent=2))

        # Print top groups
        print(f"\n  Top 15 groups:")
        for i, g in enumerate(groups[:15]):
            print(f"    [{i + 1}] {g.group_type:16s} \"{g.span_text}\"  "
                  f"score={g.priority_score:.0f}  "
                  f"+{len(g.positive_examples)}/-{len(g.negative_examples)}  "
                  f"concepts={g.concept_ids[:3]}")

        if args.phase == "group":
            save_state(run_dir, phase="group", rules=rules,
                       holdout_note_ids=sorted(holdout_ids))
            print(f"\nGroups saved. Run --phase generate to continue.")
            return

    # ------------------------------------------------------------------
    # Phase: Generate (rule generation + testing loop)
    # ------------------------------------------------------------------
    if args.phase in ("generate", "all"):
        backend = args.rule_gen_backend
        best_of_n = args.best_of_n
        backend_label = (
            f"vLLM ({args.model.split('/')[-1]})"
            if backend == "vllm"
            else f"Sonnet ({args.sonnet_model})"
        )
        bon_label = f", best-of-{best_of_n}" if best_of_n > 1 else ""
        print(f"\n{'='*70}")
        print(f"PHASE 3: RULE GENERATION + TESTING  "
              f"[backend: {backend_label}{bon_label}]")
        print(f"{'='*70}")

        # Build chunk lookup
        chunk_lookup: dict[tuple[str, int], ChunkInfo] = {}
        for c in chunks:
            chunk_lookup[(c.note_id, c.chunk_idx)] = c

        accepted_rules = list(rules)
        total_accepted = 0
        total_rejected = 0

        for round_idx in range(args.rounds):
            round_accepted = 0
            round_rejected = 0

            print(f"\n  {'='*60}")
            print(f"  ROUND {round_idx + 1}/{args.rounds}  "
                  f"(rules so far: {len(accepted_rules)})")
            print(f"  {'='*60}")

            for gi, group in enumerate(groups):
                if len(accepted_rules) >= args.max_rules:
                    print(f"\n  Reached max rules ({args.max_rules}), stopping.")
                    break

                print(f"\n  {'─'*50}")
                print(f"  [{round_idx + 1}/{args.rounds}] "
                      f"Group {gi + 1}/{len(groups)}: {group.group_type} "
                      f"\"{group.span_text}\"  "
                      f"(score={group.priority_score:.0f}, "
                      f"concepts={group.concept_ids[:3]})")

                # Generate candidate rule sets
                n_label = f"{best_of_n} samples" if best_of_n > 1 else ""
                print(f"    Generating rules via {backend_label} "
                      f"{'(' + n_label + ') ' if n_label else ''}...")
                t0 = time.time()
                rule_sets = generate_rules_for_group(
                    group, accepted_rules,
                    backend=backend,
                    sonnet_model=args.sonnet_model,
                    vllm_url=args.vllm_url,
                    vllm_model=args.model,
                    max_context=args.max_context,
                    best_of_n=best_of_n,
                )
                gen_elapsed = time.time() - t0
                n_sets = len(rule_sets)
                total_rules_across = sum(len(rs) for rs in rule_sets)
                print(f"    Generated {n_sets} rule set(s) "
                      f"({total_rules_across} rules total) "
                      f"in {gen_elapsed:.1f}s")

                if not rule_sets:
                    continue

                # Find affected chunks
                affected = [
                    chunk_lookup[key]
                    for key in group.affected_chunk_keys
                    if key in chunk_lookup
                ]
                if not affected:
                    print(f"    No affected chunks found, skipping.")
                    continue

                print(f"    Testing {n_sets} set(s) on "
                      f"{len(affected)} affected chunks ...")

                # --- Best-of-N: test each rule set, pick the best ---
                set_scores: list[tuple[int, int, int, list[dict]]] = []
                # (net, n_fixed, n_broken, rule_set)

                for si, rule_set in enumerate(rule_sets):
                    set_label = f"set {si+1}/{n_sets}" if n_sets > 1 else ""
                    if n_sets > 1:
                        rules_preview = ", ".join(
                            r.get("id", "?") for r in rule_set
                        )
                        print(f"    Testing {set_label} "
                              f"[{rules_preview}] ...")

                    # Test all rules in this set together by injecting
                    # them as a group
                    set_fixed = 0
                    set_broken = 0
                    for rule in rule_set:
                        t0 = time.time()
                        n_fixed, n_broken, fixed_d, broken_d = asyncio.run(
                            test_candidate_rule(
                                rule, accepted_rules, affected,
                                vllm_url=args.vllm_url,
                                model=args.model,
                                max_concurrent=args.max_concurrent,
                            )
                        )
                        net = n_fixed - n_broken
                        elapsed = time.time() - t0
                        set_fixed += n_fixed
                        set_broken += n_broken

                        prefix = f"      [{set_label}] " if n_sets > 1 else "    "
                        print(f"{prefix}{rule.get('id', '?')}: "
                              f"+{n_fixed}/-{n_broken} net={net}  "
                              f"({elapsed:.1f}s)")

                    set_net = set_fixed - set_broken
                    set_scores.append(
                        (set_net, set_fixed, set_broken, rule_set)
                    )

                    if n_sets > 1:
                        print(f"      {set_label} total: "
                              f"+{set_fixed}/-{set_broken} "
                              f"net={set_net}")

                # Pick the best set
                set_scores.sort(key=lambda x: x[0], reverse=True)
                best_net, best_fixed, best_broken, best_set = set_scores[0]

                if n_sets > 1:
                    best_ids = ", ".join(r.get("id", "?") for r in best_set)
                    print(f"    BEST: [{best_ids}] "
                          f"net={best_net} "
                          f"(+{best_fixed}/-{best_broken})")
                    if n_sets > 1:
                        worst_net = set_scores[-1][0]
                        print(f"    (worst net={worst_net}, "
                              f"spread={best_net - worst_net})")

                # Accept/reject each rule in the best set
                for rule in best_set:
                    # Re-test individual rule for its own stats (for logging)
                    # We already have per-rule stats from above for best_set
                    # but since we tested independently, use the set-level
                    # decision: accept the whole set if net >= 1
                    kept = best_net >= 1

                    log_rule_change(
                        run_dir, gi, rule,
                        action="accept" if kept else "reject",
                        n_fixed=best_fixed, n_broken=best_broken,
                    )

                    if kept:
                        accepted_rules.append(rule)
                        round_accepted += 1

                if best_net < 1:
                    round_rejected += len(best_set)
                    action_str = "REJECT ALL"
                else:
                    action_str = f"ACCEPT {len(best_set)} rules"
                print(f"    → {action_str} (best net={best_net})")

                # Save state after each group
                save_rules(accepted_rules, args.rules_file)
                save_state(
                    run_dir, phase="generate", group_idx=gi,
                    rules=accepted_rules,
                    processed_groups=[g.span_text for g in groups[:gi + 1]],
                    holdout_note_ids=sorted(holdout_ids),
                )

            total_accepted += round_accepted
            total_rejected += round_rejected
            print(f"\n  Round {round_idx + 1} complete: "
                  f"+{round_accepted} accepted, -{round_rejected} rejected, "
                  f"total rules: {len(accepted_rules)}")

            if round_accepted == 0:
                print(f"  No rules accepted this round — stopping early.")
                break

        print(f"\n{'='*70}")
        print(f"DONE")
        print(f"{'='*70}")
        print(f"  Rules accepted: {total_accepted}")
        print(f"  Rules rejected: {total_rejected}")
        print(f"  Total rules: {len(accepted_rules)}")
        print(f"  Rules file: {args.rules_file}")
        print(f"  Run dir: {run_dir}")


if __name__ == "__main__":
    main()
