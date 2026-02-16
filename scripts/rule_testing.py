#!/usr/bin/env python3
"""Script 2: Test Opus-generated rules via a Sonnet agent (Claude Agent SDK).

Two-pass approach per annotation:
  Pass 1: Sonnet generates search terms based on rules + context
  Python: Executes SNOMED retrieval with those search terms
  Pass 2: Sonnet selects the best concept from the retrieved candidates

Measures character-level IoU against the gold standard.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pandas as pd
from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    TextBlock,
    query,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))
sys.path.insert(0, str(REPO_ROOT))

from engine import get_section_for_pos, segment_sections  # noqa: E402
from runtime_scoring import macro_char_iou  # noqa: E402

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
TERMINOLOGY_CSV = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"
RULES_PATH = Path(__file__).parent / "rules_output.json"
RESULTS_PATH = Path(__file__).parent / "test_results.json"

CONTEXT_BEFORE = 300
CONTEXT_AFTER = 100
SONNET_MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_first_note() -> tuple[str, str, pd.DataFrame]:
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    ann_df = pd.read_csv(SPLIT_DIR / "train_annotations.csv")

    first_note_id = notes_df.iloc[0]["note_id"]
    note_text = notes_df.iloc[0]["text"]

    note_anns = (
        ann_df[ann_df["note_id"] == first_note_id]
        .copy()
        .sort_values("start")
    )
    for col in ["start", "end", "concept_id", "annotation_id"]:
        note_anns[col] = note_anns[col].astype(int)

    return first_note_id, note_text, note_anns


def load_concept_names() -> dict[int, tuple[str, str]]:
    ft = pd.read_csv(TERMINOLOGY_CSV)
    return {
        int(row.concept_id): (row.concept_name, str(row.hierarchy))
        for row in ft.itertuples()
    }


# ---------------------------------------------------------------------------
# SNOMED retrieval (lazy init)
# ---------------------------------------------------------------------------

_retrieval_ready = False
_faiss_index = None
_faiss_sctids = None
_bm25_index = None
_bm25_sctids = None
_encoder = None
_concept_names: dict[int, tuple[str, str]] = {}


def _init_retrieval() -> None:
    global _retrieval_ready, _faiss_index, _faiss_sctids
    global _bm25_index, _bm25_sctids, _encoder, _concept_names

    if _retrieval_ready:
        return

    from snomed_ct_entity_linking.recall_analysis.config import Config
    from snomed_ct_entity_linking.recall_analysis.encoder import SapBERTEncoder
    from snomed_ct_entity_linking.recall_analysis.index_builder import load_indexes

    cfg = Config()
    _faiss_index, _faiss_sctids, _bm25_index, _bm25_sctids, _ = load_indexes(cfg)
    _encoder = SapBERTEncoder(cfg.embedding_model)
    _concept_names = load_concept_names()
    _retrieval_ready = True


def snomed_search(query_text: str, top_k: int = 10) -> list[dict]:
    """Execute hybrid SNOMED retrieval."""
    from snomed_ct_entity_linking.recall_analysis.config import Config
    from snomed_ct_entity_linking.recall_analysis.retrieval import (
        dense_search,
        reciprocal_rank_fusion,
        sparse_search,
    )

    _init_retrieval()
    top_k = min(max(top_k, 1), 20)
    cfg = Config()

    query_emb = _encoder.encode([query_text], batch_size=1)
    dense_res = dense_search(query_emb, _faiss_index, _faiss_sctids, cfg.dense_top_k)
    sparse_res = sparse_search(
        [query_text], _bm25_index, _bm25_sctids, cfg.sparse_top_k,
        n_threads=1,
    )
    fused = reciprocal_rank_fusion(dense_res[0], sparse_res[0], cfg.rrf_k, top_k)

    results = []
    for sctid, score in fused[:top_k]:
        cname, hierarchy = _concept_names.get(int(sctid), ("Unknown", "unknown"))
        results.append({
            "concept_id": int(sctid),
            "concept_name": cname,
            "hierarchy": hierarchy,
            "score": round(float(score), 4),
        })
    return results


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def get_context(note_text: str, start: int, end: int) -> tuple[str, str, str]:
    ctx_start = max(0, start - CONTEXT_BEFORE)
    ctx_end = min(len(note_text), end + CONTEXT_AFTER)
    return note_text[ctx_start:start], note_text[start:end], note_text[end:ctx_end]


# ---------------------------------------------------------------------------
# Format rules
# ---------------------------------------------------------------------------

def format_rules(g_rules: list[dict], numbered_rules: list[dict]) -> str:
    text = "=== UNIVERSAL RULES ===\n"
    for rule in g_rules:
        text += f"\n{rule['id']}: {rule['rule']}\n"

    if numbered_rules:
        text += "\n=== SPECIFIC RULES ===\n"
        for rule in numbered_rules:
            text += f"\n{rule['id']}: {rule['rule']}\n"
            for ex in rule.get("examples", []):
                text += f"  Example: {ex}\n"
    return text


# ---------------------------------------------------------------------------
# Pass 1: Generate search terms
# ---------------------------------------------------------------------------

SEARCH_SYSTEM = """\
You are a clinical NLP agent. Given a text excerpt from a clinical note with a \
highlighted region and annotation rules, generate search terms for querying a \
SNOMED CT terminology index.

Respond with ONLY a JSON object:
{"search_terms": ["term1", "term2", "term3"]}

Generate 1-3 search terms, ordered from most to least likely to match. \
Include the exact span text as one term, plus expanded/alternative forms.\
"""


async def generate_search_terms(
    before: str, span: str, after: str,
    section_header: str,
    rules_text: str,
) -> list[str]:
    """Pass 1: Ask Sonnet to generate search terms."""

    user_prompt = f"""\
{rules_text}

Section: {section_header}

Excerpt: ...{before}>>>{span}<<<{after}...

The text between >>> and <<< is the region of interest.
Generate search terms to find the matching SNOMED concept.\
"""

    options = ClaudeCodeOptions(
        model=SONNET_MODEL,
        permission_mode="bypassPermissions",
        max_turns=1,
        system_prompt=SEARCH_SYSTEM,
    )

    raw_text = ""
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    raw_text += block.text

    # Parse search terms from JSON
    try:
        match = re.search(r"\{[^{}]*\"search_terms\"[^{}]*\}", raw_text)
        if match:
            result = json.loads(match.group())
            terms = result.get("search_terms", [])
            if terms:
                return terms[:3]
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: use the span text directly
    return [span.strip()]


# ---------------------------------------------------------------------------
# Pass 2: Select concept from candidates
# ---------------------------------------------------------------------------

SELECT_SYSTEM = """\
You are a clinical NLP agent. Given annotation rules, a text excerpt with a \
highlighted region, and SNOMED CT search results, select the best matching concept.

Respond with ONLY a JSON object:
{"concept_id": <integer>, "span_start": <int>, "span_end": <int>}

Choose the concept whose meaning best matches the clinical text in context. \
The span_start and span_end should be the exact character positions provided.\
"""


async def select_concept(
    before: str, span: str, after: str,
    section_header: str,
    ann_start: int, ann_end: int,
    rules_text: str,
    candidates: list[dict],
) -> tuple[int, int, int] | None:
    """Pass 2: Ask Sonnet to select best concept from candidates."""

    candidates_text = "\n".join(
        f"  {i+1}. [{c['concept_id']}] {c['concept_name']} ({c['hierarchy']}) — score: {c['score']}"
        for i, c in enumerate(candidates)
    )

    user_prompt = f"""\
{rules_text}

Section: {section_header}

Excerpt: ...{before}>>>{span}<<<{after}...

The text between >>> and <<< is the region of interest.
Span position: start={ann_start}, end={ann_end}

SNOMED search results:
{candidates_text}

Select the best matching concept and confirm span boundaries.\
"""

    options = ClaudeCodeOptions(
        model=SONNET_MODEL,
        permission_mode="bypassPermissions",
        max_turns=1,
        system_prompt=SELECT_SYSTEM,
    )

    raw_text = ""
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    raw_text += block.text

    return _parse_response(raw_text)


def _parse_response(text: str) -> tuple[int, int, int] | None:
    # Try code fence first
    fence_match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1))
            return (
                int(result["concept_id"]),
                int(result.get("span_start", -1)),
                int(result.get("span_end", -1)),
            )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

    # Try inline JSON
    try:
        match = re.search(r"\{[^{}]*\"concept_id\"[^{}]*\}", text)
        if match:
            result = json.loads(match.group())
            return (
                int(result["concept_id"]),
                int(result.get("span_start", -1)),
                int(result.get("span_end", -1)),
            )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Fallback: extract concept_id only
    cid_match = re.search(r"\"concept_id\"\s*:\s*(\d+)", text)
    if cid_match:
        return (int(cid_match.group(1)), -1, -1)

    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def single_annotation_iou(
    gold_start: int, gold_end: int, gold_cid: int,
    pred_start: int, pred_end: int, pred_cid: int,
) -> float:
    if gold_cid != pred_cid:
        return 0.0
    inter_start = max(gold_start, pred_start)
    inter_end = min(gold_end, pred_end)
    intersection = max(0, inter_end - inter_start)
    union = (gold_end - gold_start) + (pred_end - pred_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain() -> None:
    # Load rules
    print(f"Loading rules from {RULES_PATH} ...")
    with open(RULES_PATH) as f:
        rules = json.load(f)
    g_rules = rules["g_rules"]
    numbered_rules_by_id = {r["id"]: r for r in rules["numbered_rules"]}
    ann_rule_map = rules["annotation_rule_map"]

    # Load data
    print("Loading note and annotations ...")
    note_id, note_text, note_anns = load_first_note()
    sections = segment_sections(note_text)

    print(f"Note: {note_id}  |  {len(note_anns)} annotations")
    print(f"Rules: {len(g_rules)} global, {len(numbered_rules_by_id)} numbered")

    # Initialize retrieval
    print("Initializing SNOMED retrieval system ...")
    _init_retrieval()

    # Process each annotation
    predictions: list[dict] = []
    per_ann_results: list[dict] = []
    total_cost = 0.0

    for idx, (_, row) in enumerate(note_anns.iterrows()):
        ann_id = str(int(row["annotation_id"]))
        gold_start = int(row["start"])
        gold_end = int(row["end"])
        gold_cid = int(row["concept_id"])
        gold_span = note_text[gold_start:gold_end]

        # Section header
        sec = get_section_for_pos(gold_start, sections)
        section_header = sec.header if sec else "unknown"

        # Applicable rules
        mapped_rule_ids = ann_rule_map.get(ann_id, [])
        applicable_numbered = [
            numbered_rules_by_id[rid]
            for rid in mapped_rule_ids
            if rid in numbered_rules_by_id
        ]
        rules_text = format_rules(g_rules, applicable_numbered)

        # Context
        before, span, after = get_context(note_text, gold_start, gold_end)

        # === PASS 1: Generate search terms ===
        search_terms = await generate_search_terms(
            before, span, after, section_header, rules_text,
        )

        # === RETRIEVAL: Execute SNOMED search for each term, merge results ===
        all_candidates: dict[int, dict] = {}
        for term in search_terms:
            results = snomed_search(term, top_k=10)
            for c in results:
                cid = c["concept_id"]
                if cid not in all_candidates or c["score"] > all_candidates[cid]["score"]:
                    all_candidates[cid] = c
        candidates = sorted(all_candidates.values(), key=lambda x: x["score"], reverse=True)[:15]

        # === PASS 2: Select concept from candidates ===
        result = await select_concept(
            before, span, after, section_header,
            gold_start, gold_end,
            rules_text, candidates,
        )

        if result:
            pred_cid, pred_start, pred_end = result
            if pred_start < 0 or pred_end < 0:
                pred_start, pred_end = gold_start, gold_end

            predictions.append({
                "note_id": note_id,
                "start": pred_start,
                "end": pred_end,
                "concept_id": pred_cid,
            })

            iou = single_annotation_iou(
                gold_start, gold_end, gold_cid,
                pred_start, pred_end, pred_cid,
            )
            concept_match = pred_cid == gold_cid

            per_ann_results.append({
                "annotation_id": ann_id,
                "gold_span": gold_span,
                "gold_concept": gold_cid,
                "pred_concept": pred_cid,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "concept_match": concept_match,
                "iou": iou,
                "section": section_header,
                "search_terms": search_terms,
            })

            status = "MATCH" if concept_match else "MISS"
            print(f"  [{idx+1:2d}/{len(note_anns)}] '{gold_span}' -> {status}  iou={iou:.3f}  searches={search_terms}")
        else:
            per_ann_results.append({
                "annotation_id": ann_id,
                "gold_span": gold_span,
                "gold_concept": gold_cid,
                "pred_concept": None,
                "pred_start": None,
                "pred_end": None,
                "concept_match": False,
                "iou": 0.0,
                "section": section_header,
                "search_terms": search_terms,
            })
            print(f"  [{idx+1:2d}/{len(note_anns)}] '{gold_span}' -> FAIL  searches={search_terms}")

    # --- Aggregate scoring ---
    if predictions:
        pred_df = pd.DataFrame(predictions)
        for col in ["start", "end", "concept_id"]:
            pred_df[col] = pred_df[col].astype(int)

        gold_df = note_anns[["note_id", "start", "end", "concept_id"]].copy()
        for col in ["start", "end", "concept_id"]:
            gold_df[col] = gold_df[col].astype(int)

        agg_iou = macro_char_iou(pred_df, gold_df)
    else:
        agg_iou = 0.0

    n_matches = sum(1 for r in per_ann_results if r["concept_match"])
    n_total = len(per_ann_results)
    avg_iou = sum(r["iou"] for r in per_ann_results) / max(n_total, 1)

    # --- Report ---
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total annotations:  {n_total}")
    print(f"Concept matches:    {n_matches}/{n_total} ({100 * n_matches / max(n_total, 1):.1f}%)")
    print(f"Avg per-ann IoU:    {avg_iou:.4f}")
    print(f"Macro char IoU:     {agg_iou:.4f}")

    # Per-section breakdown
    section_groups: dict[str, list[dict]] = {}
    for r in per_ann_results:
        section_groups.setdefault(r["section"], []).append(r)

    print(f"\nPer-section accuracy:")
    for sec in sorted(section_groups):
        group = section_groups[sec]
        n = len(group)
        matches = sum(1 for r in group if r["concept_match"])
        sec_iou = sum(r["iou"] for r in group) / n
        print(f"  {sec:40s} n={n:3d}  concept_acc={100 * matches / n:5.1f}%  avg_iou={sec_iou:.3f}")

    # --- Save results ---
    output = {
        "note_id": note_id,
        "n_annotations": n_total,
        "n_concept_matches": n_matches,
        "concept_accuracy": n_matches / max(n_total, 1),
        "avg_per_annotation_iou": avg_iou,
        "macro_char_iou": agg_iou,
        "per_annotation": per_ann_results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {RESULTS_PATH}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
