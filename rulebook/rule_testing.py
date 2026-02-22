#!/usr/bin/env python3
"""Script 2: Test Opus-generated rules with a two-pass LLM agent.

Supports two backends:
  --backend sonnet   Claude Sonnet via Claude Agent SDK (sequential)
  --backend vllm     gpt-oss-20b via local vLLM server (concurrent)

Two-pass approach per annotation:
  Pass 1: LLM generates search terms based on rules + context
  Python: Executes SNOMED retrieval with those search terms
  Pass 2: LLM selects the best concept from the retrieved candidates

Measures character-level IoU against the gold standard.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))
sys.path.insert(0, str(REPO_ROOT))

from engine import get_section_for_pos, segment_sections  # noqa: E402
from runtime_scoring import macro_char_iou  # noqa: E402

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
TERMINOLOGY_CSV = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"
DEFAULT_RULES_PATH = Path(__file__).parent / "rules_output.json"

DEFAULT_CONTEXT_BEFORE = 300
DEFAULT_CONTEXT_AFTER = 100


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_note(note_id: str | None = None) -> tuple[str, str, pd.DataFrame]:
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    ann_df = pd.read_csv(SPLIT_DIR / "train_annotations.csv")

    if note_id is None:
        note_id = notes_df.iloc[0]["note_id"]
        note_text = notes_df.iloc[0]["text"]
    else:
        match = notes_df[notes_df["note_id"] == note_id]
        if match.empty:
            raise ValueError(f"Note {note_id!r} not found in train_notes.csv")
        note_text = match.iloc[0]["text"]

    note_anns = (
        ann_df[ann_df["note_id"] == note_id]
        .copy()
        .sort_values("start")
    )
    for col in ["start", "end", "concept_id", "annotation_id"]:
        note_anns[col] = note_anns[col].astype(int)

    return note_id, note_text, note_anns


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
_index_server_url: str | None = None  # set via --index-server


def _init_retrieval() -> None:
    global _retrieval_ready, _faiss_index, _faiss_sctids
    global _bm25_index, _bm25_sctids, _encoder, _concept_names

    if _retrieval_ready:
        return

    if _index_server_url:
        import requests
        resp = requests.get(f"{_index_server_url}/health", timeout=10)
        resp.raise_for_status()
        h = resp.json()
        print(f"Connected to index server at {_index_server_url}: "
              f"{h['faiss_vectors']:,} concepts, {h['bm25_docs']:,} descriptions ({h['device']})")
        _concept_names = load_concept_names()
        _retrieval_ready = True
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
    """Execute hybrid SNOMED retrieval for a single query."""
    results = snomed_search_batch([query_text], top_k=top_k)
    return results[0]


def snomed_search_batch(queries: list[str], top_k: int = 10) -> list[list[dict]]:
    """Batch SNOMED retrieval: encode + FAISS + BM25 all queries at once."""
    _init_retrieval()
    top_k = min(max(top_k, 1), 100)

    if _index_server_url:
        import requests
        resp = requests.post(
            f"{_index_server_url}/search/hybrid",
            json={"queries": queries, "fusion_top_k": top_k},
            timeout=120,
        )
        resp.raise_for_status()
        all_results = []
        for query_hits in resp.json()["results"]:
            results = []
            for hit in query_hits:
                sctid = hit["sctid"]
                cname, hierarchy = _concept_names.get(sctid, (hit.get("name") or "Unknown", "unknown"))
                results.append({
                    "concept_id": sctid,
                    "concept_name": cname,
                    "hierarchy": hierarchy,
                    "score": round(hit["rrf_score"], 4),
                })
            all_results.append(results)
        return all_results

    from snomed_ct_entity_linking.recall_analysis.config import Config
    from snomed_ct_entity_linking.recall_analysis.retrieval import (
        dense_search,
        reciprocal_rank_fusion,
        sparse_search,
    )

    cfg = Config()
    query_embs = _encoder.encode(queries, batch_size=max(len(queries), 1))
    dense_results = dense_search(
        query_embs, _faiss_index, _faiss_sctids, cfg.dense_top_k, verbose=False,
        oversample=4,
    )
    sparse_results = sparse_search(
        queries, _bm25_index, _bm25_sctids, cfg.sparse_top_k,
        n_threads=min(len(queries), 16), verbose=False,
    )

    all_results = []
    for i in range(len(queries)):
        fused = reciprocal_rank_fusion(
            dense_results[i], sparse_results[i], cfg.rrf_k, top_k,
        )
        results = []
        for sctid, score in fused[:top_k]:
            cname, hierarchy = _concept_names.get(int(sctid), ("Unknown", "unknown"))
            results.append({
                "concept_id": int(sctid),
                "concept_name": cname,
                "hierarchy": hierarchy,
                "score": round(float(score), 4),
            })
        all_results.append(results)
    return all_results


def snomed_search_batch_items(
    items: list[tuple[str, list[str]]],
    top_k: int = 10,
) -> dict[str, list[dict]]:
    """Batch SNOMED retrieval using the batch_hybrid endpoint.

    Each item is (id, queries) — multiple search terms per annotation.
    The server does a single SapBERT encode + FAISS + BM25 pass and
    per-item cross-query RRF fusion.

    Returns {id: [candidate_dicts]} with results already fused and sorted.
    """
    _init_retrieval()
    top_k = min(max(top_k, 1), 100)

    if _index_server_url:
        import requests
        payload = {
            "items": [
                {"id": item_id, "queries": queries, "top_k": top_k}
                for item_id, queries in items
                if queries  # skip items with no search terms
            ],
        }
        if not payload["items"]:
            return {item_id: [] for item_id, _ in items}

        resp = requests.post(
            f"{_index_server_url}/search/batch_hybrid",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()

        result_map: dict[str, list[dict]] = {}
        for item_result in resp.json()["results"]:
            candidates = []
            for hit in item_result["results"]:
                sctid = hit["sctid"]
                cname, hierarchy = _concept_names.get(
                    sctid, (hit.get("name") or "Unknown", "unknown")
                )
                candidates.append({
                    "concept_id": sctid,
                    "concept_name": cname,
                    "hierarchy": hierarchy,
                    "score": round(hit["rrf_score"], 4),
                })
            result_map[item_result["id"]] = candidates

        # Ensure every input item has an entry
        for item_id, _ in items:
            result_map.setdefault(item_id, [])
        return result_map

    # Fallback: in-process retrieval — flatten, search, regroup
    all_queries: list[str] = []
    item_slices: list[tuple[str, int, int]] = []
    for item_id, queries in items:
        start = len(all_queries)
        all_queries.extend(queries)
        item_slices.append((item_id, start, len(all_queries)))

    if not all_queries:
        return {item_id: [] for item_id, _ in items}

    batch_results = snomed_search_batch(all_queries, top_k=top_k)

    result_map = {}
    for item_id, qstart, qend in item_slices:
        merged: dict[int, dict] = {}
        for results in batch_results[qstart:qend]:
            for c in results:
                cid = c["concept_id"]
                if cid not in merged or c["score"] > merged[cid]["score"]:
                    merged[cid] = c
        result_map[item_id] = sorted(
            merged.values(), key=lambda x: x["score"], reverse=True
        )[:top_k]
    return result_map


def check_gold_in_wider_retrieval(per_ann_results: list[dict]) -> None:
    """For retrieval misses, check if gold concept is in wider top-100 retrieval.

    Mutates per_ann_results in place, adding gold_in_top100 / gold_rank_top100
    fields to retrieval miss entries.
    """
    misses = [r for r in per_ann_results if r.get("failure_reason") == "retrieval_miss"]
    if not misses:
        return

    print(f"\n  Checking {len(misses)} retrieval misses against wider top-100 pool ...")
    _init_retrieval()

    # One batch call — server does cross-query fusion per miss
    items = [
        (str(i), miss["search_terms"])
        for i, miss in enumerate(misses)
    ]
    result_map = snomed_search_batch_items(items, top_k=100)

    for i, miss in enumerate(misses):
        gold_cid = miss["gold_concept"]
        candidates = result_map.get(str(i), [])
        candidate_ids = [c["concept_id"] for c in candidates]

        found = gold_cid in candidate_ids
        rank = candidate_ids.index(gold_cid) + 1 if found else None

        miss["gold_in_top100"] = found
        miss["gold_rank_top100"] = rank
        miss["n_candidates_top100"] = len(candidates)

        status = f"rank={rank}" if found else "NOT FOUND"
        print(f"    '{miss['gold_span']}' (gold={gold_cid}): {status} in {len(candidates)} candidates")


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def get_context(
    note_text: str, start: int, end: int,
    context_before: int = DEFAULT_CONTEXT_BEFORE,
    context_after: int = DEFAULT_CONTEXT_AFTER,
) -> tuple[str, str, str]:
    ctx_start = max(0, start - context_before)
    ctx_end = min(len(note_text), end + context_after)

    # Snap ctx_start back to a word boundary so we don't cut mid-word
    while ctx_start > 0 and not note_text[ctx_start - 1].isspace():
        ctx_start -= 1

    # Snap ctx_end forward to a word boundary so we don't cut mid-word
    while ctx_end < len(note_text) and not note_text[ctx_end].isspace():
        ctx_end += 1

    return note_text[ctx_start:start], note_text[start:end], note_text[end:ctx_end]


# ---------------------------------------------------------------------------
# Format rules
# ---------------------------------------------------------------------------

def _is_grule_for_stage(g_rule: dict, stage: str) -> bool:
    """Check if a G-rule applies to a given stage."""
    stages = g_rule.get("stages")
    if stages is None:
        return True  # no stages field = universal
    return stage in stages


def format_rules_for_search(
    g_rules: list[dict],
    structured_rules: list[dict],
) -> str:
    """Format rules for Stage 2: Search Term Generation."""
    text = "=== UNIVERSAL RULES (Search) ===\n"
    for rule in g_rules:
        if _is_grule_for_stage(rule, "stage_2_search"):
            text += f"\n{rule['id']}: {rule['rule']}\n"

    applicable = [r for r in structured_rules if "stage_2_search" in r]
    if applicable:
        text += "\n=== SEARCH RULES ===\n"
        for rule in applicable:
            s2 = rule["stage_2_search"]
            text += f"\n{rule['rule_id']} [{rule['concept_type']}]:"
            if s2.get("filtering_logic"):
                text += f"\n  Filter: {s2['filtering_logic']}"
            text += f"\n  Translation: {s2['intent_translation']}\n"
            for ex in rule.get("examples", []):
                text += f"  Example: {ex}\n"
    return text


def format_rules_for_select(
    g_rules: list[dict],
    structured_rules: list[dict],
) -> str:
    """Format rules for Stage 3: Concept Disambiguation."""
    text = "=== UNIVERSAL RULES (Selection) ===\n"
    for rule in g_rules:
        if _is_grule_for_stage(rule, "stage_3_select"):
            text += f"\n{rule['id']}: {rule['rule']}\n"

    applicable = [r for r in structured_rules if "stage_3_select" in r]
    if applicable:
        text += "\n=== DISAMBIGUATION RULES ===\n"
        for rule in applicable:
            s3 = rule["stage_3_select"]
            text += f"\n{rule['rule_id']} [{rule['concept_type']}]:"
            text += f"\n  Logic: {s3['disambiguation_logic']}"
            if s3.get("preferred_hierarchy"):
                text += f"\n  Prefer: {s3['preferred_hierarchy']}"
            if s3.get("reject_hierarchies"):
                text += f"\n  Reject: {', '.join(s3['reject_hierarchies'])}"
            text += "\n"
            for ex in rule.get("examples", []):
                text += f"  Example: {ex}\n"
    return text


def format_rules(g_rules: list[dict], numbered_rules: list[dict]) -> str:
    """Legacy format_rules for backward compatibility with v1.0 rules."""
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
# Shared system prompts
# ---------------------------------------------------------------------------

SEARCH_SYSTEM = """\
You are a clinical NLP agent. Given a text excerpt from a clinical note with a \
highlighted region and SEARCH-SPECIFIC annotation rules, generate search terms \
for querying a SNOMED CT terminology index.

The rules below are specifically curated for search term generation. Follow \
the filtering and intent translation instructions carefully.

Respond with ONLY a JSON object:
{"search_terms": ["term1", "term2", "term3"]}

Generate 1-3 search terms, ordered from most to least likely to match. \
Include the exact span text as one term, plus expanded/alternative forms.\
"""

SELECT_SYSTEM = """\
You are a clinical NLP agent. Given DISAMBIGUATION-SPECIFIC annotation rules, \
a text excerpt with a highlighted region, and SNOMED CT search results, select \
the best matching concept.

The rules below are specifically curated for concept disambiguation. Follow \
the hierarchy preferences and disambiguation logic carefully.

Respond with ONLY a JSON object:
{"choice": <integer>}

where <integer> is the 1-based index of the best matching candidate from the \
numbered list. Choose the concept whose meaning best matches the clinical text \
in context.\
"""


# ---------------------------------------------------------------------------
# Prompt builders (shared by both backends)
# ---------------------------------------------------------------------------

def build_search_prompt(
    before: str, span: str, after: str,
    section_header: str, rules_text: str,
    prompt_style: str = "standard",
) -> str:
    excerpt = f"...{before}>>>{span}<<<{after}..."
    if prompt_style == "repeated-text":
        return f"""\
You will annotate the following text:
TEXT (Section: {section_header}): {excerpt}

According to the rules provided below:
{rules_text}

For reference, this is the text again:
TEXT (Section: {section_header}): {excerpt}

The text between >>> and <<< is the region of interest.
Generate search terms to find the matching SNOMED concept.\
"""
    # standard (default)
    return f"""\
{rules_text}

Section: {section_header}

Excerpt: {excerpt}

The text between >>> and <<< is the region of interest.
Generate search terms to find the matching SNOMED concept.\
"""


def build_select_prompt(
    before: str, span: str, after: str,
    section_header: str, ann_start: int, ann_end: int,
    rules_text: str, candidates: list[dict],
    prompt_style: str = "standard",
) -> str:
    excerpt = f"...{before}>>>{span}<<<{after}..."
    candidates_text = "\n".join(
        f"  {i+1}. {c['concept_name']}"
        for i, c in enumerate(candidates)
    )
    if prompt_style == "repeated-text":
        return f"""\
You will annotate the following text:
TEXT (Section: {section_header}): {excerpt}

According to the rules provided below:
{rules_text}

For reference, this is the text again:
TEXT (Section: {section_header}): {excerpt}

The text between >>> and <<< is the region of interest.

SNOMED candidates:
{candidates_text}

Select the best matching concept by number.\
"""
    # standard (default)
    return f"""\
{rules_text}

Section: {section_header}

Excerpt: {excerpt}

The text between >>> and <<< is the region of interest.

SNOMED candidates:
{candidates_text}

Select the best matching concept by number.\
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_search_terms(text: str, fallback_span: str) -> list[str]:
    try:
        match = re.search(r"\{[^{}]*\"search_terms\"[^{}]*\}", text)
        if match:
            result = json.loads(match.group())
            terms = result.get("search_terms", [])
            if terms:
                return terms[:3]
    except (json.JSONDecodeError, KeyError):
        pass
    return [fallback_span.strip()]


def parse_concept_response(
    text: str,
    candidates: list[dict] | None = None,
) -> tuple[int, int, int] | None:
    """Parse the LLM's concept selection response.

    Supports two formats:
      - Index-based: {"choice": N}  (1-indexed into candidates list)
      - Legacy:      {"concept_id": N, "span_start": M, "span_end": K}
    When candidates is provided, index-based parsing is tried first.
    """

    def _try_json(raw: str) -> dict | None:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

    def _extract_json(text: str) -> dict | None:
        # Try code fence first
        fence_match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
        if fence_match:
            result = _try_json(fence_match.group(1))
            if result:
                return result
        # Try inline JSON (look for any {...})
        for m in re.finditer(r"\{[^{}]*\}", text):
            result = _try_json(m.group())
            if result:
                return result
        return None

    parsed = _extract_json(text)

    # Index-based format: {"choice": N}
    if parsed and candidates:
        choice = parsed.get("choice")
        if choice is not None:
            try:
                idx = int(choice) - 1  # 1-indexed -> 0-indexed
                if 0 <= idx < len(candidates):
                    return (candidates[idx]["concept_id"], -1, -1)
            except (ValueError, TypeError):
                pass

    # Legacy format: {"concept_id": N, ...}
    if parsed and "concept_id" in parsed:
        try:
            return (
                int(parsed["concept_id"]),
                int(parsed.get("span_start", -1)),
                int(parsed.get("span_end", -1)),
            )
        except (ValueError, TypeError):
            pass

    # Fallback: bare "choice": N in text
    if candidates:
        choice_match = re.search(r"\"choice\"\s*:\s*(\d+)", text)
        if choice_match:
            idx = int(choice_match.group(1)) - 1
            if 0 <= idx < len(candidates):
                return (candidates[idx]["concept_id"], -1, -1)

    # Fallback: bare "concept_id": N in text
    cid_match = re.search(r"\"concept_id\"\s*:\s*(\d+)", text)
    if cid_match:
        return (int(cid_match.group(1)), -1, -1)

    return None


# ---------------------------------------------------------------------------
# Backend: Claude Sonnet (sequential, via Claude Agent SDK)
# ---------------------------------------------------------------------------

async def _sonnet_call(system_prompt: str, user_prompt: str) -> str:
    from claude_code_sdk import (
        AssistantMessage,
        ClaudeCodeOptions,
        TextBlock,
        query,
    )

    options = ClaudeCodeOptions(
        model="claude-sonnet-4-20250514",
        permission_mode="bypassPermissions",
        max_turns=1,
        system_prompt=system_prompt,
    )

    raw_text = ""
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    raw_text += block.text
    return raw_text


async def sonnet_search_terms_batch(prompts: list[str], spans: list[str]) -> list[list[str]]:
    """Sequential Pass 1 for Sonnet backend."""
    results = []
    for i, (prompt, span) in enumerate(zip(prompts, spans)):
        raw = await _sonnet_call(SEARCH_SYSTEM, prompt)
        terms = parse_search_terms(raw, span)
        results.append(terms)
        if (i + 1) % 10 == 0:
            print(f"  [search] {i+1}/{len(prompts)} done")
    return results


async def sonnet_select_batch(
    prompts: list[str],
    candidates_per_ann: list[list[dict]],
) -> list[tuple[int, int, int] | None]:
    """Sequential Pass 2 for Sonnet backend."""
    results = []
    for i, (prompt, cands) in enumerate(zip(prompts, candidates_per_ann)):
        raw = await _sonnet_call(SELECT_SYSTEM, prompt)
        results.append(parse_concept_response(raw, cands))
        if (i + 1) % 10 == 0:
            print(f"  [select] {i+1}/{len(prompts)} done")
    return results


# ---------------------------------------------------------------------------
# Backend: vLLM / gpt-oss-20b (concurrent, via AsyncOpenAI)
# ---------------------------------------------------------------------------


def vllm_pipeline(
    annotations: list[dict],
    base_url: str = "http://localhost:8000",
    model: str = "openai/gpt-oss-20b",
    max_concurrent: int = 0,
    reasoning_effort: str = "low",
    prompt_style: str = "standard",
    *,
    precomputed_search_terms: list[list[str]] | None = None,
    precomputed_candidates: list[list[dict]] | None = None,
    search_only: bool = False,
    search_terms_only: bool = False,
) -> tuple[list[list[str]], list[list[dict]], list[tuple[int, int, int] | None], list[dict]]:
    """Run the search->retrieve->select pipeline with continuous batching.

    Each annotation independently flows through:
      1. LLM generates search terms (Pass 1)
      2. Batched SNOMED retrieval (queries accumulated across annotations)
      3. LLM selects concept from candidates (Pass 2)

    If precomputed_search_terms AND precomputed_candidates are provided,
    phases 1+2 are skipped entirely and only the select phase runs.

    If search_only=True, only phases 1+2 run (no select LLM call).
    Selections will be None for all annotations.

    If search_terms_only=True, only phase 1 runs (LLM calls only, no
    retrieval at all). candidates_list will be empty lists. Use this to
    batch many LLM calls efficiently, then do retrieval separately.
    """
    from openai import AsyncOpenAI

    extra_body = {}
    if reasoning_effort and reasoning_effort != "none":
        extra_body["reasoning_effort"] = reasoning_effort

    select_only = (precomputed_search_terms is not None
                   and precomputed_candidates is not None)

    n = len(annotations)
    conc_label = "unlimited" if max_concurrent <= 0 else str(max_concurrent)
    if search_terms_only:
        search_only = True  # implies search_only
    mode_label = ("select-only" if select_only
                  else "search-terms-only" if search_terms_only
                  else "search-only" if search_only
                  else "full")
    print(f"  Pipeline: {n} annotations "
          f"(max_concurrent={conc_label}, model={model}, "
          f"reasoning_effort={reasoning_effort}, mode={mode_label})")

    # Scale max_tokens with reasoning effort — reasoning tokens count against the budget.
    # Reasoning models (e.g. gpt-oss-20b) put chain-of-thought into reasoning_content
    # which counts against the token budget.  The actual JSON output is tiny (~20-50 tokens)
    # but reasoning can consume thousands.  With insufficient budget, content=None and
    # finish_reason="length".
    max_tokens = {"low": 1024, "medium": 4096, "high": 16384}.get(reasoning_effort, 256)

    # Pre-build search prompts (not needed in select-only mode)
    search_prompts = None if select_only else [
        build_search_prompt(a["before"], a["span"], a["after"],
                            a["section_header"], a["search_rules_text"],
                            prompt_style=prompt_style)
        for a in annotations
    ]

    t0 = time.time()

    # Output arrays (filled by each task)
    out_search_terms: list[list[str]] = [[] for _ in range(n)]
    out_candidates: list[list[dict]] = [[] for _ in range(n)]
    out_selections: list[tuple[int, int, int] | None] = [None] * n

    # Per-annotation timing arrays
    ann_timings: list[dict] = [{"search_s": 0.0, "retrieval_s": 0.0, "select_s": 0.0, "total_s": 0.0, "mapping_bypass": False} for _ in range(n)]

    # Token usage tracking (prompt_tokens per LLM call)
    _prompt_token_counts: list[int] = []

    async def _run():
        async_client = AsyncOpenAI(
            base_url=f"{base_url}/v1", api_key="unused",
            max_retries=2,
            timeout=300.0,
        )
        sem = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

        _prompt_logged: set[str] = set()

        async def _llm_call(system_prompt: str, user_prompt: str, max_tokens: int, *, call_type: str = "") -> str:
            if sem is not None:
                await sem.acquire()
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                if call_type not in _prompt_logged:
                    _prompt_logged.add(call_type)
                    tqdm.write(f"\n{'='*60}\nPROMPT SAMPLE [{call_type}]\n{'='*60}")
                    tqdm.write(f"[SYSTEM]\n{system_prompt}")
                    tqdm.write(f"\n[USER]\n{system_prompt + chr(10)*2 + user_prompt}")
                    tqdm.write(f"{'='*60}\n")

                tok_budget = max_tokens
                for attempt in range(2):
                    try:
                        resp = await async_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            max_tokens=tok_budget,
                            temperature=0.0,
                            frequency_penalty=0.3,
                            seed=42,  # determinism also requires VLLM_BATCH_INVARIANT=1 on server
                            extra_body=extra_body if extra_body else None,
                        )
                        if resp.usage and resp.usage.prompt_tokens:
                            _prompt_token_counts.append(resp.usage.prompt_tokens)
                        choice = resp.choices[0]
                        content = choice.message.content or ""
                        reasoning = getattr(choice.message, "reasoning_content", None) or ""
                        return content.strip() if content.strip() else reasoning.strip()
                    except Exception as e:
                        err_str = str(e)
                        # Retry once if max_tokens exceeds available context headroom.
                        # Error format: "has N input tokens (M > CTX - N)"
                        if attempt == 0 and "max_tokens" in err_str and "maximum context length" in err_str:
                            m = re.search(r"\((\d+) > (\d+) - (\d+)\)", err_str)
                            if m:
                                available = int(m.group(2)) - int(m.group(3)) - 2
                                if available >= 16:
                                    tqdm.write(f"  [{call_type}] context headroom={available} tokens; retrying")
                                    tok_budget = available
                                    continue
                        tqdm.write(f"  ERROR: {e}")
                        return ""
                return ""
            finally:
                if sem is not None:
                    sem.release()

        pbar = tqdm(total=n, desc="Annotations", unit="ann",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        if select_only:
            # ---------------------------------------------------------
            # Select-only mode: reuse precomputed search terms + cands
            # ---------------------------------------------------------
            for idx in range(n):
                out_search_terms[idx] = precomputed_search_terms[idx]
                out_candidates[idx] = precomputed_candidates[idx]

            async def _select_only(idx: int):
                a = annotations[idx]
                candidates = out_candidates[idx]
                t_sel = time.monotonic()
                select_prompt = build_select_prompt(
                    a["before"], a["span"], a["after"],
                    a["section_header"], a["gold_start"], a["gold_end"],
                    a["select_rules_text"], candidates,
                    prompt_style=prompt_style,
                )
                raw_select = await _llm_call(
                    SELECT_SYSTEM, select_prompt,
                    max_tokens=max_tokens, call_type="select",
                )
                out_selections[idx] = parse_concept_response(raw_select, candidates)
                ann_timings[idx]["select_s"] = time.monotonic() - t_sel
                ann_timings[idx]["total_s"] = ann_timings[idx]["select_s"]
                pbar.update(1)

            await asyncio.gather(*[_select_only(i) for i in range(n)])
        else:
            # ---------------------------------------------------------
            # Full streaming pipeline: search → retrieve → select
            # ---------------------------------------------------------
            if search_terms_only:
                # ---------------------------------------------------------
                # Search-terms-only mode: LLM calls only, no retrieval
                # ---------------------------------------------------------
                async def _search_only_ann(idx: int):
                    a = annotations[idx]
                    t_start = time.monotonic()
                    if a.get("mapping_hit"):
                        search_terms = [a["mapping_hit"], a["span"]]
                        ann_timings[idx]["mapping_bypass"] = True
                    else:
                        raw_search = await _llm_call(
                            SEARCH_SYSTEM, search_prompts[idx],
                            max_tokens=max_tokens, call_type="search",
                        )
                        search_terms = parse_search_terms(raw_search, a["span"])
                    ann_timings[idx]["search_s"] = time.monotonic() - t_start
                    ann_timings[idx]["total_s"] = ann_timings[idx]["search_s"]
                    out_search_terms[idx] = search_terms
                    pbar.update(1)

                await asyncio.gather(*[_search_only_ann(i) for i in range(n)])
            else:
                # ---------------------------------------------------------
                # Streaming pipeline: search → retrieve (→ select)
                # ---------------------------------------------------------
                _retr_queue: asyncio.Queue[tuple[int, list[str], asyncio.Future]] = asyncio.Queue()
                _retr_stop = False
                _retr_n_queries = 0
                _retr_n_batches = 0

                async def _retrieval_worker(batch_size: int = 512, flush_interval: float = 0.25):
                    nonlocal _retr_n_queries, _retr_n_batches, _retr_stop
                    loop = asyncio.get_event_loop()
                    while not _retr_stop or not _retr_queue.empty():
                        batch: list[tuple[int, list[str], asyncio.Future]] = []
                        try:
                            item = await asyncio.wait_for(_retr_queue.get(), timeout=0.5)
                            batch.append(item)
                        except asyncio.TimeoutError:
                            continue
                        deadline = loop.time() + flush_interval
                        while len(batch) < batch_size:
                            remaining = deadline - loop.time()
                            if remaining <= 0:
                                break
                            try:
                                item = await asyncio.wait_for(_retr_queue.get(), timeout=remaining)
                                batch.append(item)
                            except asyncio.TimeoutError:
                                break
                        if not batch:
                            continue
                        items = [(str(idx), queries) for idx, queries, _ in batch]
                        try:
                            result_map = await loop.run_in_executor(
                                None, snomed_search_batch_items, items, 10,
                            )
                        except Exception as e:
                            tqdm.write(f"  Retrieval ERROR: {e}")
                            result_map = {}
                        for idx, _, future in batch:
                            candidates = result_map.get(str(idx), [])[:15]
                            if not future.done():
                                future.set_result(candidates)
                        _retr_n_queries += sum(len(q) for _, q, _ in batch)
                        _retr_n_batches += 1

                retr_task = asyncio.create_task(_retrieval_worker())

                async def _process_annotation(idx: int):
                    a = annotations[idx]
                    t_ann_start = time.monotonic()

                    # --- Pass 1: search terms ---
                    t_search = time.monotonic()
                    if a.get("mapping_hit"):
                        search_terms = [a["mapping_hit"], a["span"]]
                        ann_timings[idx]["mapping_bypass"] = True
                    else:
                        raw_search = await _llm_call(
                            SEARCH_SYSTEM, search_prompts[idx],
                            max_tokens=max_tokens, call_type="search",
                        )
                        search_terms = parse_search_terms(raw_search, a["span"])
                    ann_timings[idx]["search_s"] = time.monotonic() - t_search
                    out_search_terms[idx] = search_terms

                    # --- Retrieval (via micro-batcher) ---
                    t_retr = time.monotonic()
                    future = asyncio.get_event_loop().create_future()
                    await _retr_queue.put((idx, search_terms, future))
                    candidates = await future
                    ann_timings[idx]["retrieval_s"] = time.monotonic() - t_retr
                    out_candidates[idx] = candidates

                    # --- Pass 2: select concept (skipped in search_only mode) ---
                    if not search_only:
                        t_sel = time.monotonic()
                        select_prompt = build_select_prompt(
                            a["before"], a["span"], a["after"],
                            a["section_header"], a["gold_start"], a["gold_end"],
                            a["select_rules_text"], candidates,
                            prompt_style=prompt_style,
                        )
                        raw_select = await _llm_call(
                            SELECT_SYSTEM, select_prompt,
                            max_tokens=max_tokens, call_type="select",
                        )
                        out_selections[idx] = parse_concept_response(raw_select, candidates)
                        ann_timings[idx]["select_s"] = time.monotonic() - t_sel

                    ann_timings[idx]["total_s"] = time.monotonic() - t_ann_start
                    pbar.update(1)

                await asyncio.gather(*[_process_annotation(i) for i in range(n)])

                _retr_stop = True
                await retr_task
                tqdm.write(f"  Retrieval: {_retr_n_queries} queries in {_retr_n_batches} batch calls")

        pbar.close()
        await async_client.close()

    asyncio.run(_run())

    elapsed = time.time() - t0
    if _prompt_token_counts:
        peak = max(_prompt_token_counts)
        avg = sum(_prompt_token_counts) / len(_prompt_token_counts)
        print(f"  Context: peak={peak} tokens, avg={avg:.0f} tokens "
              f"({len(_prompt_token_counts)} LLM calls)")
    print(f"  Pipeline done in {elapsed:.1f}s ({n / elapsed:.1f} ann/s)")
    return out_search_terms, out_candidates, out_selections, ann_timings


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
# Report printing
# ---------------------------------------------------------------------------

def print_summary_report(
    per_ann_results: list[dict],
    *,
    backend: str,
    model: str,
    temperature: float,
    reasoning_effort: str | None,
    macro_char_iou: float,
    total_time_s: float | None = None,
    replay: bool = False,
) -> None:
    """Print the results summary report."""
    n_total = len(per_ann_results)
    n_matches = sum(1 for r in per_ann_results if r["concept_match"])
    avg_iou = sum(r["iou"] for r in per_ann_results) / max(n_total, 1)

    n_gold_in_candidates = sum(1 for r in per_ann_results if r["gold_in_candidates"])
    n_retrieval_miss = sum(1 for r in per_ann_results if r.get("failure_reason") == "retrieval_miss")
    n_selection_miss = sum(1 for r in per_ann_results if r.get("failure_reason") == "selection_miss")
    n_parse_fail = sum(1 for r in per_ann_results if r.get("failure_reason") == "parse_fail")
    avg_candidates = sum(r["n_candidates"] for r in per_ann_results) / max(n_total, 1)
    gold_ranks = [r["gold_rank"] for r in per_ann_results if r["gold_rank"] is not None]
    avg_gold_rank = sum(gold_ranks) / len(gold_ranks) if gold_ranks else 0

    label = "RESULTS SUMMARY (replay)" if replay else "RESULTS SUMMARY"
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    print(f"Backend:            {backend}")
    print(f"Model:              {model}")
    print(f"Temperature:        {temperature}")
    if reasoning_effort:
        print(f"Reasoning effort:   {reasoning_effort}")
    print(f"Total annotations:  {n_total}")
    print(f"Concept matches:    {n_matches}/{n_total} ({100 * n_matches / max(n_total, 1):.1f}%)")
    print(f"Avg per-ann IoU:    {avg_iou:.4f}")
    print(f"Macro char IoU:     {macro_char_iou:.4f}")
    if total_time_s is not None:
        print(f"Total time:         {total_time_s:.1f}s")

    print(f"\nRetrieval:")
    print(f"  Avg candidates:   {avg_candidates:.1f}")
    print(f"  Gold in top-k:    {n_gold_in_candidates}/{n_total} ({100 * n_gold_in_candidates / max(n_total, 1):.1f}%)")
    if gold_ranks:
        print(f"  Avg gold rank:    {avg_gold_rank:.1f} (when found)")

    print(f"\nFailure breakdown (stage-targeted):")
    print(f"  Correct:          {n_matches}")
    print(f"  Stage 2 miss:     {n_retrieval_miss}  (search rules -> gold concept not in candidates)")
    print(f"  Stage 3 miss:     {n_selection_miss}  (select rules -> gold in candidates, LLM picked wrong)")
    if n_parse_fail:
        print(f"  Parse fail:       {n_parse_fail}  (LLM response unparseable)")
    print(f"  Stage 1 (span):   not tested (NER model not yet integrated)")

    # Wider retrieval analysis
    top100_checked = [r for r in per_ann_results if "gold_in_top100" in r]
    if top100_checked:
        n_found = sum(1 for r in top100_checked if r["gold_in_top100"])
        n_not_found = len(top100_checked) - n_found
        print(f"\nRetrieval miss analysis (top-100 check):")
        print(f"  Found in top-100:     {n_found}/{len(top100_checked)}  (recoverable with wider retrieval)")
        print(f"  Not in top-100:       {n_not_found}/{len(top100_checked)}  (true retrieval gap)")
        top100_ranks = [r["gold_rank_top100"] for r in top100_checked if r.get("gold_rank_top100") is not None]
        if top100_ranks:
            print(f"  Avg rank when found:  {sum(top100_ranks) / len(top100_ranks):.1f}")

    # Per-section breakdown
    section_groups: dict[str, list[dict]] = {}
    for r in per_ann_results:
        section_groups.setdefault(r["section"], []).append(r)

    print(f"\nPer-section accuracy:")
    for sec in sorted(section_groups):
        group = section_groups[sec]
        n = len(group)
        matches = sum(1 for r in group if r["concept_match"])
        retr = sum(1 for r in group if r["gold_in_candidates"])
        sel_miss = sum(1 for r in group if r.get("failure_reason") == "selection_miss")
        sec_iou = sum(r["iou"] for r in group) / n
        print(f"  {sec:40s} n={n:3d}  acc={100 * matches / n:5.1f}%  retrieval={100 * retr / n:5.1f}%  sel_miss={sel_miss}  iou={sec_iou:.3f}")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay(results_path: str) -> None:
    """Replay results from a saved JSON file without re-running inference."""
    path = Path(results_path)
    print(f"Loading saved results from {path} ...")

    with open(path) as f:
        saved = json.load(f)

    per_ann_results = saved["per_annotation"]

    # Run wider retrieval check for misses
    check_gold_in_wider_retrieval(per_ann_results)

    print_summary_report(
        per_ann_results,
        backend=saved.get("backend", "unknown"),
        model=saved.get("model", "unknown"),
        temperature=saved.get("temperature", 0.0),
        reasoning_effort=saved.get("reasoning_effort"),
        macro_char_iou=saved.get("macro_char_iou", 0.0),
        total_time_s=saved.get("total_time_s"),
        replay=True,
    )

    # Save updated results (with top-100 check data)
    saved["per_annotation"] = per_ann_results
    with open(path, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"\nUpdated results saved to {path}")


# ---------------------------------------------------------------------------
# Stage-targeted error diagnostics
# ---------------------------------------------------------------------------

def _suggest_rule_improvements(
    per_ann_results: list[dict],
    structured_by_id: dict[str, dict],
    ann_rule_map: dict,
) -> dict:
    """Analyze failures and suggest which rule stages need improvement."""
    stage_2_misses: list[dict] = []
    stage_3_misses: list[dict] = []

    for r in per_ann_results:
        ann_id = r["annotation_id"]
        rule_ids = ann_rule_map.get(ann_id, [])
        applicable_rules = [structured_by_id[rid] for rid in rule_ids if rid in structured_by_id]

        if r.get("failure_reason") == "retrieval_miss":
            stage_2_misses.append({
                "annotation_id": ann_id,
                "gold_span": r["gold_span"],
                "gold_concept": r["gold_concept"],
                "search_terms": r["search_terms"],
                "rules_applied": [
                    {
                        "rule_id": sr["rule_id"],
                        "concept_type": sr["concept_type"],
                        "intent_translation": sr.get("stage_2_search", {}).get("intent_translation", "N/A"),
                    }
                    for sr in applicable_rules if "stage_2_search" in sr
                ],
            })
        elif r.get("failure_reason") == "selection_miss":
            stage_3_misses.append({
                "annotation_id": ann_id,
                "gold_span": r["gold_span"],
                "gold_concept": r["gold_concept"],
                "pred_concept": r["pred_concept"],
                "gold_rank": r["gold_rank"],
                "rules_applied": [
                    {
                        "rule_id": sr["rule_id"],
                        "concept_type": sr["concept_type"],
                        "disambiguation_logic": sr.get("stage_3_select", {}).get("disambiguation_logic", "N/A"),
                    }
                    for sr in applicable_rules if "stage_3_select" in sr
                ],
            })

    return {
        "stage_2_search_failures": stage_2_misses,
        "stage_3_select_failures": stage_3_misses,
        "stage_2_failure_count": len(stage_2_misses),
        "stage_3_failure_count": len(stage_3_misses),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_rules_v2(rules: dict) -> tuple[list[dict], dict[str, dict], dict | None, dict]:
    """Load v2.0/v3.0 structured rules."""
    g_rules = rules["g_rules"]
    structured_by_id = {r["rule_id"]: r for r in rules["structured_rules"]}
    ann_rule_map = rules.get("annotation_rule_map")
    mappings = rules.get("mappings", {})
    return g_rules, structured_by_id, ann_rule_map, mappings


def _load_rules_v1(rules: dict) -> tuple[list[dict], dict[str, dict], dict, dict]:
    """Load v1.0 flat rules, converting to v2.0 structure for compatibility."""
    g_rules = rules["g_rules"]
    structured_by_id = {}
    for r in rules["numbered_rules"]:
        structured_by_id[r["id"]] = {
            "rule_id": r["id"],
            "concept_type": "legacy",
            "stage_2_search": {
                "filtering_logic": None,
                "intent_translation": r["rule"],
            },
            "stage_3_select": {
                "disambiguation_logic": r["rule"],
            },
            "examples": r.get("examples", []),
        }
    ann_rule_map = rules["annotation_rule_map"]
    mappings: dict = {}
    return g_rules, structured_by_id, ann_rule_map, mappings


def run(args: argparse.Namespace) -> None:
    bench: dict[str, object] = {}
    bench_wall_start = time.monotonic()

    # Load rules
    t = time.monotonic()
    rules_path = Path(args.rules) if args.rules else DEFAULT_RULES_PATH
    print(f"Loading rules from {rules_path} ...")
    with open(rules_path) as f:
        rules = json.load(f)

    version = rules.get("version", "1.0")
    print(f"Rules version: {version}")

    if version in ("2.0", "3.0", "4.0"):
        g_rules, structured_by_id, ann_rule_map, mappings = _load_rules_v2(rules)
    else:
        g_rules, structured_by_id, ann_rule_map, mappings = _load_rules_v1(rules)
    bench["load_rules_s"] = time.monotonic() - t

    # For v4.0, load subsumption index for applies_to matching
    subsumption_idx = None
    use_subsumption = version == "4.0"
    if use_subsumption:
        from snomed_subsumption import SubsumptionIndex
        t = time.monotonic()
        print("Loading SNOMED subsumption index for v4.0 rule matching ...")
        subsumption_idx = SubsumptionIndex.load()
        bench["load_subsumption_index_s"] = time.monotonic() - t

    # Build case-insensitive mapping lookup
    mappings_lower = {k.lower(): v for k, v in mappings.items()}

    # Load data
    t = time.monotonic()
    print("Loading note and annotations ...")
    note_id, note_text, note_anns = load_note(args.note)
    sections = segment_sections(note_text)
    bench["load_note_s"] = time.monotonic() - t

    print(f"Note: {note_id}  |  {len(note_anns)} annotations")
    print(f"Rules: {len(g_rules)} global, {len(structured_by_id)} structured, {len(mappings)} mappings")
    print(f"Backend: {args.backend}")
    print(f"Prompt style: {args.prompt_style}")
    print(f"Context window: {args.context_before} before / {args.context_after} after")

    # Initialize retrieval
    t = time.monotonic()
    print("Initializing SNOMED retrieval system ...")
    _init_retrieval()
    bench["init_retrieval_s"] = time.monotonic() - t

    # --- Prepare all annotation data up front ---
    t = time.monotonic()
    annotations = []
    n_subsumption_matched = 0
    n_map_matched = 0
    for _, row in note_anns.iterrows():
        ann_id = str(int(row["annotation_id"]))
        gold_start = int(row["start"])
        gold_end = int(row["end"])
        gold_cid = int(row["concept_id"])
        gold_span = note_text[gold_start:gold_end]

        sec = get_section_for_pos(gold_start, sections)
        section_header = sec.header if sec else "unknown"

        if use_subsumption and subsumption_idx is not None:
            # v4.0: compute applicable rules via subsumption
            applicable_structured = []
            for rule in structured_by_id.values():
                applies_to = rule.get("applies_to")
                if applies_to and subsumption_idx.match_rule_applies_to(
                    gold_cid, section_header, applies_to
                ):
                    applicable_structured.append(rule)
            if applicable_structured:
                n_subsumption_matched += 1
        else:
            # v2.0/v3.0: use annotation_rule_map
            mapped_rule_ids = (ann_rule_map or {}).get(ann_id, [])
            applicable_structured = [
                structured_by_id[rid]
                for rid in mapped_rule_ids
                if rid in structured_by_id
            ]
            if mapped_rule_ids:
                n_map_matched += 1

        search_rules_text = format_rules_for_search(g_rules, applicable_structured)
        select_rules_text = format_rules_for_select(g_rules, applicable_structured)

        before, span, after = get_context(
            note_text, gold_start, gold_end,
            args.context_before, args.context_after,
        )

        # Check if span matches a mapping (bypass LLM search)
        mapping_hit = mappings_lower.get(gold_span.strip().lower())

        annotations.append({
            "ann_id": ann_id,
            "gold_start": gold_start,
            "gold_end": gold_end,
            "gold_cid": gold_cid,
            "gold_span": gold_span,
            "section_header": section_header,
            "search_rules_text": search_rules_text,
            "select_rules_text": select_rules_text,
            "before": before,
            "span": span,
            "after": after,
            "mapping_hit": mapping_hit,
        })

    bench["prepare_annotations_s"] = time.monotonic() - t

    if use_subsumption:
        print(f"Subsumption matching: {n_subsumption_matched}/{len(annotations)} annotations matched at least one rule")
    else:
        print(f"Rule map matching: {n_map_matched}/{len(annotations)} annotations matched at least one rule")

    t0 = time.time()

    if args.backend == "vllm":
        # Pipelined: each annotation flows search→retrieve→select independently
        # so vLLM's continuous batching keeps the GPU saturated throughout.
        print(f"\n=== PIPELINED: search → retrieve → select ({len(annotations)} annotations) ===")
        all_search_terms, all_candidates, all_selections, pipeline_ann_timings = vllm_pipeline(
            annotations,
            base_url=args.vllm_url, model=args.vllm_model,
            max_concurrent=args.concurrency,
            reasoning_effort=args.reasoning_effort,
            prompt_style=args.prompt_style,
        )
        bench["pipeline_ann_timings"] = pipeline_ann_timings
    else:
        pipeline_ann_timings = None
        # Sonnet: sequential three-phase approach
        # Pre-resolve mappings, only send unmapped annotations to LLM
        n_mapped = sum(1 for a in annotations if a.get("mapping_hit"))
        print(f"\n=== PASS 1: Generating search terms ({len(annotations)} annotations, {n_mapped} pre-mapped) ===")

        all_search_terms: list[list[str]] = []
        unmapped_indices = []
        unmapped_prompts = []
        unmapped_spans = []

        for i, a in enumerate(annotations):
            if a.get("mapping_hit"):
                all_search_terms.append([a["mapping_hit"], a["span"]])
            else:
                all_search_terms.append([])  # placeholder
                unmapped_indices.append(i)
                unmapped_prompts.append(
                    build_search_prompt(a["before"], a["span"], a["after"],
                                        a["section_header"], a["search_rules_text"],
                                        prompt_style=args.prompt_style)
                )
                unmapped_spans.append(a["span"])

        if unmapped_prompts:
            llm_search_terms = asyncio.run(
                sonnet_search_terms_batch(unmapped_prompts, unmapped_spans)
            )
            for idx, terms in zip(unmapped_indices, llm_search_terms):
                all_search_terms[idx] = terms

        print(f"\n=== RETRIEVAL: Running SNOMED searches (batch_hybrid) ===")
        n_queries = sum(len(terms) for terms in all_search_terms)
        items = [(str(idx), terms) for idx, terms in enumerate(all_search_terms)]
        print(f"  {n_queries} queries from {len(annotations)} annotations (1 batch call)")
        result_map = snomed_search_batch_items(items, top_k=10)

        all_candidates: list[list[dict]] = []
        for idx in range(len(annotations)):
            all_candidates.append(result_map.get(str(idx), [])[:15])

        print(f"\n=== PASS 2: Selecting concepts ({len(annotations)} annotations) ===")
        select_prompts = [
            build_select_prompt(
                a["before"], a["span"], a["after"],
                a["section_header"], a["gold_start"], a["gold_end"],
                a["select_rules_text"], candidates,
                prompt_style=args.prompt_style,
            )
            for a, candidates in zip(annotations, all_candidates)
        ]
        all_selections = asyncio.run(
            sonnet_select_batch(select_prompts, all_candidates)
        )

    bench["pipeline_total_s"] = time.time() - t0

    # --- Score results ---
    t = time.monotonic()
    predictions: list[dict] = []
    per_ann_results: list[dict] = []

    for idx, (a, search_terms, candidates, result) in enumerate(
        zip(annotations, all_search_terms, all_candidates, all_selections)
    ):
        gold_start = a["gold_start"]
        gold_end = a["gold_end"]
        gold_cid = a["gold_cid"]
        gold_span = a["gold_span"]

        # Retrieval diagnostics
        n_candidates = len(candidates)
        candidate_ids = [c["concept_id"] for c in candidates]
        gold_in_candidates = gold_cid in candidate_ids
        gold_rank = candidate_ids.index(gold_cid) + 1 if gold_in_candidates else None

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

            # Failure categorization
            if concept_match:
                failure_reason = None
            elif not gold_in_candidates:
                failure_reason = "retrieval_miss"
            else:
                failure_reason = "selection_miss"

            per_ann_results.append({
                "annotation_id": a["ann_id"],
                "gold_span": gold_span,
                "gold_concept": gold_cid,
                "pred_concept": pred_cid,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "concept_match": concept_match,
                "iou": iou,
                "section": a["section_header"],
                "search_terms": search_terms,
                "n_candidates": n_candidates,
                "gold_in_candidates": gold_in_candidates,
                "gold_rank": gold_rank,
                "failure_reason": failure_reason,
            })

            tag = "MATCH" if concept_match else ("RETR_MISS" if failure_reason == "retrieval_miss" else "SEL_MISS")
            rank_str = f"rank={gold_rank}" if gold_rank else "not_found"
            print(f"  [{idx+1:2d}/{len(annotations)}] '{gold_span}' -> {tag}  iou={iou:.3f}  candidates={n_candidates} ({rank_str})  searches={search_terms}")
        else:
            failure_reason = "retrieval_miss" if not gold_in_candidates else "parse_fail"

            per_ann_results.append({
                "annotation_id": a["ann_id"],
                "gold_span": gold_span,
                "gold_concept": gold_cid,
                "pred_concept": None,
                "pred_start": None,
                "pred_end": None,
                "concept_match": False,
                "iou": 0.0,
                "section": a["section_header"],
                "search_terms": search_terms,
                "n_candidates": n_candidates,
                "gold_in_candidates": gold_in_candidates,
                "gold_rank": gold_rank,
                "failure_reason": failure_reason,
            })
            print(f"  [{idx+1:2d}/{len(annotations)}] '{gold_span}' -> FAIL ({failure_reason})  candidates={n_candidates}  searches={search_terms}")

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

    total_elapsed = time.time() - t0
    bench["scoring_s"] = time.monotonic() - t

    # Check wider retrieval for misses
    t = time.monotonic()
    check_gold_in_wider_retrieval(per_ann_results)
    bench["wider_retrieval_check_s"] = time.monotonic() - t

    # --- Model info ---
    if args.backend == "vllm":
        model_name = args.vllm_model
        reasoning = args.reasoning_effort
    else:
        model_name = "claude-sonnet-4-20250514"
        reasoning = None

    print_summary_report(
        per_ann_results,
        backend=args.backend,
        model=model_name,
        temperature=0.0,
        reasoning_effort=reasoning,
        macro_char_iou=agg_iou,
        total_time_s=total_elapsed,
    )

    # --- Save results ---
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    n_total = len(per_ann_results)
    n_matches = sum(1 for r in per_ann_results if r["concept_match"])
    avg_iou = sum(r["iou"] for r in per_ann_results) / max(n_total, 1)

    n_mapping_resolved = sum(1 for a in annotations if a.get("mapping_hit"))

    output = {
        "timestamp": timestamp,
        "note_id": note_id,
        "rules_version": version,
        "backend": args.backend,
        "model": model_name,
        "temperature": 0.0,
        "reasoning_effort": reasoning,
        "prompt_style": args.prompt_style,
        "context_before": args.context_before,
        "context_after": args.context_after,
        "n_annotations": n_total,
        "n_mapping_resolved": n_mapping_resolved,
        "n_concept_matches": n_matches,
        "concept_accuracy": n_matches / max(n_total, 1),
        "avg_per_annotation_iou": avg_iou,
        "macro_char_iou": agg_iou,
        "total_time_s": round(total_elapsed, 1),
        "per_annotation": per_ann_results,
    }

    # Add stage-targeted diagnostics for v2.0+ rules
    if version in ("2.0", "3.0"):
        output["rule_improvement_suggestions"] = _suggest_rule_improvements(
            per_ann_results, structured_by_id, ann_rule_map,
        )

    # Save results alongside the rules file when --rules is specified
    results_dir = rules_path.parent
    if args.backend == "vllm":
        results_path = results_dir / f"test_results_vllm_{timestamp}.json"
    else:
        results_path = results_dir / f"test_results_sonnet_{timestamp}.json"

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")

    # --- Save benchmarks ---
    bench["total_wall_s"] = time.monotonic() - bench_wall_start
    bench["n_annotations"] = len(annotations)
    bench["n_mapping_resolved"] = n_mapping_resolved
    bench["rules_version"] = version
    bench["backend"] = args.backend
    bench["note_id"] = note_id

    # Aggregate per-annotation pipeline timings (vllm only)
    if pipeline_ann_timings:
        search_times = [t["search_s"] for t in pipeline_ann_timings]
        retrieval_times = [t["retrieval_s"] for t in pipeline_ann_timings]
        select_times = [t["select_s"] for t in pipeline_ann_timings]
        total_times = [t["total_s"] for t in pipeline_ann_timings]
        n_bypassed = sum(1 for t in pipeline_ann_timings if t["mapping_bypass"])

        bench["per_annotation_stats"] = {
            "search_llm": {
                "mean_s": sum(search_times) / len(search_times),
                "min_s": min(search_times),
                "max_s": max(search_times),
                "p50_s": sorted(search_times)[len(search_times) // 2],
                "p95_s": sorted(search_times)[int(len(search_times) * 0.95)],
                "total_s": sum(search_times),
            },
            "retrieval": {
                "mean_s": sum(retrieval_times) / len(retrieval_times),
                "min_s": min(retrieval_times),
                "max_s": max(retrieval_times),
                "p50_s": sorted(retrieval_times)[len(retrieval_times) // 2],
                "p95_s": sorted(retrieval_times)[int(len(retrieval_times) * 0.95)],
                "total_s": sum(retrieval_times),
            },
            "select_llm": {
                "mean_s": sum(select_times) / len(select_times),
                "min_s": min(select_times),
                "max_s": max(select_times),
                "p50_s": sorted(select_times)[len(select_times) // 2],
                "p95_s": sorted(select_times)[int(len(select_times) * 0.95)],
                "total_s": sum(select_times),
            },
            "per_annotation_total": {
                "mean_s": sum(total_times) / len(total_times),
                "min_s": min(total_times),
                "max_s": max(total_times),
                "p50_s": sorted(total_times)[len(total_times) // 2],
                "p95_s": sorted(total_times)[int(len(total_times) * 0.95)],
            },
            "n_mapping_bypassed": n_bypassed,
        }

    # Remove raw per-annotation timings from bench (too verbose)
    bench.pop("pipeline_ann_timings", None)

    # Round all floats
    def _round_floats(obj):
        if isinstance(obj, float):
            return round(obj, 3)
        if isinstance(obj, dict):
            return {k: _round_floats(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_floats(v) for v in obj]
        return obj

    bench = _round_floats(bench)

    bench_path = results_dir / f"benchmark_testing_{timestamp}.json"
    with open(bench_path, "w") as f:
        json.dump(bench, f, indent=2)
    print(f"Saved benchmarks to {bench_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test annotation rules with LLM agent")
    parser.add_argument(
        "--backend", choices=["sonnet", "vllm"], default="sonnet",
        help="LLM backend: 'sonnet' (Claude Agent SDK) or 'vllm' (gpt-oss-20b)",
    )
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--vllm-model", default="openai/gpt-oss-20b",
        help="vLLM model name (default: openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=0,
        help="Max concurrent requests for vLLM (default: 0 = unlimited, let vLLM batch)",
    )
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high", "none"], default="low",
        help="Reasoning effort for gpt-oss-20b (default: low). Use 'none' to disable (e.g. for non-reasoning models)",
    )
    parser.add_argument(
        "--prompt-style", choices=["standard", "repeated-text"], default="standard",
        help="Prompt template style: 'standard' (rules then excerpt) or 'repeated-text' (excerpt sandwiched around rules) (default: standard)",
    )
    parser.add_argument(
        "--rules", type=str, default=None, metavar="PATH",
        help="Path to rules JSON file (default: scripts/rules_output.json)",
    )
    parser.add_argument(
        "--note", type=str, default=None, metavar="NOTE_ID",
        help="Note ID to test (default: first note in train_notes.csv)",
    )
    parser.add_argument(
        "--replay", type=str, default=None, metavar="PATH",
        help="Replay results from a saved JSON file (no inference, just report + top-100 check)",
    )
    parser.add_argument(
        "--index-server", type=str, default=None, metavar="URL",
        help="URL of a running snomed_index_server.py (e.g. http://127.0.0.1:8421). "
             "Skips local FAISS/BM25/SapBERT loading entirely.",
    )
    parser.add_argument(
        "--context-before", type=int, default=None, metavar="N",
        help=f"Characters of context before the span (default: {DEFAULT_CONTEXT_BEFORE})",
    )
    parser.add_argument(
        "--context-after", type=int, default=None, metavar="N",
        help=f"Characters of context after the span (default: {DEFAULT_CONTEXT_AFTER})",
    )
    parser.add_argument(
        "--config", type=str, default=None, metavar="PATH",
        help="Path to a JSON config file (can set context_before, context_after)",
    )
    args = parser.parse_args()

    # Load config file, then apply CLI overrides on top
    config: dict = {}
    if args.config:
        with open(args.config) as _f:
            config = json.load(_f)
    args.context_before = args.context_before if args.context_before is not None else config.get("context_before", DEFAULT_CONTEXT_BEFORE)
    args.context_after = args.context_after if args.context_after is not None else config.get("context_after", DEFAULT_CONTEXT_AFTER)
    if "reasoning_effort" in config and args.reasoning_effort == "low":  # "low" is the argparse default
        args.reasoning_effort = config["reasoning_effort"]
    if "vllm_model" in config and args.vllm_model == "openai/gpt-oss-20b":
        args.vllm_model = config["vllm_model"]
    if "vllm_url" in config and args.vllm_url == "http://localhost:8000":
        args.vllm_url = config["vllm_url"]
    if "concurrency" in config and args.concurrency == 0:
        args.concurrency = config["concurrency"]
    if "prompt_style" in config and args.prompt_style == "standard":
        args.prompt_style = config["prompt_style"]
    global _index_server_url
    _index_server_url = args.index_server

    if args.replay:
        replay(args.replay)
    else:
        run(args)


if __name__ == "__main__":
    main()
