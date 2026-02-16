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
RULES_PATH = Path(__file__).parent / "rules_output.json"
RESULTS_PATH = Path(__file__).parent / "test_results.json"

CONTEXT_BEFORE = 300
CONTEXT_AFTER = 100


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
    """Execute hybrid SNOMED retrieval for a single query."""
    results = snomed_search_batch([query_text], top_k=top_k)
    return results[0]


def snomed_search_batch(queries: list[str], top_k: int = 10) -> list[list[dict]]:
    """Batch SNOMED retrieval: encode + FAISS + BM25 all queries at once."""
    from snomed_ct_entity_linking.recall_analysis.config import Config
    from snomed_ct_entity_linking.recall_analysis.retrieval import (
        dense_search,
        reciprocal_rank_fusion,
        sparse_search,
    )

    _init_retrieval()
    top_k = min(max(top_k, 1), 20)
    cfg = Config()

    query_embs = _encoder.encode(queries, batch_size=max(len(queries), 1))
    dense_results = dense_search(
        query_embs, _faiss_index, _faiss_sctids, cfg.dense_top_k, verbose=False,
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
# Shared system prompts
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

SELECT_SYSTEM = """\
You are a clinical NLP agent. Given annotation rules, a text excerpt with a \
highlighted region, and SNOMED CT search results, select the best matching concept.

Respond with ONLY a JSON object:
{"concept_id": <integer>, "span_start": <int>, "span_end": <int>}

Choose the concept whose meaning best matches the clinical text in context. \
The span_start and span_end should be the exact character positions provided.\
"""


# ---------------------------------------------------------------------------
# Prompt builders (shared by both backends)
# ---------------------------------------------------------------------------

def build_search_prompt(
    before: str, span: str, after: str,
    section_header: str, rules_text: str,
) -> str:
    return f"""\
{rules_text}

Section: {section_header}

Excerpt: ...{before}>>>{span}<<<{after}...

The text between >>> and <<< is the region of interest.
Generate search terms to find the matching SNOMED concept.\
"""


def build_select_prompt(
    before: str, span: str, after: str,
    section_header: str, ann_start: int, ann_end: int,
    rules_text: str, candidates: list[dict],
) -> str:
    candidates_text = "\n".join(
        f"  {i+1}. [{c['concept_id']}] {c['concept_name']} ({c['hierarchy']}) — score: {c['score']}"
        for i, c in enumerate(candidates)
    )
    return f"""\
{rules_text}

Section: {section_header}

Excerpt: ...{before}>>>{span}<<<{after}...

The text between >>> and <<< is the region of interest.
Span position: start={ann_start}, end={ann_end}

SNOMED search results:
{candidates_text}

Select the best matching concept and confirm span boundaries.\
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


def parse_concept_response(text: str) -> tuple[int, int, int] | None:
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


async def sonnet_select_batch(prompts: list[str]) -> list[tuple[int, int, int] | None]:
    """Sequential Pass 2 for Sonnet backend."""
    results = []
    for i, prompt in enumerate(prompts):
        raw = await _sonnet_call(SELECT_SYSTEM, prompt)
        results.append(parse_concept_response(raw))
        if (i + 1) % 10 == 0:
            print(f"  [select] {i+1}/{len(prompts)} done")
    return results


# ---------------------------------------------------------------------------
# Backend: vLLM / gpt-oss-20b (concurrent, via AsyncOpenAI)
# ---------------------------------------------------------------------------

class _AsyncBatchRetriever:
    """Accumulates SNOMED search queries from concurrent tasks and processes in batches.

    Instead of each annotation calling snomed_search() individually (one SapBERT
    encode + one FAISS search + one BM25 search per query), this collects queries
    arriving from concurrent annotations and processes them in efficient batches.
    """

    def __init__(self, batch_size: int = 32, flush_interval: float = 0.15):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self.n_queries = 0
        self.n_batches = 0
        self._stop = False

    async def search_multi(self, queries: list[str], top_k: int = 10) -> list[list[dict]]:
        """Submit multiple queries and await all results."""
        loop = asyncio.get_event_loop()
        futures = []
        for q in queries:
            future = loop.create_future()
            await self._queue.put((q, top_k, future))
            futures.append(future)
        return [await f for f in futures]

    def stop(self):
        self._stop = True

    async def worker(self):
        """Background worker: collects queries into batches and processes them."""
        loop = asyncio.get_event_loop()
        while not self._stop or not self._queue.empty():
            batch = []

            # Wait for first item (with timeout to check stop flag)
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                batch.append(item)
            except asyncio.TimeoutError:
                continue

            # Collect more items within flush_interval window
            deadline = loop.time() + self._flush_interval
            while len(batch) < self._batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            if not batch:
                continue

            # Process batch
            queries = [q for q, _, _ in batch]
            top_k = max(tk for _, tk, _ in batch)

            results = await loop.run_in_executor(
                None, snomed_search_batch, queries, top_k,
            )

            for (_, _, future), result in zip(batch, results):
                if not future.done():
                    future.set_result(result)

            self.n_queries += len(batch)
            self.n_batches += 1


def vllm_pipeline(
    annotations: list[dict],
    base_url: str = "http://localhost:8000",
    model: str = "openai/gpt-oss-20b",
    max_concurrent: int = 0,
    reasoning_effort: str = "low",
) -> tuple[list[list[str]], list[list[dict]], list[tuple[int, int, int] | None]]:
    """Run the full search->retrieve->select pipeline with continuous batching.

    Each annotation independently flows through:
      1. LLM generates search terms (Pass 1)
      2. Batched SNOMED retrieval (queries accumulated across annotations)
      3. LLM selects concept from candidates (Pass 2)

    All LLM requests share a single AsyncOpenAI client so vLLM's continuous
    batching keeps the GPU saturated. Retrieval queries are accumulated by an
    async batch retriever for efficient SapBERT/FAISS/BM25 processing.
    """
    from openai import AsyncOpenAI

    extra_body = {}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort

    n = len(annotations)
    conc_label = "unlimited" if max_concurrent <= 0 else str(max_concurrent)
    print(f"  Pipeline: {n} annotations "
          f"(max_concurrent={conc_label}, model={model}, "
          f"reasoning_effort={reasoning_effort})")

    # Scale max_tokens with reasoning effort — reasoning tokens count against the budget
    max_tokens = {"low": 256, "medium": 1024, "high": 2048}.get(reasoning_effort, 256)

    # Pre-build search prompts
    search_prompts = [
        build_search_prompt(a["before"], a["span"], a["after"],
                            a["section_header"], a["rules_text"])
        for a in annotations
    ]

    t0 = time.time()

    # Output arrays (filled by each task)
    out_search_terms: list[list[str]] = [[] for _ in range(n)]
    out_candidates: list[list[dict]] = [[] for _ in range(n)]
    out_selections: list[tuple[int, int, int] | None] = [None] * n

    async def _run():
        async_client = AsyncOpenAI(
            base_url=f"{base_url}/v1", api_key="unused",
            max_retries=2,
            timeout=300.0,
        )
        sem = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        batcher = _AsyncBatchRetriever(batch_size=32, flush_interval=0.15)
        worker_task = asyncio.create_task(batcher.worker())

        pbar = tqdm(total=n, desc="Annotations", unit="ann",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        async def _llm_call(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
            if sem is not None:
                await sem.acquire()
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                resp = await async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    frequency_penalty=0.3,
                    extra_body=extra_body if extra_body else None,
                )
                choice = resp.choices[0]
                content = choice.message.content or ""
                reasoning = getattr(choice.message, "reasoning_content", None) or ""
                return content.strip() if content.strip() else reasoning.strip()
            except Exception as e:
                tqdm.write(f"  ERROR: {e}")
                return ""
            finally:
                if sem is not None:
                    sem.release()

        async def _process_annotation(idx: int):
            a = annotations[idx]

            # --- Pass 1: search terms ---
            raw_search = await _llm_call(SEARCH_SYSTEM, search_prompts[idx], max_tokens=max_tokens)
            search_terms = parse_search_terms(raw_search, a["span"])
            out_search_terms[idx] = search_terms

            # --- Retrieval (batched across annotations) ---
            per_term_results = await batcher.search_multi(search_terms, top_k=10)
            merged: dict[int, dict] = {}
            for results in per_term_results:
                for c in results:
                    cid = c["concept_id"]
                    if cid not in merged or c["score"] > merged[cid]["score"]:
                        merged[cid] = c
            candidates = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:15]
            out_candidates[idx] = candidates

            # --- Pass 2: select concept ---
            select_prompt = build_select_prompt(
                a["before"], a["span"], a["after"],
                a["section_header"], a["gold_start"], a["gold_end"],
                a["rules_text"], candidates,
            )
            raw_select = await _llm_call(SELECT_SYSTEM, select_prompt, max_tokens=max_tokens)
            out_selections[idx] = parse_concept_response(raw_select)

            pbar.update(1)

        tasks = [_process_annotation(i) for i in range(n)]
        await asyncio.gather(*tasks)

        # Shut down batcher
        batcher.stop()
        await worker_task

        pbar.close()
        await async_client.close()

        tqdm.write(f"  Retrieval: {batcher.n_queries} queries in {batcher.n_batches} batches "
                    f"(avg {batcher.n_queries / max(batcher.n_batches, 1):.1f} queries/batch)")

    asyncio.run(_run())

    elapsed = time.time() - t0
    print(f"  Pipeline done in {elapsed:.1f}s ({n / elapsed:.1f} ann/s)")
    return out_search_terms, out_candidates, out_selections


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

def run(args: argparse.Namespace) -> None:
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
    print(f"Backend: {args.backend}")

    # Initialize retrieval
    print("Initializing SNOMED retrieval system ...")
    _init_retrieval()

    # --- Prepare all annotation data up front ---
    annotations = []
    for _, row in note_anns.iterrows():
        ann_id = str(int(row["annotation_id"]))
        gold_start = int(row["start"])
        gold_end = int(row["end"])
        gold_cid = int(row["concept_id"])
        gold_span = note_text[gold_start:gold_end]

        sec = get_section_for_pos(gold_start, sections)
        section_header = sec.header if sec else "unknown"

        mapped_rule_ids = ann_rule_map.get(ann_id, [])
        applicable_numbered = [
            numbered_rules_by_id[rid]
            for rid in mapped_rule_ids
            if rid in numbered_rules_by_id
        ]
        rules_text = format_rules(g_rules, applicable_numbered)

        before, span, after = get_context(note_text, gold_start, gold_end)

        annotations.append({
            "ann_id": ann_id,
            "gold_start": gold_start,
            "gold_end": gold_end,
            "gold_cid": gold_cid,
            "gold_span": gold_span,
            "section_header": section_header,
            "rules_text": rules_text,
            "before": before,
            "span": span,
            "after": after,
        })

    t0 = time.time()

    if args.backend == "vllm":
        # Pipelined: each annotation flows search→retrieve→select independently
        # so vLLM's continuous batching keeps the GPU saturated throughout.
        print(f"\n=== PIPELINED: search → retrieve → select ({len(annotations)} annotations) ===")
        all_search_terms, all_candidates, all_selections = vllm_pipeline(
            annotations,
            base_url=args.vllm_url, model=args.vllm_model,
            max_concurrent=args.concurrency,
            reasoning_effort=args.reasoning_effort,
        )
    else:
        # Sonnet: sequential three-phase approach
        print(f"\n=== PASS 1: Generating search terms ({len(annotations)} annotations) ===")
        search_prompts = [
            build_search_prompt(a["before"], a["span"], a["after"],
                                a["section_header"], a["rules_text"])
            for a in annotations
        ]
        spans = [a["span"] for a in annotations]
        all_search_terms = asyncio.run(
            sonnet_search_terms_batch(search_prompts, spans)
        )

        print(f"\n=== RETRIEVAL: Running SNOMED searches (batched) ===")
        # Collect all search terms and batch-retrieve at once
        all_queries = []
        query_ann_map = []  # (annotation_idx, term_idx)
        for idx, search_terms in enumerate(all_search_terms):
            for term_idx, term in enumerate(search_terms):
                all_queries.append(term)
                query_ann_map.append((idx, term_idx))

        print(f"  {len(all_queries)} queries from {len(annotations)} annotations")
        batch_results = snomed_search_batch(all_queries, top_k=10)

        # Map results back to annotations and merge per-annotation
        ann_term_results: dict[int, list[list[dict]]] = {}
        for (ann_idx, _), results in zip(query_ann_map, batch_results):
            ann_term_results.setdefault(ann_idx, []).append(results)

        all_candidates: list[list[dict]] = []
        for idx in range(len(annotations)):
            merged: dict[int, dict] = {}
            for results in ann_term_results.get(idx, []):
                for c in results:
                    cid = c["concept_id"]
                    if cid not in merged or c["score"] > merged[cid]["score"]:
                        merged[cid] = c
            candidates = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:15]
            all_candidates.append(candidates)

        print(f"\n=== PASS 2: Selecting concepts ({len(annotations)} annotations) ===")
        select_prompts = [
            build_select_prompt(
                a["before"], a["span"], a["after"],
                a["section_header"], a["gold_start"], a["gold_end"],
                a["rules_text"], candidates,
            )
            for a, candidates in zip(annotations, all_candidates)
        ]
        all_selections = asyncio.run(
            sonnet_select_batch(select_prompts)
        )

    # --- Score results ---
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

    n_matches = sum(1 for r in per_ann_results if r["concept_match"])
    n_total = len(per_ann_results)
    avg_iou = sum(r["iou"] for r in per_ann_results) / max(n_total, 1)
    total_elapsed = time.time() - t0

    # Retrieval & failure stats
    n_gold_in_candidates = sum(1 for r in per_ann_results if r["gold_in_candidates"])
    n_retrieval_miss = sum(1 for r in per_ann_results if r["failure_reason"] == "retrieval_miss")
    n_selection_miss = sum(1 for r in per_ann_results if r["failure_reason"] == "selection_miss")
    n_parse_fail = sum(1 for r in per_ann_results if r["failure_reason"] == "parse_fail")
    avg_candidates = sum(r["n_candidates"] for r in per_ann_results) / max(n_total, 1)
    gold_ranks = [r["gold_rank"] for r in per_ann_results if r["gold_rank"] is not None]
    avg_gold_rank = sum(gold_ranks) / len(gold_ranks) if gold_ranks else 0

    # --- Model info ---
    if args.backend == "vllm":
        model_name = args.vllm_model
        temperature = 0.0
        reasoning = args.reasoning_effort
    else:
        model_name = "claude-sonnet-4-20250514"
        temperature = 0.0
        reasoning = None

    # --- Report ---
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Backend:            {args.backend}")
    print(f"Model:              {model_name}")
    print(f"Temperature:        {temperature}")
    if reasoning:
        print(f"Reasoning effort:   {reasoning}")
    print(f"Total annotations:  {n_total}")
    print(f"Concept matches:    {n_matches}/{n_total} ({100 * n_matches / max(n_total, 1):.1f}%)")
    print(f"Avg per-ann IoU:    {avg_iou:.4f}")
    print(f"Macro char IoU:     {agg_iou:.4f}")
    print(f"Total time:         {total_elapsed:.1f}s")

    print(f"\nRetrieval:")
    print(f"  Avg candidates:   {avg_candidates:.1f}")
    print(f"  Gold in top-k:    {n_gold_in_candidates}/{n_total} ({100 * n_gold_in_candidates / max(n_total, 1):.1f}%)")
    if gold_ranks:
        print(f"  Avg gold rank:    {avg_gold_rank:.1f} (when found)")

    print(f"\nFailure breakdown:")
    print(f"  Correct:          {n_matches}")
    print(f"  Retrieval miss:   {n_retrieval_miss}  (gold concept not in candidates)")
    print(f"  Selection miss:   {n_selection_miss}  (gold in candidates, LLM picked wrong)")
    if n_parse_fail:
        print(f"  Parse fail:       {n_parse_fail}  (LLM response unparseable)")

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
        sel_miss = sum(1 for r in group if r["failure_reason"] == "selection_miss")
        sec_iou = sum(r["iou"] for r in group) / n
        print(f"  {sec:40s} n={n:3d}  acc={100 * matches / n:5.1f}%  retrieval={100 * retr / n:5.1f}%  sel_miss={sel_miss}  iou={sec_iou:.3f}")

    # --- Save results ---
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    output = {
        "timestamp": timestamp,
        "note_id": note_id,
        "backend": args.backend,
        "model": model_name,
        "temperature": temperature,
        "reasoning_effort": reasoning,
        "n_annotations": n_total,
        "n_concept_matches": n_matches,
        "concept_accuracy": n_matches / max(n_total, 1),
        "avg_per_annotation_iou": avg_iou,
        "macro_char_iou": agg_iou,
        "total_time_s": round(total_elapsed, 1),
        "retrieval": {
            "avg_candidates": round(avg_candidates, 1),
            "gold_in_topk": n_gold_in_candidates,
            "gold_in_topk_pct": round(100 * n_gold_in_candidates / max(n_total, 1), 1),
            "avg_gold_rank": round(avg_gold_rank, 1) if gold_ranks else None,
        },
        "failure_breakdown": {
            "correct": n_matches,
            "retrieval_miss": n_retrieval_miss,
            "selection_miss": n_selection_miss,
            "parse_fail": n_parse_fail,
        },
        "per_annotation": per_ann_results,
    }

    if args.backend == "vllm":
        results_path = RESULTS_PATH.with_name(f"test_results_vllm_{timestamp}.json")
    else:
        results_path = RESULTS_PATH

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {results_path}")


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
        "--reasoning-effort", choices=["low", "medium", "high"], default="low",
        help="Reasoning effort for gpt-oss-20b (default: low)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
