#!/usr/bin/env python3
"""Search term oracle: have an LLM generate 'ideal' search terms for each annotation.

For each annotation, shows an LLM the clinical context AND the correct SNOMED
concept, then asks it to generate realistic search terms an agent would use.
We then run those terms through retrieval to measure the retrieval ceiling.

This separates two failure modes:
  1. Agent generates poor search terms  → rule/prompt problem
  2. Retrieval can't find the concept even with good terms → index problem

Usage:
  python scripts/search_term_oracle.py --backend vllm
  python scripts/search_term_oracle.py --backend vllm --reasoning-effort high
  python scripts/search_term_oracle.py --backend sonnet
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))
sys.path.insert(0, str(REPO_ROOT))

from engine import get_section_for_pos, segment_sections  # noqa: E402

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
TERMINOLOGY_CSV = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"

CONTEXT_BEFORE = 300
CONTEXT_AFTER = 100


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_note(index: int = 0) -> tuple[str, str, pd.DataFrame]:
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    ann_df = pd.read_csv(SPLIT_DIR / "train_annotations.csv")
    if index < 0 or index >= len(notes_df):
        raise ValueError(f"Note index {index} out of range (0-{len(notes_df)-1})")
    note_id = notes_df.iloc[index]["note_id"]
    note_text = notes_df.iloc[index]["text"]
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


def get_context(note_text: str, start: int, end: int) -> tuple[str, str, str]:
    ctx_start = max(0, start - CONTEXT_BEFORE)
    ctx_end = min(len(note_text), end + CONTEXT_AFTER)
    return note_text[ctx_start:start], note_text[start:end], note_text[end:ctx_end]


# ---------------------------------------------------------------------------
# SNOMED retrieval (same pattern as rule_testing.py)
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
# LLM prompt
# ---------------------------------------------------------------------------

ORACLE_SYSTEM = """\
You are a clinical NLP expert analyzing annotated medical text. Given a clinical \
text excerpt with a highlighted region and the correct SNOMED CT concept for that \
region, generate 1-3 search terms that a clinical NLP agent would realistically \
use to find this concept in a terminology search index.

Guidelines:
- Think about what terms a skilled agent would derive from reading the clinical text
- Do NOT just copy the SNOMED concept name verbatim — that would be cheating
- Consider what the clinical text actually says: abbreviations, shorthand, \
implicit meanings
- Your terms should be what someone reading the text would naturally search for
- Include the literal span text as one term if it's clinically meaningful
- Add clinical expansions or synonyms the agent might reasonably infer

Respond with ONLY a JSON object:
{"search_terms": ["term1", "term2", "term3"]}\
"""


def build_oracle_prompt(
    before: str, span: str, after: str,
    section_header: str,
    concept_name: str, hierarchy: str,
) -> str:
    return f"""\
Section: {section_header}

Excerpt: ...{before}>>>{span}<<<{after}...

The text between >>> and <<< is the annotated span.

Correct SNOMED concept: {concept_name} ({hierarchy})

Generate realistic search terms an agent would use to find this concept.\
"""


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


# ---------------------------------------------------------------------------
# Backend: vLLM (concurrent)
# ---------------------------------------------------------------------------

def vllm_generate_terms(
    prompts: list[str],
    fallback_spans: list[str],
    base_url: str = "http://localhost:8000",
    model: str = "openai/gpt-oss-20b",
    max_concurrent: int = 0,
    reasoning_effort: str = "low",
) -> list[list[str]]:
    """Generate oracle search terms for all annotations concurrently via vLLM."""
    from openai import AsyncOpenAI

    extra_body = {}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort

    max_tokens = {"low": 256, "medium": 1024, "high": 2048}.get(reasoning_effort, 256)
    n = len(prompts)
    results: list[list[str]] = [[] for _ in range(n)]

    async def _run():
        client = AsyncOpenAI(
            base_url=f"{base_url}/v1", api_key="unused",
            max_retries=2, timeout=300.0,
        )
        sem = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        pbar = tqdm(total=n, desc="Oracle terms", unit="ann")

        async def _call(idx: int):
            if sem is not None:
                await sem.acquire()
            try:
                messages = [
                    {"role": "system", "content": ORACLE_SYSTEM},
                    {"role": "user", "content": prompts[idx]},
                ]
                resp = await client.chat.completions.create(
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
                raw = content.strip() if content.strip() else reasoning.strip()
            except Exception as e:
                tqdm.write(f"  ERROR [{idx}]: {e}")
                raw = ""
            finally:
                if sem is not None:
                    sem.release()

            results[idx] = parse_search_terms(raw, fallback_spans[idx])
            pbar.update(1)

        await asyncio.gather(*[_call(i) for i in range(n)])
        pbar.close()
        await client.close()

    asyncio.run(_run())
    return results


# ---------------------------------------------------------------------------
# Backend: Sonnet (sequential, via Claude Agent SDK)
# ---------------------------------------------------------------------------

def sonnet_generate_terms(
    prompts: list[str],
    fallback_spans: list[str],
) -> list[list[str]]:
    """Generate oracle search terms sequentially via Claude Sonnet."""
    from claude_code_sdk import (
        AssistantMessage,
        ClaudeCodeOptions,
        TextBlock,
        query,
    )

    async def _call(prompt: str) -> str:
        options = ClaudeCodeOptions(
            model="claude-sonnet-4-20250514",
            permission_mode="bypassPermissions",
            max_turns=1,
            system_prompt=ORACLE_SYSTEM,
        )
        raw = ""
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        raw += block.text
        return raw

    results = []
    for i, (prompt, span) in enumerate(zip(prompts, fallback_spans)):
        raw = asyncio.run(_call(prompt))
        terms = parse_search_terms(raw, span)
        results.append(terms)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)} done")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # Load data
    print("Loading data ...")
    note_id, note_text, note_anns = load_note(args.note_index)
    sections = segment_sections(note_text)
    concept_names = load_concept_names()

    print(f"Note: {note_id}  |  {len(note_anns)} annotations")
    print(f"Backend: {args.backend}")

    # Initialize retrieval
    print("Initializing SNOMED retrieval ...")
    _init_retrieval()

    # Prepare annotation data
    annotations = []
    for _, row in note_anns.iterrows():
        ann_id = str(int(row["annotation_id"]))
        gold_start, gold_end = int(row["start"]), int(row["end"])
        gold_cid = int(row["concept_id"])
        gold_span = note_text[gold_start:gold_end]

        sec = get_section_for_pos(gold_start, sections)
        section_header = sec.header if sec else "unknown"
        before, span, after = get_context(note_text, gold_start, gold_end)

        cname, hierarchy = concept_names.get(gold_cid, ("Unknown", "unknown"))

        annotations.append({
            "ann_id": ann_id,
            "gold_start": gold_start,
            "gold_end": gold_end,
            "gold_cid": gold_cid,
            "gold_span": gold_span,
            "gold_concept_name": cname,
            "gold_hierarchy": hierarchy,
            "section_header": section_header,
            "before": before,
            "span": span,
            "after": after,
        })

    # Build prompts
    prompts = [
        build_oracle_prompt(
            a["before"], a["span"], a["after"],
            a["section_header"],
            a["gold_concept_name"], a["gold_hierarchy"],
        )
        for a in annotations
    ]
    fallback_spans = [a["span"] for a in annotations]

    # --- Phase 1: Generate search terms ---
    t0 = time.time()

    if args.span_only:
        print(f"\n=== Phase 1: Using literal span text ({len(annotations)} annotations) ===")
        all_search_terms = [[a["gold_span"].strip()] for a in annotations]
        print(f"  Phase 1 done (no LLM needed)")
    else:
        print(f"\n=== Phase 1: Generating oracle search terms ({len(annotations)} annotations) ===")
        if args.backend == "vllm":
            all_search_terms = vllm_generate_terms(
                prompts, fallback_spans,
                base_url=args.vllm_url, model=args.vllm_model,
                max_concurrent=args.concurrency,
                reasoning_effort=args.reasoning_effort,
            )
        else:
            all_search_terms = sonnet_generate_terms(prompts, fallback_spans)

        phase1_time = time.time() - t0
        print(f"  Phase 1 done in {phase1_time:.1f}s")

    # --- Phase 2: Batch retrieval for all terms ---
    print(f"\n=== Phase 2: Batch retrieval ===")
    t1 = time.time()

    # Optionally limit to first N terms per annotation
    if args.max_terms:
        all_search_terms = [terms[:args.max_terms] for terms in all_search_terms]

    # Flatten all search terms with source tracking
    all_queries = []
    query_map = []  # (annotation_idx, term_idx)
    for idx, terms in enumerate(all_search_terms):
        for term_idx, term in enumerate(terms):
            all_queries.append(term)
            query_map.append((idx, term_idx))

    print(f"  {len(all_queries)} queries from {len(annotations)} annotations")
    top_k = 20  # search deeper to see where gold falls
    batch_results = snomed_search_batch(all_queries, top_k=top_k)

    phase2_time = time.time() - t1
    print(f"  Phase 2 done in {phase2_time:.1f}s")

    # --- Phase 3: Analyze results ---
    print(f"\n=== Phase 3: Analysis ===")

    # Map results back per annotation, per term
    ann_term_results: dict[int, list[tuple[str, list[dict]]]] = {}
    for (ann_idx, term_idx), results in zip(query_map, batch_results):
        term = all_queries[query_map.index((ann_idx, term_idx))]
        ann_term_results.setdefault(ann_idx, []).append((term, results))

    per_ann: list[dict] = []
    for idx, a in enumerate(annotations):
        gold_cid = a["gold_cid"]
        term_results = ann_term_results.get(idx, [])

        per_term = []
        best_rank = None
        gold_found = False

        for term, candidates in term_results:
            candidate_ids = [c["concept_id"] for c in candidates]
            if gold_cid in candidate_ids:
                rank = candidate_ids.index(gold_cid) + 1
                found = True
            else:
                rank = None
                found = False

            per_term.append({
                "term": term,
                "gold_rank": rank,
                "gold_found": found,
                "n_candidates": len(candidates),
                "top_5": [
                    {"concept_id": c["concept_id"], "concept_name": c["concept_name"]}
                    for c in candidates[:5]
                ],
            })

            if found:
                gold_found = True
                if best_rank is None or rank < best_rank:
                    best_rank = rank

        per_ann.append({
            "ann_id": a["ann_id"],
            "gold_span": a["gold_span"],
            "gold_cid": gold_cid,
            "gold_concept_name": a["gold_concept_name"],
            "gold_hierarchy": a["gold_hierarchy"],
            "section": a["section_header"],
            "search_terms": [t for t, _ in term_results],
            "per_term": per_term,
            "best_rank": best_rank,
            "gold_found": gold_found,
        })

    # --- Print per-annotation detail ---
    for r in per_ann:
        tag = "FOUND" if r["gold_found"] else "MISS "
        rank_str = f"rank={r['best_rank']}" if r["best_rank"] else "not_found"
        terms_str = ", ".join(f'"{t}"' for t in r["search_terms"])
        print(f"  [{r['ann_id']:>3s}] {tag} {rank_str:>12s}  "
              f"'{r['gold_span'][:35]:35s}'  [{r['gold_concept_name'][:40]}]  "
              f"terms=[{terms_str}]")

    # --- Aggregate stats ---
    n = len(per_ann)
    n_found = sum(1 for r in per_ann if r["gold_found"])
    gold_ranks = [r["best_rank"] for r in per_ann if r["best_rank"] is not None]
    avg_rank = sum(gold_ranks) / len(gold_ranks) if gold_ranks else 0
    rank_at_1 = sum(1 for r in gold_ranks if r == 1)
    rank_at_5 = sum(1 for r in gold_ranks if r <= 5)
    rank_at_10 = sum(1 for r in gold_ranks if r <= 10)

    # Per-term stats (how often each individual term finds gold)
    all_term_found = sum(1 for r in per_ann for t in r["per_term"] if t["gold_found"])
    all_term_total = sum(len(r["per_term"]) for r in per_ann)

    # Per-section breakdown
    section_groups: dict[str, list[dict]] = {}
    for r in per_ann:
        section_groups.setdefault(r["section"], []).append(r)

    total_elapsed = time.time() - t0

    # Model info
    if args.span_only:
        model_name = "none (span-only)"
        reasoning = None
    elif args.backend == "vllm":
        model_name = args.vllm_model
        reasoning = args.reasoning_effort
    else:
        model_name = "claude-sonnet-4-20250514"
        reasoning = None

    print(f"\n{'='*80}")
    print("ORACLE SEARCH TERM RESULTS")
    print(f"{'='*80}")
    print(f"Backend:              {args.backend}")
    print(f"Model:                {model_name}")
    if reasoning:
        print(f"Reasoning effort:     {reasoning}")
    print(f"Annotations:          {n}")
    print(f"Retrieval top-k:      {top_k}")
    print(f"Total time:           {total_elapsed:.1f}s")

    print(f"\nRetrieval with oracle terms:")
    print(f"  Gold found (any term):  {n_found}/{n} ({100*n_found/n:.1f}%)")
    print(f"  Gold at rank 1:         {rank_at_1}/{n} ({100*rank_at_1/n:.1f}%)")
    print(f"  Gold in top 5:          {rank_at_5}/{n} ({100*rank_at_5/n:.1f}%)")
    print(f"  Gold in top 10:         {rank_at_10}/{n} ({100*rank_at_10/n:.1f}%)")
    if gold_ranks:
        print(f"  Avg gold rank:          {avg_rank:.1f} (when found)")

    print(f"\nPer-term stats:")
    print(f"  Total terms generated:  {all_term_total}")
    print(f"  Terms finding gold:     {all_term_found}/{all_term_total} "
          f"({100*all_term_found/max(all_term_total,1):.1f}%)")
    print(f"  Avg terms/annotation:   {all_term_total/n:.1f}")

    print(f"\nPer-section breakdown:")
    for sec in sorted(section_groups):
        group = section_groups[sec]
        ns = len(group)
        found = sum(1 for r in group if r["gold_found"])
        ranks = [r["best_rank"] for r in group if r["best_rank"] is not None]
        avg_r = sum(ranks) / len(ranks) if ranks else 0
        print(f"  {sec:40s} n={ns:3d}  found={100*found/ns:5.1f}%  "
              f"avg_rank={avg_r:.1f}")

    # Missed concepts (retrieval failures)
    misses = [r for r in per_ann if not r["gold_found"]]
    if misses:
        print(f"\nRetrieval misses ({len(misses)}):")
        for r in misses:
            terms_str = ", ".join(f'"{t}"' for t in r["search_terms"])
            print(f"  '{r['gold_span'][:30]:30s}'  "
                  f"concept={r['gold_concept_name'][:50]}  "
                  f"terms=[{terms_str}]")

    # --- Save results ---
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    output = {
        "timestamp": timestamp,
        "note_id": note_id,
        "backend": args.backend,
        "model": model_name,
        "reasoning_effort": reasoning,
        "retrieval_top_k": top_k,
        "n_annotations": n,
        "n_found": n_found,
        "found_pct": round(100 * n_found / n, 1),
        "rank_at_1": rank_at_1,
        "rank_at_5": rank_at_5,
        "rank_at_10": rank_at_10,
        "avg_gold_rank": round(avg_rank, 1) if gold_ranks else None,
        "total_terms": all_term_total,
        "terms_finding_gold": all_term_found,
        "total_time_s": round(total_elapsed, 1),
        "per_annotation": per_ann,
    }

    results_path = Path(__file__).parent / f"oracle_search_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle search term generation + retrieval analysis",
    )
    parser.add_argument(
        "--backend", choices=["sonnet", "vllm"], default="vllm",
        help="LLM backend (default: vllm)",
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
        help="Max concurrent requests (default: 0 = unlimited)",
    )
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default="low",
        help="Reasoning effort for gpt-oss-20b (default: low)",
    )
    parser.add_argument(
        "--note-index", type=int, default=0,
        help="Index of the note to analyze (default: 0 = first note)",
    )
    parser.add_argument(
        "--max-terms", type=int, default=None,
        help="Max search terms per annotation (default: all). Use 1 for first-term-only.",
    )
    parser.add_argument(
        "--span-only", action="store_true",
        help="Skip LLM — search using the literal annotation span text only.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
