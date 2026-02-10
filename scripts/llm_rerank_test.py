#!/usr/bin/env python3
"""
LLM Reranking Test: Can an LLM pick the correct SNOMED concept?

Tests whether a local LLM can improve R@1 by selecting the right concept
from the candidate list, especially for ambiguous/hard cases the pipeline
and dictionary can't auto-resolve.

Supports two backends:
  - llamacpp: llama-cpp-python with GGUF models (sequential inference)
  - vllm:     vLLM server with OpenAI-compatible API (batched inference)

Usage:
    # llama-cpp backend (default)
    python scripts/llm_rerank_test.py [--max-queries N]

    # vLLM backend (model is MXFP4 quantised — no --quantization flag needed)
    # Start server:
    #   vllm serve openai/gpt-oss-20b --enable-prefix-caching --enable-chunked-prefill
    # Blackwell GPUs:
    #   VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 vllm serve openai/gpt-oss-20b \
    #       --enable-prefix-caching --enable-chunked-prefill
    python scripts/llm_rerank_test.py --backend vllm [--reasoning-effort low]
"""

import argparse
import concurrent.futures
import gc
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snomed_ct_entity_linking.recall_analysis.config import Config
from snomed_ct_entity_linking.recall_analysis.snomed_loader import load_snomed, load_validation_data
from snomed_ct_entity_linking.recall_analysis.index_builder import (
    load_indexes, indexes_exist, load_raw_embeddings,
)
from snomed_ct_entity_linking.recall_analysis.retrieval import (
    encode_queries, hybrid_retrieve, expand_with_hierarchy,
    dense_search, sparse_search, reciprocal_rank_fusion,
)
from snomed_ct_entity_linking.recall_analysis.encoder import SapBERTEncoder
from snomed_ct_entity_linking.recall_analysis.reranker import SapBERTReranker


# ---------------------------------------------------------------------------
# Prompt templates — split into system (shared/cacheable) + user (per-query)
# so that vLLM's automatic prefix caching (APC) can reuse the KV cache for
# the system message across all requests in a batch.
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """\
Select the SNOMED CT concept that best matches a medical term from a discharge note.

Example:
Term: chest pain
Context: ...presented to ED with acute onset chest pain radiating to left arm...
Candidates:
0: Chest pain
1: Acute pain
2: Pain in left arm
3: Angina pectoris
Answer: 0

Now classify:"""

CLASSIFY_USER = """\
Term: {mention}
Context: ...{context}...
Candidates:
{candidates}
Answer:"""

CHECKER_SYSTEM = """\
You are reviewing another model's answer to a medical concept matching task.

Carefully review whether the chosen candidate is the best match for the medical \
term given the clinical context. Consider whether a more general or more specific \
concept would be more appropriate.

If the answer is correct, respond with the same number. If a different candidate \
is a better match, respond with that number instead."""

CHECKER_USER = """\
Original task:
{original_prompt}

The model responded:
{raw_response}

It chose candidate {choice}: {choice_concept}

Answer:"""

SEARCH_SYSTEM = """\
Select the SNOMED CT concept that best matches a medical term from a discharge note.

If a candidate matches, respond with its number. If none of the candidates are a \
good match, you may request a new search by responding with SEARCH: <better search term>.

Example 1 — good match found:
Term: chest pain
Context: ...presented to ED with acute onset chest pain radiating to left arm...
Search term used: chest pain
Candidates:
0: Chest pain
1: Acute pain
2: Pain in left arm
Answer: 0

Example 2 — no good match, request new search:
Term: A&O x 3
Context: ...alert and oriented, A&O x 3, following commands...
Search term used: A&O x 3
Candidates:
0: Monosomy X
1: XX males
2: Para 3
Answer: SEARCH: alert and oriented to person place and time

Now classify:"""

SEARCH_USER = """\
Term: {mention}
Context: ...{context}...
Search term used: {search_term}
Candidates:
{candidates}
Answer:"""

SEARCH_FOLLOWUP_SYSTEM = """\
Select the SNOMED CT concept that best matches a medical term from a discharge note.

The original search did not yield a good match, so a new search was performed \
with a better term. Select the best matching candidate from the new results."""

SEARCH_FOLLOWUP_USER = """\
Term: {mention}
Context: ...{context}...
Original search term: {original_search_term}
New search term: {new_search_term}
Candidates:
{candidates}
Answer:"""

def build_train_dictionary(cfg: Config) -> dict:
    """Build dictionary from training split annotations."""
    ann = pd.read_csv(cfg.train_annotations, dtype={"concept_id": int})
    train = ann[ann["annotation_type"] == "train"]
    span_groups = defaultdict(list)
    for _, row in train.iterrows():
        span_groups[row["span"].lower()].append(row["concept_id"])

    dictionary = {}
    for span_lower, concept_ids in span_groups.items():
        counter = Counter(concept_ids)
        most_common_id, most_common_count = counter.most_common(1)[0]
        total = len(concept_ids)
        dictionary[span_lower] = {
            "concept_id": most_common_id,
            "count": total,
            "confidence": most_common_count / total,
        }
    return dictionary


def classify_tiers(
    gold_sctids, mention_spans, reranked_results, dictionary, indexed_sctids,
    dict_conf=1.0, dict_count=3, pipeline_gap=0.10,
):
    """Classify each query into tier 1 (dict), tier 2 (pipeline), or tier 3 (LLM)."""
    tiers = []
    for i in range(len(gold_sctids)):
        span_lower = mention_spans[i].lower()
        candidates = reranked_results[i]

        # Tier 1: dictionary
        entry = dictionary.get(span_lower)
        if (entry and entry["confidence"] >= dict_conf
                and entry["count"] >= dict_count
                and entry["concept_id"] in indexed_sctids):
            tiers.append(1)
            continue

        # Tier 2: pipeline high confidence
        if len(candidates) >= 2:
            gap = candidates[0][1] - candidates[1][1]
        elif len(candidates) == 1:
            gap = candidates[0][1]
        else:
            gap = 0.0

        if gap >= pipeline_gap:
            tiers.append(2)
            continue

        tiers.append(3)
    return tiers


def build_prompt(mention, context, candidates, sctid_to_fsn, top_k=10):
    """Build the user-message part of the classification prompt."""
    return CLASSIFY_USER.format(
        mention=mention,
        context=context[:500],
        candidates=_format_candidate_lines(candidates, sctid_to_fsn, top_k),
    )


def _format_candidate_lines(candidates, sctid_to_fsn, top_k=10):
    """Format candidate list as numbered lines."""
    lines = []
    for idx, (sctid, score) in enumerate(candidates[:top_k]):
        name = sctid_to_fsn.get(sctid, "Unknown concept")
        lines.append(f"{idx}: {name}")
    return "\n".join(lines)


def build_search_prompt(mention, context, candidates, sctid_to_fsn, top_k=10):
    """Build the user-message part of the search-enabled prompt."""
    return SEARCH_USER.format(
        mention=mention,
        context=context[:500],
        search_term=mention,
        candidates=_format_candidate_lines(candidates, sctid_to_fsn, top_k),
    )


def build_search_followup_prompt(mention, context, original_search_term,
                                  new_search_term, candidates, sctid_to_fsn,
                                  top_k=10):
    """Build the user-message part of the follow-up prompt."""
    return SEARCH_FOLLOWUP_USER.format(
        mention=mention,
        context=context[:500],
        original_search_term=original_search_term,
        new_search_term=new_search_term,
        candidates=_format_candidate_lines(candidates, sctid_to_fsn, top_k),
    )


def parse_search_response(text, max_idx):
    """Parse a search-enabled LLM response.

    Returns:
        ("choice", int)   — model picked a candidate number
        ("search", str)   — model requested a new search term
        ("abstain", None) — unparseable
    """
    text = text.strip()

    # Check for SEARCH: directive
    search_match = re.search(r"SEARCH:\s*(.+)", text, re.IGNORECASE)
    if search_match:
        new_term = search_match.group(1).strip().strip('"\'')
        if new_term:
            return ("search", new_term)

    # Otherwise try to parse as a numeric choice
    choice = parse_llm_choice(text, max_idx)
    if choice >= 0:
        return ("choice", choice)

    return ("abstain", None)


def rerun_retrieval_batch(search_terms, encoder, faiss_index, faiss_sctids,
                          bm25_index, bm25_sctids, reranker, parent_map,
                          indexed_sctids, cfg):
    """Re-run the full retrieval pipeline for a batch of new search terms.

    Returns a list of reranked result lists, one per search term.
    """
    print(f"  Re-retrieving for {len(search_terms)} new search terms ...")
    t0 = time.time()

    # 1. Encode
    query_embeddings = encoder.encode(search_terms, batch_size=cfg.query_batch_size)

    # 2. Dense search
    dense_results = dense_search(query_embeddings, faiss_index, faiss_sctids,
                                 cfg.dense_top_k)

    # 3. Sparse search
    sparse_results = sparse_search(search_terms, bm25_index, bm25_sctids,
                                   cfg.sparse_top_k, cfg.bm25_threads)

    # 4. RRF fusion
    fused = []
    for d, s in zip(dense_results, sparse_results):
        fused.append(reciprocal_rank_fusion(d, s, k=cfg.rrf_k,
                                            top_n=cfg.fusion_top_k))

    # 5. Hierarchy expansion
    fused = expand_with_hierarchy(fused, parent_map, indexed_sctids)

    # 6. SapBERT reranking
    candidate_sctids = [[s for s, _ in f] for f in fused]
    reranked = reranker.rerank(query_embeddings, candidate_sctids)

    elapsed = time.time() - t0
    print(f"  Re-retrieval done in {elapsed:.1f}s")
    return reranked


def run_llm_llamacpp(model_path, prompts, max_tokens=2048):
    """Run LLM inference using llama-cpp-python with GGUF model on GPU."""
    from llama_cpp import Llama

    print(f"  Loading GGUF model: {model_path} ...")
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=-1,  # all layers on GPU
        n_ctx=4096,
        n_batch=2048,     # larger batch for faster prompt eval
        flash_attn=True,
        verbose=False,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    print(f"  Running inference on {len(prompts)} prompts ...")
    t0 = time.time()
    outputs = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        text = result["choices"][0]["message"]["content"].strip()
        outputs.append(text)

        if (i + 1) % 10 == 0 or i + 1 == len(prompts):
            elapsed = time.time() - t0
            print(f"    {i + 1}/{len(prompts)} done ({elapsed:.1f}s, "
                  f"{(i + 1) / elapsed:.1f} queries/s)")

    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s ({len(prompts) / elapsed:.1f} queries/s)")

    del llm
    gc.collect()

    return outputs


def _make_vllm_client(base_url, model):
    """Create an OpenAI client pointed at a vLLM server and verify connectivity.

    The model (openai/gpt-oss-20b) ships as MXFP4 quantised weights — do NOT
    pass --quantization to the server.  Start the server with:

        vllm serve openai/gpt-oss-20b \\
            --enable-prefix-caching --enable-chunked-prefill

    On Blackwell GPUs (RTX 5090, B200, …) enable the optimised MoE kernel:

        VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 \\
            vllm serve openai/gpt-oss-20b \\
            --enable-prefix-caching --enable-chunked-prefill
    """
    from openai import OpenAI

    client = OpenAI(base_url=f"{base_url}/v1", api_key="unused")

    print(f"  Connecting to vLLM server at {base_url} (model: {model}) ...")
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        print(f"  Available models: {available}")
        if model not in available:
            print(f"  WARNING: requested model '{model}' not in available models")
    except Exception as e:
        print(f"  ERROR: Could not connect to vLLM server at {base_url}: {e}")
        sys.exit(1)

    return client


def _call_vllm_batch(client, model, prompts, reasoning_effort="low",
                     max_tokens=2048, max_concurrent=256, label="",
                     system_prompt=None, frequency_penalty=0.3):
    """Run batched inference against a vLLM server using async I/O.

    Submits ALL requests concurrently (up to max_concurrent in-flight) so
    vLLM's continuous batching keeps the GPU saturated. Uses AsyncOpenAI
    for efficient I/O multiplexing — no thread-per-request overhead.

    When system_prompt is provided, it is sent as a separate system message
    before each user message. With --enable-prefix-caching on the vLLM
    server, the KV cache for this shared prefix is computed once and reused
    across all requests in the batch.

    frequency_penalty penalises repeated tokens to prevent degenerate
    repetition loops (e.g. "Afebrile" repeated hundreds of times).
    """
    import asyncio
    from openai import AsyncOpenAI

    extra_body = {}
    if reasoning_effort:
        extra_body["reasoning_effort"] = reasoning_effort

    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}Firing {len(prompts)} prompts "
          f"(max_concurrent={max_concurrent}, reasoning_effort={reasoning_effort})")

    t0 = time.time()

    async def _run():
        async_client = AsyncOpenAI(
            base_url=str(client.base_url), api_key="unused",
            max_retries=2,
            timeout=300.0,
        )
        sem = asyncio.Semaphore(max_concurrent)
        completed = [0]
        in_flight = [0]

        async def _call_single(idx, prompt):
            async with sem:
                in_flight[0] += 1
                try:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})

                    resp = await async_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=0.0,
                        frequency_penalty=frequency_penalty,
                        extra_body=extra_body if extra_body else None,
                    )
                    choice = resp.choices[0]
                    content = choice.message.content or ""
                    reasoning = getattr(choice.message, "reasoning_content", None) or ""
                    text = content.strip() if content.strip() else reasoning.strip()
                    return idx, text
                except Exception as e:
                    print(f"  {prefix}ERROR on prompt {idx}: {e}")
                    return idx, ""
                finally:
                    in_flight[0] -= 1
                    completed[0] += 1
                    done = completed[0]
                    if done % 50 == 0 or done == len(prompts):
                        elapsed = time.time() - t0
                        qps = done / elapsed if elapsed > 0 else 0
                        print(f"  {prefix}{done}/{len(prompts)} done, "
                              f"{in_flight[0]} in-flight "
                              f"({elapsed:.1f}s, {qps:.1f} q/s)")

        tasks = [_call_single(i, p) for i, p in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        await async_client.close()

        outputs = [None] * len(prompts)
        for idx, text in results:
            outputs[idx] = text
        return outputs

    outputs = asyncio.run(_run())

    elapsed = time.time() - t0
    print(f"  {prefix}Done in {elapsed:.1f}s ({len(prompts) / elapsed:.1f} queries/s)")
    return outputs


def run_llm_vllm(prompts, base_url="http://localhost:8000", model="openai/gpt-oss-20b",
                 max_tokens=2048, max_concurrent=256, reasoning_effort="low",
                 system_prompt=None):
    """Run LLM inference via a vLLM server (convenience wrapper)."""
    client = _make_vllm_client(base_url, model)
    return _call_vllm_batch(client, model, prompts, reasoning_effort,
                            max_tokens, max_concurrent,
                            system_prompt=system_prompt)


def parse_llm_choice(text: str, max_idx: int) -> int:
    """Parse the LLM's numeric choice from its output text.

    MedGemma outputs chain-of-thought then the answer. We look for:
    1. A number at the very start (if thinking was stripped)
    2. The last number in the output (after thinking)
    """
    text = text.strip()

    # Try to find a number at the start
    match = re.match(r"^(-?\d+)", text)
    if match:
        num = int(match.group(1))
        if -1 <= num <= max_idx:
            return num

    # MedGemma thinking mode: look for the last standalone number
    # It typically ends with "Answer: X" or just "X" after reasoning
    all_nums = re.findall(r"(?:^|\s)(-?\d+)\s*$", text, re.MULTILINE)
    if all_nums:
        num = int(all_nums[-1])
        if -1 <= num <= max_idx:
            return num

    # Try "Answer: X" or "choose X" or "candidate X" patterns
    answer_match = re.search(r"(?:answer|choose|select|candidate)[:\s]+(\d+)", text, re.IGNORECASE)
    if answer_match:
        num = int(answer_match.group(1))
        if 0 <= num <= max_idx:
            return num

    # MedGemma often ends with "N: ConceptName" (number stuck to period/text)
    tail_match = re.search(r"(\d+):\s+\S+[^.]*$", text[-100:])
    if tail_match:
        num = int(tail_match.group(1))
        if 0 <= num <= max_idx:
            return num

    return -1  # unparseable → no selection


def main():
    parser = argparse.ArgumentParser(description="LLM reranking test")
    parser.add_argument(
        "--backend", type=str, choices=["llamacpp", "vllm"], default="llamacpp",
        help="Inference backend: 'llamacpp' (GGUF) or 'vllm' (OpenAI API) (default: llamacpp)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model path (llamacpp) or model name (vllm). "
             "Defaults: llamacpp='models/medgemma-1.5-4b-it-Q8_0.gguf', vllm='openai/gpt-oss-20b'",
    )
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000",
                        help="vLLM server base URL (default: http://localhost:8000)")
    parser.add_argument("--batch-size", type=int, default=256, dest="batch_size",
                        help="Max concurrent in-flight requests to vLLM (default: 256)")
    parser.add_argument("--reasoning-effort", type=str, default="low",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for gpt-oss models (default: low)")
    parser.add_argument("--strategy", type=str, default="single",
                        choices=["single", "adaptive", "checker",
                                 "adaptive-checker", "search",
                                 "adaptive-search"],
                        help="Inference strategy: 'single' (one pass), "
                             "'adaptive' (low + re-prompt non-rank-0), "
                             "'checker' (low + verify all with checker prompt), "
                             "'adaptive-checker' (low + checker prompt only non-rank-0), "
                             "'search' (low with search option + re-retrieve + pick), "
                             "'adaptive-search' (fast low → search prompt for non-rank-0 "
                             "→ re-retrieve + pick) "
                             "(default: single)")
    parser.add_argument("--checker-effort", type=str, default="medium",
                        choices=["medium", "high"],
                        help="Reasoning effort for the escalation/checker pass "
                             "(default: medium)")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Max tier-3 queries to test (0 = all, default: 0)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of candidates to show LLM (default: 10)")
    parser.add_argument("--note-id", type=str, default=None,
                        help="Filter to tier-3 queries from a single note_id")
    parser.add_argument("--max-notes", type=int, default=0,
                        help="Limit to tier-3 queries from the first N distinct notes (0 = all)")
    parser.add_argument("--save-predictions", type=str, default=None,
                        help="Save per-query predictions to a TSV file")
    args = parser.parse_args()

    # Set default model based on backend if not explicitly provided
    if args.model is None:
        if args.backend == "vllm":
            args.model = "openai/gpt-oss-20b"
        else:
            args.model = "models/medgemma-1.5-4b-it-Q8_0.gguf"

    cfg = Config()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # =========================================================================
    # Step 1: Run pipeline (load cached indexes, retrieve, rerank)
    # =========================================================================
    print("\n" + "=" * 65)
    print("  STEP 1: Loading SNOMED + Indexes + Validation Data")
    print("=" * 65)
    descriptions_df, sctid_to_fsn, sctid_to_tag, parent_map = load_snomed(cfg)

    if not indexes_exist(cfg):
        print("ERROR: No cached indexes. Run recall analysis pipeline first.")
        sys.exit(1)
    faiss_index, faiss_sctids, bm25_index, bm25_sctids, bm25_terms = load_indexes(cfg)

    val_df = load_validation_data(cfg)
    val_df["semantic_tag"] = val_df["concept_id"].map(sctid_to_tag).fillna("unknown")
    indexed_sctids = set(faiss_sctids)
    val_df["in_index"] = val_df["concept_id"].isin(indexed_sctids)
    val_df = val_df[val_df["in_index"]].reset_index(drop=True)
    print(f"  {len(val_df):,} test annotations with gold in index")

    gold_sctids = val_df["concept_id"].tolist()
    mention_spans = val_df["span"].tolist()
    contexts = val_df["context"].tolist()

    print("\n" + "=" * 65)
    print("  STEP 2: Hybrid Retrieval + SapBERT Reranking")
    print("=" * 65)
    query_embeddings = encode_queries(mention_spans, cfg)
    fused_results, dense_all, sparse_all = hybrid_retrieve(
        query_embeddings, mention_spans,
        faiss_index, faiss_sctids, bm25_index, bm25_sctids, cfg,
    )
    fused_results = expand_with_hierarchy(fused_results, parent_map, indexed_sctids)

    raw_data = load_raw_embeddings(cfg)
    if raw_data is None:
        print("ERROR: No raw embeddings found.")
        sys.exit(1)
    raw_embeddings, raw_sctids = raw_data
    reranker = SapBERTReranker(raw_embeddings, raw_sctids)
    candidate_sctids_per_query = [[s for s, _ in fused] for fused in fused_results]
    reranked_results = reranker.rerank(query_embeddings, candidate_sctids_per_query)
    if args.strategy in ("search", "adaptive-search"):
        # Keep reranker, embeddings, and encoder alive for re-retrieval
        search_encoder = SapBERTEncoder(cfg.embedding_model)
        del query_embeddings
        torch.cuda.empty_cache()
    else:
        reranker.close()
        del query_embeddings, raw_embeddings
        torch.cuda.empty_cache()

    # Baseline R@1
    baseline_correct = sum(
        1 for g, r in zip(gold_sctids, reranked_results) if r and r[0][0] == g
    )
    print(f"\n  Pipeline baseline R@1: {baseline_correct}/{len(gold_sctids)} "
          f"({baseline_correct / len(gold_sctids) * 100:.1f}%)")

    # =========================================================================
    # Step 3: Tier classification
    # =========================================================================
    print("\n" + "=" * 65)
    print("  STEP 3: Tier Classification")
    print("=" * 65)
    dictionary = build_train_dictionary(cfg)
    tiers = classify_tiers(
        gold_sctids, mention_spans, reranked_results, dictionary, indexed_sctids,
    )

    tier_counts = Counter(tiers)
    print(f"  Tier 1 (dictionary): {tier_counts[1]:,}")
    print(f"  Tier 2 (pipeline):   {tier_counts[2]:,}")
    print(f"  Tier 3 (LLM):        {tier_counts[3]:,}")

    # Collect tier-3 indices
    tier3_indices = [i for i, t in enumerate(tiers) if t == 3]

    # Filter to a single note if requested
    note_ids = val_df["note_id"].tolist()
    if args.note_id is not None:
        tier3_indices = [i for i in tier3_indices if note_ids[i] == args.note_id]
        print(f"  Filtered to note_id={args.note_id}: {len(tier3_indices)} tier-3 queries")

    # Filter to first N distinct notes if requested
    if args.max_notes:
        seen_notes = []
        seen_set = set()
        for i in tier3_indices:
            nid = note_ids[i]
            if nid not in seen_set:
                seen_set.add(nid)
                seen_notes.append(nid)
        selected_notes = set(seen_notes[:args.max_notes])
        tier3_indices = [i for i in tier3_indices if note_ids[i] in selected_notes]
        print(f"  Filtered to {len(selected_notes)} notes: {len(tier3_indices)} tier-3 queries")

    # Subsample for testing
    if args.max_queries and len(tier3_indices) > args.max_queries:
        np.random.seed(42)
        tier3_indices = sorted(np.random.choice(
            tier3_indices, size=args.max_queries, replace=False,
        ))
    print(f"  Testing LLM on {len(tier3_indices):,} tier-3 queries")

    # =========================================================================
    # Step 4: LLM Inference
    # =========================================================================
    print("\n" + "=" * 65)
    print(f"  STEP 4: LLM Inference via {args.backend} "
          f"(strategy={args.strategy}, model={args.model})")
    print("=" * 65)

    # Build prompts
    prompts = []
    for idx in tier3_indices:
        prompt = build_prompt(
            mention_spans[idx], contexts[idx],
            reranked_results[idx], sctid_to_fsn, top_k=args.top_k,
        )
        prompts.append(prompt)

    # Track which queries were escalated/checked (for reporting)
    escalated_flags = [False] * len(prompts)

    if args.backend != "vllm" or args.strategy == "single":
        # Single-pass: original behavior
        if args.backend == "vllm":
            raw_outputs = run_llm_vllm(
                prompts, base_url=args.vllm_url, model=args.model,
                max_concurrent=args.batch_size,
                reasoning_effort=args.reasoning_effort,
                system_prompt=CLASSIFY_SYSTEM,
            )
        else:
            raw_outputs = run_llm_llamacpp(args.model, prompts)

    elif args.strategy == "adaptive":
        # Stage 1: fast pass with low reasoning
        client = _make_vllm_client(args.vllm_url, args.model)
        raw_outputs = _call_vllm_batch(
            client, args.model, prompts, reasoning_effort="low",
            max_tokens=2048, max_concurrent=args.batch_size, label="low",
            system_prompt=CLASSIFY_SYSTEM,
        )

        # Identify non-rank-0 answers to escalate
        escalate_indices = []
        for j, raw in enumerate(raw_outputs):
            candidates = reranked_results[tier3_indices[j]][:args.top_k]
            choice = parse_llm_choice(raw, len(candidates) - 1)
            if choice != 0:
                escalate_indices.append(j)

        print(f"\n  Adaptive: {len(escalate_indices)}/{len(prompts)} picked "
              f"non-rank-0 → escalating with {args.checker_effort} reasoning")

        if escalate_indices:
            esc_prompts = [prompts[j] for j in escalate_indices]
            esc_outputs = _call_vllm_batch(
                client, args.model, esc_prompts,
                reasoning_effort=args.checker_effort,
                max_tokens=2048, max_concurrent=args.batch_size, label="escalate",
                system_prompt=CLASSIFY_SYSTEM,
            )
            for k, j in enumerate(escalate_indices):
                raw_outputs[j] = esc_outputs[k]
                escalated_flags[j] = True

    elif args.strategy == "checker":
        # Stage 1: fast pass with low reasoning
        client = _make_vllm_client(args.vllm_url, args.model)
        low_outputs = _call_vllm_batch(
            client, args.model, prompts, reasoning_effort="low",
            max_tokens=2048, max_concurrent=args.batch_size, label="low",
            system_prompt=CLASSIFY_SYSTEM,
        )

        # Stage 2: build checker prompts for all queries
        checker_prompts = []
        for j, raw in enumerate(low_outputs):
            candidates = reranked_results[tier3_indices[j]][:args.top_k]
            choice = parse_llm_choice(raw, len(candidates) - 1)
            if 0 <= choice < len(candidates):
                choice_sctid = candidates[choice][0]
                choice_concept = sctid_to_fsn.get(choice_sctid, "Unknown")
            else:
                choice_concept = "None (abstained/unparseable)"
            checker_prompts.append(CHECKER_USER.format(
                original_prompt=prompts[j],
                raw_response=raw,
                choice=choice,
                choice_concept=choice_concept,
            ))

        print(f"\n  Checker: verifying all {len(prompts)} answers "
              f"with {args.checker_effort} reasoning")

        raw_outputs = _call_vllm_batch(
            client, args.model, checker_prompts,
            reasoning_effort=args.checker_effort,
            max_tokens=2048, max_concurrent=args.batch_size, label="checker",
            system_prompt=CHECKER_SYSTEM,
        )
        escalated_flags = [True] * len(prompts)

    elif args.strategy == "adaptive-checker":
        # Stage 1: fast pass with low reasoning
        client = _make_vllm_client(args.vllm_url, args.model)
        raw_outputs = _call_vllm_batch(
            client, args.model, prompts, reasoning_effort="low",
            max_tokens=2048, max_concurrent=args.batch_size, label="low",
            system_prompt=CLASSIFY_SYSTEM,
        )

        # Identify non-rank-0 answers to escalate via checker prompt
        escalate_indices = []
        for j, raw in enumerate(raw_outputs):
            candidates = reranked_results[tier3_indices[j]][:args.top_k]
            choice = parse_llm_choice(raw, len(candidates) - 1)
            if choice != 0:
                escalate_indices.append(j)

        print(f"\n  Adaptive-checker: {len(escalate_indices)}/{len(prompts)} picked "
              f"non-rank-0 → checking with {args.checker_effort} reasoning")

        if escalate_indices:
            checker_prompts = []
            for j in escalate_indices:
                raw = raw_outputs[j]
                candidates = reranked_results[tier3_indices[j]][:args.top_k]
                choice = parse_llm_choice(raw, len(candidates) - 1)
                if 0 <= choice < len(candidates):
                    choice_sctid = candidates[choice][0]
                    choice_concept = sctid_to_fsn.get(choice_sctid, "Unknown")
                else:
                    choice_concept = "None (abstained/unparseable)"
                checker_prompts.append(CHECKER_USER.format(
                    original_prompt=prompts[j],
                    raw_response=raw,
                    choice=choice,
                    choice_concept=choice_concept,
                ))

            esc_outputs = _call_vllm_batch(
                client, args.model, checker_prompts,
                reasoning_effort=args.checker_effort,
                max_tokens=2048, max_concurrent=args.batch_size, label="checker",
                system_prompt=CHECKER_SYSTEM,
            )
            for k, j in enumerate(escalate_indices):
                raw_outputs[j] = esc_outputs[k]
                escalated_flags[j] = True

    elif args.strategy == "search":
        # Stage 1: search-enabled prompt with low reasoning
        client = _make_vllm_client(args.vllm_url, args.model)

        # Build search-enabled prompts instead of standard ones
        search_prompts = []
        for idx in tier3_indices:
            search_prompts.append(build_search_prompt(
                mention_spans[idx], contexts[idx],
                reranked_results[idx], sctid_to_fsn, top_k=args.top_k,
            ))

        low_outputs = _call_vllm_batch(
            client, args.model, search_prompts, reasoning_effort="low",
            max_tokens=2048, max_concurrent=args.batch_size, label="low+search",
            system_prompt=SEARCH_SYSTEM,
        )

        # Parse responses: choice or search term
        raw_outputs = list(low_outputs)  # will be overwritten for re-searched queries
        research_indices = []
        new_search_terms_map = {}  # j -> new_term

        for j, raw in enumerate(low_outputs):
            candidates = reranked_results[tier3_indices[j]][:args.top_k]
            action, value = parse_search_response(raw, len(candidates) - 1)
            if action == "search":
                research_indices.append(j)
                new_search_terms_map[j] = value

        print(f"\n  Search: {len(research_indices)}/{len(search_prompts)} "
              f"requested new search terms")

        if research_indices:
            # Stage 2a: re-retrieve with new search terms
            new_terms = [new_search_terms_map[j] for j in research_indices]
            for j, term in zip(research_indices, new_terms):
                print(f"    [{j}] \"{mention_spans[tier3_indices[j]]}\" → "
                      f"SEARCH: \"{term}\"")

            new_results = rerun_retrieval_batch(
                new_terms, search_encoder,
                faiss_index, faiss_sctids, bm25_index, bm25_sctids,
                reranker, parent_map, indexed_sctids, cfg,
            )

            # Stage 2b: build follow-up prompts with new candidates
            followup_prompts = []
            for k, j in enumerate(research_indices):
                idx = tier3_indices[j]
                followup_prompts.append(build_search_followup_prompt(
                    mention=mention_spans[idx],
                    context=contexts[idx],
                    original_search_term=mention_spans[idx],
                    new_search_term=new_search_terms_map[j],
                    candidates=new_results[k],
                    sctid_to_fsn=sctid_to_fsn,
                    top_k=args.top_k,
                ))

            followup_outputs = _call_vllm_batch(
                client, args.model, followup_prompts,
                reasoning_effort=args.checker_effort,
                max_tokens=2048, max_concurrent=args.batch_size, label="followup",
                system_prompt=SEARCH_FOLLOWUP_SYSTEM,
            )

            # Replace outputs and update reranked_results for re-searched queries
            for k, j in enumerate(research_indices):
                raw_outputs[j] = followup_outputs[k]
                # Update candidate list so eval uses the new candidates
                reranked_results[tier3_indices[j]] = new_results[k]
                escalated_flags[j] = True

        # Cleanup deferred resources
        search_encoder.close()
        reranker.close()
        del raw_embeddings
        torch.cuda.empty_cache()

    elif args.strategy == "adaptive-search":
        # Two-pass approach:
        #   Pass 1: search-enabled prompt for ALL queries (one big GPU burst)
        #   Pass 2: re-retrieve + follow-up for queries that requested SEARCH

        client = _make_vllm_client(args.vllm_url, args.model)

        # Build search-enabled prompts for all queries
        search_prompts_all = []
        for j in range(len(prompts)):
            idx = tier3_indices[j]
            search_prompts_all.append(build_search_prompt(
                mention_spans[idx], contexts[idx],
                reranked_results[idx], sctid_to_fsn, top_k=args.top_k,
            ))

        # Pass 1: fire all prompts at once — vLLM continuous batching keeps GPU fed
        # Use low reasoning + small max_tokens: model just needs to output a
        # number or "SEARCH: <term>", not a full chain-of-thought.
        raw_outputs = _call_vllm_batch(
            client, args.model, search_prompts_all,
            reasoning_effort="low",
            max_tokens=384, max_concurrent=args.batch_size, label="pass1",
            system_prompt=SEARCH_SYSTEM,
        )

        # Parse all results: identify SEARCH requests
        research_indices = []
        new_search_terms_map = {}
        n_picked = 0
        for j, raw in enumerate(raw_outputs):
            candidates = reranked_results[tier3_indices[j]][:args.top_k]
            action, value = parse_search_response(raw, len(candidates) - 1)
            if action == "search":
                research_indices.append(j)
                new_search_terms_map[j] = value
            elif action == "choice":
                n_picked += 1
                escalated_flags[j] = True

        n_abstain = len(raw_outputs) - n_picked - len(research_indices)
        print(f"\n  Pass 1: {n_picked} picked, "
              f"{len(research_indices)} SEARCH, "
              f"{n_abstain} abstain "
              f"(of {len(raw_outputs)} total)")

        # Pass 2a: re-retrieve and follow-up for SEARCH requests
        if research_indices:
            new_terms = [new_search_terms_map[j] for j in research_indices]
            for j, term in zip(research_indices[:20], new_terms[:20]):
                print(f"    [{j}] \"{mention_spans[tier3_indices[j]]}\" → "
                      f"SEARCH: \"{term}\"")
            if len(research_indices) > 20:
                print(f"    ... and {len(research_indices) - 20} more")

            new_results = rerun_retrieval_batch(
                new_terms, search_encoder,
                faiss_index, faiss_sctids, bm25_index, bm25_sctids,
                reranker, parent_map, indexed_sctids, cfg,
            )

            followup_prompts = []
            for k, j in enumerate(research_indices):
                idx = tier3_indices[j]
                followup_prompts.append(build_search_followup_prompt(
                    mention=mention_spans[idx],
                    context=contexts[idx],
                    original_search_term=mention_spans[idx],
                    new_search_term=new_search_terms_map[j],
                    candidates=new_results[k],
                    sctid_to_fsn=sctid_to_fsn,
                    top_k=args.top_k,
                ))

            followup_outputs = _call_vllm_batch(
                client, args.model, followup_prompts,
                reasoning_effort=args.checker_effort,
                max_tokens=1024, max_concurrent=args.batch_size,
                label="pass2-search",
                system_prompt=SEARCH_FOLLOWUP_SYSTEM,
            )

            for k, j in enumerate(research_indices):
                raw_outputs[j] = followup_outputs[k]
                reranked_results[tier3_indices[j]] = new_results[k]
                escalated_flags[j] = True

        # Cleanup deferred resources
        search_encoder.close()
        reranker.close()
        del raw_embeddings
        torch.cuda.empty_cache()

    # Track new search terms for save-predictions
    search_terms_used = {}
    if args.strategy in ("search", "adaptive-search"):
        search_terms_used = new_search_terms_map if research_indices else {}

    n_escalated = sum(escalated_flags)
    if n_escalated > 0:
        print(f"\n  Queries escalated/checked: {n_escalated}/{len(prompts)}")

    # =========================================================================
    # Step 5: Evaluate
    # =========================================================================
    print("\n" + "=" * 65)
    print("  STEP 5: Evaluation")
    print("=" * 65)

    llm_correct = 0
    llm_wrong = 0
    llm_abstain = 0  # chose -1
    llm_unparseable = 0
    pipeline_would_correct = 0

    # Failure mode tracking: gold in candidates vs not
    wrong_gold_in_topk = 0       # LLM picked wrong candidate (gold was available)
    wrong_gold_not_in_topk = 0   # retrieval failure (gold wasn't in candidates)
    abstain_gold_in_topk = 0     # abstained but gold was there
    abstain_gold_not_in_topk = 0 # abstained and gold wasn't there either
    # Deeper retrieval analysis: gold not in top-k but in full reranked list?
    miss_but_in_full = 0         # not in top-k shown, but in full reranked list
    miss_not_in_full = 0         # not in full reranked list at all

    examples_correct = []
    examples_wrong = []

    for j, idx in enumerate(tier3_indices):
        gold = gold_sctids[idx]
        candidates = reranked_results[idx][:args.top_k]
        raw_text = raw_outputs[j]
        choice = parse_llm_choice(raw_text, len(candidates) - 1)

        # Check if gold is even in the top-k candidates
        candidate_ids = [s for s, _ in candidates]
        gold_in_topk = gold in candidate_ids
        # Check full reranked list for deeper retrieval analysis
        all_candidate_ids = [s for s, _ in reranked_results[idx]]
        gold_in_full = gold in all_candidate_ids

        # Pipeline's pick (always rank 0)
        pipeline_pick = candidate_ids[0] if candidate_ids else None
        if pipeline_pick == gold:
            pipeline_would_correct += 1

        if choice == -1:
            llm_abstain += 1
            pred = None
            if gold_in_topk:
                abstain_gold_in_topk += 1
            else:
                abstain_gold_not_in_topk += 1
                if gold_in_full:
                    miss_but_in_full += 1
                else:
                    miss_not_in_full += 1
        elif 0 <= choice < len(candidates):
            pred = candidate_ids[choice]
        else:
            llm_unparseable += 1
            pred = None
            if gold_in_topk:
                abstain_gold_in_topk += 1
            else:
                abstain_gold_not_in_topk += 1
                if gold_in_full:
                    miss_but_in_full += 1
                else:
                    miss_not_in_full += 1

        if pred == gold:
            llm_correct += 1
            if len(examples_correct) < 5:
                examples_correct.append({
                    "mention": mention_spans[idx],
                    "gold": sctid_to_fsn.get(gold, "?"),
                    "choice": choice,
                    "raw": raw_text,
                })
        elif pred is not None:
            llm_wrong += 1
            if gold_in_topk:
                wrong_gold_in_topk += 1
            else:
                wrong_gold_not_in_topk += 1
                if gold_in_full:
                    miss_but_in_full += 1
                else:
                    miss_not_in_full += 1
            if len(examples_wrong) < 10:
                examples_wrong.append({
                    "mention": mention_spans[idx],
                    "context": contexts[idx][:200],
                    "gold": f"{gold} ({sctid_to_fsn.get(gold, '?')})",
                    "pred": f"{pred} ({sctid_to_fsn.get(pred, '?')})" if pred else "ABSTAIN/UNPARSEABLE",
                    "gold_in_topk": gold_in_topk,
                    "choice": choice,
                    "raw": raw_text,
                    "candidates": [f"{s} ({sctid_to_fsn.get(s, '?')})" for s, _ in candidates],
                })

    # Optionally save per-query predictions
    if args.save_predictions:
        import csv
        with open(args.save_predictions, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([
                "query_idx", "mention", "gold_sctid", "gold_fsn",
                "pred_sctid", "pred_fsn", "choice", "correct",
                "gold_in_topk", "n_candidates", "pipeline_rank0_correct",
                "mention_len_tokens", "strategy", "escalated",
                "new_search_term", "raw_output",
            ])
            for j, idx in enumerate(tier3_indices):
                gold = gold_sctids[idx]
                candidates = reranked_results[idx][:args.top_k]
                candidate_ids = [s for s, _ in candidates]
                raw_text = raw_outputs[j]
                choice = parse_llm_choice(raw_text, len(candidates) - 1)
                if choice == -1 or choice >= len(candidates):
                    pred = None
                else:
                    pred = candidate_ids[choice]
                writer.writerow([
                    idx,
                    mention_spans[idx],
                    gold,
                    sctid_to_fsn.get(gold, ""),
                    pred or "",
                    sctid_to_fsn.get(pred, "") if pred else "",
                    choice,
                    int(pred == gold) if pred else 0,
                    int(gold in candidate_ids),
                    len(candidates),
                    int(candidate_ids[0] == gold) if candidate_ids else 0,
                    len(mention_spans[idx].split()),
                    args.strategy,
                    int(escalated_flags[j]),
                    search_terms_used.get(j, ""),
                    raw_text.replace("\t", " ").replace("\n", " "),
                ])
        print(f"\n  Predictions saved to {args.save_predictions}")

    n_tested = len(tier3_indices)
    strategy_desc = args.strategy
    if args.strategy != "single":
        strategy_desc += f" (checker_effort={args.checker_effort})"
    print(f"\n  LLM Results on {n_tested} tier-3 queries "
          f"[strategy={strategy_desc}]:")
    print(f"    Correct:      {llm_correct:4d} ({llm_correct / n_tested * 100:.1f}%)")
    print(f"    Wrong:        {llm_wrong:4d} ({llm_wrong / n_tested * 100:.1f}%)")
    print(f"    Abstain (-1): {llm_abstain:4d}")
    print(f"    Unparseable:  {llm_unparseable:4d}")
    if n_escalated > 0:
        print(f"    Escalated:    {n_escalated:4d} ({n_escalated / n_tested * 100:.1f}%)")
    print(f"    Pipeline R@1 on same queries: {pipeline_would_correct:4d} "
          f"({pipeline_would_correct / n_tested * 100:.1f}%)")

    # Failure mode analysis: gold in candidates vs retrieval failure
    n_failures = llm_wrong + llm_abstain + llm_unparseable
    print(f"\n  Failure mode analysis ({n_failures} failures):")
    print(f"    Gold IN top-{args.top_k} but LLM picked wrong: {wrong_gold_in_topk:4d}"
          + (f" ({wrong_gold_in_topk / n_failures * 100:.1f}%)" if n_failures else ""))
    print(f"    Gold NOT in top-{args.top_k} (retrieval miss):  {wrong_gold_not_in_topk:4d}"
          + (f" ({wrong_gold_not_in_topk / n_failures * 100:.1f}%)" if n_failures else ""))
    print(f"    Abstain/unparse, gold in top-{args.top_k}:      {abstain_gold_in_topk:4d}"
          + (f" ({abstain_gold_in_topk / n_failures * 100:.1f}%)" if n_failures else ""))
    print(f"    Abstain/unparse, gold NOT in top-{args.top_k}:  {abstain_gold_not_in_topk:4d}"
          + (f" ({abstain_gold_not_in_topk / n_failures * 100:.1f}%)" if n_failures else ""))
    recoverable = wrong_gold_in_topk + abstain_gold_in_topk
    print(f"    → Recoverable (gold was in candidates): {recoverable}/{n_failures}"
          + (f" ({recoverable / n_failures * 100:.1f}%)" if n_failures else ""))
    unreachable = wrong_gold_not_in_topk + abstain_gold_not_in_topk
    print(f"    → Unreachable (retrieval ceiling):      {unreachable}/{n_failures}"
          + (f" ({unreachable / n_failures * 100:.1f}%)" if n_failures else ""))
    # Deeper: of unreachable, how many had gold in the full reranked list (top-50)?
    n_full = len(reranked_results[tier3_indices[0]]) if tier3_indices else 0
    print(f"\n    Retrieval depth analysis (of {unreachable} not-in-top-{args.top_k}):")
    print(f"      Gold in top-{n_full} (fixable by showing more): {miss_but_in_full:4d}"
          + (f" ({miss_but_in_full / unreachable * 100:.1f}%)" if unreachable else ""))
    print(f"      Gold not in top-{n_full} (true retrieval miss): {miss_not_in_full:4d}"
          + (f" ({miss_not_in_full / unreachable * 100:.1f}%)" if unreachable else ""))

    # Tier 1 & 2 accuracy on the full set
    t1_correct = sum(
        1 for i, t in enumerate(tiers)
        if t == 1 and dictionary[mention_spans[i].lower()]["concept_id"] == gold_sctids[i]
    )
    t2_correct = sum(
        1 for i, t in enumerate(tiers)
        if t == 2 and reranked_results[i] and reranked_results[i][0][0] == gold_sctids[i]
    )

    print(f"\n  Full routing summary:")
    print(f"    Tier 1: {t1_correct}/{tier_counts[1]} correct ({t1_correct / tier_counts[1] * 100:.1f}%)")
    print(f"    Tier 2: {t2_correct}/{tier_counts[2]} correct ({t2_correct / tier_counts[2] * 100:.1f}%)")
    print(f"    Tier 3 (LLM, {n_tested} tested): {llm_correct}/{n_tested} "
          f"({llm_correct / n_tested * 100:.1f}%)")

    # Projected combined R@1
    # For tier-3 queries not tested, assume pipeline accuracy
    tier3_not_tested = tier_counts[3] - n_tested
    tier3_pipeline_est = sum(
        1 for i, t in enumerate(tiers)
        if t == 3 and i not in set(tier3_indices)
        and reranked_results[i] and reranked_results[i][0][0] == gold_sctids[i]
    )
    # Extrapolate LLM accuracy to full tier-3
    llm_rate = llm_correct / n_tested if n_tested > 0 else 0
    projected_llm_correct = int(llm_rate * tier_counts[3])

    projected_total = t1_correct + t2_correct + projected_llm_correct
    n_total = len(gold_sctids)
    print(f"\n  Projected combined R@1 (if LLM handles all tier 3):")
    print(f"    {projected_total}/{n_total} ({projected_total / n_total * 100:.1f}%)")
    print(f"    vs baseline pipeline: {baseline_correct}/{n_total} "
          f"({baseline_correct / n_total * 100:.1f}%)")
    print(f"    Lift: +{(projected_total / n_total - baseline_correct / n_total) * 100:.1f}pp")

    # Examples
    if examples_correct:
        print(f"\n  ─── LLM Correct Examples ───")
        for ex in examples_correct:
            print(f"    \"{ex['mention']}\" → choice {ex['choice']}: {ex['gold']} (raw: \"{ex['raw']}\")")

    if examples_wrong:
        print(f"\n  ─── LLM Wrong Examples ───")
        for ex in examples_wrong:
            print(f"    \"{ex['mention']}\"")
            print(f"      Context: {ex['context'][:150]}...")
            print(f"      Gold: {ex['gold']} (in top-{args.top_k}: {ex['gold_in_topk']})")
            print(f"      LLM picked: {ex['pred']} (raw: \"{ex['raw']}\")")
            if ex["gold_in_topk"]:
                print(f"      Candidates: {ex['candidates']}")
            print()


if __name__ == "__main__":
    main()
