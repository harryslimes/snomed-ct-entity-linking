"""Hybrid retrieval: dense (FAISS) + sparse (BM25S) with Reciprocal Rank Fusion."""

import time

import bm25s
import faiss
import numpy as np

from .config import Config
from .encoder import SapBERTEncoder


def encode_queries(
    mention_spans: list[str],
    cfg: Config,
) -> np.ndarray:
    """Encode mention spans using SapBERT for dense retrieval."""
    encoder = SapBERTEncoder(cfg.embedding_model)

    print(f"Encoding {len(mention_spans):,} mention spans (batch_size={cfg.query_batch_size}) ...")
    t0 = time.time()
    embeddings = encoder.encode(mention_spans, batch_size=cfg.query_batch_size, show_progress=True)
    elapsed = time.time() - t0
    print(f"  Encoded in {elapsed:.1f}s ({len(mention_spans) / elapsed:.0f} queries/s)")

    encoder.close()
    return embeddings


def dense_search(
    query_embeddings: np.ndarray,
    faiss_index: faiss.Index,
    faiss_sctids: list[int],
    top_k: int,
    verbose: bool = True,
    oversample: int = 1,
) -> list[list[tuple[int, float, int]]]:
    """
    Batch FAISS nearest-neighbor search.

    Args:
        oversample: Fetch top_k * oversample raw vectors before deduplicating
            per concept. Use oversample > 1 with a multi-vector index (one
            vector per description rather than one per concept) so that
            top_k unique concepts are reliably returned after dedup.

    Returns:
        List of per-query results, each a list of (sctid, score, rank).
        Each SCTID appears at most once (first/best hit kept).
    """
    search_k = min(top_k * oversample, faiss_index.ntotal)
    if verbose:
        print(f"FAISS search: {len(query_embeddings):,} queries x top-{search_k} ...")
    t0 = time.time()
    scores, indices = faiss_index.search(query_embeddings, search_k)
    if verbose:
        print(f"  FAISS search done in {time.time() - t0:.1f}s")

    results = []
    for i in range(len(query_embeddings)):
        seen: set[int] = set()
        query_results = []
        rank = 1
        for idx, score in zip(indices[i], scores[i]):
            if idx == -1:
                break
            sctid = faiss_sctids[idx]
            if sctid not in seen:
                seen.add(sctid)
                query_results.append((sctid, float(score), rank))
                rank += 1
                if rank > top_k:
                    break
        results.append(query_results)
    return results


def sparse_search(
    mention_spans: list[str],
    bm25_index: bm25s.BM25,
    bm25_sctids: list[int],
    top_k: int,
    n_threads: int = 16,
    verbose: bool = True,
) -> list[list[tuple[int, float, int]]]:
    """
    BM25S batch search using mention spans (keyword matching).
    Uses multi-threaded retrieval for speed.

    Returns:
        List of per-query results, each a list of (sctid, score, rank).
    """
    if verbose:
        print(f"BM25S search: {len(mention_spans):,} queries x top-{top_k} (n_threads={n_threads}) ...")
    t0 = time.time()

    query_tokens = bm25s.tokenize(mention_spans, show_progress=False)
    # Pass corpus=None to get integer indices instead of corpus strings
    doc_indices, doc_scores = bm25_index.retrieve(
        query_tokens, k=top_k, n_threads=n_threads, show_progress=False,
        corpus=None,
    )

    results = []
    for i in range(len(mention_spans)):
        query_results = []
        for rank in range(doc_indices.shape[1]):
            idx = int(doc_indices[i, rank])
            score = float(doc_scores[i, rank])
            if score <= 0:
                break
            query_results.append((bm25_sctids[idx], score, rank + 1))
        results.append(query_results)

    elapsed = time.time() - t0
    if verbose:
        print(f"  BM25S search done in {elapsed:.1f}s ({len(mention_spans) / elapsed:.0f} queries/s)")
    return results


def reciprocal_rank_fusion(
    dense_results: list[tuple[int, float, int]],
    sparse_results: list[tuple[int, float, int]],
    k: int,
    top_n: int,
) -> list[tuple[int, float]]:
    """
    Combine dense and sparse results using Reciprocal Rank Fusion.

    Score(sctid) = 1/(k + rank_dense) + 1/(k + rank_sparse)

    Returns unique SCTIDs sorted by RRF score, limited to top_n.
    """
    dense_rank = _best_rank_by_sctid(dense_results)
    sparse_rank = _best_rank_by_sctid(sparse_results)

    all_sctids = set(dense_rank.keys()) | set(sparse_rank.keys())
    max_rank = 10000  # penalty for missing results

    scored = []
    for sctid in all_sctids:
        dr = dense_rank.get(sctid, max_rank)
        sr = sparse_rank.get(sctid, max_rank)
        rrf_score = 1.0 / (k + dr) + 1.0 / (k + sr)
        scored.append((sctid, rrf_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def _best_rank_by_sctid(
    results: list[tuple[int, float, int]],
) -> dict[int, int]:
    """Get the best (lowest) rank for each unique SCTID in results."""
    best = {}
    for sctid, _score, rank in results:
        if sctid not in best or rank < best[sctid]:
            best[sctid] = rank
    return best


def hybrid_retrieve(
    query_embeddings: np.ndarray,
    mention_spans: list[str],
    faiss_index: faiss.Index,
    faiss_sctids: list[int],
    bm25_index: bm25s.BM25,
    bm25_sctids: list[int],
    cfg: Config,
) -> tuple[list[list[tuple[int, float]]], list[list[tuple[int, float, int]]], list[list[tuple[int, float, int]]]]:
    """
    Full hybrid retrieval pipeline: dense + sparse + RRF.

    Returns:
        fused_results: Per-query list of (sctid, rrf_score) after fusion, top-50.
        dense_results_all: Raw dense results for failure analysis.
        sparse_results_all: Raw sparse results for failure analysis.
    """
    dense_results_all = dense_search(query_embeddings, faiss_index, faiss_sctids, cfg.dense_top_k)
    sparse_results_all = sparse_search(
        mention_spans, bm25_index, bm25_sctids, cfg.sparse_top_k,
        n_threads=cfg.bm25_threads,
    )

    print(f"Applying RRF fusion (k={cfg.rrf_k}, top-{cfg.fusion_top_k}) ...")
    fused_results = []
    for dense_res, sparse_res in zip(dense_results_all, sparse_results_all):
        fused = reciprocal_rank_fusion(dense_res, sparse_res, cfg.rrf_k, cfg.fusion_top_k)
        fused_results.append(fused)

    return fused_results, dense_results_all, sparse_results_all


def expand_with_hierarchy(
    fused_results: list[list[tuple[int, float]]],
    parent_map: dict[int, set[int]],
    indexed_sctids: set[int],
) -> list[list[tuple[int, float]]]:
    """
    Expand candidate lists by adding parent concepts from IS-A hierarchy.

    For each candidate in the fused results, adds its direct IS-A parents
    (if they're in the indexed set) with a discounted score. This catches
    granularity mismatches where we retrieved a child but the gold is a parent.
    """
    expanded = []
    n_added = 0

    for candidates in fused_results:
        existing = {sctid for sctid, _ in candidates}
        new_candidates = list(candidates)

        for sctid, score in candidates:
            for parent in parent_map.get(sctid, set()):
                if parent not in existing and parent in indexed_sctids:
                    new_candidates.append((parent, score * 0.5))
                    existing.add(parent)
                    n_added += 1

        new_candidates.sort(key=lambda x: x[1], reverse=True)
        expanded.append(new_candidates)

    print(f"  Hierarchy expansion: added {n_added:,} parent concepts "
          f"across {len(fused_results):,} queries")
    return expanded
