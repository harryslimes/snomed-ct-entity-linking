#!/usr/bin/env python3
"""
Mechanical Evaluation Pipeline: Recall@K Ceiling Analysis
for SNOMED CT Clinical Finding Hierarchy.

Measures how often the correct concept appears in the top-N candidates
using hybrid retrieval (dense + BM25) and cross-encoder reranking.

Usage:
    python -m snomed_ct_entity_linking.recall_analysis.run [--rebuild-index] [--skip-rerank]
"""

import argparse
import os
import sys
import time

from .config import Config
from .snomed_loader import load_snomed, load_validation_data
from .index_builder import (
    build_indexes, save_indexes, load_indexes, indexes_exist,
    rebuild_faiss_from_cache, load_raw_embeddings,
)
from .retrieval import encode_queries, hybrid_retrieve, expand_with_hierarchy
from .reranker import SapBERTReranker, CrossEncoderReranker
from .evaluate import (
    compute_recall_at_k,
    compute_recall_at_k_pre_rerank,
    generate_failure_csv,
    print_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Recall@K ceiling analysis for SNOMED CT entity linking"
    )
    parser.add_argument(
        "--rebuild-index", action="store_true",
        help="Force rebuild FAISS and BM25 indexes even if cached.",
    )
    parser.add_argument(
        "--skip-rerank", action="store_true",
        help="Skip reranking (evaluate hybrid retrieval only).",
    )
    parser.add_argument(
        "--reranker", type=str, default="sapbert",
        choices=["sapbert", "cross-encoder", "none"],
        help="Reranker to use (default: sapbert max-synonym).",
    )
    parser.add_argument(
        "--embedding-model", type=str, default=None,
        help="Override embedding model name.",
    )
    parser.add_argument(
        "--cross-encoder-model", type=str, default=None,
        help="Override cross-encoder model name.",
    )
    parser.add_argument(
        "--preferred-weight", type=float, default=2.0,
        help="Weight for preferred terms in concept averaging (default: 2.0).",
    )
    parser.add_argument(
        "--index-batch-size", type=int, default=None,
        help="Batch size for encoding SNOMED descriptions.",
    )
    parser.add_argument(
        "--query-batch-size", type=int, default=None,
        help="Batch size for encoding validation queries.",
    )
    parser.add_argument(
        "--rerank-batch-size", type=int, default=None,
        help="Batch size for cross-encoder reranking.",
    )
    args = parser.parse_args()

    if args.skip_rerank:
        args.reranker = "none"

    cfg = Config()
    if args.embedding_model:
        cfg.embedding_model = args.embedding_model
    if args.cross_encoder_model:
        cfg.cross_encoder_model = args.cross_encoder_model
    if args.index_batch_size:
        cfg.index_batch_size = args.index_batch_size
    if args.query_batch_size:
        cfg.query_batch_size = args.query_batch_size
    if args.rerank_batch_size:
        cfg.rerank_batch_size = args.rerank_batch_size

    os.makedirs(cfg.output_dir, exist_ok=True)
    pipeline_start = time.time()

    # =========================================================================
    # Step 1: Load SNOMED CT data
    # =========================================================================
    print("\n" + "=" * 65)
    print("  STEP 1: Loading SNOMED CT Terminology")
    print("=" * 65)
    descriptions_df, sctid_to_fsn, sctid_to_tag, parent_map = load_snomed(cfg)

    # =========================================================================
    # Step 2: Build or load indexes
    # =========================================================================
    print("\n" + "=" * 65)
    print("  STEP 2: Building / Loading Indexes")
    print("=" * 65)
    if indexes_exist(cfg) and not args.rebuild_index:
        print("Cached indexes found. Loading from disk ...")
        faiss_index, faiss_sctids, bm25_index, bm25_sctids, bm25_terms = load_indexes(cfg)

        # If preferred_weight differs from default, rebuild FAISS from cached embeddings
        if args.preferred_weight != 2.0:
            result = rebuild_faiss_from_cache(cfg, args.preferred_weight)
            if result:
                faiss_index, faiss_sctids = result
            else:
                print("  WARNING: No cached embeddings, rebuilding full index ...")
                faiss_index, faiss_sctids, bm25_index, bm25_sctids, bm25_terms = build_indexes(
                    descriptions_df, cfg, args.preferred_weight,
                )
                save_indexes(faiss_index, faiss_sctids, bm25_index, bm25_sctids, bm25_terms, cfg)
    else:
        faiss_index, faiss_sctids, bm25_index, bm25_sctids, bm25_terms = build_indexes(
            descriptions_df, cfg, args.preferred_weight,
        )
        save_indexes(faiss_index, faiss_sctids, bm25_index, bm25_sctids, bm25_terms, cfg)

    # =========================================================================
    # Step 3: Load validation data
    # =========================================================================
    print("\n" + "=" * 65)
    print("  STEP 3: Loading Validation Data")
    print("=" * 65)
    val_df = load_validation_data(cfg)

    # Label each gold annotation with its semantic tag
    val_df["semantic_tag"] = val_df["concept_id"].map(sctid_to_tag).fillna("unknown")
    indexed_sctids = set(faiss_sctids)
    val_df["in_index"] = val_df["concept_id"].isin(indexed_sctids)
    target_tags = set(cfg.semantic_tags)

    # Diagnostic breakdown
    n_total = len(val_df)
    tag_counts = val_df["semantic_tag"].value_counts()
    n_target_tag = val_df["semantic_tag"].isin(target_tags).sum()
    n_other_tag = n_total - n_target_tag
    n_in_index = val_df["in_index"].sum()
    n_target_but_missing = (val_df["semantic_tag"].isin(target_tags) & ~val_df["in_index"]).sum()

    print(f"\n  Gold annotation breakdown ({n_total:,} total):")
    print(f"    Finding/disorder:  {n_target_tag:,} ({n_target_tag / n_total * 100:.1f}%)")
    print(f"    Other tags:        {n_other_tag:,} ({n_other_tag / n_total * 100:.1f}%)")
    print(f"    In index:          {n_in_index:,} ({n_in_index / n_total * 100:.1f}%)")
    if n_target_but_missing > 0:
        print(f"    WARNING: {n_target_but_missing:,} finding/disorder annotations "
              f"NOT in index (inactive or missing descriptions)")
    print(f"\n  Top semantic tags in gold annotations:")
    for tag, count in tag_counts.head(15).items():
        marker = " *" if tag in target_tags else ""
        print(f"    {tag:30s} {count:6,} ({count / n_total * 100:5.1f}%){marker}")

    # Filter to only annotations whose gold concept is in the indexed set
    val_df = val_df[val_df["in_index"]].reset_index(drop=True)
    print(f"\n  Evaluating on {len(val_df):,} / {n_total:,} annotations "
          f"with gold concept in index")

    gold_sctids = val_df["concept_id"].tolist()
    contexts = val_df["context"].tolist()
    mention_spans = val_df["span"].tolist()

    # =========================================================================
    # Step 4: Hybrid Retrieval (mention spans for dense, keywords for BM25)
    # =========================================================================
    print("\n" + "=" * 65)
    print("  STEP 4: Hybrid Retrieval (Dense + BM25 + RRF)")
    print("=" * 65)
    # SapBERT encodes mention spans (short clinical phrases) for dense retrieval
    query_embeddings = encode_queries(mention_spans, cfg)
    fused_results, dense_results_all, sparse_results_all = hybrid_retrieve(
        query_embeddings, mention_spans,
        faiss_index, faiss_sctids, bm25_index, bm25_sctids, cfg,
    )

    # Pre-reranking recall (before hierarchy expansion)
    recall_pre_raw = compute_recall_at_k_pre_rerank(gold_sctids, fused_results, cfg.recall_k_values)
    print("\n  Recall BEFORE hierarchy expansion:")
    for k, v in sorted(recall_pre_raw.items()):
        print(f"    R@{k:<4d} {v*100:.1f}%")

    # Expand candidates with IS-A parent concepts
    print("\nApplying IS-A hierarchy expansion ...")
    fused_results = expand_with_hierarchy(fused_results, parent_map, indexed_sctids)

    # Pre-reranking recall (after hierarchy expansion)
    recall_pre = compute_recall_at_k_pre_rerank(gold_sctids, fused_results, cfg.recall_k_values)
    print("  Recall AFTER hierarchy expansion:")
    for k, v in sorted(recall_pre.items()):
        print(f"    R@{k:<4d} {v*100:.1f}%")

    # =========================================================================
    # Step 5: Reranking
    # =========================================================================
    if args.reranker == "none":
        print("\n[Skipping reranking]")
        reranked_results = fused_results
        recall_post = recall_pre
    elif args.reranker == "sapbert":
        print("\n" + "=" * 65)
        print("  STEP 5: SapBERT Max-Synonym Reranking")
        print("=" * 65)
        raw_data = load_raw_embeddings(cfg)
        if raw_data is None:
            print("  WARNING: No raw embeddings found, skipping reranking.")
            reranked_results = fused_results
            recall_post = recall_pre
        else:
            raw_embeddings, raw_sctids = raw_data
            reranker = SapBERTReranker(raw_embeddings, raw_sctids)
            candidate_sctids_per_query = [[sctid for sctid, _ in fused] for fused in fused_results]
            reranked_results = reranker.rerank(query_embeddings, candidate_sctids_per_query)
            reranker.close()
            recall_post = compute_recall_at_k(gold_sctids, reranked_results, cfg.recall_k_values)
    elif args.reranker == "cross-encoder":
        print("\n" + "=" * 65)
        print("  STEP 5: MedCPT Cross-Encoder Reranking")
        print("=" * 65)
        reranker = CrossEncoderReranker(cfg)
        candidate_sctids_per_query = [[sctid for sctid, _ in fused] for fused in fused_results]
        reranked_results = reranker.rerank(mention_spans, candidate_sctids_per_query, sctid_to_fsn)
        reranker.close()
        recall_post = compute_recall_at_k(gold_sctids, reranked_results, cfg.recall_k_values)

    # =========================================================================
    # Step 6: Evaluation & Failure Analysis
    # =========================================================================
    print("\n" + "=" * 65)
    print("  STEP 6: Evaluation")
    print("=" * 65)
    failure_df = generate_failure_csv(
        val_df, reranked_results,
        dense_results_all, sparse_results_all,
        sctid_to_fsn, cfg,
    )

    report = print_report(recall_pre, recall_post, len(gold_sctids), len(failure_df))

    # Save report
    report_path = cfg.output_dir / "recall_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    elapsed = time.time() - pipeline_start
    print(f"\nTotal pipeline time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
