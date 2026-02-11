#!/usr/bin/env python3
"""
Multi-tier routing analysis: Dictionary + Pipeline + Agent consultation.

Evaluates a routing strategy that combines:
  Tier 1: Dictionary auto-accept (high confidence, high training frequency)
  Tier 2: Pipeline high-confidence (large score gap between rank-1 and rank-2)
  Tier 3: Agent consultation (everything else - needs LLM review)

Runs the full pipeline on the test split and combines with a train-split dictionary.
"""

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from snomed_ct_entity_linking.recall_analysis.config import Config
from snomed_ct_entity_linking.recall_analysis.snomed_loader import load_snomed, load_validation_data
from snomed_ct_entity_linking.recall_analysis.index_builder import (
    load_indexes, indexes_exist, load_raw_embeddings,
)
from snomed_ct_entity_linking.recall_analysis.retrieval import encode_queries, hybrid_retrieve, expand_with_hierarchy
from snomed_ct_entity_linking.recall_analysis.reranker import SapBERTReranker


def build_train_dictionary(cfg: Config) -> dict:
    """
    Build a dictionary from the training split annotations.

    Returns dict mapping span_lower -> {
        'concept_id': most_common_concept_id,
        'count': number of training occurrences of that span,
        'confidence': fraction of occurrences mapping to the most common concept,
        'all_concepts': Counter of all concept_ids for this span,
    }
    """
    ann = pd.read_csv(cfg.train_annotations, dtype={"concept_id": int})
    train = ann[ann["annotation_type"] == "train"]

    # Group by lowercased span
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
            "all_concepts": counter,
        }

    print(f"  Train dictionary: {len(dictionary):,} unique spans")
    print(f"  Total training annotations used: {len(train):,}")
    return dictionary


def evaluate_routing(
    gold_sctids: list[int],
    mention_spans: list[str],
    reranked_results: list[list[tuple[int, float]]],
    dictionary: dict,
    dict_confidence_threshold: float,
    dict_count_threshold: int,
    pipeline_gap_threshold: float,
    sctid_to_fsn: dict[int, str],
    indexed_sctids: set[int],
) -> dict:
    """
    Evaluate a multi-tier routing strategy.

    Tier 1: Dictionary auto-accept if confidence >= threshold AND count >= count_threshold
    Tier 2: Pipeline auto-accept if score gap (rank1 - rank2) >= gap_threshold
    Tier 3: Agent consultation (everything else)

    Returns detailed per-tier statistics.
    """
    n = len(gold_sctids)
    tier1_correct = 0
    tier1_wrong = 0
    tier2_correct = 0
    tier2_wrong = 0
    tier3_queries = 0
    tier3_gold_in_candidates = 0

    # Per-query decisions
    decisions = []

    for i in range(n):
        gold = gold_sctids[i]
        span_lower = mention_spans[i].lower()
        candidates = reranked_results[i]

        # Tier 1: Dictionary lookup
        dict_entry = dictionary.get(span_lower)
        if (dict_entry
                and dict_entry["confidence"] >= dict_confidence_threshold
                and dict_entry["count"] >= dict_count_threshold
                and dict_entry["concept_id"] in indexed_sctids):
            pred = dict_entry["concept_id"]
            correct = (pred == gold)
            if correct:
                tier1_correct += 1
            else:
                tier1_wrong += 1
            decisions.append({
                "tier": 1, "correct": correct,
                "pred": pred, "gold": gold,
                "dict_confidence": dict_entry["confidence"],
                "dict_count": dict_entry["count"],
            })
            continue

        # Tier 2: Pipeline high-confidence
        if len(candidates) >= 2:
            score_gap = candidates[0][1] - candidates[1][1]
        elif len(candidates) == 1:
            score_gap = candidates[0][1]  # Only one candidate = high confidence
        else:
            score_gap = 0.0

        if score_gap >= pipeline_gap_threshold:
            pred = candidates[0][0] if candidates else None
            correct = (pred == gold)
            if correct:
                tier2_correct += 1
            else:
                tier2_wrong += 1
            decisions.append({
                "tier": 2, "correct": correct,
                "pred": pred, "gold": gold,
                "score_gap": score_gap,
            })
            continue

        # Tier 3: Agent consultation
        tier3_queries += 1
        # Check if gold is in the candidate list (agent could find it)
        candidate_sctids = [c[0] for c in candidates]
        if gold in candidate_sctids:
            tier3_gold_in_candidates += 1
        decisions.append({
            "tier": 3, "correct": None,
            "gold": gold,
            "gold_in_candidates": gold in candidate_sctids,
        })

    tier1_total = tier1_correct + tier1_wrong
    tier2_total = tier2_correct + tier2_wrong

    return {
        "n_total": n,
        "tier1_total": tier1_total,
        "tier1_correct": tier1_correct,
        "tier1_wrong": tier1_wrong,
        "tier1_accuracy": tier1_correct / tier1_total if tier1_total > 0 else 0,
        "tier2_total": tier2_total,
        "tier2_correct": tier2_correct,
        "tier2_wrong": tier2_wrong,
        "tier2_accuracy": tier2_correct / tier2_total if tier2_total > 0 else 0,
        "tier3_total": tier3_queries,
        "tier3_gold_in_candidates": tier3_gold_in_candidates,
        "tier3_gold_in_pct": tier3_gold_in_candidates / tier3_queries if tier3_queries > 0 else 0,
        "auto_resolved": tier1_total + tier2_total,
        "auto_resolved_correct": tier1_correct + tier2_correct,
        "auto_resolved_accuracy": (tier1_correct + tier2_correct) / (tier1_total + tier2_total) if (tier1_total + tier2_total) > 0 else 0,
        "decisions": decisions,
    }


def main():
    cfg = Config()
    pipeline_start = time.time()

    # =========================================================================
    # Step 1: Load SNOMED CT data
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 1: Loading SNOMED CT Terminology")
    print("=" * 70)
    descriptions_df, sctid_to_fsn, sctid_to_tag, parent_map = load_snomed(cfg)

    # =========================================================================
    # Step 2: Load cached indexes
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 2: Loading Cached Indexes")
    print("=" * 70)
    if not indexes_exist(cfg):
        print("ERROR: No cached indexes found. Run the recall analysis pipeline first.")
        sys.exit(1)
    faiss_index, faiss_sctids, bm25_index, bm25_sctids, bm25_terms = load_indexes(cfg)

    # =========================================================================
    # Step 3: Load validation data and filter
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 3: Loading Validation Data")
    print("=" * 70)
    val_df = load_validation_data(cfg)
    val_df["semantic_tag"] = val_df["concept_id"].map(sctid_to_tag).fillna("unknown")
    indexed_sctids = set(faiss_sctids)
    val_df["in_index"] = val_df["concept_id"].isin(indexed_sctids)
    val_df = val_df[val_df["in_index"]].reset_index(drop=True)
    print(f"  Evaluating on {len(val_df):,} annotations with gold concept in index")

    gold_sctids = val_df["concept_id"].tolist()
    mention_spans = val_df["span"].tolist()

    # =========================================================================
    # Step 4: Hybrid Retrieval
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 4: Hybrid Retrieval (Dense + BM25 + RRF)")
    print("=" * 70)
    query_embeddings = encode_queries(mention_spans, cfg)
    fused_results, dense_results_all, sparse_results_all = hybrid_retrieve(
        query_embeddings, mention_spans,
        faiss_index, faiss_sctids, bm25_index, bm25_sctids, cfg,
    )

    # Hierarchy expansion
    fused_results = expand_with_hierarchy(fused_results, parent_map, indexed_sctids)

    # =========================================================================
    # Step 5: SapBERT Reranking
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 5: SapBERT Max-Synonym Reranking")
    print("=" * 70)
    raw_data = load_raw_embeddings(cfg)
    if raw_data is None:
        print("ERROR: No raw embeddings found.")
        sys.exit(1)
    raw_embeddings, raw_sctids = raw_data
    reranker = SapBERTReranker(raw_embeddings, raw_sctids)
    candidate_sctids_per_query = [[sctid for sctid, _ in fused] for fused in fused_results]
    reranked_results = reranker.rerank(query_embeddings, candidate_sctids_per_query)
    reranker.close()

    # Compute baseline pipeline R@1
    baseline_correct = sum(
        1 for gold, res in zip(gold_sctids, reranked_results)
        if res and res[0][0] == gold
    )
    baseline_r1 = baseline_correct / len(gold_sctids)
    print(f"\n  Pipeline baseline R@1: {baseline_r1*100:.1f}% ({baseline_correct:,}/{len(gold_sctids):,})")

    # =========================================================================
    # Step 6: Build Train-Split Dictionary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 6: Building Train-Split Dictionary")
    print("=" * 70)
    dictionary = build_train_dictionary(cfg)

    # Dictionary coverage on test set
    n_covered = sum(1 for s in mention_spans if s.lower() in dictionary)
    print(f"  Dictionary coverage on test: {n_covered:,}/{len(mention_spans):,} "
          f"({n_covered/len(mention_spans)*100:.1f}%)")

    # =========================================================================
    # Step 7: Multi-Tier Routing Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("  STEP 7: Multi-Tier Routing Analysis")
    print("=" * 70)

    # --- Sweep parameters ---
    confidence_thresholds = [1.0]  # Only auto-accept at 100% confidence
    count_thresholds = [1, 2, 3, 5, 10]
    gap_thresholds = [0.03, 0.05, 0.08, 0.10, 0.15]

    print("\n  ─── Dictionary-Only Tier (Tier 1) ───")
    print(f"  {'Conf':>5s}  {'Count≥':>6s}  {'Auto':>6s}  {'Correct':>8s}  {'Errors':>6s}  {'Acc%':>6s}  {'Remaining':>10s}")
    print("  " + "─" * 60)
    for conf in confidence_thresholds:
        for cnt in count_thresholds:
            r = evaluate_routing(
                gold_sctids, mention_spans, reranked_results, dictionary,
                dict_confidence_threshold=conf,
                dict_count_threshold=cnt,
                pipeline_gap_threshold=999.0,  # disable tier 2
                sctid_to_fsn=sctid_to_fsn,
                indexed_sctids=indexed_sctids,
            )
            print(f"  {conf:5.0%}  {cnt:6d}  {r['tier1_total']:6,d}  {r['tier1_correct']:8,d}  "
                  f"{r['tier1_wrong']:6,d}  {r['tier1_accuracy']:5.1%}  {r['tier3_total']:10,d}")

    print("\n  ─── Pipeline-Only Tier (Tier 2) ───")
    print(f"  {'Gap≥':>6s}  {'Auto':>6s}  {'Correct':>8s}  {'Errors':>6s}  {'Acc%':>6s}  {'Remaining':>10s}")
    print("  " + "─" * 55)
    for gap in gap_thresholds:
        r = evaluate_routing(
            gold_sctids, mention_spans, reranked_results, dictionary,
            dict_confidence_threshold=999.0,  # disable tier 1
            dict_count_threshold=999999,
            pipeline_gap_threshold=gap,
            sctid_to_fsn=sctid_to_fsn,
            indexed_sctids=indexed_sctids,
        )
        print(f"  {gap:6.2f}  {r['tier2_total']:6,d}  {r['tier2_correct']:8,d}  "
              f"{r['tier2_wrong']:6,d}  {r['tier2_accuracy']:5.1%}  {r['tier3_total']:10,d}")

    # --- Combined multi-tier sweep ---
    print("\n" + "=" * 70)
    print("  COMBINED MULTI-TIER ROUTING")
    print("=" * 70)
    print(f"\n  Tier 1: Dictionary (confidence=100%)")
    print(f"  Tier 2: Pipeline (score gap ≥ threshold)")
    print(f"  Tier 3: Agent consultation (remainder)")
    print()
    print(f"  {'Dict≥':>5s}  {'Gap≥':>5s}  │ {'T1':>5s}  {'T2':>5s}  {'T3':>5s}  │ "
          f"{'Auto':>5s}  {'AutoAcc':>7s}  {'AutoErr':>7s}  │ "
          f"{'T3 gold':>7s}  {'T3 R@50':>7s}")
    print("  " + "─" * 85)

    best_config = None
    best_score = 0

    for cnt in count_thresholds:
        for gap in gap_thresholds:
            r = evaluate_routing(
                gold_sctids, mention_spans, reranked_results, dictionary,
                dict_confidence_threshold=1.0,
                dict_count_threshold=cnt,
                pipeline_gap_threshold=gap,
                sctid_to_fsn=sctid_to_fsn,
                indexed_sctids=indexed_sctids,
            )

            auto = r["auto_resolved"]
            auto_acc = r["auto_resolved_accuracy"]
            auto_err = r["tier1_wrong"] + r["tier2_wrong"]
            t3_gold = r["tier3_gold_in_candidates"]

            # Score: maximize auto-resolved at high accuracy (penalize errors)
            score = r["auto_resolved_correct"] - 5 * auto_err  # each error costs 5 correct

            print(f"  {cnt:5d}  {gap:5.2f}  │ {r['tier1_total']:5,d}  {r['tier2_total']:5,d}  "
                  f"{r['tier3_total']:5,d}  │ {auto:5,d}  {auto_acc:6.1%}  {auto_err:7,d}  │ "
                  f"{t3_gold:7,d}  {r['tier3_gold_in_pct']:6.1%}")

            if score > best_score:
                best_score = score
                best_config = (cnt, gap, r)

    # =========================================================================
    # Step 8: Best Configuration Deep-Dive
    # =========================================================================
    if best_config:
        cnt, gap, r = best_config
        print("\n" + "=" * 70)
        print(f"  BEST CONFIGURATION: Dict count≥{cnt}, Pipeline gap≥{gap:.2f}")
        print("=" * 70)
        print(f"\n  Tier 1 (Dictionary auto-accept):")
        print(f"    Queries handled: {r['tier1_total']:,d} ({r['tier1_total']/r['n_total']*100:.1f}%)")
        print(f"    Correct: {r['tier1_correct']:,d} | Errors: {r['tier1_wrong']:,d} | Accuracy: {r['tier1_accuracy']:.1%}")
        print(f"\n  Tier 2 (Pipeline auto-accept):")
        print(f"    Queries handled: {r['tier2_total']:,d} ({r['tier2_total']/r['n_total']*100:.1f}%)")
        print(f"    Correct: {r['tier2_correct']:,d} | Errors: {r['tier2_wrong']:,d} | Accuracy: {r['tier2_accuracy']:.1%}")
        print(f"\n  Tier 3 (Agent consultation):")
        print(f"    Queries remaining: {r['tier3_total']:,d} ({r['tier3_total']/r['n_total']*100:.1f}%)")
        print(f"    Gold in candidates: {r['tier3_gold_in_candidates']:,d} ({r['tier3_gold_in_pct']:.1%})")
        print(f"\n  Combined auto-resolved:")
        auto = r['tier1_total'] + r['tier2_total']
        auto_correct = r['tier1_correct'] + r['tier2_correct']
        auto_err = r['tier1_wrong'] + r['tier2_wrong']
        print(f"    Total: {auto:,d}/{r['n_total']:,d} ({auto/r['n_total']*100:.1f}%)")
        print(f"    Correct: {auto_correct:,d} | Errors: {auto_err:,d} | Accuracy: {auto_correct/auto*100:.1f}%")

        # Effective R@1 if agent is perfect on tier 3 candidates
        oracle_agent_correct = r['tier3_gold_in_candidates']
        oracle_total = auto_correct + oracle_agent_correct
        print(f"\n  Effective R@1 (oracle agent on tier 3): {oracle_total/r['n_total']*100:.1f}%")
        print(f"  Baseline pipeline R@1: {baseline_r1*100:.1f}%")
        print(f"  Lift: +{(oracle_total/r['n_total'] - baseline_r1)*100:.1f}pp")

        # Error analysis on auto-resolved errors
        decisions = r["decisions"]
        tier1_errors = [d for d in decisions if d["tier"] == 1 and not d["correct"]]
        tier2_errors = [d for d in decisions if d["tier"] == 2 and not d["correct"]]

        if tier1_errors:
            print(f"\n  ─── Tier 1 Error Examples (Dictionary, {len(tier1_errors)} total) ───")
            for d in tier1_errors[:10]:
                gold_name = sctid_to_fsn.get(d["gold"], "?")
                pred_name = sctid_to_fsn.get(d["pred"], "?")
                print(f"    Predicted: {d['pred']} ({pred_name})")
                print(f"    Gold:      {d['gold']} ({gold_name})")
                print(f"    Conf={d['dict_confidence']:.0%} Count={d['dict_count']}")
                print()

        if tier2_errors:
            print(f"\n  ─── Tier 2 Error Examples (Pipeline, {len(tier2_errors)} total) ───")
            for d in tier2_errors[:10]:
                gold_name = sctid_to_fsn.get(d["gold"], "?")
                pred_name = sctid_to_fsn.get(d["pred"], "?")
                print(f"    Predicted: {d['pred']} ({pred_name})")
                print(f"    Gold:      {d['gold']} ({gold_name})")
                print(f"    Gap={d['score_gap']:.3f}")
                print()

    elapsed = time.time() - pipeline_start
    print(f"\nTotal analysis time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
