"""Evaluation metrics: Recall@K and failure analysis."""

import os

import pandas as pd

from .config import Config


def compute_recall_at_k(
    gold_sctids: list[int],
    ranked_results: list[list[tuple[int, float]]],
    k_values: tuple[int, ...],
) -> dict[int, float]:
    """
    Compute Recall@K for multiple K values.

    Args:
        gold_sctids: List of gold-standard SCTIDs, one per query.
        ranked_results: Per-query list of (sctid, score) sorted by score desc.
        k_values: Tuple of K values to evaluate.

    Returns:
        Dict mapping K -> recall fraction.
    """
    n = len(gold_sctids)
    hits = {k: 0 for k in k_values}

    for gold, candidates in zip(gold_sctids, ranked_results):
        candidate_sctids = [c[0] for c in candidates]
        for k in k_values:
            if gold in candidate_sctids[:k]:
                hits[k] += 1

    return {k: hits[k] / n for k in k_values}


def compute_recall_at_k_pre_rerank(
    gold_sctids: list[int],
    fused_results: list[list[tuple[int, float]]],
    k_values: tuple[int, ...],
) -> dict[int, float]:
    """Compute Recall@K on the pre-reranking fused candidates."""
    return compute_recall_at_k(gold_sctids, fused_results, k_values)


def find_dense_rank(
    gold_sctid: int,
    dense_results: list[tuple[int, float, int]],
) -> int | None:
    """Find the rank of the gold SCTID in dense results, or None if missing."""
    for sctid, _score, rank in dense_results:
        if sctid == gold_sctid:
            return rank
    return None


def find_sparse_rank(
    gold_sctid: int,
    sparse_results: list[tuple[int, float, int]],
) -> int | None:
    """Find the rank of the gold SCTID in sparse results, or None if missing."""
    for sctid, _score, rank in sparse_results:
        if sctid == gold_sctid:
            return rank
    return None


def generate_failure_csv(
    val_df: pd.DataFrame,
    ranked_results: list[list[tuple[int, float]]],
    dense_results_all: list[list[tuple[int, float, int]]],
    sparse_results_all: list[list[tuple[int, float, int]]],
    sctid_to_fsn: dict[int, str],
    cfg: Config,
) -> pd.DataFrame:
    """
    Generate retrieval_failures.csv for cases where gold SCTID is NOT in top-50.

    Columns: Mention, Context, Gold_SCTID, Gold_FSN, Top_Predicted_SCTID,
             Top_Predicted_FSN, Dense_Rank, BM25_Rank
    """
    max_k = max(cfg.recall_k_values)
    failures = []

    for i, (_, row) in enumerate(val_df.iterrows()):
        gold = row["concept_id"]
        candidates = [c[0] for c in ranked_results[i]]

        if gold not in candidates[:max_k]:
            top_pred_sctid = candidates[0] if candidates else None
            top_pred_fsn = sctid_to_fsn.get(top_pred_sctid, "N/A") if top_pred_sctid else "N/A"

            dr = find_dense_rank(gold, dense_results_all[i])
            sr = find_sparse_rank(gold, sparse_results_all[i])

            failures.append({
                "Mention": row["span"],
                "Context": row["context"][:300],  # truncate for readability
                "Gold_SCTID": gold,
                "Gold_FSN": sctid_to_fsn.get(gold, "NOT IN INDEX"),
                "Top_Predicted_SCTID": top_pred_sctid,
                "Top_Predicted_FSN": top_pred_fsn,
                "Dense_Rank": dr if dr is not None else "NOT_FOUND",
                "BM25_Rank": sr if sr is not None else "NOT_FOUND",
            })

    df = pd.DataFrame(failures)
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_path = cfg.output_dir / "retrieval_failures.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFailure analysis: {len(df):,} cases where gold not in top-{max_k}")
    print(f"  Saved to {out_path}")
    return df


def print_report(
    recall_pre: dict[int, float],
    recall_post: dict[int, float],
    n_queries: int,
    n_failures: int,
) -> str:
    """Print and return a formatted evaluation report."""
    lines = [
        "",
        "=" * 65,
        "  RECALL@K ANALYSIS REPORT",
        "=" * 65,
        f"  Total validation queries: {n_queries:,}",
        "",
        "  --- Pre-Reranking (Hybrid Retrieval: Dense + BM25 + RRF) ---",
    ]
    for k, v in sorted(recall_pre.items()):
        bar = "█" * int(v * 40)
        lines.append(f"    R@{k:<4d} {v:.4f}  ({v*100:.1f}%)  {bar}")

    lines.append("")
    lines.append("  --- Post-Reranking (MedCPT Cross-Encoder) ---")
    for k, v in sorted(recall_post.items()):
        bar = "█" * int(v * 40)
        lines.append(f"    R@{k:<4d} {v:.4f}  ({v*100:.1f}%)  {bar}")

    lines.append("")
    lines.append(f"  Retrieval failures (not in top-{max(recall_post.keys())}): "
                  f"{n_failures:,} ({n_failures/n_queries*100:.1f}%)")
    lines.append("=" * 65)
    lines.append("")

    report = "\n".join(lines)
    print(report)
    return report
