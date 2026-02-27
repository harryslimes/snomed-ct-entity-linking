#!/usr/bin/env python3
"""Embed 366 chunk-level extraction rules and cluster with UMAP + HDBSCAN.

Produces:
  1. A 2D UMAP scatter plot (PNG) coloured by cluster
  2. A CSV with rule ID, cluster label, rule text, span, rule type
  3. Per-cluster summaries printed to stdout
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def get_span(rule_text: str) -> str:
    m = re.search(r"""['"](.+?)['"]""", rule_text)
    return m.group(1) if m else ""


def get_rule_type(rule_text: str) -> str:
    low = rule_text.strip().lower()
    if low.startswith("extract"):
        return "extract"
    elif low.startswith("skip"):
        return "skip"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=Path("rulebook/chunk_rules_largescale.json"),
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name",
    )
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/chunk_rule_runs/rule_clusters"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load rules
    data = json.load(open(args.input))
    rules = data["rules"]
    print(f"Loaded {len(rules)} rules from {args.input}")

    # Extract metadata
    rule_texts = [r["rule"] for r in rules]
    rule_ids = [r["id"] for r in rules]
    spans = [get_span(r["rule"]) for r in rules]
    types = [get_rule_type(r["rule"]) for r in rules]

    print(f"  Extract rules: {types.count('extract')}")
    print(f"  Skip rules: {types.count('skip')}")
    print(f"  Unique spans: {len(set(spans))}")

    # Embed
    print(f"\nEmbedding with {args.model}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    embeddings = model.encode(rule_texts, show_progress_bar=True, normalize_embeddings=True)
    print(f"  Embeddings shape: {embeddings.shape}")

    # UMAP
    print(f"\nRunning UMAP (n_neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})...")
    import umap
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    print(f"  UMAP coords shape: {coords.shape}")

    # HDBSCAN clustering
    print(f"\nClustering with HDBSCAN (min_cluster_size={args.min_cluster_size})...")
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Clusters: {n_clusters}, Noise points: {n_noise}")

    # Save CSV
    csv_path = args.output_dir / "rule_clusters.csv"
    with open(csv_path, "w") as f:
        f.write("rule_id,cluster,umap_x,umap_y,span,type,rule_text\n")
        for i, r in enumerate(rules):
            # Escape CSV
            text_escaped = r["rule"].replace('"', '""')
            f.write(
                f'{rule_ids[i]},{labels[i]},{coords[i,0]:.4f},{coords[i,1]:.4f},'
                f'"{spans[i]}",{types[i]},"{text_escaped}"\n'
            )
    print(f"\nSaved CSV: {csv_path}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Plot 1: Colored by cluster
    ax = axes[0]
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            ax.scatter(coords[mask, 0], coords[mask, 1], c="lightgray", s=20,
                      alpha=0.5, label=f"noise ({mask.sum()})")
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(label % 20)], s=30,
                      alpha=0.7, label=f"C{label} ({mask.sum()})")
    ax.set_title(f"UMAP + HDBSCAN: {n_clusters} clusters, {n_noise} noise")
    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    # Plot 2: Colored by rule type (extract vs skip)
    ax = axes[1]
    type_colors = {"extract": "tab:blue", "skip": "tab:red", "other": "tab:gray"}
    for t in ["extract", "skip", "other"]:
        mask = np.array([tp == t for tp in types])
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], c=type_colors[t], s=30,
                      alpha=0.6, label=f"{t} ({mask.sum()})")
    ax.set_title("UMAP colored by rule type (extract vs skip)")
    ax.legend(fontsize=8)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    plt.tight_layout()
    plot_path = args.output_dir / "rule_clusters.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {plot_path}")

    # Print cluster summaries
    print(f"\n{'='*80}")
    print("CLUSTER SUMMARIES")
    print(f"{'='*80}")

    cluster_rules = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_rules[label].append(i)

    for label in sorted(cluster_rules.keys()):
        indices = cluster_rules[label]
        cluster_spans = [spans[i] for i in indices]
        cluster_types = [types[i] for i in indices]

        type_counts = Counter(cluster_types)
        span_counts = Counter(cluster_spans)

        label_str = f"Cluster {label}" if label >= 0 else "NOISE"
        print(f"\n--- {label_str} ({len(indices)} rules) ---")
        print(f"  Types: {dict(type_counts)}")
        print(f"  Top spans: {span_counts.most_common(8)}")

        # Show 3 example rules
        for idx in indices[:3]:
            print(f"    [{rule_ids[idx]}] {rule_texts[idx][:120]}...")

    # Save cluster data as JSON for further processing
    cluster_data = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_rules": len(rules),
        "clusters": {}
    }
    for label in sorted(cluster_rules.keys()):
        indices = cluster_rules[label]
        cluster_data["clusters"][str(label)] = {
            "rule_ids": [rule_ids[i] for i in indices],
            "spans": list(set(spans[i] for i in indices)),
            "types": dict(Counter(types[i] for i in indices)),
            "size": len(indices),
        }
    json_path = args.output_dir / "cluster_summary.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import json as json_mod
    class NumpyEncoder(json_mod.JSONEncoder):
        def default(self, obj):
            v = convert(obj)
            if v is not obj:
                return v
            return super().default(obj)

    json_path.write_text(json.dumps(cluster_data, indent=2, cls=NumpyEncoder))
    print(f"\nSaved cluster JSON: {json_path}")


if __name__ == "__main__":
    main()
