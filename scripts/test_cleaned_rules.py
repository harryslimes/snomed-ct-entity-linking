#!/usr/bin/env python3
"""Evaluate a rules file against the saved holdout pairs from an abbrev_rule_loop run.

Usage:
    python scripts/test_cleaned_rules.py \
        --run-dir rulebook/runs/20260219T171514Z \
        --rules super-dictionary/abbrev_rules_cleaned.json \
        --vllm-url http://localhost:8000 \
        --index-server http://127.0.0.1:8421

Compare multiple rule files by running twice and comparing the printed accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

# Import the loop's pipeline functions directly
from abbrev_rule_loop import (  # noqa: E402
    _run_and_evaluate,
    build_reps_for_batch,
    extract_abbreviation_items,
    load_state,
)
import abbrev_rule_loop as arl_mod  # noqa: E402

import pandas as pd  # noqa: E402

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
DEFAULT_MODEL = "/workspaces/snomed-ct-entity-linking/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rules on holdout pairs")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run dir with state.json")
    parser.add_argument("--rules", type=Path, required=True, help="Rules JSON file to evaluate")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--index-server", default="http://127.0.0.1:8421")
    parser.add_argument("--context-before", type=int, default=200)
    parser.add_argument("--context-after", type=int, default=100)
    parser.add_argument("--reasoning-effort", default="none",
                        choices=["none", "low", "medium", "high"])
    args = parser.parse_args()

    # Pre-flight: verify vLLM server is reachable
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(f"{args.vllm_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.load(resp)
            model_ids = [m["id"] for m in data.get("data", [])]
            print(f"vLLM server OK at {args.vllm_url}  model(s): {model_ids}")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"ERROR: Cannot reach vLLM server at {args.vllm_url}")
        print(f"  {e}")
        print("  Start the vLLM server first, then re-run this script.")
        sys.exit(1)

    # Pre-flight: verify SNOMED index server is reachable
    try:
        req = urllib.request.Request(f"{args.index_server}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5):
            print(f"Index server OK at {args.index_server}")
    except (urllib.error.URLError, OSError) as e:
        print(f"ERROR: Cannot reach SNOMED index server at {args.index_server}")
        print(f"  {e}")
        print("  Start snomed_index_server.py first, then re-run this script.")
        sys.exit(1)

    # Wire index server
    arl_mod.rt._index_server_url = args.index_server

    # Load state for holdout pairs
    run_dir = args.run_dir.resolve()
    batch_idx, _state_rules, holdout_pairs, holdout_history = load_state(run_dir)
    if not holdout_pairs:
        print("ERROR: No holdout pairs in state.json — cannot evaluate.")
        sys.exit(1)
    print(f"Loaded {len(holdout_pairs)} holdout pairs from {run_dir.name}")

    # Load rules to test
    with open(args.rules) as f:
        rules = json.load(f)
    print(f"Testing {len(rules)} rules from {args.rules}")

    # Load data
    print(f"\nLoading notes and annotations from {SPLIT_DIR} ...")
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    ann_df = pd.read_csv(SPLIT_DIR / "train_annotations.csv")
    all_items = extract_abbreviation_items(ann_df, notes_df, args.context_before, args.context_after)
    print(f"  {len(all_items):,} abbreviation instances extracted")

    # Load concept names
    print("\nLoading concept names ...")
    concept_names = arl_mod.rt.load_concept_names()

    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Evaluating on {len(holdout_pairs)} holdout pairs ...")
    print(f"{'='*60}")

    summary, failures, *_ = _run_and_evaluate(
        all_items,
        [tuple(p) for p in holdout_pairs],
        rules,
        concept_names,
        args.vllm_url,
        args.model,
        args.max_concurrent,
        args.reasoning_effort,
        retry=False,  # stable metric — no retry for holdout
    )

    print(f"\n{'='*60}")
    print(f"HOLDOUT RESULT: {100 * summary['accuracy']:.1f}%  ({summary['n_correct']}/{summary['n_total']})")
    print(f"{'='*60}")

    if holdout_history:
        hist_accs = [f"{100 * h['accuracy']:.1f}%" for h in holdout_history[-5:]]
        print(f"Loop history (last 5): {' → '.join(hist_accs)}")


if __name__ == "__main__":
    main()
