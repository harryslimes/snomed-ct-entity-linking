#!/usr/bin/env python3
"""Sweep context window sizes to measure impact on rule testing performance.

Runs rule_testing with multiple (context_before, context_after) configurations
in a single process, reusing the loaded retrieval index across runs.

Usage:
    python rulebook/sweep_context.py --rules rulebook/rules/20260217T134141Z/rules.json
    python rulebook/sweep_context.py --rules rulebook/rules/20260217T134141Z/rules.json --note 18568215-DS-21
    python rulebook/sweep_context.py --configs rulebook/context_sweep_configs.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import rulebook.rule_testing as rt

# Default sweep configurations: (label, context_before, context_after)
DEFAULT_SWEEP = [
    ("300/100 (baseline)", 300, 100),
    ("200/75",             200,  75),
    ("100/50",             100,  50),
    ("50/25",               50,  25),
    ("25/10",               25,  10),
    ("0/0 (span only)",      0,   0),
]


def run_sweep(
    sweep_configs: list[tuple[str, int, int]],
    rules_path: str,
    note_id: str | None,
    backend: str,
    vllm_url: str,
    vllm_model: str,
    concurrency: int,
    reasoning_effort: str,
) -> None:
    results: list[dict] = []

    print("=" * 70)
    print("CONTEXT WINDOW SWEEP")
    print(f"Rules: {rules_path}")
    print(f"Backend: {backend}  |  Model: {vllm_model}")
    print(f"Configs to test: {len(sweep_configs)}")
    print("=" * 70)

    for label, ctx_before, ctx_after in sweep_configs:
        print(f"\n{'─' * 70}")
        print(f"Config: {label}  (context_before={ctx_before}, context_after={ctx_after})")
        print(f"{'─' * 70}")

        args = SimpleNamespace(
            backend=backend,
            vllm_url=vllm_url,
            vllm_model=vllm_model,
            concurrency=concurrency,
            reasoning_effort=reasoning_effort,
            rules=rules_path,
            note=note_id,
            context_before=ctx_before,
            context_after=ctx_after,
        )

        t0 = time.time()
        try:
            rt.run(args)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "label": label,
                "context_before": ctx_before,
                "context_after": ctx_after,
                "error": str(e),
            })
            continue
        elapsed = time.time() - t0

        # Pull the last saved results file from the rules dir
        rules_dir = Path(rules_path).parent
        result_files = sorted(rules_dir.glob(f"test_results_{backend}_*.json"))
        if result_files:
            with open(result_files[-1]) as f:
                saved = json.load(f)
            results.append({
                "label": label,
                "context_before": ctx_before,
                "context_after": ctx_after,
                "macro_char_iou": saved.get("macro_char_iou", 0.0),
                "concept_accuracy": saved.get("concept_accuracy", 0.0),
                "n_concept_matches": saved.get("n_concept_matches", 0),
                "n_annotations": saved.get("n_annotations", 0),
                "total_time_s": saved.get("total_time_s", elapsed),
            })
        else:
            results.append({
                "label": label,
                "context_before": ctx_before,
                "context_after": ctx_after,
                "total_time_s": elapsed,
            })

    # --- Comparison table ---
    print("\n" + "=" * 70)
    print("SWEEP RESULTS")
    print("=" * 70)
    header = f"{'Config':<22} {'Ctx B':>6} {'Ctx A':>6} {'IoU':>8} {'Acc%':>7} {'Matches':>9} {'Time(s)':>9}"
    print(header)
    print("─" * 70)
    for r in results:
        if "error" in r:
            print(f"  {r['label']:<20} ERROR: {r['error']}")
            continue
        print(
            f"  {r['label']:<20} "
            f"{r['context_before']:>6} "
            f"{r['context_after']:>6} "
            f"{r.get('macro_char_iou', 0):>8.4f} "
            f"{100 * r.get('concept_accuracy', 0):>6.1f}% "
            f"{r.get('n_concept_matches', 0):>4}/{r.get('n_annotations', 0):<4} "
            f"{r.get('total_time_s', 0):>8.1f}s"
        )

    # Save sweep summary
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(rules_path).parent / f"sweep_context_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({"sweep_configs": [
            {"label": l, "context_before": b, "context_after": a}
            for l, b, a in sweep_configs
        ], "results": results}, f, indent=2)
    print(f"\nSweep summary saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep context window sizes for rule testing")
    parser.add_argument(
        "--rules", type=str, default=None, metavar="PATH",
        help="Path to rules JSON file",
    )
    parser.add_argument(
        "--note", type=str, default=None, metavar="NOTE_ID",
        help="Note ID to test (default: first note in train_notes.csv)",
    )
    parser.add_argument(
        "--configs", type=str, default=None, metavar="PATH",
        help="JSON file with custom sweep configs: [{\"label\": ..., \"context_before\": ..., \"context_after\": ...}]",
    )
    parser.add_argument(
        "--backend", choices=["sonnet", "vllm"], default="vllm",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--vllm-model", default="openai/gpt-oss-20b")
    parser.add_argument("--concurrency", type=int, default=0)
    parser.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default="low",
    )
    args = parser.parse_args()

    if args.configs:
        with open(args.configs) as f:
            raw = json.load(f)
        sweep_configs = [(c["label"], c["context_before"], c["context_after"]) for c in raw]
    else:
        sweep_configs = DEFAULT_SWEEP

    if args.rules is None:
        # Default to the latest rules dir
        rules_root = REPO_ROOT / "rulebook" / "rules"
        dirs = sorted(rules_root.iterdir())
        if not dirs:
            print("No rules directories found under rulebook/rules/")
            sys.exit(1)
        args.rules = str(dirs[-1] / "rules.json")
        print(f"Defaulting to rules: {args.rules}")

    run_sweep(
        sweep_configs=sweep_configs,
        rules_path=args.rules,
        note_id=args.note,
        backend=args.backend,
        vllm_url=args.vllm_url,
        vllm_model=args.vllm_model,
        concurrency=args.concurrency,
        reasoning_effort=args.reasoning_effort,
    )


if __name__ == "__main__":
    main()
