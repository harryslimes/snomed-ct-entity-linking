#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.llm_postprocess.prompting import PromptConfig
from scripts.llm_postprocess.run_agent import _load_notes, run_agent_over_predictions
from scripts.llm_postprocess.schema import AgentConstraints, json_schema_any_object
from scripts.llm_postprocess.sglang_runner import (
    SGLangServerConfig,
    build_program,
    make_backend,
    shutdown_backend,
)
from scripts.super_dictionary.runtime_scoring import class_char_iou


def _note_iou(pred: pd.DataFrame, gold: pd.DataFrame) -> float:
    from note_scoring import iou_per_note

    ious = iou_per_note(pred[["note_id", "concept_id"]], gold[["note_id", "concept_id"]])
    return float(np.mean(list(ious.values()))) if ious else 0.0


def _macro_char_iou(pred: pd.DataFrame, gold: pd.DataFrame) -> float:
    cls = class_char_iou(
        pred[["note_id", "start", "end", "concept_id"]],
        gold[["note_id", "start", "end", "concept_id"]],
    )
    return float(cls.loc[cls["union"] > 0, "iou"].mean())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Evaluate (and optionally search over) prompt variants.")
    ap.add_argument("--pred-csv", required=True, help="Base KIRI *_pred.csv to edit.")
    ap.add_argument(
        "--gold-csv",
        default="1st Place/data/interim/train_annotations_cln.csv",
        help="Gold annotations CSV (note_id,start,end,concept_id).",
    )
    ap.add_argument(
        "--notes-csv",
        default="1st Place/data/raw/mimic-iv_notes_training_set.csv",
        help="Notes CSV (note_id,text).",
    )
    ap.add_argument("--prompt-dir", default="scripts/llm_postprocess/prompts", help="Directory of prompt .md files.")
    ap.add_argument("--out-dir", default="outputs/llm_postprocess_prompt_eval", help="Output directory.")
    ap.add_argument("--val-size", type=int, default=64, help="Validation note count (random).")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--backend", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", help="HF model id used for generation.")
    ap.add_argument("--endpoint-url", default=None, help="OpenAI-compatible endpoint URL.")
    ap.add_argument("--sglang-url", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--pp-size", type=int, default=1)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--mem-fraction-static", type=float, default=None)
    ap.add_argument("--max-running-requests", type=int, default=None)
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--server-extra-args-json", default="{}")
    ap.add_argument("--request-timeout-s", type=float, default=120.0)
    ap.add_argument("--max-parallel-http", type=int, default=16)
    ap.add_argument("--cache-dir", default="outputs/llm_postprocess_cache", help="Shared cache base dir.")
    args = ap.parse_args(argv)

    pred = pd.read_csv(args.pred_csv)
    gold = pd.read_csv(args.gold_csv)
    notes = _load_notes(Path(args.notes_csv))

    note_ids = pred["note_id"].astype(str).unique().tolist()
    rng = np.random.default_rng(args.seed)
    rng.shuffle(note_ids)
    val_ids = set(note_ids[: max(0, min(int(args.val_size), len(note_ids)))])

    pred_val = pred[pred["note_id"].astype(str).isin(val_ids)].copy()
    gold_val = gold[gold["note_id"].astype(str).isin(val_ids)].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_paths = sorted(Path(args.prompt_dir).glob("*.md"))
    if not prompt_paths:
        raise SystemExit(f"No prompts found in {args.prompt_dir}")

    backend = None
    program = build_program(json_schema=json_schema_any_object()) if args.backend == "sglang" else None
    endpoint_url = args.endpoint_url or args.sglang_url
    results = []
    try:
        model_key = (args.model or "model").replace("/", "__").replace(":", "_")
        if endpoint_url:
            model_key = ("endpoint_" + endpoint_url).replace("/", "_").replace(":", "_")

        backend = make_backend(
            sglang_url=endpoint_url,
            api_key=args.api_key,
            server=SGLangServerConfig(
                model_path=args.model or "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                backend=args.backend,
                tp_size=args.tp_size,
                pp_size=args.pp_size,
                dtype=args.dtype,
                quantization=args.quantization,
                trust_remote_code=args.trust_remote_code,
                mem_fraction_static=args.mem_fraction_static,
                max_running_requests=args.max_running_requests,
                host=args.host,
                port=args.port,
                extra_args_json=args.server_extra_args_json,
                request_timeout_s=args.request_timeout_s,
                max_parallel_http=args.max_parallel_http,
            ),
        )

        constraints = AgentConstraints(max_edits_per_note=10, max_shift_chars=30)
        prompt_cfg = PromptConfig(context_chars=96, max_spans_in_prompt=200)

        for prompt_path in prompt_paths:
            label = prompt_path.stem
            run_dir = out_dir / label
            run_dir.mkdir(parents=True, exist_ok=True)

            template = prompt_path.read_text(encoding="utf-8")
            out_jsonl = run_dir / "scripts.jsonl"
            out_pred = run_dir / "pred_out.csv"

            t0 = time.perf_counter()
            prompt_hash = run_agent_over_predictions(
                pred=pred_val,
                notes=notes,
                template=template,
                out_jsonl=out_jsonl,
                out_pred_csv=out_pred,
                cache_base_dir=Path(args.cache_dir) / model_key,
                backend=backend,
                program=program,
                system_prompt="You are a careful medical annotation editor.",
                prompt_cfg=prompt_cfg,
                constraints=constraints,
                batch_size=16,
                json_schema=json_schema_any_object(),
                max_new_tokens=512,
            )
            dt = time.perf_counter() - t0

            pred_out = pd.read_csv(out_pred)
            macro_iou = _macro_char_iou(pred_out, gold_val)
            note_iou = _note_iou(pred_out, gold_val)
            results.append(
                {
                    "prompt": label,
                    "prompt_hash": prompt_hash,
                    "macro_char_iou": macro_iou,
                    "note_iou": note_iou,
                    "seconds": dt,
                    "out_dir": str(run_dir),
                }
            )
    finally:
        shutdown_backend(backend)

    df = pd.DataFrame(results).sort_values(["macro_char_iou", "note_iou"], ascending=False)
    df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "results.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
