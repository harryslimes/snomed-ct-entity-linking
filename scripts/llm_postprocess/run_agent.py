#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from scripts.llm_postprocess.apply_edits import apply_edit_script_to_note_pred, _stable_note_spans
from scripts.llm_postprocess.prompting import PromptConfig, build_pred_rows_for_prompt, render_template
from scripts.llm_postprocess.schema import (
    AgentConstraints,
    EditScript,
    ScriptError,
    coerce_edit_script,
    json_schema_any_object,
    json_schema_delete_only,
    parse_script,
)
from scripts.llm_postprocess.sglang_runner import (
    SGLangServerConfig,
    backend_model_name,
    build_program,
    make_backend,
    run_batch_requests,
    shutdown_backend,
)


DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_notes(notes_csv: Path) -> pd.Series:
    df = pd.read_csv(notes_csv)
    if "note_id" not in df.columns or "text" not in df.columns:
        raise ValueError("notes csv must have columns: note_id,text")
    return df.set_index("note_id")["text"]


def run_agent_over_predictions(
    *,
    pred: pd.DataFrame,
    notes: pd.Series,
    template: str,
    out_jsonl: Path,
    out_pred_csv: Path | None,
    cache_base_dir: Path,
    backend,
    program,
    system_prompt: str,
    prompt_cfg: PromptConfig,
    constraints: AgentConstraints,
    batch_size: int,
    json_schema: str | None,
    max_new_tokens: int = 512,
    delete_only: bool = False,
    dry_run: bool = False,
) -> str:
    prompt_hash = _sha256_text(template)
    cache_dir = cache_base_dir / prompt_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    note_ids = list(pred["note_id"].astype(str).unique())

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    edited_notes: list[pd.DataFrame] = []

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        batch = []
        batch_note_ids: list[str] = []
        batch_pred_notes: list[pd.DataFrame] = []

        def _flush():
            nonlocal batch, batch_note_ids, batch_pred_notes, edited_notes
            if not batch:
                return
            if dry_run:
                rets = [{"raw": ""} for _ in batch]
            else:
                rets = run_batch_requests(
                    backend=backend,
                    program=program,
                    batch=batch,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    json_schema=json_schema,
                )
            for note_id, pred_note, ret in zip(batch_note_ids, batch_pred_notes, rets, strict=True):
                raw = ret.get("raw", "") if isinstance(ret, dict) else str(ret)
                parsed = None
                errors: list[str] = []
                warnings: list[str] = []
                try:
                    script_obj = parse_script(raw)
                    script = coerce_edit_script(script_obj)
                    if delete_only and script.edits:
                        script = EditScript(
                            note_id=script.note_id,
                            edits=tuple(e for e in script.edits if e.op == "delete"),
                        )
                    if len(script.edits) > constraints.max_edits_per_note:
                        script = EditScript(
                            note_id=script.note_id,
                            edits=script.edits[: constraints.max_edits_per_note],
                        )
                    parsed = {"note_id": script.note_id, "edits": [e.__dict__ for e in script.edits]}
                except (ScriptError, Exception) as e:
                    errors.append(str(e))
                    script = None

                cache_file = cache_dir / f"{note_id}.json"
                record = {
                    "note_id": note_id,
                    "prompt_hash": prompt_hash,
                    "model": backend_model_name(backend),
                    "raw": raw,
                    "parsed": parsed,
                    "errors": errors,
                    "warnings": warnings,
                }
                cache_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                if out_pred_csv is not None:
                    note_text = str(notes.get(note_id, ""))
                    note_len = len(note_text) if note_text else None
                    if script is None:
                        script = EditScript(note_id=note_id, edits=())
                    edited_note, _ = apply_edit_script_to_note_pred(
                        note_id=note_id,
                        pred_note=pred_note,
                        script=script,
                        note_len=note_len,
                        constraints=constraints,
                    )
                    edited_notes.append(edited_note)

            batch = []
            batch_note_ids = []
            batch_pred_notes = []

        for note_id in note_ids:
            cache_file = cache_dir / f"{note_id}.json"
            if cache_file.exists():
                record = json.loads(cache_file.read_text(encoding="utf-8"))
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                if out_pred_csv is not None:
                    pred_note = pred[pred["note_id"].astype(str) == note_id]
                    parsed = record.get("parsed")
                    if isinstance(parsed, dict):
                        try:
                            script = coerce_edit_script(parsed)
                        except Exception:
                            script = EditScript(note_id=note_id, edits=())
                    else:
                        script = EditScript(note_id=note_id, edits=())
                    note_text = str(notes.get(note_id, ""))
                    note_len = len(note_text) if note_text else None
                    edited_note, _ = apply_edit_script_to_note_pred(
                        note_id=note_id,
                        pred_note=pred_note,
                        script=script,
                        note_len=note_len,
                        constraints=constraints,
                    )
                    edited_notes.append(edited_note)
                continue

            note_text = str(notes.get(note_id, ""))
            pred_note = pred[pred["note_id"].astype(str) == note_id]
            pred_note_sorted = _stable_note_spans(pred_note)

            pred_rows = build_pred_rows_for_prompt(
                note_text=note_text,
                pred_note_sorted=pred_note_sorted,
                cfg=prompt_cfg,
            )

            user_prompt = render_template(
                template,
                {
                    "NOTE_ID": note_id,
                    "NOTE_LEN": str(len(note_text)),
                    "MAX_EDITS": str(constraints.max_edits_per_note),
                    "MAX_SHIFT_CHARS": str(constraints.max_shift_chars),
                    "PRED_ROWS_JSON": pred_rows,
                },
            )

            if dry_run:
                cache_file.write_text(
                    json.dumps(
                        {
                            "note_id": note_id,
                            "prompt_hash": prompt_hash,
                            "raw": "",
                            "parsed": None,
                            "errors": ["dry_run"],
                            "warnings": [],
                            "prompt_preview": user_prompt[:2000],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                out_f.write(cache_file.read_text(encoding="utf-8") + "\n")
                continue

            batch.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
            batch_note_ids.append(note_id)
            batch_pred_notes.append(pred_note_sorted)
            if len(batch) >= batch_size:
                _flush()
        _flush()

    if out_pred_csv is not None:
        out_pred = pd.concat(edited_notes, ignore_index=True) if edited_notes else pred.iloc[:0].copy()
        out_pred_csv.parent.mkdir(parents=True, exist_ok=True)
        out_pred.to_csv(out_pred_csv, index=False)

    return prompt_hash


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Run constrained LLM post-process agent over KIRI predictions.")
    ap.add_argument("--pred-csv", required=True, help="Input KIRI *_pred.csv.")
    ap.add_argument(
        "--notes-csv",
        default="1st Place/data/raw/mimic-iv_notes_training_set.csv",
        help="Notes CSV with columns (note_id,text).",
    )
    ap.add_argument("--prompt", default="scripts/llm_postprocess/prompts/base.md", help="Prompt template file.")
    ap.add_argument("--out-jsonl", required=True, help="Output JSONL with raw + parsed scripts per note.")
    ap.add_argument("--out-pred-csv", default=None, help="Optional: write edited predictions CSV.")
    ap.add_argument("--cache-dir", default="outputs/llm_postprocess_cache", help="Cache dir keyed by note/prompt hash.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of notes (debug).")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for run_batch.")
    ap.add_argument("--context-chars", type=int, default=64, help="Chars of context around each span.")
    ap.add_argument("--max-spans", type=int, default=80, help="Max spans included in a single prompt.")
    ap.add_argument(
        "--suspicious-only",
        action="store_true",
        help="Only include suspicious spans in the prompt (faster iteration). Indices remain stable.",
    )
    ap.add_argument("--max-edits", type=int, default=10)
    ap.add_argument("--max-shift", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate per note.")
    ap.add_argument("--system", default="You are a careful medical annotation editor.", help="System prompt.")
    ap.add_argument("--backend", choices=["vllm", "sglang"], default="vllm")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF model id used for generation.")
    ap.add_argument(
        "--endpoint-url",
        default=None,
        help="OpenAI-compatible base URL. Required for --backend vllm, optional for --backend sglang.",
    )
    ap.add_argument("--sglang-url", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--api-key", default=None, help="Optional API key for endpoint authentication.")
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
    ap.add_argument("--no-json-schema", action="store_true", help="Disable JSON-schema constrained decoding.")
    ap.add_argument("--delete-only", action="store_true", help="Only allow delete edits; ignore shifts.")
    ap.add_argument("--dry-run", action="store_true", help="Render prompts and exit without calling the model.")
    args = ap.parse_args(argv)

    pred = pd.read_csv(args.pred_csv)
    notes = _load_notes(Path(args.notes_csv))

    template = _read_text(Path(args.prompt))
    cfg = PromptConfig(
        context_chars=args.context_chars,
        max_spans_in_prompt=args.max_spans,
        suspicious_only=bool(args.suspicious_only),
    )
    constraints = AgentConstraints(
        max_edits_per_note=args.max_edits,
        max_shift_chars=args.max_shift,
    )

    if args.no_json_schema:
        json_schema = None
    elif args.delete_only:
        json_schema = json_schema_delete_only(max_edits=int(args.max_edits))
    else:
        json_schema = json_schema_any_object(max_edits=int(args.max_edits))
    endpoint_url = args.endpoint_url or args.sglang_url
    program = build_program(json_schema=json_schema) if args.backend == "sglang" else None

    backend = None
    t0 = time.perf_counter()
    try:
        if not args.dry_run:
            backend = make_backend(
                sglang_url=endpoint_url,
                api_key=args.api_key,
                server=SGLangServerConfig(
                    model_path=args.model,
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

        model_key = (args.model or "model").replace("/", "__").replace(":", "_")
        if endpoint_url:
            model_key = ("endpoint_" + endpoint_url).replace("/", "_").replace(":", "_")

        if args.limit is not None:
            keep = set(pred["note_id"].astype(str).unique().tolist()[: max(0, int(args.limit))])
            pred = pred[pred["note_id"].astype(str).isin(keep)].copy()

        prompt_hash = run_agent_over_predictions(
            pred=pred,
            notes=notes,
            template=template,
            out_jsonl=Path(args.out_jsonl),
            out_pred_csv=Path(args.out_pred_csv) if args.out_pred_csv is not None else None,
            cache_base_dir=Path(args.cache_dir) / model_key,
            backend=backend,
            program=program,
            system_prompt=args.system,
            prompt_cfg=cfg,
            constraints=constraints,
            batch_size=args.batch_size,
            json_schema=json_schema,
            max_new_tokens=int(args.max_new_tokens),
            delete_only=bool(args.delete_only),
            dry_run=args.dry_run,
        )

    finally:
        shutdown_backend(backend)

    dt = time.perf_counter() - t0
    print(f"Done in {dt:0.1f}s. Prompt hash: {prompt_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
