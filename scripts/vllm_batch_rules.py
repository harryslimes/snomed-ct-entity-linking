#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL = "medgemma-1.5-4b-it-nvfp4"
DEFAULT_NVFP4_QUANTIZATION = "modelopt_fp4"

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "rule_based_instructions": {"type": "array", "items": {"type": "string"}},
        "ignorance_concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept_id": {"type": "integer"},
                    "term": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["concept_id", "evidence"],
                "additionalProperties": True,
            },
        },
        "error_patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "evidence": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                },
                "required": ["pattern", "evidence", "suggested_fix"],
                "additionalProperties": True,
            },
        },
    },
    "required": ["rule_based_instructions", "ignorance_concepts", "error_patterns"],
    "additionalProperties": True,
}


@dataclass(frozen=True)
class PromptBundle:
    entry: str | None
    note_id: str
    messages: list[dict]
    source: dict


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compact_record(rec: dict, *, max_concepts: int, max_errors: int) -> dict:
    # Keep only what's relevant for diagnosis and future matching.
    out = {
        "note_id": rec.get("note_id"),
        "summary": rec.get("summary", {}),
        "concepts": {
            "missing": (rec.get("concepts", {}).get("missing") or [])[:max_concepts],
            "extra": (rec.get("concepts", {}).get("extra") or [])[:max_concepts],
            "global_miss_in_note": (rec.get("concepts", {}).get("global_miss_in_note") or [])[:max_concepts],
        },
        "errors": {},
    }
    errs = rec.get("errors", {}) or {}
    for k in [
        "wrong_concept_overlap",
        "boundary_mismatch",
        "correct_concept_no_overlap",
        "missed_spans",
        "spurious_spans",
    ]:
        v = errs.get(k) or []
        out["errors"][k] = v[:max_errors]
    out["source"] = rec.get("source", {})
    return out


def build_messages(compact: dict) -> list[dict]:
    system = (
        "You are a clinical NLP error analyst. "
        "Your job is to generate rule-based correction instructions for an agent that will post-process outputs of an entity linking system. "
        "You must be concrete, testable, and avoid vague advice.\n\n"
        "Important constraints:\n"
        "- Only cite medical concepts (concept_id/term) that appear in the provided input.\n"
        "- Do NOT invent SNOMED IDs or terms.\n"
        "- If a mistake is due to model ignorance (concept never predicted anywhere), list the medical concepts that reflect that ignorance "
        "using the provided `global_miss_in_note` list only.\n"
    )

    user = {
        "task": (
            "Analyze this single-note diagnostic record and propose correction-agent rules.\n"
            "Focus on what is relevant to fix the observed errors in this note.\n"
        ),
        "input": compact,
        "output_format": {
            "rule_based_instructions": [
                "string, imperative, specific; include guardrails and examples when helpful; 5-15 items",
            ],
            "ignorance_concepts": [
                {
                    "concept_id": "integer",
                    "term": "string if present in input",
                    "evidence": "why this reflects ignorance (must cite global_miss_in_note / missed patterns)",
                }
            ],
            "error_patterns": [
                {
                    "pattern": "short name",
                    "evidence": "cite specific spans/concepts from input",
                    "suggested_fix": "what rule would address it",
                }
            ],
        },
        "return": "Return a single JSON object ONLY (no markdown, no extra text).",
    }

    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


def extract_json_object(text: str) -> dict | None:
    # Try strict first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find first {...} blob
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def main() -> int:
    # vLLM may spawn worker processes that import optional multimodal modules which
    # depend on OpenCV. In minimal containers, importing the real OpenCV wheel can
    # fail due to missing system libraries (e.g. libxcb). If this repo provides a
    # local `cv2` stub, ensure it's on PYTHONPATH for spawned processes.
    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / "cv2" / "__init__.py").exists():
        existing = os.environ.get("PYTHONPATH", "")
        prefix = str(repo_root)
        if not existing:
            os.environ["PYTHONPATH"] = prefix
        elif not existing.split(":")[0] == prefix and prefix not in existing.split(":"):
            os.environ["PYTHONPATH"] = f"{prefix}:{existing}"

    parser = argparse.ArgumentParser(
        description="Batch-run a vLLM model over llm_note_diagnoser JSONL and produce rule-based correction guidance."
    )
    parser.add_argument(
        "--entry",
        choices=["1st", "2nd", "custom"],
        default="",
        help=(
            "Optional tag used only to choose default input/output filenames. "
            "If set and you did not explicitly pass --input/--output, the defaults become "
            "outputs/llm_note_diagnosis_<entry>.jsonl and outputs/llm_rules_<entry>.jsonl."
        ),
    )
    default_input = "outputs/llm_note_diagnosis.jsonl"
    default_output = "outputs/llm_rules.jsonl"
    parser.add_argument(
        "--input",
        default=default_input,
        help="Input JSONL from scripts/llm_note_diagnoser.py",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Output JSONL",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model id (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--quantization",
        default="",
        help=(
            "vLLM quantization override. If unset and the model name contains 'nvfp4', "
            f"this script sets it to '{DEFAULT_NVFP4_QUANTIZATION}'."
        ),
    )
    parser.add_argument("--dtype", default="auto", help="vLLM dtype (default: auto)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size (default: 1)")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override max model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU memory utilization (default: 0.90)")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs/compilation and run eager mode (faster startup, potentially lower throughput).",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow trust_remote_code when loading tokenizer/model")
    parser.add_argument("--batch-size", type=int, default=16, help="Prompts per batch (default: 16)")
    parser.add_argument("--max-concepts", type=int, default=40, help="Max concepts per list in prompt (default: 40)")
    parser.add_argument("--max-errors", type=int, default=10, help="Max items per error bucket in prompt (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (default: 0.9)")
    parser.add_argument("--max-tokens", type=int, default=800, help="Max tokens to generate (default: 800)")
    parser.add_argument("--note-limit", type=int, default=0, help="Only process first N notes (0=all)")
    parser.add_argument(
        "--structured-mode",
        choices=["off", "object", "schema"],
        default="object",
        help="Structured output mode (default: object).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not run vLLM; just render prompts for the first batch and write them to --output as JSONL.",
    )
    args = parser.parse_args()

    if args.entry:
        if args.input == default_input:
            args.input = f"outputs/llm_note_diagnosis_{args.entry}.jsonl"
        if args.output == default_output:
            args.output = f"outputs/llm_rules_{args.entry}.jsonl"

    resolved_quantization = args.quantization
    if not resolved_quantization and "nvfp4" in (args.model or "").lower():
        resolved_quantization = DEFAULT_NVFP4_QUANTIZATION

    inp = Path(args.input)
    if not inp.exists():
        if args.entry:
            raise FileNotFoundError(
                f"Input not found: {inp}. Generate it with "
                f"`python scripts/llm_note_diagnoser.py --entry {args.entry} --output {inp}` "
                "(after creating an eval JSON with the corresponding `scripts/evaluate_*.py`)."
            )
        raise FileNotFoundError(
            f"Input not found: {inp}. Generate it with `python scripts/llm_note_diagnoser.py --output {inp}`."
        )

    # Local imports to avoid hard dependency unless used.
    tokenizer = None
    has_chat_template = False
    if not args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        has_chat_template = bool(
            hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)
        )

    llm = None
    sampling = None
    if not args.dry_run:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        llm_kwargs = {
            "model": args.model,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "trust_remote_code": args.trust_remote_code,
        }
        if resolved_quantization:
            llm_kwargs["quantization"] = resolved_quantization
        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
        if args.enforce_eager:
            llm_kwargs["enforce_eager"] = True

        llm = LLM(**llm_kwargs)
        if args.structured_mode == "schema":
            structured = StructuredOutputsParams(json=OUTPUT_SCHEMA, disable_fallback=False)
        elif args.structured_mode == "object":
            structured = StructuredOutputsParams(json_object=True, disable_fallback=False)
        else:
            structured = None
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            structured_outputs=structured,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def render_prompt(messages: list[dict]) -> str:
        # Prefer the tokenizer's chat template when available.
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Fallback: Llama 3 ChatML-style prompt.
        # This is required for some community Llama 3 checkpoints whose tokenizers
        # do not ship a `chat_template`, and vLLM refuses to use a default template.
        parts = ["<|begin_of_text|>"]
        for m in messages:
            role = (m.get("role") or "user").strip()
            content = m.get("content") or ""
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)

    bundles: list[PromptBundle] = []
    written = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, rec in enumerate(load_jsonl(inp), start=1):
            if args.note_limit and i > args.note_limit:
                break
            note_id = rec.get("note_id") or f"note_{i}"
            compact = compact_record(rec, max_concepts=args.max_concepts, max_errors=args.max_errors)
            messages = build_messages(compact)
            entry = rec.get("entry")
            bundles.append(
                PromptBundle(
                    entry=entry if isinstance(entry, str) else None,
                    note_id=note_id,
                    messages=messages,
                    source=rec.get("source", {}),
                )
            )

            if len(bundles) < args.batch_size:
                continue

            if args.dry_run:
                for b in bundles:
                    out_f.write(
                        json.dumps(
                            {
                                "entry": b.entry,
                                "note_id": b.note_id,
                                "model": args.model,
                                "quantization": resolved_quantization,
                                "prompt": render_prompt(b.messages),
                                "source": b.source,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1
                print(f"[dry-run] wrote {written} prompts to: {out_path}")
                return 0

            if has_chat_template:
                message_batches = [b.messages for b in bundles]
                outputs = llm.chat(message_batches, sampling_params=sampling)
            else:
                prompts = [render_prompt(b.messages) for b in bundles]
                outputs = llm.generate(prompts, sampling_params=sampling)
            for b, o in zip(bundles, outputs, strict=False):
                text = (o.outputs[0].text if o and o.outputs else "").strip()
                parsed = extract_json_object(text)
                out_rec = {
                    "entry": b.entry,
                    "note_id": b.note_id,
                    "model": args.model,
                    "quantization": resolved_quantization,
                    "response_text": text,
                    "response_json": parsed,
                    "source": b.source,
                }
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                written += 1
            bundles = []

        # Flush remainder
        if bundles and args.dry_run:
            for b in bundles:
                out_f.write(
                    json.dumps(
                        {
                            "entry": b.entry,
                            "note_id": b.note_id,
                            "model": args.model,
                            "quantization": resolved_quantization,
                            "prompt": render_prompt(b.messages),
                            "source": b.source,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
            print(f"[dry-run] wrote {written} prompts to: {out_path}")
            return 0

        if bundles:
            if has_chat_template:
                message_batches = [b.messages for b in bundles]
                outputs = llm.chat(message_batches, sampling_params=sampling)
            else:
                prompts = [render_prompt(b.messages) for b in bundles]
                outputs = llm.generate(prompts, sampling_params=sampling)
            for b, o in zip(bundles, outputs, strict=False):
                text = (o.outputs[0].text if o and o.outputs else "").strip()
                parsed = extract_json_object(text)
                out_rec = {
                    "entry": b.entry,
                    "note_id": b.note_id,
                    "model": args.model,
                    "quantization": resolved_quantization,
                    "response_text": text,
                    "response_json": parsed,
                    "source": b.source,
                }
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} records to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
