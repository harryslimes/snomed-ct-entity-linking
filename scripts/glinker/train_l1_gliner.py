#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gliner import GLiNER  # type: ignore
from gliner.training.trainer import Trainer  # type: ignore

from scripts.glinker.run_l1_inference import normalize_l1_type


@dataclass
class BuildStats:
    notes_total: int = 0
    notes_with_entities: int = 0
    token_count_total: int = 0
    ann_rows_total: int = 0
    ann_rows_used: int = 0
    ann_rows_unknown_l1: int = 0
    ann_rows_unmapped_char_span: int = 0
    ann_rows_excluded_section: int = 0
    examples_total: int = 0
    examples_with_entities: int = 0


class StrictGLiNERTrainer(Trainer):
    """Fail fast on batch errors instead of silently returning zero loss."""

    def training_step(self, model, inputs, *args, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps


def _resolve_device(device: str) -> str:
    d = str(device).strip().lower()
    if d and d != "auto":
        return d
    try:
        import torch  # type: ignore

        if bool(torch.cuda.is_available()):
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_torch_dtype(dtype: str):
    raw = str(dtype).strip().lower()
    if raw in {"", "none", "off", "false", "0", "auto"}:
        return None
    import torch  # type: ignore

    if raw in {"float16", "fp16"}:
        return torch.float16
    if raw in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if raw in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported --load-dtype value: {dtype}")


def _load_model(model_path: str, *, map_location: str, attn_impl: str, load_dtype: str):
    kwargs: dict[str, Any] = {"map_location": map_location}
    attn = str(attn_impl).strip().lower()
    if attn not in {"", "auto", "default", "none"}:
        kwargs["_attn_implementation"] = attn

    resolved_dtype = _resolve_torch_dtype(load_dtype)
    if resolved_dtype is not None:
        kwargs["torch_dtype"] = resolved_dtype
        kwargs["dtype"] = resolved_dtype

    attempts: list[dict[str, Any]] = [dict(kwargs)]
    optional_keys = ["_attn_implementation", "torch_dtype", "dtype"]
    for key in optional_keys:
        if key in kwargs:
            k = dict(kwargs)
            k.pop(key, None)
            attempts.append(k)
    for i in range(len(optional_keys)):
        for j in range(i + 1, len(optional_keys)):
            k1, k2 = optional_keys[i], optional_keys[j]
            if k1 in kwargs and k2 in kwargs:
                k = dict(kwargs)
                k.pop(k1, None)
                k.pop(k2, None)
                attempts.append(k)
    if any(k in kwargs for k in optional_keys):
        k = dict(kwargs)
        for key in optional_keys:
            k.pop(key, None)
        attempts.append(k)

    uniq: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for a in attempts:
        key = tuple(sorted((str(k), str(v)) for k, v in a.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)

    last_err: Exception | None = None
    for a in uniq:
        try:
            return GLiNER.from_pretrained(model_path, **a)
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return GLiNER.from_pretrained(model_path, **kwargs)


def _load_l1_map(allowed_concepts_csv: Path) -> dict[str, str]:
    df = pd.read_csv(allowed_concepts_csv, usecols=["concept_id", "l1_type"])
    out: dict[str, str] = {}
    for cid, l1 in df[["concept_id", "l1_type"]].itertuples(index=False, name=None):
        key = str(cid).strip()
        val = normalize_l1_type(str(l1))
        if key and val is not None:
            out[key] = val
    return out


def _canonical_section(header: str) -> str:
    h = str(header or "").strip().lower().rstrip(":")
    if not h:
        return "unknown"
    if "history of present illness" in h or h == "hpi":
        return "hpi"
    if "past medical history" in h or h == "pmh":
        return "pmh"
    if "discharge medication" in h or "medications" in h:
        return "medications"
    if "hospital course" in h:
        return "hospital_course"
    if "physical exam" in h:
        return "physical_exam"
    if ("assessment" in h and "plan" in h) or h == "a/p":
        return "assessment_plan"
    if "diagnosis" in h:
        return "diagnosis"
    if "procedure" in h:
        return "procedures"
    if "allerg" in h:
        return "allergies"
    if "social history" in h:
        return "social_history"
    if "family history" in h:
        return "family_history"
    if "review of systems" in h or h == "ros":
        return "review_of_systems"
    if "lab" in h:
        return "labs"
    if "imaging" in h or "radiology" in h:
        return "imaging"
    return "other"


def _iter_section_headers(text: str) -> tuple[list[int], list[str]]:
    starts: list[int] = []
    labels: list[str] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        stripped = raw.strip()
        if stripped and len(stripped) <= 120:
            is_header = False
            if stripped.endswith(":"):
                is_header = True
            else:
                alpha = [ch for ch in stripped if ch.isalpha()]
                if alpha and all(ch.isupper() for ch in alpha):
                    is_header = True
            if is_header:
                starts.append(pos)
                labels.append(_canonical_section(stripped))
        pos += len(line)
    return starts, labels


def _section_for_offset(section_starts: list[int], section_labels: list[str], char_pos: int) -> str:
    if not section_starts:
        return "unknown"
    idx = bisect_right(section_starts, int(char_pos)) - 1
    if idx < 0:
        return "unknown"
    return section_labels[idx]


def _parse_excluded_sections(value: str) -> set[str]:
    out: set[str] = set()
    for part in str(value or "").split(","):
        s = str(part).strip().lower().replace("-", "_").replace(" ", "_")
        if s:
            out.add(s)
    return out


def _tokenize_with_offsets(words_splitter, text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    for tok, s, e in words_splitter(text):
        tokens.append(str(tok))
        offsets.append((int(s), int(e)))
    return tokens, offsets


def _char_to_token_span(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    hits = [i for i, (s, e) in enumerate(offsets) if s < end and e > start]
    if not hits:
        return None
    return (min(hits), max(hits))


def _window_examples(
    tokens: list[str],
    spans: list[tuple[int, int, str]],
    *,
    window_tokens: int,
    overlap_tokens: int,
    include_empty_windows: bool,
) -> list[dict[str, Any]]:
    if window_tokens <= 0 or len(tokens) <= window_tokens:
        return [{"tokenized_text": tokens, "ner": spans}]

    stride = max(1, int(window_tokens) - max(0, int(overlap_tokens)))
    out: list[dict[str, Any]] = []
    pos = 0
    n = len(tokens)
    while pos < n:
        end = min(n, pos + int(window_tokens))
        sub_tokens = tokens[pos:end]
        sub_spans = [(s - pos, e - pos, label) for s, e, label in spans if s >= pos and e < end]
        if include_empty_windows or sub_spans:
            out.append({"tokenized_text": sub_tokens, "ner": sub_spans})
        if end >= n:
            break
        pos += stride
    return out


def _build_examples(
    *,
    notes_csv: Path,
    annotations_csv: Path,
    l1_map: dict[str, str],
    words_splitter,
    window_tokens: int,
    overlap_tokens: int,
    include_empty_windows: bool,
    excluded_sections: set[str] | None = None,
) -> tuple[list[dict[str, Any]], BuildStats]:
    notes = pd.read_csv(notes_csv)
    ann = pd.read_csv(annotations_csv)
    ann["concept_id"] = ann["concept_id"].astype(str)

    ann_by_note: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for row in ann.itertuples(index=False):
        note_id = str(getattr(row, "note_id"))
        concept_id = str(getattr(row, "concept_id"))
        try:
            start = int(getattr(row, "start"))
            end = int(getattr(row, "end"))
        except Exception:
            continue
        ann_by_note[note_id].append((start, end, concept_id))

    stats = BuildStats()
    examples: list[dict[str, Any]] = []

    for row in notes.itertuples(index=False):
        note_id = str(getattr(row, "note_id"))
        text = str(getattr(row, "text"))
        stats.notes_total += 1

        tokens, offsets = _tokenize_with_offsets(words_splitter, text)
        stats.token_count_total += len(tokens)
        raw_spans = ann_by_note.get(note_id, [])
        stats.ann_rows_total += len(raw_spans)
        section_starts: list[int] = []
        section_labels: list[str] = []
        if excluded_sections:
            section_starts, section_labels = _iter_section_headers(text)

        mapped: list[tuple[int, int, str]] = []
        for start, end, concept_id in raw_spans:
            if excluded_sections:
                section_name = _section_for_offset(section_starts, section_labels, start)
                if section_name in excluded_sections:
                    stats.ann_rows_excluded_section += 1
                    continue
            l1 = l1_map.get(concept_id)
            if l1 is None:
                stats.ann_rows_unknown_l1 += 1
                continue
            tspan = _char_to_token_span(offsets, start, end)
            if tspan is None:
                stats.ann_rows_unmapped_char_span += 1
                continue
            mapped.append((tspan[0], tspan[1], l1))
            stats.ann_rows_used += 1

        if mapped:
            stats.notes_with_entities += 1

        dedup = sorted(set(mapped), key=lambda x: (x[0], x[1], x[2]))
        note_examples = _window_examples(
            tokens,
            dedup,
            window_tokens=window_tokens,
            overlap_tokens=overlap_tokens,
            include_empty_windows=include_empty_windows,
        )
        examples.extend(note_examples)

    stats.examples_total = len(examples)
    stats.examples_with_entities = sum(1 for ex in examples if ex.get("ner"))
    return examples, stats


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Fine-tune GLiNER L1 span detector from note/annotation CSVs."
    )
    ap.add_argument("--train-notes-csv", required=True)
    ap.add_argument("--train-annotations-csv", required=True)
    ap.add_argument("--val-notes-csv", required=True)
    ap.add_argument("--val-annotations-csv", required=True)
    ap.add_argument("--allowed-concepts-csv", default="data/interim/glinker/allowed_concepts.csv")
    ap.add_argument("--model-path", default="knowledgator/gliner-bi-large-v2.0")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--window-tokens", type=int, default=1800)
    ap.add_argument("--overlap-tokens", type=int, default=256)
    ap.add_argument("--include-empty-windows", action="store_true")
    ap.add_argument(
        "--exclude-sections-canonical",
        default="",
        help=(
            "Comma-separated canonical section names to exclude from TRAIN annotations only "
            "(e.g. medications,hospital_course)."
        ),
    )
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--max-width", type=int, default=12)

    ap.add_argument("--device", default="auto")
    ap.add_argument("--attn-impl", default="auto")
    ap.add_argument("--load-dtype", default="auto")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--train-batch-size", type=int, default=2)
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum-steps", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--logging-steps", type=int, default=20)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--early-stopping-patience", type=int, default=0)
    ap.add_argument("--early-stopping-threshold", type=float, default=0.0)
    ap.add_argument("--load-best-model-at-end", action="store_true")
    ap.add_argument("--seed", type=int, default=20260207)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    map_location = _resolve_device(args.device)
    model = _load_model(
        str(args.model_path),
        map_location=map_location,
        attn_impl=str(args.attn_impl),
        load_dtype=str(args.load_dtype),
    )
    model.config.max_len = int(args.max_len)
    model.config.max_width = int(args.max_width)

    l1_map = _load_l1_map(Path(args.allowed_concepts_csv))
    excluded_sections = _parse_excluded_sections(str(args.exclude_sections_canonical))

    train_examples, train_stats = _build_examples(
        notes_csv=Path(args.train_notes_csv),
        annotations_csv=Path(args.train_annotations_csv),
        l1_map=l1_map,
        words_splitter=model.data_processor.words_splitter,
        window_tokens=int(args.window_tokens),
        overlap_tokens=int(args.overlap_tokens),
        include_empty_windows=bool(args.include_empty_windows),
        excluded_sections=excluded_sections,
    )
    val_examples, val_stats = _build_examples(
        notes_csv=Path(args.val_notes_csv),
        annotations_csv=Path(args.val_annotations_csv),
        l1_map=l1_map,
        words_splitter=model.data_processor.words_splitter,
        window_tokens=int(args.window_tokens),
        overlap_tokens=int(args.overlap_tokens),
        include_empty_windows=bool(args.include_empty_windows),
        excluded_sections=None,
    )

    if not train_examples:
        raise RuntimeError("No train examples were generated.")
    if not val_examples:
        raise RuntimeError("No validation examples were generated.")

    use_best_model = bool(args.load_best_model_at_end) or int(args.early_stopping_patience) > 0
    create_training_kwargs = dict(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        per_device_train_batch_size=int(args.train_batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        gradient_accumulation_steps=int(args.grad_accum_steps),
        max_steps=int(args.max_steps),
        warmup_ratio=float(args.warmup_ratio),
        logging_steps=int(args.logging_steps),
        eval_strategy="steps",
        eval_steps=int(args.eval_steps),
        save_strategy="steps",
        save_steps=int(args.save_steps),
        save_total_limit=2,
        use_cpu=(map_location == "cpu"),
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
        report_to="none",
        seed=int(args.seed),
        dataloader_num_workers=0,
    )
    if use_best_model:
        create_training_kwargs.update(
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    try:
        training_args = model.create_training_args(**create_training_kwargs)
    except TypeError:
        for key in ("load_best_model_at_end", "metric_for_best_model", "greater_is_better"):
            create_training_kwargs.pop(key, None)
        training_args = model.create_training_args(**create_training_kwargs)
        use_best_model = False

    data_collator = model._create_data_collator()  # noqa: SLF001
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_examples,
        "eval_dataset": val_examples,
        "data_collator": data_collator,
    }
    sig = inspect.signature(Trainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = model.data_processor.transformer_tokenizer
    elif "tokenizer" in sig.parameters:
        trainer_kwargs["tokenizer"] = model.data_processor.transformer_tokenizer
    if int(args.early_stopping_patience) > 0:
        from transformers import EarlyStoppingCallback  # type: ignore

        trainer_kwargs["callbacks"] = [
            EarlyStoppingCallback(
                early_stopping_patience=int(args.early_stopping_patience),
                early_stopping_threshold=float(args.early_stopping_threshold),
            )
        ]

    trainer = StrictGLiNERTrainer(**trainer_kwargs)
    trainer.train()

    final_model_dir = out_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))

    stats = {
        "config": vars(args),
        "device_resolved": map_location,
        "use_best_model": bool(use_best_model),
        "excluded_sections_canonical": sorted(excluded_sections),
        "train": asdict(train_stats),
        "val": asdict(val_stats),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
    }
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"train_examples: {len(train_examples):,}")
    print(f"val_examples: {len(val_examples):,}")
    print(f"final_model_dir: {final_model_dir}")
    print(f"stats_json: {out_dir / 'train_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
