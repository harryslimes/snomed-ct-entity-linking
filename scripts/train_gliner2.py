#!/usr/bin/env python3
"""Train GLiNER2 on SNOMED clinical entity data.

Loads prepared data from prepare_gliner2_data.py and trains with LoRA.

Usage:
    python scripts/train_gliner2.py [--data-dir data/gliner2] [--output-dir models/gliner2-snomed]
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import torch
from gliner2 import GLiNER2
from gliner2.training.data import InputExample
from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Monkey-patches for GLiNER2 trainer bugs (see github.com/fastino-ai/GLiNER2/issues/70)
# ---------------------------------------------------------------------------

def patch_early_stopping(trainer: GLiNER2Trainer):
    """Fix early stopping bug: _evaluate() updates best_metric before
    _check_early_stopping() compares against it, so the patience counter
    never resets. Fix: track previous best_metric before _evaluate runs."""

    original_train = trainer.train

    def patched_train(train_data, eval_data=None, **kwargs):
        original_evaluate = trainer._evaluate

        def fixed_evaluate(eval_dataset):
            # Save best_metric BEFORE evaluation updates it
            trainer._pre_eval_best_metric = trainer.best_metric
            return original_evaluate(eval_dataset)

        original_check = trainer._check_early_stopping

        def fixed_check_early_stopping(metrics):
            metric_value = metrics.get(
                trainer.config.metric_for_best, metrics["eval_loss"]
            )
            # Compare against the PREVIOUS best, not the one just updated
            prev_best = getattr(trainer, "_pre_eval_best_metric", trainer.best_metric)
            if trainer.config.greater_is_better:
                improved = metric_value > prev_best + trainer.config.early_stopping_threshold
            else:
                improved = metric_value < prev_best - trainer.config.early_stopping_threshold

            if improved:
                trainer.patience_counter = 0
                logger.info(f"Early stopping: metric improved ({prev_best:.4f} -> {metric_value:.4f}), patience reset")
            else:
                trainer.patience_counter += 1
                logger.info(f"Early stopping: no improvement (best={prev_best:.4f}, current={metric_value:.4f}), "
                           f"patience {trainer.patience_counter}/{trainer.config.early_stopping_patience}")

            return trainer.patience_counter >= trainer.config.early_stopping_patience

        trainer._evaluate = fixed_evaluate
        trainer._check_early_stopping = fixed_check_early_stopping
        return original_train(train_data, eval_data=eval_data, **kwargs)

    trainer.train = patched_train
    print("  Trainer patched: early stopping bug fixed (Issue #70)")


def enable_gradient_checkpointing(model: GLiNER2):
    """Enable gradient checkpointing on the DeBERTa encoder.
    Trades ~30% more compute for ~40-60% less activation memory,
    allowing larger batch sizes."""
    # GLiNER2 exposes encoder as a direct child; it's a DebertaV2Model
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    print("  Gradient checkpointing enabled on DeBERTa encoder (use_reentrant=False)")


def patch_processor_max_length(model: GLiNER2, max_seq_len: int = 512):
    """Monkey-patch GLiNER2's processor to hard-cap sequence length.

    GLiNER2's processor._pad_batch() pads to the longest sequence in the batch
    with NO truncation. DeBERTa's O(n^2) disentangled attention makes this
    catastrophic if even one sequence is unexpectedly long.

    This patch truncates input_ids and mapped_indices at the processor level,
    guaranteeing no sequence exceeds max_seq_len subword tokens.
    """
    processor = model.processor
    original_pad_batch = processor._pad_batch

    def _capped_pad_batch(records):
        # Truncate any records that exceed the cap
        for rec in records:
            if len(rec.input_ids) > max_seq_len:
                rec.input_ids = rec.input_ids[:max_seq_len]
                rec.mapped_indices = rec.mapped_indices[:max_seq_len]
        return original_pad_batch(records)

    processor._pad_batch = _capped_pad_batch
    print(f"  Processor patched: max sequence length capped at {max_seq_len}")


def load_examples(jsonl_path: Path) -> list[InputExample]:
    """Load examples from JSONL file into InputExample objects."""
    examples = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            examples.append(InputExample(
                text=record["input"],
                entities=record["output"]["entities"],
                entity_descriptions=record["output"].get("entity_descriptions"),
            ))
    return examples


def main():
    parser = argparse.ArgumentParser(description="Train GLiNER2 on SNOMED data")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/gliner2"),
        help="Directory containing train/val/test JSONL files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/gliner2-snomed"),
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--base-model", type=str, default="fastino/gliner2-large-v1",
        help="Base model to fine-tune",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--task-lr", type=float, default=5e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--no-lora", action="store_true", help="Full fine-tuning")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bf16 (default, best for Blackwell/Ampere+)")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Hard cap on total sequence length (text + schema tokens)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing (saves memory, allows larger batch)")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    print("Loading training data...")
    train_examples = load_examples(args.data_dir / "train.jsonl")
    print(f"  Train: {len(train_examples):,} examples")

    val_examples = load_examples(args.data_dir / "val.jsonl")
    print(f"  Val: {len(val_examples):,} examples")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading base model: {args.base_model} (device={device})")
    model = GLiNER2.from_pretrained(args.base_model).to(device)

    # Patch processor to enforce max sequence length
    patch_processor_max_length(model, args.max_seq_len)

    # Enable gradient checkpointing if requested (saves ~40-60% activation memory)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    # Configure training
    use_lora = not args.no_lora
    use_bf16 = args.bf16 and not args.fp16
    use_fp16 = args.fp16 and not args.bf16

    config = TrainingConfig(
        output_dir=str(args.output_dir),
        experiment_name="gliner2-snomed-clinical",
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        encoder_lr=args.encoder_lr,
        task_lr=args.task_lr,
        warmup_ratio=0.06,
        scheduler_type="cosine",
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="epoch",
        save_best=True,
        metric_for_best="eval_f1",
        greater_is_better=True,
        early_stopping=True,
        early_stopping_patience=args.patience,
        num_workers=args.num_workers,
        pin_memory=True,
        seed=args.seed,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=float(args.lora_r * 2),
        lora_dropout=0.05,
        lora_target_modules=["encoder"],
        save_adapter_only=use_lora,
    )

    eff_bs = args.batch_size * args.grad_accum
    precision = "bf16" if use_bf16 else ("fp16" if use_fp16 else "fp32")
    print(f"\nTraining config:")
    print(f"  LoRA: {use_lora} (r={args.lora_r})" if use_lora else "  Full fine-tuning")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} × {args.grad_accum} grad accum = {eff_bs} effective")
    print(f"  Encoder LR: {args.encoder_lr}, Task LR: {args.task_lr}")
    print(f"  Precision: {precision}")
    print(f"  Max seq len: {args.max_seq_len}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"  Workers: {args.num_workers}, Pin memory: True")

    # Train
    trainer = GLiNER2Trainer(model=model, config=config)

    # Patch early stopping bug (Issue #70)
    patch_early_stopping(trainer)

    results = trainer.train(train_data=train_examples, eval_data=val_examples)

    print(f"\nTraining complete. Results: {results}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
