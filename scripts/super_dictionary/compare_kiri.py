#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
KIRI_SRC = REPO_ROOT / "1st Place" / "src"
if str(KIRI_SRC) not in sys.path:
    sys.path.append(str(KIRI_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from mimic_predict import make_predictions  # noqa: E402
from mimic_train import train  # noqa: E402
from note_scoring import iou_per_note  # noqa: E402
from scripts.super_dictionary.runtime_scoring import class_char_iou  # noqa: E402


def _load_texts_and_annotations(data_dir: Path):
    texts = pd.read_csv(data_dir / "raw" / "mimic-iv_notes_training_set.csv").set_index("note_id")[
        "text"
    ]
    annotations = pd.read_csv(data_dir / "interim" / "train_annotations_cln.csv")
    return texts, annotations


def _make_split_ids(ids: list[str], train_size: int, test_size: int | None):
    np.random.seed(12345)
    np.random.shuffle(ids)
    train_ids = ids[:train_size]
    test_ids = ids[train_size:]
    if test_size is not None:
        test_ids = ids[-test_size:]
    return train_ids, test_ids


def _run_once(texts, annotations, headers, run_name, train_ids, test_ids, synonyms_path=None):
    if synonyms_path:
        os.environ["KIRI_SYNONYMS_PATH"] = str(synonyms_path)
    else:
        os.environ.pop("KIRI_SYNONYMS_PATH", None)

    t0 = time.perf_counter()
    print(f"[{run_name}] training dicts on {len(train_ids)} notes...", flush=True)
    d, uc_d = train(texts[texts.index.isin(train_ids)], annotations, headers, run_name)
    t1 = time.perf_counter()
    print(f"[{run_name}] training done in {t1 - t0:0.1f}s", flush=True)

    print(f"[{run_name}] predicting on {len(test_ids)} notes...", flush=True)
    pred = make_predictions(texts[texts.index.isin(test_ids)], d, uc_d, run_name=run_name)
    t2 = time.perf_counter()
    print(
        f"[{run_name}] prediction done in {t2 - t1:0.1f}s (total {t2 - t0:0.1f}s)",
        flush=True,
    )
    return pred


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare KIRI default vs super-dictionary synonyms."
    )
    parser.add_argument("--kiri-data-dir", default="1st Place/data")
    parser.add_argument(
        "--default-synonyms",
        default="1st Place/data/interim/flattened_terminology_syn_snomed+omop_v5.csv",
    )
    parser.add_argument(
        "--super-synonyms",
        default="1st Place/data/interim/flattened_terminology_syn_super.csv",
    )
    parser.add_argument("--run-name-default", default="kiri_default")
    parser.add_argument("--run-name-super", default="kiri_super")
    parser.add_argument("--train-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default="outputs/super_dictionary",
        help="Directory to write prediction and class-IoU outputs.",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.kiri_data_dir)
    if not (data_dir / "interim" / "train_annotations_cln.csv").exists():
        print(
            "Missing train_annotations_cln.csv. Run:\n"
            "  (cd \"1st Place\" && python src/process_data.py make-clean-annotations)",
            file=sys.stderr,
        )
        return 2

    texts, annotations = _load_texts_and_annotations(data_dir)
    ids = list(texts.index)
    train_ids, test_ids = _make_split_ids(ids, args.train_size, args.test_size)

    from mimic_common import common_headers  # noqa: E402

    print("Running default KIRI...", flush=True)
    default_pred = _run_once(
        texts,
        annotations,
        common_headers,
        args.run_name_default,
        train_ids,
        test_ids,
        synonyms_path=args.default_synonyms,
    )

    print("Running super-dictionary KIRI...", flush=True)
    super_pred = _run_once(
        texts,
        annotations,
        common_headers,
        args.run_name_super,
        train_ids,
        test_ids,
        synonyms_path=args.super_synonyms,
    )

    print("Scoring (runtime macro-char IoU + class-level IoU)...", flush=True)
    t_score = time.perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_pred_out = out_dir / f"{args.run_name_default}_pred.csv"
    super_pred_out = out_dir / f"{args.run_name_super}_pred.csv"
    default_pred.to_csv(default_pred_out, index=False)
    super_pred.to_csv(super_pred_out, index=False)

    annotated_subset = annotations[annotations["note_id"].isin(default_pred["note_id"].unique())]
    default_class = class_char_iou(
        default_pred[["note_id", "start", "end", "concept_id"]],
        annotated_subset[["note_id", "start", "end", "concept_id"]],
    )
    super_class = class_char_iou(
        super_pred[["note_id", "start", "end", "concept_id"]],
        annotated_subset[["note_id", "start", "end", "concept_id"]],
    )

    default_class_out = out_dir / f"{args.run_name_default}_class_iou.csv"
    super_class_out = out_dir / f"{args.run_name_super}_class_iou.csv"
    default_class.to_csv(default_class_out, index=False)
    super_class.to_csv(super_class_out, index=False)

    default_runtime = float(default_class.loc[default_class["union"] > 0, "iou"].mean())
    super_runtime = float(super_class.loc[super_class["union"] > 0, "iou"].mean())
    print(f"Scoring done in {time.perf_counter() - t_score:0.2f}s", flush=True)

    default_note_iou = float(
        np.mean(
            list(
                iou_per_note(
                    default_pred[["note_id", "concept_id"]],
                    annotated_subset[["note_id", "concept_id"]],
                ).values()
            )
        )
    )
    super_note_iou = float(
        np.mean(
            list(
                iou_per_note(
                    super_pred[["note_id", "concept_id"]],
                    annotated_subset[["note_id", "concept_id"]],
                ).values()
            )
        )
    )

    print("\nSummary (note-level IoU, runtime macro-averaged character IoU):")
    print(f"Default: {default_note_iou:.4f}, {default_runtime:.4f}")
    print(f"Super:   {super_note_iou:.4f}, {super_runtime:.4f}")
    print("\nRuntime macro-averaged character IoU:")
    print(f"Default: {default_runtime:.4f}")
    print(f"Super:   {super_runtime:.4f}")
    print("\nClass-level IoU outputs:")
    print(f"Default: {default_class_out}")
    print(f"Super:   {super_class_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
