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
    if train_size < 0:
        train_size = 0
    if train_size > len(ids):
        train_size = len(ids)
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


def _set_rule_env(*, linguistic: bool, stopword: bool):
    if linguistic:
        os.environ["KIRI_LINGUISTIC_RULES"] = "1"
    else:
        os.environ.pop("KIRI_LINGUISTIC_RULES", None)
    if stopword:
        os.environ["KIRI_STOPWORD_TRANSPARENT"] = "1"
    else:
        os.environ.pop("KIRI_STOPWORD_TRANSPARENT", None)


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
    parser.add_argument(
        "--super-rules",
        action="store_true",
        help=(
            "Enable additional linguistic rules for the super run only "
            "(KIRI_LINGUISTIC_RULES=1, KIRI_STOPWORD_TRANSPARENT=1)."
        ),
    )
    parser.add_argument(
        "--super-linguistic",
        action="store_true",
        help="Enable only KIRI_LINGUISTIC_RULES=1 for the super run.",
    )
    parser.add_argument(
        "--super-stopword",
        action="store_true",
        help="Enable only KIRI_STOPWORD_TRANSPARENT=1 for the super run.",
    )
    parser.add_argument(
        "--super-ablations",
        action="store_true",
        help=(
            "Run super variants in isolation: baseline, +linguistic, +stopword, +both. "
            "Writes outputs for each variant."
        ),
    )
    parser.add_argument("--train-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Train on all notes and score on the same full training set (train_ids == test_ids).",
    )
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
    if args.eval_all:
        train_ids = ids
        test_ids = ids
    else:
        train_ids, test_ids = _make_split_ids(ids, args.train_size, args.test_size)

    from mimic_common import common_headers  # noqa: E402

    print("Running default KIRI...", flush=True)
    _set_rule_env(linguistic=False, stopword=False)
    default_pred = _run_once(
        texts,
        annotations,
        common_headers,
        args.run_name_default,
        train_ids,
        test_ids,
        synonyms_path=args.default_synonyms,
    )

    print("Scoring (runtime macro-char IoU + class-level IoU)...", flush=True)
    t_score = time.perf_counter()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotated_subset = annotations[annotations["note_id"].isin(default_pred["note_id"].unique())]

    default_pred_out = out_dir / f"{args.run_name_default}_pred.csv"
    default_pred.to_csv(default_pred_out, index=False)
    default_class = class_char_iou(
        default_pred[["note_id", "start", "end", "concept_id"]],
        annotated_subset[["note_id", "start", "end", "concept_id"]],
    )
    default_class_out = out_dir / f"{args.run_name_default}_class_iou.csv"
    default_class.to_csv(default_class_out, index=False)
    default_runtime = float(default_class.loc[default_class["union"] > 0, "iou"].mean())
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

    def _score_and_write(label: str, pred: pd.DataFrame):
        pred_out = out_dir / f"{label}_pred.csv"
        pred.to_csv(pred_out, index=False)
        cls = class_char_iou(
            pred[["note_id", "start", "end", "concept_id"]],
            annotated_subset[["note_id", "start", "end", "concept_id"]],
        )
        cls_out = out_dir / f"{label}_class_iou.csv"
        cls.to_csv(cls_out, index=False)
        runtime = float(cls.loc[cls["union"] > 0, "iou"].mean())
        note_iou = float(
            np.mean(
                list(
                    iou_per_note(
                        pred[["note_id", "concept_id"]],
                        annotated_subset[["note_id", "concept_id"]],
                    ).values()
                )
            )
        )
        return note_iou, runtime, pred_out, cls_out

    super_results: list[tuple[str, float, float, Path, Path]] = []

    if args.super_ablations:
        configs = [
            ("base", False, False),
            ("linguistic", True, False),
            ("stopword", False, True),
            ("linguistic_stopword", True, True),
        ]
        for suffix, linguistic, stopword in configs:
            print(f"Running super-dictionary KIRI ({suffix})...", flush=True)
            _set_rule_env(linguistic=linguistic, stopword=stopword)
            run_name = f"{args.run_name_super}_{suffix}"
            pred = _run_once(
                texts,
                annotations,
                common_headers,
                run_name,
                train_ids,
                test_ids,
                synonyms_path=args.super_synonyms,
            )
            note_iou, runtime, pred_out, cls_out = _score_and_write(run_name, pred)
            super_results.append((run_name, note_iou, runtime, pred_out, cls_out))
    else:
        linguistic = args.super_rules or args.super_linguistic
        stopword = args.super_rules or args.super_stopword
        suffix_bits = []
        if linguistic:
            suffix_bits.append("linguistic")
        if stopword:
            suffix_bits.append("stopword")
        suffix = "_".join(suffix_bits) if suffix_bits else "base"

        print("Running super-dictionary KIRI...", flush=True)
        _set_rule_env(linguistic=linguistic, stopword=stopword)
        run_name = f"{args.run_name_super}_{suffix}"
        pred = _run_once(
            texts,
            annotations,
            common_headers,
            run_name,
            train_ids,
            test_ids,
            synonyms_path=args.super_synonyms,
        )
        note_iou, runtime, pred_out, cls_out = _score_and_write(run_name, pred)
        super_results.append((run_name, note_iou, runtime, pred_out, cls_out))

    print(f"Scoring done in {time.perf_counter() - t_score:0.2f}s", flush=True)
    print("\nSummary (note-level IoU, runtime macro-averaged character IoU):")
    print(f"Default: {default_note_iou:.4f}, {default_runtime:.4f}")
    for run_name, note_iou, runtime, _, _ in super_results:
        print(f"{run_name}: {note_iou:.4f}, {runtime:.4f}")
    print("\nOutputs:")
    print(f"Default: {default_pred_out}")
    print(f"Default: {default_class_out}")
    for run_name, _, _, pred_out, cls_out in super_results:
        print(f"{run_name}: {pred_out}")
        print(f"{run_name}: {cls_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
