#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


SMOKE_TEST_NOTE_IDS = ["12204158-DS-10", "11417242-DS-18", "11810606-DS-7"]


@dataclass(frozen=True)
class Interval:
    start: int
    end: int


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    sorted_intervals = sorted(intervals, key=lambda x: (x.start, x.end))
    merged: list[Interval] = []
    for interval in sorted_intervals:
        if interval.end <= interval.start:
            continue
        if not merged:
            merged.append(interval)
            continue
        last = merged[-1]
        if interval.start > last.end:
            merged.append(interval)
        else:
            merged[-1] = Interval(last.start, max(last.end, interval.end))
    return merged


def total_len(intervals: Iterable[Interval]) -> int:
    return sum(i.end - i.start for i in intervals)


def intersection_len(a: list[Interval], b: list[Interval]) -> int:
    i = 0
    j = 0
    total = 0
    while i < len(a) and j < len(b):
        left = max(a[i].start, b[j].start)
        right = min(a[i].end, b[j].end)
        if right > left:
            total += right - left
        if a[i].end <= b[j].end:
            i += 1
        else:
            j += 1
    return total


def ensure_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {"note_id", "start", "end", "concept_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    out = df[list(required)].copy()
    out["start"] = pd.to_numeric(out["start"], errors="coerce").astype("Int64")
    out["end"] = pd.to_numeric(out["end"], errors="coerce").astype("Int64")
    out["concept_id"] = pd.to_numeric(out["concept_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["note_id", "start", "end", "concept_id"]).copy()
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    out["concept_id"] = out["concept_id"].astype(int)
    out = out[out["end"] > out["start"]].copy()
    return out


def score_macro_iou(pred: pd.DataFrame, gold: pd.DataFrame) -> dict:
    pred = ensure_columns(pred, "pred")
    gold = ensure_columns(gold, "gold")

    cats = sorted(set(pred["concept_id"].tolist()) | set(gold["concept_id"].tolist()))
    docs = sorted(set(pred["note_id"].tolist()) | set(gold["note_id"].tolist()))

    pred_groups = pred.groupby(["note_id", "concept_id"])[["start", "end"]].apply(
        lambda x: [Interval(int(a), int(b)) for a, b in x.itertuples(index=False)]
    )
    gold_groups = gold.groupby(["note_id", "concept_id"])[["start", "end"]].apply(
        lambda x: [Interval(int(a), int(b)) for a, b in x.itertuples(index=False)]
    )

    ious: dict[int, float] = {}
    for cat in cats:
        intersection_total = 0
        union_total = 0
        for doc in docs:
            pred_ints = merge_intervals(pred_groups.get((doc, cat), []))
            gold_ints = merge_intervals(gold_groups.get((doc, cat), []))
            if not pred_ints and not gold_ints:
                continue
            intersection_total += intersection_len(pred_ints, gold_ints)
            union_total += total_len(merge_intervals([*pred_ints, *gold_ints]))
        ious[cat] = (intersection_total / union_total) if union_total else 0.0

    macro = sum(ious.values()) / len(ious) if ious else 0.0
    return {
        "macro_iou": macro,
        "n_classes": len(ious),
        "per_class_iou": ious,
    }


def run_cmd(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def build_smoke_test_data(
    train_notes: Path,
    train_annotations: Path,
    out_notes: Path,
    out_annotations: Path,
) -> None:
    notes = pd.read_csv(train_notes)
    annotations = pd.read_csv(train_annotations)
    notes = notes[notes.note_id.isin(SMOKE_TEST_NOTE_IDS)]
    annotations = annotations[annotations.note_id.isin(SMOKE_TEST_NOTE_IDS)]
    for col in ("start", "end"):
        if col in annotations.columns:
            annotations[col] = pd.to_numeric(annotations[col], errors="coerce")
    annotations = annotations.dropna(subset=["start", "end"]).copy()
    annotations["start"] = annotations["start"].astype(int)
    annotations["end"] = annotations["end"].astype(int)
    out_notes.parent.mkdir(parents=True, exist_ok=True)
    out_annotations.parent.mkdir(parents=True, exist_ok=True)
    notes.to_csv(out_notes, index=False)
    annotations.to_csv(out_annotations, index=False)


def ensure_submission_dir(submission_dir: Path, rebuild: bool) -> None:
    required = [submission_dir / "main.py", submission_dir / "assets" / "kiri_dicts.pkl"]
    if not rebuild and all(p.exists() for p in required):
        return
    run_cmd(
        ["python", str(ROOT / "1st Place" / "src" / "make_inference_env.py"), str(submission_dir)],
        cwd=ROOT,
    )
    if not all(p.exists() for p in required):
        missing = [str(p) for p in required if not p.exists()]
        raise FileNotFoundError(f"Submission build incomplete. Missing: {missing}")


def main():
    parser = argparse.ArgumentParser(description="Run 1st place and score outputs.")
    parser.add_argument("--notes", default=str(ROOT / "data" / "test_notes.csv"))
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--make-smoke-test", action="store_true")
    default_output = str(ROOT / "outputs" / "1st_place_eval.json")
    parser.add_argument("--output", default=default_output)
    parser.add_argument(
        "--tag",
        default="",
        help=(
            "Optional tag used to name the copied submission CSV and (if --output is left at its default) the eval JSON. "
            "If unset, a tag is inferred (e.g. smoke/train/test)."
        ),
    )
    parser.add_argument(
        "--note-limit",
        type=int,
        default=0,
        help="If >0, only run inference on the first N notes (and filter annotations to those notes for scoring).",
    )
    parser.add_argument("--submission-dir", default=str(ROOT / "1st Place" / "submission"))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    submission_dir = Path(args.submission_dir)
    notes_path = Path(args.notes)
    annotations_path: Optional[Path] = Path(args.annotations) if args.annotations else None

    if args.make_smoke_test:
        notes_path = ROOT / "data" / "smoke_test_notes.csv"
        annotations_path = ROOT / "data" / "smoke_test_annotations.csv"
        build_smoke_test_data(
            train_notes=ROOT / "data" / "train_notes.csv",
            train_annotations=ROOT / "data" / "train_annotations.csv",
            out_notes=notes_path,
            out_annotations=annotations_path,
        )

    inferred_tag = args.tag
    if not inferred_tag:
        if args.make_smoke_test:
            inferred_tag = "smoke"
        else:
            name = notes_path.name.lower()
            if "train" in name:
                inferred_tag = "train"
            elif "test" in name:
                inferred_tag = "test"
            else:
                inferred_tag = notes_path.stem
    if args.note_limit and args.note_limit > 0:
        inferred_tag = f"{inferred_tag}_n{args.note_limit}"

    if args.output == default_output and inferred_tag:
        args.output = str(ROOT / "outputs" / f"1st_place_{inferred_tag}_eval.json")

    ensure_submission_dir(submission_dir, rebuild=args.rebuild)

    if not notes_path.exists():
        raise FileNotFoundError(f"Notes not found: {notes_path}")

    dst_notes = submission_dir / "data" / "test_notes.csv"
    dst_notes.parent.mkdir(parents=True, exist_ok=True)
    used_notes_path = notes_path
    used_annotations_path = annotations_path
    if args.note_limit and args.note_limit > 0:
        ROOT.joinpath("outputs").mkdir(parents=True, exist_ok=True)
        used_notes_path = ROOT / "outputs" / f"1st_place_{inferred_tag}_notes.csv"
        notes_df = pd.read_csv(notes_path)
        notes_df = notes_df.head(args.note_limit).copy()
        used_note_ids = set(notes_df["note_id"].astype(str).tolist())
        notes_df.to_csv(used_notes_path, index=False)
        if annotations_path and annotations_path.exists():
            used_annotations_path = ROOT / "outputs" / f"1st_place_{inferred_tag}_annotations.csv"
            ann_df = pd.read_csv(annotations_path)
            ann_df = ann_df[ann_df["note_id"].astype(str).isin(used_note_ids)].copy()
            ann_df.to_csv(used_annotations_path, index=False)

    shutil.copyfile(used_notes_path, dst_notes)

    run_cmd(["python", "main.py"], cwd=submission_dir)

    pred_path = submission_dir / "submission.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Expected output not found: {pred_path}")

    ROOT.joinpath("outputs").mkdir(parents=True, exist_ok=True)
    copied_pred_path = ROOT / "outputs" / f"submission_1st_place_{inferred_tag}.csv"
    shutil.copyfile(pred_path, copied_pred_path)

    result = {
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "notes_path": str(used_notes_path),
        "submission_csv": str(copied_pred_path),
    }

    notes_df = pd.read_csv(used_notes_path)
    result["n_notes"] = int(len(notes_df))

    pred_df = pd.read_csv(copied_pred_path)
    result["n_predictions"] = int(len(pred_df))

    if used_annotations_path and used_annotations_path.exists():
        gold_df = pd.read_csv(used_annotations_path)
        score = score_macro_iou(pred_df, gold_df)
        result["annotations_path"] = str(used_annotations_path)
        result.update({k: v for k, v in score.items() if k != "per_class_iou"})
        result["per_class_iou"] = {str(k): v for k, v in score["per_class_iou"].items()}
    else:
        result["annotations_path"] = str(used_annotations_path) if used_annotations_path else None
        result["macro_iou"] = None
        result["n_classes"] = None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote: {out_path}")
    if result["macro_iou"] is not None:
        print(f"macro IoU: {result['macro_iou']:.4f} over {result['n_classes']} classes")


if __name__ == "__main__":
    main()
