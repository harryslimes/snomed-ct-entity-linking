#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.io import normalize_alias


def _write_train_span_aliases(train_ann: pd.DataFrame, out_tsv: Path) -> int:
    rows = []
    for cid, span in train_ann[["concept_id", "span"]].itertuples(index=False, name=None):
        concept_id = str(cid).strip()
        alias = normalize_alias(span)
        if not concept_id or not alias:
            continue
        rows.append((concept_id, alias))

    deduped = sorted(set(rows), key=lambda x: (not x[0].isdigit(), int(x[0]) if x[0].isdigit() else x[0], x[1]))
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["concept_id", "alias"])
        writer.writerows(deduped)
    return len(deduped)


def _partition_note_ids(note_ids: list[str], n_folds: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    ids = list(note_ids)
    rng.shuffle(ids)
    chunks = [[] for _ in range(n_folds)]
    for idx, note_id in enumerate(ids):
        chunks[idx % n_folds].append(note_id)
    return chunks


def build_folds(
    *,
    notes_csv: Path,
    annotations_csv: Path,
    out_dir: Path,
    n_folds: int,
    seed: int,
    write_full_train: bool,
) -> dict:
    notes = pd.read_csv(notes_csv)
    annotations = pd.read_csv(annotations_csv)
    notes["note_id"] = notes["note_id"].astype(str)
    annotations["note_id"] = annotations["note_id"].astype(str)

    note_ids = sorted(notes["note_id"].unique().tolist())
    val_chunks = _partition_note_ids(note_ids, n_folds=n_folds, seed=seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_summaries = []
    all_note_ids = set(note_ids)
    for fold_idx in range(n_folds):
        val_ids = set(val_chunks[fold_idx])
        train_ids = all_note_ids - val_ids
        if train_ids & val_ids:
            raise RuntimeError(f"Fold {fold_idx} has train/val note overlap")

        fold_dir = out_dir / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_notes = notes[notes["note_id"].isin(train_ids)].copy()
        val_notes = notes[notes["note_id"].isin(val_ids)].copy()
        train_ann = annotations[annotations["note_id"].isin(train_ids)].copy()
        val_ann = annotations[annotations["note_id"].isin(val_ids)].copy()

        train_notes.to_csv(fold_dir / "train_notes.csv", index=False)
        val_notes.to_csv(fold_dir / "val_notes.csv", index=False)
        train_ann.to_csv(fold_dir / "train_annotations.csv", index=False)
        val_ann.to_csv(fold_dir / "val_annotations.csv", index=False)
        n_aliases = _write_train_span_aliases(train_ann, fold_dir / "train_span_aliases.tsv")

        fold_summaries.append(
            {
                "fold_idx": fold_idx,
                "n_train_notes": int(len(train_notes)),
                "n_val_notes": int(len(val_notes)),
                "n_train_annotations": int(len(train_ann)),
                "n_val_annotations": int(len(val_ann)),
                "n_train_span_aliases": int(n_aliases),
            }
        )

    if write_full_train:
        full_dir = out_dir / "full_train"
        full_dir.mkdir(parents=True, exist_ok=True)
        notes.to_csv(full_dir / "train_notes.csv", index=False)
        annotations.to_csv(full_dir / "train_annotations.csv", index=False)
        _write_train_span_aliases(annotations, full_dir / "train_span_aliases.tsv")

    manifest = {
        "notes_csv": str(notes_csv),
        "annotations_csv": str(annotations_csv),
        "n_folds": n_folds,
        "seed": seed,
        "write_full_train": bool(write_full_train),
        "folds": fold_summaries,
    }
    (out_dir / "fold_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Create leakage-safe note-level CV folds and fold-specific train-span alias files."
        )
    )
    ap.add_argument("--notes-csv", default="data/train_notes.csv")
    ap.add_argument("--annotations-csv", default="data/train_annotations.csv")
    ap.add_argument("--out-dir", default="data/interim/glinker/folds")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--write-full-train", action="store_true")
    args = ap.parse_args(argv)

    manifest = build_folds(
        notes_csv=Path(args.notes_csv),
        annotations_csv=Path(args.annotations_csv),
        out_dir=Path(args.out_dir),
        n_folds=int(args.n_folds),
        seed=int(args.seed),
        write_full_train=bool(args.write_full_train),
    )
    print(f"wrote folds: {args.out_dir}")
    print(f"n_folds: {manifest['n_folds']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
