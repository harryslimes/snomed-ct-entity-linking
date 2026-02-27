#!/usr/bin/env python3
"""Character-level IoU scoring for SNOMED CT entity linking predictions.

Computes macro-averaged character-level IoU using coordinate compression
for memory-efficient evaluation on large clinical notes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


def iou_per_class(
    user_annotations: pd.DataFrame,
    target_annotations: pd.DataFrame,
) -> List[float]:
    docs = np.unique(np.concatenate([user_annotations.note_id, target_annotations.note_id]))
    doc_index_mapping = dict(zip(docs, range(len(docs))))

    cats = np.unique(np.concatenate([user_annotations.concept_id, target_annotations.concept_id]))
    max_end = np.max(np.concatenate([user_annotations.end, target_annotations.end]))

    def populate_char_mtx(n_rows, n_cols, annot_df):
        mtx = sp.lil_array((n_rows, n_cols), dtype=np.uint64)
        for row in annot_df.itertuples():
            doc_index = doc_index_mapping[row.note_id]
            mtx[doc_index, row.start : row.end] = int(row.concept_id)  # noqa: E203
        return mtx.tocsr()

    gt_mtx = populate_char_mtx(docs.shape[0], max_end, target_annotations)
    pred_mtx = populate_char_mtx(docs.shape[0], max_end, user_annotations)

    ious = []
    for cat in cats:
        gt_cat = gt_mtx == cat
        pred_cat = pred_mtx == cat
        intersection = gt_cat * pred_cat
        union = gt_cat + pred_cat
        iou = intersection.sum() / union.sum()
        ious.append(iou)

    return ious


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["concept_id"] = df["concept_id"].astype(int)
    return df


_SCORE_USER = None
_SCORE_TARGET = None
_SCORE_CAT_TO_IDX = None
_SCORE_N_CATS = None


def _build_note_map(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    if df.empty:
        return {}
    return {
        str(note_id): group[["start", "end", "concept_id"]].to_numpy(dtype=np.int64, copy=False)
        for note_id, group in df.groupby("note_id", sort=False)
    }


def _score_workers() -> int:
    raw = os.environ.get("KIRI_SCORE_WORKERS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except Exception:
            return 1
    raw_parallel = os.environ.get("KIRI_SCORE_PARALLEL", "").strip()
    if raw_parallel and str(raw_parallel).lower() not in {"0", "false", "no", "off"}:
        return min(8, (os.cpu_count() or 1))
    return 1


def _score_chunksize() -> int:
    raw = os.environ.get("KIRI_SCORE_CHUNKSIZE", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except Exception:
            return 1
    return 16


def _score_debug() -> bool:
    raw = os.environ.get("KIRI_SCORE_DEBUG", "").strip()
    if not raw:
        return False
    return str(raw).lower() not in {"0", "false", "no", "off"}


def _aggregate_note(
    gt_note: np.ndarray | None,
    pred_note: np.ndarray | None,
    cat_to_idx: Dict[int, int],
    n_cats: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if gt_note is None and pred_note is None:
        return (
            np.zeros(n_cats + 1, dtype=np.int64),
            np.zeros(n_cats + 1, dtype=np.int64),
            np.zeros(n_cats + 1, dtype=np.int64),
        )

    if gt_note is None:
        gt_note = np.empty((0, 3), dtype=np.int64)
    if pred_note is None:
        pred_note = np.empty((0, 3), dtype=np.int64)

    if gt_note.shape[0] == 0 and pred_note.shape[0] == 0:
        return (
            np.zeros(n_cats + 1, dtype=np.int64),
            np.zeros(n_cats + 1, dtype=np.int64),
            np.zeros(n_cats + 1, dtype=np.int64),
        )

    boundaries = np.unique(
        np.concatenate(
            [
                gt_note[:, 0],
                gt_note[:, 1],
                pred_note[:, 0],
                pred_note[:, 1],
            ]
        )
    )
    if boundaries.shape[0] < 2:
        return (
            np.zeros(n_cats + 1, dtype=np.int64),
            np.zeros(n_cats + 1, dtype=np.int64),
            np.zeros(n_cats + 1, dtype=np.int64),
        )

    idx_of = {int(v): i for i, v in enumerate(boundaries.tolist())}
    lengths = (boundaries[1:] - boundaries[:-1]).astype(np.int64, copy=False)

    gt_labels = np.zeros(boundaries.shape[0] - 1, dtype=np.int32)
    pred_labels = np.zeros(boundaries.shape[0] - 1, dtype=np.int32)

    for start, end, cid in gt_note:
        i = idx_of.get(int(start))
        j = idx_of.get(int(end))
        if i is None or j is None or i >= j:
            continue
        gt_labels[i:j] = cat_to_idx.get(int(cid), 0)
    for start, end, cid in pred_note:
        i = idx_of.get(int(start))
        j = idx_of.get(int(end))
        if i is None or j is None or i >= j:
            continue
        pred_labels[i:j] = cat_to_idx.get(int(cid), 0)

    gt_count = np.zeros(n_cats + 1, dtype=np.int64)
    pred_count = np.zeros(n_cats + 1, dtype=np.int64)
    inter_count = np.zeros(n_cats + 1, dtype=np.int64)

    np.add.at(gt_count, gt_labels, lengths)
    np.add.at(pred_count, pred_labels, lengths)
    eq = gt_labels == pred_labels
    if np.any(eq):
        np.add.at(inter_count, gt_labels[eq], lengths[eq])

    return gt_count, pred_count, inter_count


def _aggregate_chunk(note_ids: List[str]):
    user_by_note = _SCORE_USER
    target_by_note = _SCORE_TARGET
    cat_to_idx = _SCORE_CAT_TO_IDX
    n_cats = _SCORE_N_CATS

    total_gt = np.zeros(n_cats + 1, dtype=np.int64)
    total_pred = np.zeros(n_cats + 1, dtype=np.int64)
    total_inter = np.zeros(n_cats + 1, dtype=np.int64)

    for note_id in note_ids:
        gt_note = target_by_note.get(note_id)
        pred_note = user_by_note.get(note_id)
        gt_count, pred_count, inter_count = _aggregate_note(
            gt_note, pred_note, cat_to_idx, n_cats
        )
        total_gt += gt_count
        total_pred += pred_count
        total_inter += inter_count
    return total_gt, total_pred, total_inter


def _aggregate_counts(
    user_annotations: pd.DataFrame, target_annotations: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    user_annotations = _prepare_df(user_annotations)
    target_annotations = _prepare_df(target_annotations)

    all_cats = np.unique(
        np.concatenate([user_annotations.concept_id.values, target_annotations.concept_id.values])
    )
    if len(all_cats) == 0:
        return all_cats, np.array([]), np.array([]), np.array([])

    cat_to_idx: Dict[int, int] = {int(cat): i + 1 for i, cat in enumerate(all_cats)}
    n_cats = len(all_cats)

    user_by_note = _build_note_map(user_annotations)
    target_by_note = _build_note_map(target_annotations)

    note_ids = sorted(set(user_by_note.keys()) | set(target_by_note.keys()))
    total_gt = np.zeros(n_cats + 1, dtype=np.int64)
    total_pred = np.zeros(n_cats + 1, dtype=np.int64)
    total_inter = np.zeros(n_cats + 1, dtype=np.int64)

    workers = _score_workers()
    if _score_debug():
        print(
            f"[score] notes={len(note_ids)} cats={n_cats} workers={workers} chunksize={_score_chunksize()}"
        )
    if workers > 1 and os.name == "posix" and len(note_ids) > 0:
        global _SCORE_USER, _SCORE_TARGET, _SCORE_CAT_TO_IDX, _SCORE_N_CATS
        _SCORE_USER = user_by_note
        _SCORE_TARGET = target_by_note
        _SCORE_CAT_TO_IDX = cat_to_idx
        _SCORE_N_CATS = n_cats

        import multiprocessing as mp

        chunksize = _score_chunksize()
        chunks = [
            note_ids[i : i + chunksize] for i in range(0, len(note_ids), chunksize)
        ]
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for gt_count, pred_count, inter_count in pool.imap(_aggregate_chunk, chunks):
                total_gt += gt_count
                total_pred += pred_count
                total_inter += inter_count
    else:
        for note_id in note_ids:
            gt_note = target_by_note.get(note_id)
            pred_note = user_by_note.get(note_id)
            gt_count, pred_count, inter_count = _aggregate_note(
                gt_note, pred_note, cat_to_idx, n_cats
            )
            total_gt += gt_count
            total_pred += pred_count
            total_inter += inter_count

    return all_cats, total_gt, total_pred, total_inter


def class_char_iou(
    user_annotations: pd.DataFrame, target_annotations: pd.DataFrame
) -> pd.DataFrame:
    all_cats, total_gt, total_pred, total_inter = _aggregate_counts(
        user_annotations, target_annotations
    )
    if len(all_cats) == 0:
        return pd.DataFrame(
            columns=["concept_id", "gt_chars", "pred_chars", "intersection", "union", "iou"]
        )
    union = total_gt + total_pred - total_inter
    iou = np.zeros(len(all_cats), dtype=np.float64)
    valid = union[1:] > 0
    iou[valid] = total_inter[1:][valid] / union[1:][valid]
    return pd.DataFrame(
        {
            "concept_id": all_cats.astype(int),
            "gt_chars": total_gt[1:],
            "pred_chars": total_pred[1:],
            "intersection": total_inter[1:],
            "union": union[1:],
            "iou": iou,
        }
    )


def macro_char_iou(user_annotations: pd.DataFrame, target_annotations: pd.DataFrame) -> float:
    class_df = class_char_iou(user_annotations, target_annotations)
    if class_df.empty:
        return 0.0
    valid = class_df["union"] > 0
    if not valid.any():
        return 0.0
    return float(class_df.loc[valid, "iou"].mean())


def _load_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _prepare_df(df[["note_id", "start", "end", "concept_id"]])


def main(
    user_annotations_path: Path,
    target_annotations_path: Path,
) -> None:
    user_annotations = _load_annotations(user_annotations_path)
    target_annotations = _load_annotations(target_annotations_path)
    score = macro_char_iou(user_annotations, target_annotations)
    print(f"macro-averaged character IoU metric: {score:0.4f}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score predictions vs ground truth.")
    parser.add_argument("user_annotations", type=Path, help="Path to predictions CSV")
    parser.add_argument("target_annotations", type=Path, help="Path to ground truth CSV")
    args = parser.parse_args()
    main(args.user_annotations, args.target_annotations)
