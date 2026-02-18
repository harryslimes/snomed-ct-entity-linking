#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
KIRI_SRC = REPO_ROOT / "1st Place" / "src"
if str(KIRI_SRC) not in sys.path:
    sys.path.append(str(KIRI_SRC))


def _load_spans(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"note_id", "start", "end", "concept_id"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df[["note_id", "start", "end", "concept_id"]].copy()
    df["note_id"] = df["note_id"].astype(str)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["concept_id"] = df["concept_id"].astype(int)
    return df


def _load_class_iou(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"concept_id", "gt_chars", "pred_chars", "intersection", "union", "iou"} - set(
        df.columns
    )
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df[["concept_id", "gt_chars", "pred_chars", "intersection", "union", "iou"]].copy()
    df["concept_id"] = df["concept_id"].astype(int)
    for col in ["gt_chars", "pred_chars", "intersection", "union"]:
        df[col] = df[col].astype(np.int64)
    df["iou"] = df["iou"].astype(float)
    return df


def _concept_name_map(flattened_terminology_csv: Path) -> dict[int, str]:
    df = pd.read_csv(flattened_terminology_csv, usecols=["concept_id", "concept_name"])
    df = df.drop_duplicates("concept_id", keep="first")
    return {
        int(cid): str(name)
        for cid, name in zip(df["concept_id"].tolist(), df["concept_name"].tolist())
    }


def _iou_set(a: set[int], b: set[int]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _to_note_sets(df: pd.DataFrame) -> dict[str, set[int]]:
    if df.empty:
        return {}
    return {
        str(note_id): set(map(int, group["concept_id"].tolist()))
        for note_id, group in df.groupby("note_id", sort=False)
    }


def _to_note_spans(df: pd.DataFrame) -> dict[str, np.ndarray]:
    if df.empty:
        return {}
    return {
        str(note_id): group[["start", "end", "concept_id"]].to_numpy(dtype=np.int64, copy=False)
        for note_id, group in df.groupby("note_id", sort=False)
    }


def _labels_over_boundaries(boundaries: np.ndarray, spans: np.ndarray | None) -> np.ndarray:
    labels = np.zeros(boundaries.shape[0] - 1, dtype=np.int64)
    if spans is None or spans.shape[0] == 0:
        return labels

    idx_of = {int(v): i for i, v in enumerate(boundaries.tolist())}
    for start, end, cid in spans:
        i = idx_of.get(int(start))
        j = idx_of.get(int(end))
        if i is None or j is None or i >= j:
            continue
        labels[i:j] = int(cid)
    return labels


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".tsv"}:
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def build_note_delta_report(
    default_pred: pd.DataFrame,
    super_pred: pd.DataFrame,
    gold: pd.DataFrame,
    concept_name: dict[int, str],
    note_id_source: str = "default",
) -> pd.DataFrame:
    if note_id_source == "default":
        note_ids = sorted(set(default_pred["note_id"].unique()))
    elif note_id_source == "super":
        note_ids = sorted(set(super_pred["note_id"].unique()))
    elif note_id_source == "gold":
        note_ids = sorted(set(gold["note_id"].unique()))
    else:
        raise ValueError("--note-id-source must be one of: default, super, gold")

    gold = gold[gold["note_id"].isin(note_ids)].copy()

    default_sets = _to_note_sets(default_pred)
    super_sets = _to_note_sets(super_pred)
    gold_sets = _to_note_sets(gold)

    rows = []
    for note_id in sorted(set(gold_sets.keys())):
        g = gold_sets.get(note_id, set())
        d = default_sets.get(note_id, set())
        s = super_sets.get(note_id, set())

        new = s - d
        lost = d - s

        new_tp = sorted([cid for cid in new if cid in g])
        new_fp = sorted([cid for cid in new if cid not in g])
        lost_tp = sorted([cid for cid in lost if cid in g])
        lost_fp = sorted([cid for cid in lost if cid not in g])

        rows.append(
            {
                "note_id": note_id,
                "iou_default": _iou_set(d, g),
                "iou_super": _iou_set(s, g),
                "delta": _iou_set(s, g) - _iou_set(d, g),
                "n_gold": len(g),
                "n_pred_default": len(d),
                "n_pred_super": len(s),
                "new_tp": json.dumps(new_tp),
                "new_fp": json.dumps(new_fp),
                "lost_tp": json.dumps(lost_tp),
                "lost_fp": json.dumps(lost_fp),
                "new_tp_count": len(new_tp),
                "new_fp_count": len(new_fp),
                "lost_tp_count": len(lost_tp),
                "lost_fp_count": len(lost_fp),
            }
        )

    rep = pd.DataFrame(rows).sort_values(["delta", "note_id"], ascending=[False, True])
    # Helpful “most common added/removed concepts” aggregates for quick triage.
    def _count_concepts(col: str) -> Counter[int]:
        c: Counter[int] = Counter()
        for raw in rep[col].tolist():
            for cid in json.loads(raw):
                c[int(cid)] += 1
        return c

    rep.attrs["top_new_fp"] = [
        (cid, n, concept_name.get(cid, ""))
        for cid, n in _count_concepts("new_fp").most_common(25)
    ]
    rep.attrs["top_lost_tp"] = [
        (cid, n, concept_name.get(cid, ""))
        for cid, n in _count_concepts("lost_tp").most_common(25)
    ]
    return rep


def build_concept_delta_report(
    default_class: pd.DataFrame,
    super_class: pd.DataFrame,
    concept_name: dict[int, str],
) -> pd.DataFrame:
    merged = default_class.merge(
        super_class, on="concept_id", how="outer", suffixes=("_default", "_super")
    ).fillna(0)

    merged["delta_iou"] = merged["iou_super"] - merged["iou_default"]
    merged["delta_gt_chars"] = merged["gt_chars_super"] - merged["gt_chars_default"]
    merged["delta_pred_chars"] = merged["pred_chars_super"] - merged["pred_chars_default"]
    merged["delta_intersection"] = merged["intersection_super"] - merged["intersection_default"]
    merged["delta_union"] = merged["union_super"] - merged["union_default"]
    merged["concept_name"] = merged["concept_id"].map(lambda x: concept_name.get(int(x), ""))

    cols = [
        "concept_id",
        "concept_name",
        "iou_default",
        "iou_super",
        "delta_iou",
        "gt_chars_default",
        "pred_chars_default",
        "intersection_default",
        "union_default",
        "gt_chars_super",
        "pred_chars_super",
        "intersection_super",
        "union_super",
        "delta_gt_chars",
        "delta_pred_chars",
        "delta_intersection",
        "delta_union",
    ]
    out = merged[cols].copy()
    out = out.sort_values(
        ["delta_iou", "delta_intersection", "concept_id"], ascending=[False, False, True]
    )
    return out


def build_char_flip_report(
    default_pred: pd.DataFrame,
    super_pred: pd.DataFrame,
    gold: pd.DataFrame,
    concept_name: dict[int, str],
    note_id_source: str = "default",
    top_k: int = 50,
) -> dict[str, pd.DataFrame]:
    if note_id_source == "default":
        note_ids = sorted(set(default_pred["note_id"].unique()))
    elif note_id_source == "super":
        note_ids = sorted(set(super_pred["note_id"].unique()))
    elif note_id_source == "gold":
        note_ids = sorted(set(gold["note_id"].unique()))
    else:
        raise ValueError("--note-id-source must be one of: default, super, gold")

    gold = gold[gold["note_id"].isin(note_ids)].copy()

    gt_by_note = _to_note_spans(gold)
    def_by_note = _to_note_spans(default_pred)
    sup_by_note = _to_note_spans(super_pred)

    lost_correct: Counter[tuple[int, int]] = Counter()
    gained_correct: Counter[tuple[int, int]] = Counter()
    new_fp: Counter[int] = Counter()
    removed_fp: Counter[int] = Counter()

    notes = sorted(set(gt_by_note.keys()) | set(def_by_note.keys()) | set(sup_by_note.keys()))
    for note_id in notes:
        gt = gt_by_note.get(note_id)
        d = def_by_note.get(note_id)
        s = sup_by_note.get(note_id)

        spans = [arr for arr in (gt, d, s) if arr is not None and arr.shape[0] > 0]
        if not spans:
            continue
        boundaries = np.unique(np.concatenate([arr[:, 0] for arr in spans] + [arr[:, 1] for arr in spans]))
        if boundaries.shape[0] < 2:
            continue
        lengths = (boundaries[1:] - boundaries[:-1]).astype(np.int64, copy=False)

        gt_labels = _labels_over_boundaries(boundaries, gt)
        def_labels = _labels_over_boundaries(boundaries, d)
        sup_labels = _labels_over_boundaries(boundaries, s)

        for g, dd, ss, L in zip(gt_labels, def_labels, sup_labels, lengths):
            if L <= 0:
                continue

            if g == 0:
                if dd == 0 and ss != 0:
                    new_fp[int(ss)] += int(L)
                if dd != 0 and ss == 0:
                    removed_fp[int(dd)] += int(L)
                continue

            if dd == g and ss != g:
                lost_correct[(int(g), int(ss))] += int(L)
            if dd != g and ss == g:
                gained_correct[(int(g), int(dd))] += int(L)

    def _pairs_df(counter: Counter[tuple[int, int]], kind: str) -> pd.DataFrame:
        rows = []
        for (gt_cid, other_cid), chars in counter.most_common(top_k):
            rows.append(
                {
                    "gt_concept_id": int(gt_cid),
                    "gt_concept_name": concept_name.get(int(gt_cid), ""),
                    "other_concept_id": int(other_cid),
                    "other_concept_name": concept_name.get(int(other_cid), ""),
                    "chars": int(chars),
                    "kind": kind,
                }
            )
        return pd.DataFrame(rows)

    def _single_df(counter: Counter[int], kind: str) -> pd.DataFrame:
        rows = []
        for cid, chars in counter.most_common(top_k):
            rows.append(
                {
                    "concept_id": int(cid),
                    "concept_name": concept_name.get(int(cid), ""),
                    "chars": int(chars),
                    "kind": kind,
                }
            )
        return pd.DataFrame(rows)

    return {
        "lost_correct_pairs": _pairs_df(lost_correct, "default_correct__super_wrong"),
        "gained_correct_pairs": _pairs_df(gained_correct, "default_wrong__super_correct"),
        "new_fp": _single_df(new_fp, "new_false_positive"),
        "removed_fp": _single_df(removed_fp, "removed_false_positive"),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze where super-dictionary KIRI helps/hurts vs default KIRI."
    )
    parser.add_argument(
        "--default-pred",
        type=Path,
        default=REPO_ROOT / "outputs" / "super_dictionary" / "kiri_default_pred.csv",
    )
    parser.add_argument(
        "--super-pred",
        type=Path,
        default=REPO_ROOT / "outputs" / "super_dictionary" / "kiri_super_pred.csv",
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=REPO_ROOT / "1st Place" / "data" / "interim" / "train_annotations_cln.csv",
    )
    parser.add_argument(
        "--default-class-iou",
        type=Path,
        default=REPO_ROOT / "outputs" / "super_dictionary" / "kiri_default_class_iou.csv",
    )
    parser.add_argument(
        "--super-class-iou",
        type=Path,
        default=REPO_ROOT / "outputs" / "super_dictionary" / "kiri_super_class_iou.csv",
    )
    parser.add_argument(
        "--concept-names",
        type=Path,
        default=REPO_ROOT / "1st Place" / "data" / "interim" / "flattened_terminology.csv",
        help="CSV with columns concept_id, concept_name (for readable reports).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "super_dictionary" / "delta_report",
    )
    parser.add_argument(
        "--note-id-source",
        choices=["default", "super", "gold"],
        default="default",
        help="Which file determines the evaluated note_id set (matches compare_kiri default behavior by default).",
    )
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args(argv)

    concept_name = _concept_name_map(args.concept_names)
    gold = _load_spans(args.gold)
    default_pred = _load_spans(args.default_pred)
    super_pred = _load_spans(args.super_pred)
    default_class = _load_class_iou(args.default_class_iou)
    super_class = _load_class_iou(args.super_class_iou)

    note_rep = build_note_delta_report(
        default_pred, super_pred, gold, concept_name=concept_name, note_id_source=args.note_id_source
    )
    concept_rep = build_concept_delta_report(default_class, super_class, concept_name=concept_name)
    flip = build_char_flip_report(
        default_pred,
        super_pred,
        gold,
        concept_name=concept_name,
        note_id_source=args.note_id_source,
        top_k=args.top_k,
    )

    out_dir = args.out_dir
    _write_df(note_rep, out_dir / "note_deltas.csv")
    _write_df(concept_rep, out_dir / "concept_deltas.csv")
    for k, df in flip.items():
        _write_df(df, out_dir / f"{k}.tsv")

    improved = int((note_rep["delta"] > 0).sum())
    worsened = int((note_rep["delta"] < 0).sum())
    unchanged = int((note_rep["delta"] == 0).sum())
    print(
        f"Notes: {len(note_rep)} (improved={improved}, worsened={worsened}, unchanged={unchanged})"
    )
    print(
        f"Mean note-IoU: default={note_rep['iou_default'].mean():0.4f} super={note_rep['iou_super'].mean():0.4f} (Δ={note_rep['delta'].mean():+0.4f})"
    )
    print()
    print("Top new false-positive concepts (count of notes):")
    for cid, n, name in note_rep.attrs.get("top_new_fp", [])[:10]:
        print(f"  {cid}  {n:>3}  {name}")
    print()
    print("Top lost true-positive concepts (count of notes):")
    for cid, n, name in note_rep.attrs.get("top_lost_tp", [])[:10]:
        print(f"  {cid}  {n:>3}  {name}")
    print()
    print(f"Wrote: {out_dir}/note_deltas.csv")
    print(f"Wrote: {out_dir}/concept_deltas.csv")
    print(f"Wrote: {out_dir}/lost_correct_pairs.tsv")
    print(f"Wrote: {out_dir}/gained_correct_pairs.tsv")
    print(f"Wrote: {out_dir}/new_fp.tsv")
    print(f"Wrote: {out_dir}/removed_fp.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
