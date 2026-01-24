from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


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

