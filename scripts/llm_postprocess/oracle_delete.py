#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.llm_postprocess.apply_edits import apply_edit_script_to_note_pred, _stable_note_spans
from scripts.llm_postprocess.schema import AgentConstraints, Edit, EditScript


@dataclass(frozen=True)
class Interval:
    start: int
    end: int


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if not intervals:
        return []
    intervals_sorted = sorted(intervals, key=lambda x: (x.start, x.end))
    merged: list[Interval] = [intervals_sorted[0]]
    for itv in intervals_sorted[1:]:
        last = merged[-1]
        if itv.start <= last.end:
            merged[-1] = Interval(start=last.start, end=max(last.end, itv.end))
        else:
            merged.append(itv)
    return merged


def _intervals_total_len(intervals: list[Interval]) -> int:
    return sum(max(0, itv.end - itv.start) for itv in intervals)


def _overlap_len(span_start: int, span_end: int, intervals: list[Interval]) -> int:
    if span_end <= span_start or not intervals:
        return 0
    total = 0
    for itv in intervals:
        if itv.end <= span_start:
            continue
        if itv.start >= span_end:
            break
        total += max(0, min(span_end, itv.end) - max(span_start, itv.start))
    return total


def _gold_intervals_by_cid(gold_note: pd.DataFrame) -> dict[int, list[Interval]]:
    out: dict[int, list[Interval]] = defaultdict(list)
    for start, end, cid in gold_note[["start", "end", "concept_id"]].itertuples(
        index=False, name=None
    ):
        out[int(cid)].append(Interval(int(start), int(end)))
    return {cid: _merge_intervals(v) for cid, v in out.items()}


def _score_note_macro_iou(pred_note_clean: pd.DataFrame, gold_note: pd.DataFrame) -> float:
    pred_by_cid: dict[int, list[Interval]] = defaultdict(list)
    for start, end, cid in pred_note_clean[["start", "end", "concept_id"]].itertuples(
        index=False, name=None
    ):
        pred_by_cid[int(cid)].append(Interval(int(start), int(end)))

    gold_by_cid = _gold_intervals_by_cid(gold_note)

    cids = set(pred_by_cid.keys()) | set(gold_by_cid.keys())
    if not cids:
        return 0.0

    total = 0.0
    n = 0
    for cid in sorted(cids):
        pred_intervals = _merge_intervals(pred_by_cid.get(cid, []))
        gold_intervals = gold_by_cid.get(cid, [])
        pred_len = _intervals_total_len(pred_intervals)
        gold_len = _intervals_total_len(gold_intervals)
        inter_len = sum(
            _overlap_len(itv.start, itv.end, gold_intervals) for itv in pred_intervals
        )
        union = pred_len + gold_len - inter_len
        if union <= 0:
            continue
        total += inter_len / union
        n += 1
    return float(total / max(1, n))


def _oracle_delete_greedy(
    *,
    note_id: str,
    pred_note: pd.DataFrame,
    gold_note: pd.DataFrame,
    note_len: int | None,
    max_edits: int,
    max_candidates: int,
) -> tuple[EditScript, float, float]:
    pred_sorted = _stable_note_spans(pred_note)
    constraints = AgentConstraints(max_edits_per_note=max_edits, max_shift_chars=0)

    # Baseline (no edits) uses the same postprocess pipeline: stable sort + overlap removal.
    baseline_clean, _ = apply_edit_script_to_note_pred(
        note_id=note_id,
        pred_note=pred_sorted,
        script=EditScript(note_id=note_id, edits=()),
        note_len=note_len,
        constraints=constraints,
    )
    baseline_score = _score_note_macro_iou(baseline_clean, gold_note)

    gold_by_cid = _gold_intervals_by_cid(gold_note)

    # Candidate spans: prioritize spans with many FP chars (len - overlap with gold for same concept).
    candidates: list[tuple[int, int]] = []  # (fp_chars, idx)
    for idx, (start, end, cid) in enumerate(
        pred_sorted[["start", "end", "concept_id"]].itertuples(index=False, name=None)
    ):
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            continue
        cid_i = int(cid)
        overlap = _overlap_len(start_i, end_i, gold_by_cid.get(cid_i, []))
        fp_chars = max(0, (end_i - start_i) - overlap)
        if fp_chars <= 0:
            continue
        candidates.append((fp_chars, idx))

    candidates.sort(reverse=True)
    cand_idxs = [idx for _, idx in candidates[: max(0, int(max_candidates))]]

    chosen: list[int] = []
    best_score = baseline_score

    for _step in range(max(0, int(max_edits))):
        best_next = None
        best_next_score = best_score
        for idx in cand_idxs:
            if idx in chosen:
                continue
            trial = chosen + [idx]
            trial_script = EditScript(
                note_id=note_id,
                edits=tuple(Edit(op="delete", idx=int(i)) for i in trial),
            )
            trial_clean, _ = apply_edit_script_to_note_pred(
                note_id=note_id,
                pred_note=pred_sorted,
                script=trial_script,
                note_len=note_len,
                constraints=constraints,
            )
            s = _score_note_macro_iou(trial_clean, gold_note)
            if s > best_next_score + 1e-12:
                best_next_score = s
                best_next = idx
        if best_next is None:
            break
        chosen.append(best_next)
        best_score = best_next_score

    script = EditScript(
        note_id=note_id,
        edits=tuple(Edit(op="delete", idx=int(i)) for i in chosen),
    )
    return script, baseline_score, best_score


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Generate oracle delete-only edit scripts that maximize per-note macro IoU on a subset (uses gold)."
    )
    ap.add_argument("--pred-csv", required=True)
    ap.add_argument("--gold-csv", required=True)
    ap.add_argument("--notes-csv", required=True, help="Notes CSV with columns (note_id,text).")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-pred-csv", default=None, help="Optional: apply oracle scripts and write edited pred CSV.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-edits", type=int, default=5)
    ap.add_argument("--max-candidates", type=int, default=80)
    args = ap.parse_args(argv)

    pred = pd.read_csv(args.pred_csv)
    gold = pd.read_csv(args.gold_csv)
    notes = pd.read_csv(args.notes_csv)
    if "note_id" not in notes.columns or "text" not in notes.columns:
        raise ValueError("notes csv must have columns: note_id,text")
    note_len_by_id = {
        str(nid): int(len(str(txt))) for nid, txt in notes[["note_id", "text"]].itertuples(index=False, name=None)
    }

    note_ids = pred["note_id"].astype(str).unique().tolist()
    if args.limit is not None:
        note_ids = note_ids[: max(0, int(args.limit))]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scripts: dict[str, EditScript] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for note_id in note_ids:
            pred_note = pred[pred["note_id"].astype(str) == note_id].copy()
            gold_note = gold[gold["note_id"].astype(str) == note_id].copy()
            note_len = note_len_by_id.get(str(note_id))
            script, baseline_score, oracle_score = _oracle_delete_greedy(
                note_id=str(note_id),
                pred_note=pred_note,
                gold_note=gold_note,
                note_len=note_len,
                max_edits=int(args.max_edits),
                max_candidates=int(args.max_candidates),
            )
            scripts[str(note_id)] = script
            record = {
                "note_id": str(note_id),
                "raw": json.dumps(
                    {"note_id": script.note_id, "edits": [e.__dict__ for e in script.edits]},
                    ensure_ascii=False,
                ),
                "parsed": {"note_id": script.note_id, "edits": [e.__dict__ for e in script.edits]},
                "errors": [],
                "warnings": [],
                "baseline_score": float(baseline_score),
                "oracle_score": float(oracle_score),
                "delta": float(oracle_score - baseline_score),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.out_pred_csv:
        # Apply scripts using the same code path as the agent output application.
        from scripts.llm_postprocess.apply_edits import apply_edit_script_to_note_pred

        out_notes = []
        constraints = AgentConstraints(max_edits_per_note=int(args.max_edits), max_shift_chars=0)
        for note_id, df_note in pred.groupby("note_id", sort=False):
            note_id_s = str(note_id)
            if note_id_s not in scripts:
                continue
            note_len = note_len_by_id.get(note_id_s)
            out_note, _ = apply_edit_script_to_note_pred(
                note_id=note_id_s,
                pred_note=df_note,
                script=scripts[note_id_s],
                note_len=note_len,
                constraints=constraints,
            )
            out_notes.append(out_note)
        out_df = pd.concat(out_notes, ignore_index=True) if out_notes else pred.iloc[:0].copy()
        out_csv = Path(args.out_pred_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))

