#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.llm_postprocess.schema import (
    AgentConstraints,
    EditScript,
    Span,
    coerce_edit_script,
    parse_script,
    spans_from_pred_rows,
    validate_and_normalize_edits,
)


def _stable_note_spans(df_note: pd.DataFrame) -> pd.DataFrame:
    return df_note.sort_values(["start", "end", "concept_id"], kind="mergesort").reset_index(
        drop=True
    )


def _remove_overlaps_simple(df_note: pd.DataFrame) -> pd.DataFrame:
    if df_note.empty:
        return df_note
    df = df_note.sort_values(["start", "end", "concept_id"], kind="mergesort").reset_index(
        drop=True
    )
    to_remove: set[int] = set()
    n = len(df)
    for i in range(n):
        if i in to_remove:
            continue
        si = int(df.at[i, "start"])
        ei = int(df.at[i, "end"])
        li = ei - si
        for j in range(i + 1, n):
            sj = int(df.at[j, "start"])
            if sj >= ei:
                break
            ej = int(df.at[j, "end"])
            lj = ej - sj
            # Remove the shorter span; tie-breaker keep earlier.
            if lj < li:
                to_remove.add(j)
            else:
                to_remove.add(i)
                break
    if not to_remove:
        return df
    return df.drop(sorted(to_remove)).reset_index(drop=True)


def apply_edit_script_to_note_pred(
    *,
    note_id: str,
    pred_note: pd.DataFrame,
    script: EditScript,
    note_len: int | None,
    constraints: AgentConstraints,
) -> tuple[pd.DataFrame, list[str]]:
    pred_note = _stable_note_spans(pred_note)
    spans = spans_from_pred_rows(
        pred_note[["start", "end", "concept_id"]].itertuples(index=False, name=None)
    )
    normalized, warnings = validate_and_normalize_edits(
        script=script,
        expected_note_id=note_id,
        spans=spans,
        note_len=note_len,
        constraints=constraints,
    )

    delete_idx = {e.idx for e in normalized.edits if e.op == "delete"}
    shift_by_idx = {
        e.idx: (int(e.start), int(e.end))
        for e in normalized.edits
        if e.op == "shift" and e.start is not None and e.end is not None
    }

    rows = []
    for idx, row in enumerate(
        pred_note[["start", "end", "concept_id"]].itertuples(index=False, name=None)
    ):
        if idx in delete_idx:
            continue
        start, end, cid = (int(row[0]), int(row[1]), int(row[2]))
        if idx in shift_by_idx:
            start, end = shift_by_idx[idx]
        rows.append((note_id, start, end, cid))

    out = pd.DataFrame(rows, columns=["note_id", "start", "end", "concept_id"])
    out = _remove_overlaps_simple(out)
    return out, warnings


def load_scripts_jsonl(path: Path) -> dict[str, EditScript]:
    scripts: dict[str, EditScript] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"invalid jsonl at line {line_no}") from e
            if not isinstance(obj, dict):
                continue
            note_id = obj.get("note_id")
            raw_text = obj.get("raw")
            parsed = obj.get("parsed")
            if isinstance(parsed, dict):
                script_obj = parsed
            elif isinstance(raw_text, str):
                script_obj = parse_script(raw_text)
            elif "edits" in obj and isinstance(obj.get("edits"), list):
                # Allow direct edit-script JSONL (one script per line).
                script_obj = obj
            else:
                continue
            script = coerce_edit_script(script_obj)
            if isinstance(note_id, str) and note_id:
                scripts[note_id] = script
            else:
                scripts[script.note_id] = script
    return scripts


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Apply LLM edit scripts to KIRI *_pred.csv.")
    ap.add_argument("--pred-csv", required=True, help="Input KIRI predictions CSV.")
    ap.add_argument("--scripts-jsonl", required=True, help="JSONL of model outputs / scripts.")
    ap.add_argument("--notes-csv", default=None, help="Optional notes CSV (note_id,text).")
    ap.add_argument("--out-csv", required=True, help="Output predictions CSV after edits.")
    ap.add_argument("--max-edits", type=int, default=10)
    ap.add_argument("--max-shift", type=int, default=30)
    args = ap.parse_args(argv)

    pred = pd.read_csv(args.pred_csv)
    scripts = load_scripts_jsonl(Path(args.scripts_jsonl))

    note_len_by_id: dict[str, int] = {}
    if args.notes_csv:
        notes = pd.read_csv(args.notes_csv).set_index("note_id")["text"]
        note_len_by_id = {str(k): int(len(str(v))) for k, v in notes.items()}

    constraints = AgentConstraints(max_edits_per_note=args.max_edits, max_shift_chars=args.max_shift)
    out_notes = []
    all_warnings: list[str] = []
    for note_id, df_note in pred.groupby("note_id", sort=False):
        note_id = str(note_id)
        script = scripts.get(note_id, EditScript(note_id=note_id, edits=()))
        note_len = note_len_by_id.get(note_id)
        out_note, warnings = apply_edit_script_to_note_pred(
            note_id=note_id,
            pred_note=df_note,
            script=script,
            note_len=note_len,
            constraints=constraints,
        )
        out_notes.append(out_note)
        for w in warnings:
            all_warnings.append(f"{note_id}: {w}")

    out_df = pd.concat(out_notes, ignore_index=True) if out_notes else pred.iloc[:0].copy()
    out_df.to_csv(args.out_csv, index=False)

    warn_path = Path(args.out_csv).with_suffix(".warnings.txt")
    if all_warnings:
        warn_path.write_text("\n".join(all_warnings) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
