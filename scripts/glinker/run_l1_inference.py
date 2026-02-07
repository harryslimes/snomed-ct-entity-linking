#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.io import detect_delimiter


@dataclass(frozen=True)
class L1Span:
    note_id: str
    start_char: int
    end_char: int
    mention: str
    l1_type: str
    score: float
    raw_label: str


def normalize_l1_type(value: str | None) -> str | None:
    if value is None:
        return None
    t = str(value).strip().lower().replace("-", " ").replace("_", " ")
    t = " ".join(t.split())
    if not t:
        return None
    if t in {"finding", "clinical finding", "disorder"}:
        return "finding"
    if t in {"procedure", "regime/therapy"}:
        return "procedure"
    if t in {"body structure", "body", "morphologic abnormality", "cell structure"}:
        return "body_structure"
    return None


def load_notes(notes_csv: Path, *, note_id_col: str, text_col: str) -> list[tuple[str, str]]:
    delim = detect_delimiter(notes_csv)
    out: list[tuple[str, str]] = []
    with notes_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter=delim)
        for row in reader:
            note_id = str(row.get(note_id_col) or "").strip()
            if not note_id:
                continue
            out.append((note_id, str(row.get(text_col) or "")))
    return out


def _load_gliner_model(model_path: str):
    try:
        from gliner import GLiNER  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import gliner. Install it in your environment before running L1 inference."
        ) from e
    return GLiNER.from_pretrained(model_path)


def _predict_entities_raw(
    model,
    text: str,
    *,
    labels: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    # GLiNER APIs can differ slightly by version; try common signatures.
    try:
        out = model.predict_entities(text, labels=labels, threshold=threshold)
    except TypeError:
        out = model.predict_entities(text, labels=labels)
    if out is None:
        return []
    return [dict(x) for x in out]


def _iter_windows(text: str, *, window_size: int, overlap: int) -> Iterable[tuple[int, int]]:
    if window_size <= 0 or window_size >= len(text):
        yield (0, len(text))
        return
    stride = max(1, window_size - max(0, overlap))
    n = len(text)
    pos = 0
    while pos < n:
        end = min(n, pos + window_size)
        yield (pos, end)
        if end >= n:
            break
        pos += stride


def extract_l1_spans_for_note(
    *,
    model,
    note_id: str,
    text: str,
    labels: list[str],
    threshold: float,
    window_chars: int,
    window_overlap_chars: int,
    strict_label_filter: bool,
) -> list[L1Span]:
    spans: list[L1Span] = []
    seen: set[tuple[int, int, str]] = set()
    for start, end in _iter_windows(
        text, window_size=window_chars, overlap=window_overlap_chars
    ):
        chunk = text[start:end]
        preds = _predict_entities_raw(model, chunk, labels=labels, threshold=threshold)
        for pred in preds:
            p_start = pred.get("start")
            p_end = pred.get("end")
            raw_label = str(pred.get("label") or pred.get("entity") or "").strip()
            mention = str(pred.get("text") or "")
            score = float(pred.get("score") or 0.0)

            if p_start is None or p_end is None:
                continue
            try:
                s_local = int(p_start)
                e_local = int(p_end)
            except Exception:
                continue
            if e_local <= s_local:
                continue

            s = start + s_local
            e = start + e_local
            if e <= s or s < 0 or e > len(text):
                continue
            if not mention:
                mention = text[s:e]

            l1_type = normalize_l1_type(raw_label)
            if strict_label_filter and l1_type is None:
                continue
            if l1_type is None:
                # Keep unknowns only when not strict, but mark clearly.
                l1_type = "unknown"

            key = (s, e, l1_type)
            if key in seen:
                continue
            seen.add(key)
            spans.append(
                L1Span(
                    note_id=note_id,
                    start_char=s,
                    end_char=e,
                    mention=mention,
                    l1_type=l1_type,
                    score=score,
                    raw_label=raw_label,
                )
            )
    spans.sort(key=lambda x: (x.start_char, x.end_char, x.l1_type, -x.score))
    return spans


def run_inference(
    *,
    notes_csv: Path,
    model_path: str,
    out_spans_csv: Path,
    note_id_col: str = "note_id",
    text_col: str = "text",
    entity_types: list[str] | None = None,
    threshold: float = 0.4,
    strict_label_filter: bool = True,
    window_chars: int = 0,
    window_overlap_chars: int = 256,
    limit_notes: int = 0,
    model=None,
) -> dict[str, int]:
    labels = entity_types or ["finding", "procedure", "body_structure"]
    model_obj = model or _load_gliner_model(model_path)
    notes = load_notes(notes_csv, note_id_col=note_id_col, text_col=text_col)
    if limit_notes > 0:
        notes = notes[:limit_notes]

    out_spans_csv.parent.mkdir(parents=True, exist_ok=True)
    n_notes = 0
    n_spans = 0
    with out_spans_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "mention_id",
                "note_id",
                "start_char",
                "end_char",
                "mention",
                "l1_type",
                "l1_score",
                "l1_label_raw",
            ]
        )
        for note_idx, (note_id, text) in enumerate(notes):
            note_spans = extract_l1_spans_for_note(
                model=model_obj,
                note_id=note_id,
                text=text,
                labels=labels,
                threshold=threshold,
                window_chars=window_chars,
                window_overlap_chars=window_overlap_chars,
                strict_label_filter=strict_label_filter,
            )
            n_notes += 1
            for i, s in enumerate(note_spans):
                mention_id = f"{note_id}:{s.start_char}:{s.end_char}:{i}"
                writer.writerow(
                    [
                        mention_id,
                        s.note_id,
                        s.start_char,
                        s.end_char,
                        s.mention,
                        s.l1_type,
                        f"{s.score:.6f}",
                        s.raw_label,
                    ]
                )
                n_spans += 1
    return {"notes": n_notes, "spans": n_spans}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run L1 GLiNER span extraction over notes and write standardized spans CSV."
        )
    )
    ap.add_argument("--notes-csv", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-spans-csv", required=True)
    ap.add_argument("--note-id-col", default="note_id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--entity-types", default="finding,procedure,body_structure")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--no-strict-label-filter", action="store_true")
    ap.add_argument("--window-chars", type=int, default=0)
    ap.add_argument("--window-overlap-chars", type=int, default=256)
    ap.add_argument("--limit-notes", type=int, default=0)
    args = ap.parse_args(argv)

    entity_types = [x.strip() for x in str(args.entity_types).split(",") if x.strip()]
    stats = run_inference(
        notes_csv=Path(args.notes_csv),
        model_path=str(args.model_path),
        out_spans_csv=Path(args.out_spans_csv),
        note_id_col=str(args.note_id_col),
        text_col=str(args.text_col),
        entity_types=entity_types,
        threshold=float(args.threshold),
        strict_label_filter=not bool(args.no_strict_label_filter),
        window_chars=max(0, int(args.window_chars)),
        window_overlap_chars=max(0, int(args.window_overlap_chars)),
        limit_notes=max(0, int(args.limit_notes)),
    )
    print(f"processed notes: {stats['notes']:,}")
    print(f"predicted spans: {stats['spans']:,}")
    print(f"out_spans_csv: {args.out_spans_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

