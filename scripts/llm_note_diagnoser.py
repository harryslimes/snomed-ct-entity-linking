#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from iou_metric import ensure_columns


FSN_TYPE_ID = "900000000000003001"


@dataclass(frozen=True)
class Span:
    note_id: str
    start: int
    end: int
    concept_id: int
    idx: int  # stable row index within the loaded CSV


def overlap_len(a: Span, b: Span) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def span_iou(a: Span, b: Span) -> float:
    inter = overlap_len(a, b)
    if inter <= 0:
        return 0.0
    union = (a.end - a.start) + (b.end - b.start) - inter
    return (inter / union) if union else 0.0


def span_distance(a: Span, b: Span) -> int:
    # 0 if overlapping; otherwise the gap between intervals.
    if overlap_len(a, b) > 0:
        return 0
    if a.end <= b.start:
        return b.start - a.end
    if b.end <= a.start:
        return a.start - b.end
    return 0


def safe_slice(text: str, start: int, end: int) -> str:
    start = max(0, start)
    end = max(start, end)
    if start >= len(text):
        return ""
    end = min(len(text), end)
    return text[start:end]


def safe_context(text: str, start: int, end: int, window: int) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return safe_slice(text, left, right)


def make_spans(df: pd.DataFrame) -> list[Span]:
    spans: list[Span] = []
    for idx, (note_id, start, end, concept_id) in enumerate(
        df[["note_id", "start", "end", "concept_id"]].itertuples(index=False, name=None)
    ):
        spans.append(
            Span(note_id=str(note_id), start=int(start), end=int(end), concept_id=int(concept_id), idx=idx)
        )
    return spans


def greedy_overlap_matching(
    gold_spans: list[Span],
    pred_spans: list[Span],
    *,
    require_same_concept: bool,
) -> tuple[dict[int, int], set[int], set[int]]:
    # Returns mapping gold_idx -> pred_idx for pairs with overlap > 0, plus unmatched sets.
    candidates: list[tuple[int, float, int, int]] = []
    for g in gold_spans:
        for p in pred_spans:
            if require_same_concept and g.concept_id != p.concept_id:
                continue
            inter = overlap_len(g, p)
            if inter <= 0:
                continue
            candidates.append((inter, span_iou(g, p), g.idx, p.idx))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    matched_gold: set[int] = set()
    matched_pred: set[int] = set()
    pairs: dict[int, int] = {}
    for inter, iou, g_idx, p_idx in candidates:
        if g_idx in matched_gold or p_idx in matched_pred:
            continue
        matched_gold.add(g_idx)
        matched_pred.add(p_idx)
        pairs[g_idx] = p_idx

    unmatched_gold = {g.idx for g in gold_spans if g.idx not in matched_gold}
    unmatched_pred = {p.idx for p in pred_spans if p.idx not in matched_pred}
    return pairs, unmatched_gold, unmatched_pred


def relaxed_same_concept_matching(
    gold_spans: list[Span],
    pred_spans: list[Span],
    unmatched_gold: set[int],
    unmatched_pred: set[int],
    *,
    max_distance: int,
) -> dict[int, int]:
    # Pair (gold, pred) with same concept_id, no overlap, minimal distance.
    pred_by_concept: dict[int, list[Span]] = defaultdict(list)
    for p in pred_spans:
        if p.idx in unmatched_pred:
            pred_by_concept[p.concept_id].append(p)

    pairs: dict[int, int] = {}
    used_pred: set[int] = set()
    for g in gold_spans:
        if g.idx not in unmatched_gold:
            continue
        candidates = []
        for p in pred_by_concept.get(g.concept_id, []):
            if p.idx in used_pred:
                continue
            if overlap_len(g, p) > 0:
                continue
            dist = span_distance(g, p)
            if max_distance >= 0 and dist > max_distance:
                continue
            candidates.append((dist, abs((g.end - g.start) - (p.end - p.start)), p.idx))
        if not candidates:
            continue
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        best_pred_idx = candidates[0][2]
        pairs[g.idx] = best_pred_idx
        used_pred.add(best_pred_idx)
    return pairs


def find_snomed_description_snapshot(data_dir: Path) -> Path | None:
    for release_dir in sorted(data_dir.glob("SnomedCT_*")):
        candidate = release_dir / "Snapshot" / "Terminology"
        if not candidate.exists():
            continue
        matches = sorted(candidate.glob("sct2_Description_Snapshot-en_*_*.txt"))
        if matches:
            return matches[0]
    return None


def load_fsn_terms(
    description_path: Path,
    concept_ids: set[int],
    *,
    language_code: str = "en",
) -> dict[int, str]:
    try:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    except Exception:
        pass

    out: dict[int, str] = {}
    with description_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"active", "conceptId", "languageCode", "typeId", "term"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Unexpected SNOMED description format, missing: {sorted(missing)}")

        for row in reader:
            if row.get("active") != "1":
                continue
            if row.get("languageCode") != language_code:
                continue
            if row.get("typeId") != FSN_TYPE_ID:
                continue
            try:
                cid = int(row["conceptId"])
            except Exception:
                continue
            if cid not in concept_ids or cid in out:
                continue
            term = (row.get("term") or "").strip()
            if term:
                out[cid] = term
    return out


def load_eval_paths(eval_json: Path) -> tuple[Path | None, Path | None, Path | None]:
    payload = json.loads(eval_json.read_text())
    pred_path = payload.get("submission_csv")
    gold_path = payload.get("annotations_path")
    notes_path = payload.get("notes_path")
    return (
        Path(pred_path) if pred_path else None,
        Path(gold_path) if gold_path else None,
        Path(notes_path) if notes_path else None,
    )


def pick_eval_json_for_entry(entry: str) -> Path:
    outputs = Path("outputs")
    if entry == "2nd":
        candidate = outputs / "2nd_place_train_eval.json"
        if candidate.exists():
            return candidate
        matches = sorted(outputs.glob("2nd_place*_eval.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
        raise FileNotFoundError("Could not find outputs/2nd_place*_eval.json. Run `python scripts/evaluate_2nd_place.py --copy-submission` first.")

    if entry == "1st":
        matches = sorted(outputs.glob("1st_place*_eval.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
        raise FileNotFoundError("Could not find outputs/1st_place*_eval.json. Run `python scripts/evaluate_1st_place.py` first.")

    raise ValueError(f"Unsupported entry: {entry}")


def load_note_text(notes_csv: Path) -> dict[str, str]:
    notes = pd.read_csv(notes_csv, usecols=["note_id", "text"])
    notes["note_id"] = notes["note_id"].astype(str)
    notes["text"] = notes["text"].astype(str)
    return dict(zip(notes["note_id"].tolist(), notes["text"].tolist(), strict=False))


def to_concept_list(concept_ids: set[int], terms: dict[int, str] | None, max_items: int) -> list[dict]:
    ids_sorted = sorted(concept_ids)
    if max_items > 0:
        ids_sorted = ids_sorted[:max_items]
    out = []
    for cid in ids_sorted:
        obj = {"concept_id": cid}
        if terms is not None:
            obj["term"] = terms.get(cid, "")
        out.append(obj)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export per-note, LLM-ready diagnostic records from predictions + gold annotations."
    )
    parser.add_argument(
        "--entry",
        choices=["1st", "2nd", "custom"],
        default="2nd",
        help="Which competition entry's outputs to use by default (default: 2nd). Use custom to pass explicit CSV paths.",
    )
    parser.add_argument(
        "--eval-json",
        default=None,
        help="Evaluation JSON to derive pred/gold/notes paths (overrides --entry defaults).",
    )
    parser.add_argument("--pred-csv", default=None, help="Predictions CSV (overrides eval-json).")
    parser.add_argument("--gold-csv", default=None, help="Gold annotations CSV (overrides eval-json).")
    parser.add_argument("--notes-csv", default=None, help="Notes CSV with text (overrides eval-json).")
    parser.add_argument(
        "--output",
        default="outputs/llm_note_diagnosis.jsonl",
        help="Where to write JSONL (default: outputs/llm_note_diagnosis.jsonl). Use '-' for stdout.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory used to auto-find a SNOMED release (default: ./data).",
    )
    parser.add_argument(
        "--snomed-description",
        default=None,
        help="Path to SNOMED description snapshot TSV (overrides auto-detection).",
    )
    parser.add_argument("--no-terms", action="store_true", help="Do not include FSN terms.")
    parser.add_argument("--language-code", default="en", help="languageCode filter (default: en).")
    parser.add_argument("--context-window", type=int, default=60, help="Context window in chars (default: 60).")
    parser.add_argument(
        "--max-distance",
        type=int,
        default=50,
        help="Max character distance for relaxed same-concept, no-overlap pairing (default: 50). Use -1 for no limit.",
    )
    parser.add_argument("--note-id", default=None, help="Only export this note_id.")
    parser.add_argument("--note-limit", type=int, default=0, help="Limit number of notes exported (0 = all).")
    parser.add_argument("--max-items", type=int, default=50, help="Max concepts to list per note (default: 50). Use 0 for all.")
    parser.add_argument("--max-wrong", type=int, default=15, help="Max wrong-concept overlap pairs per note.")
    parser.add_argument("--max-boundary", type=int, default=15, help="Max boundary-mismatch pairs per note.")
    parser.add_argument("--max-no-overlap", type=int, default=15, help="Max correct-concept-no-overlap pairs per note.")
    parser.add_argument("--max-missed", type=int, default=15, help="Max missed gold spans per note.")
    parser.add_argument("--max-spurious", type=int, default=15, help="Max spurious predicted spans per note.")
    args = parser.parse_args()

    eval_json: Path | None = Path(args.eval_json) if args.eval_json else None
    if eval_json is None and args.entry != "custom":
        eval_json = pick_eval_json_for_entry(args.entry)

    pred_csv: Path | None = Path(args.pred_csv) if args.pred_csv else None
    gold_csv: Path | None = Path(args.gold_csv) if args.gold_csv else None
    notes_csv: Path | None = Path(args.notes_csv) if args.notes_csv else None

    if eval_json is not None:
        if not eval_json.exists():
            raise FileNotFoundError(f"Eval JSON not found: {eval_json}")
        default_pred, default_gold, default_notes = load_eval_paths(eval_json)
        pred_csv = pred_csv or default_pred
        gold_csv = gold_csv or default_gold
        notes_csv = notes_csv or default_notes

    if pred_csv is None or not pred_csv.exists():
        raise FileNotFoundError(
            f"Predictions CSV not found: {pred_csv}. Pass --pred-csv, or an --eval-json that contains submission_csv."
        )
    if gold_csv is None or not gold_csv.exists():
        raise FileNotFoundError(
            f"Gold annotations CSV not found: {gold_csv}. Pass --gold-csv, or an --eval-json that contains annotations_path."
        )
    if notes_csv is None or not notes_csv.exists():
        raise FileNotFoundError(
            f"Notes CSV not found: {notes_csv}. Pass --notes-csv, or an --eval-json that contains notes_path."
        )

    pred_df = ensure_columns(pd.read_csv(pred_csv), "pred")
    gold_df = ensure_columns(pd.read_csv(gold_csv), "gold")
    pred_df["note_id"] = pred_df["note_id"].astype(str)
    gold_df["note_id"] = gold_df["note_id"].astype(str)

    note_text = load_note_text(notes_csv)

    pred_spans_all = make_spans(pred_df)
    gold_spans_all = make_spans(gold_df)

    pred_by_note: dict[str, list[Span]] = defaultdict(list)
    gold_by_note: dict[str, list[Span]] = defaultdict(list)
    for p in pred_spans_all:
        pred_by_note[p.note_id].append(p)
    for g in gold_spans_all:
        gold_by_note[g.note_id].append(g)

    note_ids = sorted(set(pred_by_note.keys()) | set(gold_by_note.keys()))
    if args.note_id is not None:
        note_ids = [nid for nid in note_ids if nid == args.note_id]
    if args.note_limit and args.note_limit > 0:
        note_ids = note_ids[: args.note_limit]

    pred_concepts_global = {s.concept_id for s in pred_spans_all}
    gold_concepts_global = {s.concept_id for s in gold_spans_all}
    global_miss_concepts = gold_concepts_global - pred_concepts_global

    terms: dict[int, str] | None = None
    description_path: Path | None = None
    if not args.no_terms:
        all_concepts = pred_concepts_global | gold_concepts_global
        if args.snomed_description:
            description_path = Path(args.snomed_description)
        else:
            description_path = find_snomed_description_snapshot(Path(args.data_dir))
        if description_path and description_path.exists():
            terms = load_fsn_terms(description_path, all_concepts, language_code=args.language_code)

    def span_obj(span: Span, text: str, *, kind: str) -> dict:
        obj = {
            "start": span.start,
            "end": span.end,
            "len": span.end - span.start,
            "concept_id": span.concept_id,
            "span_id": f"{span.note_id}:{kind}{span.idx}",
            "text": safe_slice(text, span.start, span.end).replace("\n", "\\n"),
        }
        if terms is not None:
            obj["term"] = terms.get(span.concept_id, "")
        return obj

    out_f = sys.stdout if args.output == "-" else Path(args.output).open("w", encoding="utf-8")
    try:
        for nid in note_ids:
            text = note_text.get(nid, "")
            gold_spans = gold_by_note.get(nid, [])
            pred_spans = pred_by_note.get(nid, [])

            gold_concepts = {s.concept_id for s in gold_spans}
            pred_concepts = {s.concept_id for s in pred_spans}
            missing_concepts = gold_concepts - pred_concepts
            extra_concepts = pred_concepts - gold_concepts

            # Two-stage strict overlap matching: same-concept first, then any overlap.
            stage1_pairs, ug1, up1 = greedy_overlap_matching(gold_spans, pred_spans, require_same_concept=True)
            gold_remaining = [g for g in gold_spans if g.idx in ug1]
            pred_remaining = [p for p in pred_spans if p.idx in up1]
            stage2_pairs, ug2, up2 = greedy_overlap_matching(gold_remaining, pred_remaining, require_same_concept=False)

            strict_pairs: dict[int, int] = {}
            strict_pairs.update(stage1_pairs)
            strict_pairs.update(stage2_pairs)

            unmatched_gold = ug2
            unmatched_pred = up2

            relaxed_pairs = relaxed_same_concept_matching(
                gold_spans, pred_spans, unmatched_gold, unmatched_pred, max_distance=args.max_distance
            )
            unmatched_gold = unmatched_gold - set(relaxed_pairs.keys())
            unmatched_pred = unmatched_pred - set(relaxed_pairs.values())

            gold_by_idx = {g.idx: g for g in gold_spans}
            pred_by_idx = {p.idx: p for p in pred_spans}

            wrong_pairs = []
            boundary_pairs = []
            no_overlap_pairs = []

            for g_idx, p_idx in strict_pairs.items():
                g = gold_by_idx[g_idx]
                p = pred_by_idx[p_idx]
                inter = overlap_len(g, p)
                iou_val = span_iou(g, p)
                record = {
                    "gold": span_obj(g, text, kind="g"),
                    "pred": span_obj(p, text, kind="p"),
                    "overlap_len": inter,
                    "span_iou": round(iou_val, 6),
                    "span_distance": 0,
                    "context": safe_context(text, g.start, g.end, args.context_window).replace("\n", "\\n"),
                }
                if g.concept_id == p.concept_id:
                    if iou_val < 0.999999:
                        boundary_pairs.append(record)
                else:
                    wrong_pairs.append(record)

            for g_idx, p_idx in relaxed_pairs.items():
                g = gold_by_idx[g_idx]
                p = pred_by_idx[p_idx]
                dist = span_distance(g, p)
                no_overlap_pairs.append(
                    {
                        "gold": span_obj(g, text, kind="g"),
                        "pred": span_obj(p, text, kind="p"),
                        "overlap_len": 0,
                        "span_iou": 0.0,
                        "span_distance": dist,
                        "context": safe_context(text, g.start, g.end, args.context_window).replace("\n", "\\n"),
                    }
                )

            missed_spans = []
            for g_idx in sorted(unmatched_gold):
                g = gold_by_idx[g_idx]
                missed_spans.append(
                    {
                        "gold": span_obj(g, text, kind="g"),
                        "context": safe_context(text, g.start, g.end, args.context_window).replace("\n", "\\n"),
                    }
                )

            spurious_spans = []
            for p_idx in sorted(unmatched_pred):
                p = pred_by_idx[p_idx]
                spurious_spans.append(
                    {
                        "pred": span_obj(p, text, kind="p"),
                        "context": safe_context(text, p.start, p.end, args.context_window).replace("\n", "\\n"),
                    }
                )

            # Sort and truncate for relevance.
            wrong_pairs.sort(key=lambda r: (r["overlap_len"], r["span_iou"]), reverse=True)
            boundary_pairs.sort(key=lambda r: (r["span_iou"], r["overlap_len"]))
            no_overlap_pairs.sort(key=lambda r: r["span_distance"])
            missed_spans.sort(key=lambda r: r["gold"]["len"], reverse=True)
            spurious_spans.sort(key=lambda r: r["pred"]["len"], reverse=True)

            totals = {
                "wrong_concept_overlap": len(wrong_pairs),
                "boundary_mismatch": len(boundary_pairs),
                "correct_concept_no_overlap": len(no_overlap_pairs),
                "missed_spans": len(missed_spans),
                "spurious_spans": len(spurious_spans),
            }

            wrong_pairs = wrong_pairs[: args.max_wrong]
            boundary_pairs = boundary_pairs[: args.max_boundary]
            no_overlap_pairs = no_overlap_pairs[: args.max_no_overlap]
            missed_spans = missed_spans[: args.max_missed]
            spurious_spans = spurious_spans[: args.max_spurious]

            global_miss_in_note = missing_concepts & global_miss_concepts

            # Per-concept counts (helpful for later graph/embedding matching).
            gold_counts = defaultdict(int)
            for s in gold_spans:
                gold_counts[s.concept_id] += 1
            pred_counts = defaultdict(int)
            for s in pred_spans:
                pred_counts[s.concept_id] += 1

            record = {
                "entry": args.entry if args.entry != "custom" else "custom",
                "note_id": nid,
                "source": {
                    "eval_json": str(eval_json) if eval_json is not None else None,
                    "pred_csv": str(pred_csv),
                    "gold_csv": str(gold_csv),
                    "notes_csv": str(notes_csv),
                    "snomed_description": str(description_path) if description_path is not None else None,
                },
                "summary": {
                    "n_gold_spans": len(gold_spans),
                    "n_pred_spans": len(pred_spans),
                    "n_gold_concepts": len(gold_concepts),
                    "n_pred_concepts": len(pred_concepts),
                    "n_missing_concepts": len(missing_concepts),
                    "n_extra_concepts": len(extra_concepts),
                    "n_wrong_concept_overlap_total": totals["wrong_concept_overlap"],
                    "n_boundary_mismatch_total": totals["boundary_mismatch"],
                    "n_correct_concept_no_overlap_total": totals["correct_concept_no_overlap"],
                    "n_missed_spans_total": totals["missed_spans"],
                    "n_spurious_spans_total": totals["spurious_spans"],
                    "n_wrong_concept_overlap_shown": len(wrong_pairs),
                    "n_boundary_mismatch_shown": len(boundary_pairs),
                    "n_correct_concept_no_overlap_shown": len(no_overlap_pairs),
                    "n_missed_spans_shown": len(missed_spans),
                    "n_spurious_spans_shown": len(spurious_spans),
                    "n_global_miss_concepts_in_note": len(global_miss_in_note),
                },
                "concepts": {
                    "gold": to_concept_list(gold_concepts, terms, args.max_items),
                    "pred": to_concept_list(pred_concepts, terms, args.max_items),
                    "missing": to_concept_list(missing_concepts, terms, args.max_items),
                    "extra": to_concept_list(extra_concepts, terms, args.max_items),
                    "global_miss_in_note": to_concept_list(global_miss_in_note, terms, args.max_items),
                },
                "concept_counts": {
                    "gold": {str(k): int(v) for k, v in gold_counts.items()},
                    "pred": {str(k): int(v) for k, v in pred_counts.items()},
                },
                "errors": {
                    "wrong_concept_overlap": wrong_pairs,
                    "boundary_mismatch": boundary_pairs,
                    "correct_concept_no_overlap": no_overlap_pairs,
                    "missed_spans": missed_spans,
                    "spurious_spans": spurious_spans,
                },
                "llm_instructions": (
                    "You are reviewing an entity linking model on a single clinical note. "
                    "Use the provided span-level errors and concept-level differences to diagnose likely failure modes "
                    "(extraction vs boundary vs concept selection vs spurious). "
                    "Summarize key error patterns and propose concrete, testable next steps."
                ),
            }
            try:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except BrokenPipeError:
                return 0
    finally:
        if out_f is not sys.stdout:
            out_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
