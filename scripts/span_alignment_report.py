#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
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
    idx: int


def find_snomed_description_snapshot(data_dir: Path) -> Path | None:
    for release_dir in sorted(data_dir.glob("SnomedCT_*")):
        candidate = release_dir / "Snapshot" / "Terminology"
        if not candidate.exists():
            continue
        matches = sorted(candidate.glob("sct2_Description_Snapshot-en_*_*.txt"))
        if matches:
            return matches[0]
    return None


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


def load_note_text(notes_csv: Path) -> dict[str, str]:
    notes = pd.read_csv(notes_csv, usecols=["note_id", "text"])
    notes["note_id"] = notes["note_id"].astype(str)
    notes["text"] = notes["text"].astype(str)
    return dict(zip(notes["note_id"].tolist(), notes["text"].tolist(), strict=False))


def safe_slice(text: str, start: int, end: int) -> str:
    if start < 0:
        start = 0
    if end < start:
        end = start
    if start > len(text):
        return ""
    if end > len(text):
        end = len(text)
    return text[start:end]


def make_spans(df: pd.DataFrame) -> list[Span]:
    spans: list[Span] = []
    for idx, (note_id, start, end, concept_id) in enumerate(
        df[["note_id", "start", "end", "concept_id"]].itertuples(index=False, name=None)
    ):
        spans.append(
            Span(note_id=str(note_id), start=int(start), end=int(end), concept_id=int(concept_id), idx=idx)
        )
    return spans


def greedy_max_overlap_matching(
    gold_spans: list[Span],
    pred_spans: list[Span],
) -> tuple[dict[int, int], set[int], set[int]]:
    # Returns mapping gold_idx -> pred_idx for pairs with overlap > 0, plus unmatched sets.
    candidates: list[tuple[int, float, int, int]] = []
    for g in gold_spans:
        for p in pred_spans:
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
    # Only pair (gold, pred) with same concept_id, no overlap, minimal distance.
    pred_by_concept: dict[int, list[Span]] = {}
    for p in pred_spans:
        if p.idx not in unmatched_pred:
            continue
        pred_by_concept.setdefault(p.concept_id, []).append(p)

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a span-level alignment report: strict overlap matching + relaxed same-concept matching for non-overlapping spans."
    )
    parser.add_argument(
        "--eval-json",
        default="outputs/2nd_place_train_eval.json",
        help="Evaluation JSON containing submission_csv/annotations_path (default: outputs/2nd_place_train_eval.json)",
    )
    parser.add_argument("--pred-csv", default=None, help="Predictions CSV (overrides --eval-json).")
    parser.add_argument("--gold-csv", default=None, help="Gold annotations CSV (overrides --eval-json).")
    parser.add_argument(
        "--notes-csv",
        default=None,
        help="Notes CSV for snippet/context (optional). Defaults to notes_path inside --eval-json if present.",
    )
    parser.add_argument(
        "--include-snippets",
        action="store_true",
        help="Include extracted span text for gold/pred (requires notes CSV).",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include +/- window chars around the gold span (requires notes CSV).",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=40,
        help="Context window size in chars (default: 40).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory used to auto-find a SNOMED release (default: ./data)",
    )
    parser.add_argument(
        "--snomed-description",
        default=None,
        help="Path to SNOMED description snapshot TSV (overrides auto-detection).",
    )
    parser.add_argument(
        "--no-terms",
        action="store_true",
        help="Do not include FSN terms (only concept ids).",
    )
    parser.add_argument(
        "--language-code",
        default="en",
        help="languageCode filter for the description file (default: en)",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=50,
        help="Max character distance for relaxed same-concept matching (default: 50). Use -1 for no limit.",
    )
    parser.add_argument(
        "--only-mismatches",
        action="store_true",
        help="Only emit rows where concepts differ or a span is unmatched.",
    )
    args = parser.parse_args()

    eval_json = Path(args.eval_json)
    if not eval_json.exists():
        raise FileNotFoundError(f"Eval JSON not found: {eval_json}")

    default_pred, default_gold, default_notes = load_eval_paths(eval_json)
    pred_csv = Path(args.pred_csv) if args.pred_csv else default_pred
    gold_csv = Path(args.gold_csv) if args.gold_csv else default_gold
    notes_csv = Path(args.notes_csv) if args.notes_csv else default_notes

    if pred_csv is None or not pred_csv.exists():
        raise FileNotFoundError("Predictions CSV not found. Pass --pred-csv, or ensure --eval-json contains submission_csv.")
    if gold_csv is None or not gold_csv.exists():
        raise FileNotFoundError("Gold CSV not found. Pass --gold-csv, or ensure --eval-json contains annotations_path.")

    if (args.include_snippets or args.include_context) and (notes_csv is None or not notes_csv.exists()):
        raise FileNotFoundError("Notes CSV not found. Pass --notes-csv, or ensure --eval-json contains notes_path.")

    pred_df = ensure_columns(pd.read_csv(pred_csv), "pred")
    gold_df = ensure_columns(pd.read_csv(gold_csv), "gold")
    pred_df["note_id"] = pred_df["note_id"].astype(str)
    gold_df["note_id"] = gold_df["note_id"].astype(str)

    pred_spans_all = make_spans(pred_df)
    gold_spans_all = make_spans(gold_df)

    note_ids = sorted(set(s.note_id for s in pred_spans_all) | set(s.note_id for s in gold_spans_all))
    pred_by_note: dict[str, list[Span]] = {nid: [] for nid in note_ids}
    gold_by_note: dict[str, list[Span]] = {nid: [] for nid in note_ids}
    for p in pred_spans_all:
        pred_by_note[p.note_id].append(p)
    for g in gold_spans_all:
        gold_by_note[g.note_id].append(g)

    terms: dict[int, str] | None = None
    description_path: Path | None = None
    if not args.no_terms:
        all_concepts = {s.concept_id for s in pred_spans_all} | {s.concept_id for s in gold_spans_all}
        if args.snomed_description:
            description_path = Path(args.snomed_description)
        else:
            description_path = find_snomed_description_snapshot(Path(args.data_dir))
        if description_path and description_path.exists():
            terms = load_fsn_terms(description_path, all_concepts, language_code=args.language_code)

    note_text: dict[str, str] | None = None
    if (args.include_snippets or args.include_context) and notes_csv is not None:
        note_text = load_note_text(notes_csv)

    meta = f"# eval_json={eval_json}\tpred_csv={pred_csv}\tgold_csv={gold_csv}"
    if description_path:
        meta += f"\tsnomed_description={description_path}"
    if note_text is not None and notes_csv is not None:
        meta += f"\tnotes_csv={notes_csv}"
    print(meta)

    fieldnames = [
        "note_id",
        "match_type",
        "diagnosis",
        "gold_span_id",
        "pred_span_id",
        "gold_start",
        "gold_end",
        "pred_start",
        "pred_end",
        "overlap_len",
        "span_iou",
        "span_distance",
        "gold_concept_id",
        "pred_concept_id",
    ]
    if terms is not None:
        fieldnames.extend(["gold_term", "pred_term"])
    if args.include_snippets:
        fieldnames.extend(["gold_text", "pred_text"])
    if args.include_context:
        fieldnames.append("gold_context")

    w = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    try:
        w.writeheader()
        for nid in note_ids:
            gold_spans = gold_by_note.get(nid, [])
            pred_spans = pred_by_note.get(nid, [])

            strict_pairs, unmatched_gold, unmatched_pred = greedy_max_overlap_matching(gold_spans, pred_spans)
            relaxed_pairs = relaxed_same_concept_matching(
                gold_spans, pred_spans, unmatched_gold, unmatched_pred, max_distance=args.max_distance
            )

            # Build reverse maps for easy lookup.
            pred_by_idx = {p.idx: p for p in pred_spans}
            gold_by_idx = {g.idx: g for g in gold_spans}

            used_pred = set(strict_pairs.values()) | set(relaxed_pairs.values())

            def emit_row(
                *,
                match_type: str,
                gold: Span | None,
                pred: Span | None,
            ) -> None:
                gold_concept = gold.concept_id if gold else None
                pred_concept = pred.concept_id if pred else None
                inter = overlap_len(gold, pred) if gold and pred else 0
                iou_val = span_iou(gold, pred) if gold and pred else 0.0
                dist = span_distance(gold, pred) if gold and pred else None

                if match_type == "unmatched_gold":
                    diagnosis = "missed"
                elif match_type == "extra_pred":
                    diagnosis = "spurious"
                elif match_type == "relaxed_same_concept":
                    diagnosis = "correct_concept_no_overlap"
                else:  # strict_overlap
                    if gold_concept == pred_concept:
                        diagnosis = "correct" if iou_val >= 0.999999 else "boundary_mismatch"
                    else:
                        diagnosis = "wrong_concept_overlap"

                if args.only_mismatches and diagnosis == "correct":
                    return
                row = {
                    "note_id": nid,
                    "match_type": match_type,
                    "diagnosis": diagnosis,
                    "gold_span_id": (f"{nid}:g{gold.idx}" if gold else ""),
                    "pred_span_id": (f"{nid}:p{pred.idx}" if pred else ""),
                    "gold_start": (str(gold.start) if gold else ""),
                    "gold_end": (str(gold.end) if gold else ""),
                    "pred_start": (str(pred.start) if pred else ""),
                    "pred_end": (str(pred.end) if pred else ""),
                    "overlap_len": (str(inter) if gold and pred else "0"),
                    "span_iou": (f"{iou_val:.6f}" if gold and pred else "0.000000"),
                    "span_distance": (str(dist) if dist is not None else ""),
                    "gold_concept_id": (str(gold.concept_id) if gold else ""),
                    "pred_concept_id": (str(pred.concept_id) if pred else ""),
                }
                if terms is not None:
                    row["gold_term"] = terms.get(gold.concept_id, "") if gold else ""
                    row["pred_term"] = terms.get(pred.concept_id, "") if pred else ""
                if note_text is not None:
                    text = note_text.get(nid, "")
                    if args.include_snippets:
                        row["gold_text"] = safe_slice(text, gold.start, gold.end).replace("\n", "\\n") if gold else ""
                        row["pred_text"] = safe_slice(text, pred.start, pred.end).replace("\n", "\\n") if pred else ""
                    if args.include_context:
                        if gold:
                            left = max(0, gold.start - args.context_window)
                            right = min(len(text), gold.end + args.context_window)
                            row["gold_context"] = safe_slice(text, left, right).replace("\n", "\\n")
                        else:
                            row["gold_context"] = ""
                w.writerow(row)

            # Emit matched gold rows first.
            for g_idx, p_idx in strict_pairs.items():
                emit_row(match_type="strict_overlap", gold=gold_by_idx[g_idx], pred=pred_by_idx[p_idx])
            for g_idx, p_idx in relaxed_pairs.items():
                emit_row(match_type="relaxed_same_concept", gold=gold_by_idx[g_idx], pred=pred_by_idx[p_idx])

            # Remaining unmatched gold spans.
            still_unmatched_gold = unmatched_gold - set(relaxed_pairs.keys())
            for g_idx in sorted(still_unmatched_gold):
                emit_row(match_type="unmatched_gold", gold=gold_by_idx[g_idx], pred=None)

            # Extra/unmatched predicted spans.
            remaining_pred = unmatched_pred - set(relaxed_pairs.values())
            for p_idx in sorted(remaining_pred):
                emit_row(match_type="extra_pred", gold=None, pred=pred_by_idx[p_idx])
    except BrokenPipeError:
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
