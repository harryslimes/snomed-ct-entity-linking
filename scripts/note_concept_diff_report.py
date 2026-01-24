#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

from iou_metric import ensure_columns


FSN_TYPE_ID = "900000000000003001"


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


def format_concepts(
    concept_ids: set[int],
    *,
    terms: dict[int, str] | None,
    include_terms: bool,
    max_items: int,
) -> str:
    ids_sorted = sorted(concept_ids)
    shown = ids_sorted if max_items <= 0 else ids_sorted[:max_items]
    if include_terms and terms is not None:
        parts = [f"{cid}|{terms.get(cid, '')}".rstrip("|") for cid in shown]
    else:
        parts = [str(cid) for cid in shown]
    suffix = ""
    if max_items > 0 and len(ids_sorted) > max_items:
        suffix = f",...(+{len(ids_sorted) - max_items} more)"
    return ",".join(parts) + suffix


def load_note_text(notes_csv: Path) -> dict[str, str]:
    notes = pd.read_csv(notes_csv, usecols=["note_id", "text"])
    notes["note_id"] = notes["note_id"].astype(str)
    notes["text"] = notes["text"].astype(str).str.replace("\r", " ").str.replace("\n", " ")
    return dict(zip(notes["note_id"].tolist(), notes["text"].tolist(), strict=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Produce a per-note report of gold vs predicted SNOMED concepts, including missing and extra."
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
        help="Notes CSV (for optional text column). Defaults to notes_path inside --eval-json if present.",
    )
    parser.add_argument("--include-text", action="store_true", help="Include a note text column (can be large).")
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
        help="Do not append FSN terms (only concept ids).",
    )
    parser.add_argument(
        "--language-code",
        default="en",
        help="languageCode filter for the description file (default: en)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=50,
        help="Max concepts to show per list column (default: 50). Use 0 to show all.",
    )
    parser.add_argument(
        "--sort",
        choices=["note_id", "missing_desc", "extra_desc", "gold_desc", "pred_desc"],
        default="missing_desc",
        help="Sort order (default: missing_desc).",
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
    if args.include_text and (notes_csv is None or not notes_csv.exists()):
        raise FileNotFoundError("Notes CSV not found. Pass --notes-csv, or ensure --eval-json contains notes_path.")

    pred_df = ensure_columns(pd.read_csv(pred_csv), "pred")
    gold_df = ensure_columns(pd.read_csv(gold_csv), "gold")
    pred_df["note_id"] = pred_df["note_id"].astype(str)
    gold_df["note_id"] = gold_df["note_id"].astype(str)

    pred_sets = pred_df.groupby("note_id")["concept_id"].apply(lambda s: set(map(int, s.tolist())))
    gold_sets = gold_df.groupby("note_id")["concept_id"].apply(lambda s: set(map(int, s.tolist())))

    note_ids = sorted(set(pred_sets.index.tolist()) | set(gold_sets.index.tolist()))
    all_concepts: set[int] = set()
    for nid in note_ids:
        all_concepts |= set(gold_sets.get(nid, set()))
        all_concepts |= set(pred_sets.get(nid, set()))

    terms: dict[int, str] | None = None
    description_path: Path | None = None
    if not args.no_terms:
        if args.snomed_description:
            description_path = Path(args.snomed_description)
        else:
            description_path = find_snomed_description_snapshot(Path(args.data_dir))
        if description_path and description_path.exists():
            terms = load_fsn_terms(description_path, all_concepts, language_code=args.language_code)

    text_by_note: dict[str, str] | None = None
    if args.include_text and notes_csv is not None:
        text_by_note = load_note_text(notes_csv)

    rows = []
    for nid in note_ids:
        gold = set(gold_sets.get(nid, set()))
        pred = set(pred_sets.get(nid, set()))
        missing = gold - pred
        extra = pred - gold
        rows.append(
            {
                "note_id": nid,
                "n_gold": len(gold),
                "n_pred": len(pred),
                "n_missing": len(missing),
                "n_extra": len(extra),
                "gold_concepts": format_concepts(gold, terms=terms, include_terms=terms is not None, max_items=args.max_items),
                "pred_concepts": format_concepts(pred, terms=terms, include_terms=terms is not None, max_items=args.max_items),
                "missing_concepts": format_concepts(missing, terms=terms, include_terms=terms is not None, max_items=args.max_items),
                "extra_concepts": format_concepts(extra, terms=terms, include_terms=terms is not None, max_items=args.max_items),
                "text": (text_by_note.get(nid, "") if text_by_note is not None else None),
            }
        )

    if args.sort == "note_id":
        rows.sort(key=lambda r: r["note_id"])
    elif args.sort == "missing_desc":
        rows.sort(key=lambda r: (r["n_missing"], r["n_extra"], r["note_id"]), reverse=True)
    elif args.sort == "extra_desc":
        rows.sort(key=lambda r: (r["n_extra"], r["n_missing"], r["note_id"]), reverse=True)
    elif args.sort == "gold_desc":
        rows.sort(key=lambda r: (r["n_gold"], r["note_id"]), reverse=True)
    elif args.sort == "pred_desc":
        rows.sort(key=lambda r: (r["n_pred"], r["note_id"]), reverse=True)

    meta = f"# eval_json={eval_json}\tpred_csv={pred_csv}\tgold_csv={gold_csv}"
    if description_path:
        meta += f"\tsnomed_description={description_path}"
    if args.include_text and notes_csv:
        meta += f"\tnotes_csv={notes_csv}"
    print(meta)

    fieldnames = [
        "note_id",
        "n_gold",
        "n_pred",
        "n_missing",
        "n_extra",
        "gold_concepts",
        "pred_concepts",
        "missing_concepts",
        "extra_concepts",
    ]
    if args.include_text:
        fieldnames.append("text")

    w = csv.DictWriter(sys.stdout, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    try:
        w.writeheader()
        for row in rows:
            if not args.include_text:
                row.pop("text", None)
            w.writerow(row)
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
