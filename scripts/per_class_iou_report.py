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
SYNONYM_TYPE_ID = "900000000000013009"


@dataclass(frozen=True)
class Term:
    term: str
    type_id: str
    rank: int


def find_snomed_description_snapshot(data_dir: Path) -> Path | None:
    # Mirror `scripts/evaluate_2nd_place.py` release-dir convention.
    for release_dir in sorted(data_dir.glob("SnomedCT_*")):
        candidate = release_dir / "Snapshot" / "Terminology"
        if not candidate.exists():
            continue
        matches = sorted(candidate.glob("sct2_Description_Snapshot-en_*_*.txt"))
        if matches:
            return matches[0]
    return None


def load_per_class_iou(eval_json: Path) -> dict[int, float]:
    payload = json.loads(eval_json.read_text())
    per_class = payload.get("per_class_iou")
    if not isinstance(per_class, dict):
        raise ValueError(f"Missing or invalid per_class_iou in: {eval_json}")
    out: dict[int, float] = {}
    for k, v in per_class.items():
        try:
            cid = int(k)
        except Exception:
            continue
        try:
            out[cid] = float(v)
        except Exception:
            out[cid] = float("nan")
    return out


def load_eval_paths(eval_json: Path) -> tuple[Path | None, Path | None]:
    payload = json.loads(eval_json.read_text())
    pred_path = payload.get("submission_csv")
    gold_path = payload.get("annotations_path")
    return (Path(pred_path) if pred_path else None, Path(gold_path) if gold_path else None)


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return min(a_end, b_end) > max(a_start, b_start)


def compute_per_class_counts(pred_csv: Path, gold_csv: Path) -> dict[int, dict[str, int]]:
    pred_df = ensure_columns(pd.read_csv(pred_csv), "pred")
    gold_df = ensure_columns(pd.read_csv(gold_csv), "gold")

    pred_groups: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for note_id, concept_id, start, end in pred_df[["note_id", "concept_id", "start", "end"]].itertuples(
        index=False, name=None
    ):
        pred_groups[(int(concept_id), str(note_id))].append((int(start), int(end)))

    gold_groups: dict[tuple[int, str], list[tuple[int, int]]] = defaultdict(list)
    for note_id, concept_id, start, end in gold_df[["note_id", "concept_id", "start", "end"]].itertuples(
        index=False, name=None
    ):
        gold_groups[(int(concept_id), str(note_id))].append((int(start), int(end)))

    keys = set(pred_groups.keys()) | set(gold_groups.keys())
    out: dict[int, dict[str, int]] = {}

    for (concept_id, note_id) in keys:
        pred_spans = pred_groups.get((concept_id, note_id), [])
        gold_spans = gold_groups.get((concept_id, note_id), [])

        stats = out.setdefault(
            concept_id,
            {
                "gold_spans": 0,
                "pred_spans": 0,
                "gold_hit_spans": 0,
                "pred_hit_spans": 0,
            },
        )
        stats["gold_spans"] += len(gold_spans)
        stats["pred_spans"] += len(pred_spans)

        if gold_spans and pred_spans:
            for g_start, g_end in gold_spans:
                if any(spans_overlap(g_start, g_end, p_start, p_end) for p_start, p_end in pred_spans):
                    stats["gold_hit_spans"] += 1
            for p_start, p_end in pred_spans:
                if any(spans_overlap(p_start, p_end, g_start, g_end) for g_start, g_end in gold_spans):
                    stats["pred_hit_spans"] += 1

    for concept_id, stats in out.items():
        stats["gold_missed_spans"] = stats["gold_spans"] - stats["gold_hit_spans"]
        stats["pred_wrong_spans"] = stats["pred_spans"] - stats["pred_hit_spans"]
    return out


def type_rank(type_id: str) -> int:
    if type_id == FSN_TYPE_ID:
        return 0
    if type_id == SYNONYM_TYPE_ID:
        return 1
    return 2


def type_label(type_id: str) -> str:
    if type_id == FSN_TYPE_ID:
        return "FSN"
    if type_id == SYNONYM_TYPE_ID:
        return "SYNONYM"
    return type_id


def load_terms_from_description_snapshot(
    description_path: Path,
    concept_ids: set[int],
    *,
    language_code: str = "en",
) -> dict[int, Term]:
    best: dict[int, Term] = {}
    # Some SNOMED releases include very long fields; raise CSV field limit.
    try:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    except Exception:
        pass
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
            try:
                cid = int(row["conceptId"])
            except Exception:
                continue
            if cid not in concept_ids:
                continue

            t = (row.get("term") or "").strip()
            if not t:
                continue

            type_id = row.get("typeId") or ""
            candidate = Term(term=t, type_id=type_id, rank=type_rank(type_id))
            current = best.get(cid)
            if current is None or candidate.rank < current.rank:
                best[cid] = candidate
    return best


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pretty-print per-class IoU with SNOMED concept terms and miss/wrong counts, sorted by IoU ascending."
    )
    parser.add_argument(
        "--eval-json",
        default="outputs/2nd_place_train_eval.json",
        help="Evaluation JSON containing per_class_iou (default: outputs/2nd_place_train_eval.json)",
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
        "--pred-csv",
        default=None,
        help="Predictions CSV (default: submission_csv inside --eval-json).",
    )
    parser.add_argument(
        "--gold-csv",
        default=None,
        help="Gold annotations CSV (default: annotations_path inside --eval-json).",
    )
    parser.add_argument(
        "--language-code",
        default="en",
        help="languageCode filter for the description file (default: en)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Max rows to print (default: 200). Use 0 to print all.",
    )
    parser.add_argument(
        "--max-iou",
        type=float,
        default=None,
        help="Only include concepts with IoU <= this value (useful to focus on worst classes).",
    )
    args = parser.parse_args()

    eval_json = Path(args.eval_json)
    if not eval_json.exists():
        raise FileNotFoundError(f"Eval JSON not found: {eval_json}")

    per_class = load_per_class_iou(eval_json)
    concept_ids = set(per_class.keys())

    default_pred_csv, default_gold_csv = load_eval_paths(eval_json)
    pred_csv = Path(args.pred_csv) if args.pred_csv else default_pred_csv
    gold_csv = Path(args.gold_csv) if args.gold_csv else default_gold_csv
    if pred_csv is None or not pred_csv.exists():
        raise FileNotFoundError(
            "Predictions CSV not found. Pass --pred-csv, or ensure --eval-json contains submission_csv."
        )
    if gold_csv is None or not gold_csv.exists():
        raise FileNotFoundError(
            "Gold annotations CSV not found. Pass --gold-csv, or ensure --eval-json contains annotations_path."
        )

    counts = compute_per_class_counts(pred_csv, gold_csv)

    description_path: Path | None
    if args.snomed_description:
        description_path = Path(args.snomed_description)
    else:
        description_path = find_snomed_description_snapshot(Path(args.data_dir))

    terms: dict[int, Term] = {}
    if description_path and description_path.exists():
        terms = load_terms_from_description_snapshot(
            description_path,
            concept_ids,
            language_code=args.language_code,
        )
    else:
        description_path = None

    rows = []
    for cid, iou in per_class.items():
        if args.max_iou is not None and not (iou <= args.max_iou):
            continue
        term = terms.get(cid)
        c = counts.get(
            cid,
            {
                "gold_spans": 0,
                "pred_spans": 0,
                "gold_hit_spans": 0,
                "pred_hit_spans": 0,
                "gold_missed_spans": 0,
                "pred_wrong_spans": 0,
            },
        )
        rows.append(
            (
                iou,
                cid,
                term.term if term else "",
                term.type_id if term else "",
                int(c["gold_spans"]),
                int(c["pred_spans"]),
                int(c["gold_missed_spans"]),
                int(c["pred_wrong_spans"]),
            )
        )
    rows.sort(key=lambda x: (x[0], x[1]))

    limit = args.limit if args.limit and args.limit > 0 else None
    out_rows = rows[:limit] if limit is not None else rows

    meta = f"# eval_json={eval_json}"
    if description_path:
        meta += f"\tsnomed_description={description_path}"
    meta += f"\tpred_csv={pred_csv}\tgold_csv={gold_csv}"
    print(meta)

    w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    w.writerow(
        [
            "iou",
            "concept_id",
            "term",
            "description_type",
            "gold_spans",
            "pred_spans",
            "gold_missed_spans",
            "pred_wrong_spans",
            "pct_gold_missed",
            "pct_pred_wrong",
            "global_miss",
            "extra_only",
        ]
    )
    for iou, cid, term, type_id, gold_spans, pred_spans, gold_missed, pred_wrong in out_rows:
        pct_gold_missed = (100.0 * gold_missed / gold_spans) if gold_spans else None
        pct_pred_wrong = (100.0 * pred_wrong / pred_spans) if pred_spans else None
        global_miss = 1 if (gold_spans > 0 and pred_spans == 0) else 0
        extra_only = 1 if (gold_spans == 0 and pred_spans > 0) else 0
        w.writerow(
            [
                f"{iou:.6f}",
                str(cid),
                term,
                type_label(type_id),
                str(gold_spans),
                str(pred_spans),
                str(gold_missed),
                str(pred_wrong),
                (f"{pct_gold_missed:.2f}" if pct_gold_missed is not None else ""),
                (f"{pct_pred_wrong:.2f}" if pct_pred_wrong is not None else ""),
                str(global_miss),
                str(extra_only),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
