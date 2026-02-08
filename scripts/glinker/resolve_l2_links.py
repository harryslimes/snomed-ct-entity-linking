#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.io import detect_delimiter


@dataclass(frozen=True)
class ResolverConfig:
    allow_fuzzy_top1: bool = True
    require_l1_type_match: bool = False
    min_top1_score_exact: float = 0.2
    min_top1_score_fuzzy: float = 6.0
    min_top1_score_l3: float = 0.0
    min_top1_score_l4: float = 0.0
    min_score_margin: float = 0.0
    max_second_to_first_ratio: float = 1.0
    route_min_top1_score: dict[str, float] = field(default_factory=dict)
    route_min_score_margin: dict[str, float] = field(default_factory=dict)
    route_max_second_to_first_ratio: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryPostprocessConfig:
    trim_non_alnum_edges: bool = True
    trim_history_of_prefix: bool = True


_HISTORY_OF_RE = re.compile(r"^\s*history of\s+", flags=re.IGNORECASE)


def _to_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(str(value)))
        except Exception:
            return None


def _to_float(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


def _parse_route_float_overrides(values: list[str], *, flag: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in values:
        token = str(raw or "").strip()
        if not token:
            continue
        if "=" in token:
            route, score_text = token.split("=", 1)
        elif ":" in token:
            route, score_text = token.split(":", 1)
        else:
            raise ValueError(f"Invalid {flag} value '{token}'. Expected route=number.")
        route = route.strip()
        if not route:
            raise ValueError(f"Invalid {flag} value '{token}'. Route cannot be empty.")
        try:
            out[route] = float(score_text.strip())
        except ValueError as e:
            raise ValueError(f"Invalid {flag} value '{token}'. Score must be numeric.") from e
    return out


def _looks_like_history_concept(top1: dict) -> bool:
    probe = " ".join(
        str(top1.get(k) or "")
        for k in ("matched_alias", "canonical_name", "name", "term", "display_name")
    ).strip()
    if not probe:
        return False
    s = probe.lower()
    return ("history of " in s) or s.startswith("h/o ")


def _trim_span_by_mention_text(
    *,
    start: int,
    end: int,
    mention: str,
    top1: dict,
    cfg: BoundaryPostprocessConfig,
) -> tuple[int, int, str]:
    if not mention:
        return (start, end, "")
    # Only remap offsets when mention length aligns exactly with the original span.
    if len(mention) != (end - start):
        return (start, end, "unaligned_mention")

    s = start
    e = end
    left = 0
    right = 0
    n = len(mention)

    if cfg.trim_non_alnum_edges:
        while left < n and not mention[left].isalnum():
            left += 1
        while right < (n - left) and not mention[n - 1 - right].isalnum():
            right += 1
        if left > 0:
            s += left
        if right > 0:
            e -= right

    if e <= s:
        return (start, end, "invalid_after_edge_trim")

    core = mention[left : n - right] if (left > 0 or right > 0) else mention
    if cfg.trim_history_of_prefix and core:
        m = _HISTORY_OF_RE.match(core)
        if m and not _looks_like_history_concept(top1):
            shift = int(m.end())
            if shift < len(core):
                s += shift
            else:
                return (start, end, "invalid_after_history_trim")

    if e <= s:
        return (start, end, "invalid_after_history_trim")
    if s == start and e == end:
        return (s, e, "")
    return (s, e, "trimmed")


def _load_allowed_concepts(path: Path) -> set[str]:
    if not path.exists() and path.suffix.lower() == ".parquet":
        fallback = path.with_suffix(".csv")
        if fallback.exists():
            path = fallback
    if not path.exists():
        raise FileNotFoundError(f"Allowed concepts file not found: {path}")

    out: set[str] = set()
    if path.suffix.lower() == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Reading parquet allowed_concepts requires pandas+pyarrow in this environment."
            ) from e
        df = pd.read_parquet(path)
        if "concept_id" not in df.columns:
            raise ValueError(f"{path} must contain concept_id column")
        out = {str(x).strip() for x in df["concept_id"].tolist() if str(x).strip()}
        return out

    delim = detect_delimiter(path)
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter=delim)
        for row in reader:
            cid = str(row.get("concept_id") or "").strip()
            if cid:
                out.add(cid)
    return out


def resolve_record(
    record: dict,
    *,
    cfg: ResolverConfig,
    allowed_concepts: set[str] | None = None,
    boundary_cfg: BoundaryPostprocessConfig | None = None,
) -> tuple[bool, str, dict | None, str]:
    cands = record.get("final_candidates") or []
    note_id = str(record.get("note_id") or "").strip()
    start = _to_int(record.get("start_char"))
    end = _to_int(record.get("end_char"))
    mention_id = str(record.get("mention_id") or "").strip()
    mention = str(record.get("mention") or "")
    mention_l1 = str(record.get("l1_type") or "").strip() or None
    route = str(record.get("route") or "").strip()

    if not note_id:
        return (False, "missing_note_id", None, "")
    if start is None or end is None or end <= start:
        return (False, "invalid_span", None, "")
    if not cands:
        return (False, "no_candidates", None, "")

    top1 = cands[0]
    top2 = cands[1] if len(cands) > 1 else None

    concept_id = str(top1.get("concept_id") or "").strip()
    cand_l1 = str(top1.get("l1_type") or "").strip() or None
    top1_method = str(top1.get("method") or "").strip()
    top1_score = _to_float(top1.get("score"))
    if not concept_id:
        return (False, "missing_concept_id", None, "")

    if allowed_concepts is not None and concept_id not in allowed_concepts:
        return (False, "not_allowed", None, "")

    if (not cfg.allow_fuzzy_top1) and top1_method != "l2_exact":
        return (False, "fuzzy_top1_disabled", None, "")

    if cfg.require_l1_type_match and mention_l1 and cand_l1 and cand_l1 != mention_l1:
        return (False, "l1_type_mismatch", None, "")

    if top1_method == "l2_exact":
        min_score = cfg.min_top1_score_exact
    elif top1_method == "l3_biencoder":
        min_score = cfg.min_top1_score_l3
    elif top1_method == "l4_cross_rerank":
        min_score = cfg.min_top1_score_l4
    else:
        min_score = cfg.min_top1_score_fuzzy
    if route and route in cfg.route_min_top1_score:
        min_score = float(cfg.route_min_top1_score[route])
    if top1_score < min_score:
        return (False, "low_score", None, "")

    margin = math.inf
    second_ratio = 0.0
    min_margin = cfg.min_score_margin
    max_second_ratio = cfg.max_second_to_first_ratio
    if route and route in cfg.route_min_score_margin:
        min_margin = float(cfg.route_min_score_margin[route])
    if route and route in cfg.route_max_second_to_first_ratio:
        max_second_ratio = float(cfg.route_max_second_to_first_ratio[route])
    if top2 is not None:
        top2_score = _to_float(top2.get("score"))
        margin = top1_score - top2_score
        if top1_score > 0:
            second_ratio = top2_score / top1_score
        top2_cid = str(top2.get("concept_id") or "").strip()
        if top2_cid and top2_cid != concept_id:
            if margin < min_margin:
                return (False, "low_margin", None, "")
            if second_ratio > max_second_ratio:
                return (False, "high_second_ratio", None, "")

    postprocess_reason = ""
    if boundary_cfg is not None:
        start, end, postprocess_reason = _trim_span_by_mention_text(
            start=start,
            end=end,
            mention=mention,
            top1=top1,
            cfg=boundary_cfg,
        )

    decision = {
        "mention_id": mention_id,
        "note_id": note_id,
        "start_char": start,
        "end_char": end,
        "concept_id": concept_id,
        "score": top1_score,
        "method": top1_method,
        "l1_type": cand_l1 or "",
        "margin_to_second": margin if margin != math.inf else "",
        "second_to_first_ratio": second_ratio if top2 is not None else "",
        "postprocess": postprocess_reason,
        "route": route,
    }
    return (True, "accepted", decision, postprocess_reason)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Resolve L2 candidate bundles to one concept per span using threshold/margin "
            "rules, then write submission-ready resolved CSV."
        )
    )
    ap.add_argument("--candidates-jsonl", required=True)
    ap.add_argument("--out-resolved-csv", required=True)
    ap.add_argument("--out-decisions-csv", default="")
    ap.add_argument("--allowed-concepts", default="")
    ap.add_argument("--no-fuzzy-top1", action="store_true")
    ap.add_argument("--require-l1-type-match", action="store_true")
    ap.add_argument("--min-top1-score-exact", type=float, default=0.2)
    ap.add_argument("--min-top1-score-fuzzy", type=float, default=6.0)
    ap.add_argument("--min-top1-score-l3", type=float, default=0.0)
    ap.add_argument("--min-top1-score-l4", type=float, default=0.0)
    ap.add_argument("--min-score-margin", type=float, default=0.0)
    ap.add_argument("--max-second-to-first-ratio", type=float, default=1.0)
    ap.add_argument(
        "--route-min-top1-score",
        action="append",
        default=[],
        help="Per-route score threshold override. Format: route=score",
    )
    ap.add_argument(
        "--route-min-score-margin",
        action="append",
        default=[],
        help="Per-route score margin override. Format: route=margin",
    )
    ap.add_argument(
        "--route-max-second-to-first-ratio",
        action="append",
        default=[],
        help="Per-route 2nd/1st score ratio override. Format: route=ratio",
    )
    ap.add_argument("--no-trim-non-alnum-edges", action="store_true")
    ap.add_argument("--no-trim-history-of-prefix", action="store_true")
    args = ap.parse_args(argv)

    cfg = ResolverConfig(
        allow_fuzzy_top1=not bool(args.no_fuzzy_top1),
        require_l1_type_match=bool(args.require_l1_type_match),
        min_top1_score_exact=float(args.min_top1_score_exact),
        min_top1_score_fuzzy=float(args.min_top1_score_fuzzy),
        min_top1_score_l3=float(args.min_top1_score_l3),
        min_top1_score_l4=float(args.min_top1_score_l4),
        min_score_margin=float(args.min_score_margin),
        max_second_to_first_ratio=float(args.max_second_to_first_ratio),
        route_min_top1_score=_parse_route_float_overrides(
            list(args.route_min_top1_score), flag="--route-min-top1-score"
        ),
        route_min_score_margin=_parse_route_float_overrides(
            list(args.route_min_score_margin), flag="--route-min-score-margin"
        ),
        route_max_second_to_first_ratio=_parse_route_float_overrides(
            list(args.route_max_second_to_first_ratio),
            flag="--route-max-second-to-first-ratio",
        ),
    )
    boundary_cfg = BoundaryPostprocessConfig(
        trim_non_alnum_edges=not bool(args.no_trim_non_alnum_edges),
        trim_history_of_prefix=not bool(args.no_trim_history_of_prefix),
    )

    allowed: set[str] | None = None
    if args.allowed_concepts:
        allowed = _load_allowed_concepts(Path(args.allowed_concepts))

    in_path = Path(args.candidates_jsonl)
    out_resolved = Path(args.out_resolved_csv)
    out_resolved.parent.mkdir(parents=True, exist_ok=True)
    out_decisions = Path(args.out_decisions_csv) if args.out_decisions_csv else None
    if out_decisions is not None:
        out_decisions.parent.mkdir(parents=True, exist_ok=True)

    reason_counts: Counter[str] = Counter()
    postprocess_counts: Counter[str] = Counter()
    seen_rows: set[tuple[str, int, int, str]] = set()
    accepted_rows: list[dict] = []
    decision_rows: list[dict] = []

    with in_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ok, reason, row, postprocess_reason = resolve_record(
                rec,
                cfg=cfg,
                allowed_concepts=allowed,
                boundary_cfg=boundary_cfg,
            )
            reason_counts[reason] += 1
            if ok and row is not None:
                if postprocess_reason:
                    postprocess_counts[postprocess_reason] += 1
                dedupe_key = (
                    str(row["note_id"]),
                    int(row["start_char"]),
                    int(row["end_char"]),
                    str(row["concept_id"]),
                )
                if dedupe_key not in seen_rows:
                    seen_rows.add(dedupe_key)
                    accepted_rows.append(row)
                if out_decisions is not None:
                    row_copy = dict(row)
                    row_copy["decision"] = "accepted"
                    row_copy["reason"] = reason
                    decision_rows.append(row_copy)
            elif out_decisions is not None:
                decision_rows.append(
                    {
                        "mention_id": str(rec.get("mention_id") or ""),
                        "note_id": str(rec.get("note_id") or ""),
                        "start_char": _to_int(rec.get("start_char")) or "",
                        "end_char": _to_int(rec.get("end_char")) or "",
                        "concept_id": "",
                        "score": "",
                        "method": "",
                        "l1_type": str(rec.get("l1_type") or ""),
                        "margin_to_second": "",
                        "second_to_first_ratio": "",
                        "postprocess": "",
                        "route": str(rec.get("route") or ""),
                        "decision": "rejected",
                        "reason": reason,
                    }
                )

    accepted_rows.sort(key=lambda r: (r["note_id"], int(r["start_char"]), int(r["end_char"]), str(r["concept_id"])))
    with out_resolved.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["note_id", "start_char", "end_char", "concept_id"])
        for r in accepted_rows:
            writer.writerow([r["note_id"], r["start_char"], r["end_char"], r["concept_id"]])

    if out_decisions is not None:
        with out_decisions.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "mention_id",
                    "note_id",
                    "start_char",
                    "end_char",
                    "concept_id",
                    "score",
                    "method",
                    "l1_type",
                    "margin_to_second",
                    "second_to_first_ratio",
                    "postprocess",
                    "route",
                    "decision",
                    "reason",
                ],
            )
            writer.writeheader()
            for r in decision_rows:
                writer.writerow(r)

    total = sum(reason_counts.values())
    print(f"processed candidate bundles: {total:,}")
    print(f"accepted: {len(accepted_rows):,}")
    for reason, n in sorted(reason_counts.items()):
        print(f"reason[{reason}]={n}")
    for reason, n in sorted(postprocess_counts.items()):
        print(f"postprocess[{reason}]={n}")
    print(f"resolved_csv: {out_resolved}")
    if out_decisions is not None:
        print(f"decisions_csv: {out_decisions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
