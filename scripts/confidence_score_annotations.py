#!/usr/bin/env python3
"""Heuristic confidence scoring for super dictionary annotations.

Assigns a [0, 1] confidence score to each annotation produced by the super
dictionary and sweeps multiple thresholds to find the best FP-suppression
operating point for the hybrid dictionary + LLM pipeline.

Two signals are combined:

1. **Span heuristics** (length + token count + hard-zero rules):
   Pure length/token score applied to each individual prediction.

2. **OOF concept precision** (primary signal when available):
   For each concept_id, how reliably did the dictionary predict it correctly
   during training (out-of-fold)?  Concepts with high OOF IoU are trustworthy;
   pure-FP concepts in OOF (gt_chars=0) get near-zero confidence.

The final score is a weighted combination of the two signals.

Usage
-----
PYTHONPATH=super-dictionary python scripts/confidence_score_annotations.py \\
  --pred   outputs/old_challenge_split/kiri_super_pred.csv \\
  --notes  data/old-challenge-split/test_notes.csv \\
  --gold   data/old-challenge-split/test_annotations.csv \\
  --fp-blocklist outputs/old_challenge_split/fp_blocklist_common.csv \\
  --oof-pred outputs/old_challenge_split/kiri_super_oof_pred.csv \\
  --oof-gold data/old-challenge-split/train_annotations.csv \\
  --output outputs/old_challenge_split/kiri_super_pred_scored.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Stopwords used for hard-zero detection
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "to", "the", "a", "an", "of", "in", "on", "at", "for",
    "with", "by", "from", "is", "are", "was", "be", "and", "or",
    "as", "it", "its", "this", "that", "these", "those", "not",
}

# Digits-only / lab-value pattern (e.g. "5.2", "135", "0.9")
_LAB_VALUE_RE = re.compile(r"^\d+(\.\d+)?%?$")


# ---------------------------------------------------------------------------
# Length / token scoring tables
# ---------------------------------------------------------------------------

def _len_score(char_len: int) -> float:
    if char_len >= 20:
        return 1.00
    elif char_len >= 12:
        return 0.85
    elif char_len >= 8:
        return 0.65
    elif char_len >= 5:
        return 0.30
    elif char_len >= 3:
        return 0.10
    else:
        return 0.00


def _token_mult(token_count: int, char_len: int) -> float:
    if token_count >= 3:
        return 1.00
    elif token_count == 2:
        return 0.90
    else:  # single token
        if char_len >= 10:
            return 0.80
        elif char_len >= 7:
            return 0.55
        else:
            return 0.20


# ---------------------------------------------------------------------------
# Signal builders: one dict per signal, all keyed by int concept_id
# ---------------------------------------------------------------------------

def build_oof_iou_signal(
    oof_pred_df: pd.DataFrame, oof_gold_df: pd.DataFrame
) -> dict[int, float]:
    """Per-concept signal from OOF character-level IoU.

    Mapping:
    - gt_chars=0, pred>0 → pure OOF FP → 0.05
    - gt_chars>0, iou>0  → correctly predicted (at least partially) → iou
    - gt_chars>0, iou=0  → in gold but never matched → 0.10
    - absent from OOF    → no data; caller handles
    """
    from runtime_scoring import class_char_iou  # type: ignore

    oof_p = oof_pred_df[["note_id", "start", "end", "concept_id"]].copy()
    oof_g = oof_gold_df[["note_id", "start", "end", "concept_id"]].copy()
    for df in (oof_p, oof_g):
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        df["concept_id"] = df["concept_id"].astype(int)

    iou_df = class_char_iou(oof_p, oof_g)
    out: dict[int, float] = {}
    for _, r in iou_df.iterrows():
        cid = int(r["concept_id"])
        gt, pred, iou = r["gt_chars"], r["pred_chars"], float(r["iou"])
        if gt == 0 and pred > 0:
            out[cid] = 0.05
        elif gt > 0 and iou > 0:
            out[cid] = max(iou, 0.05)
        elif gt > 0 and iou == 0:
            out[cid] = 0.10
    return out


def build_train_freq_signal(train_gold_df: pd.DataFrame) -> dict[int, float]:
    """Per-concept signal from training annotation frequency.

    Concepts that are frequently annotated in training data are reliable.
    Concepts never annotated are suspect (likely FPs).

    Mapping:
    - 0 times  → 0.05  (never in gold; may be test-only or a pure FP)
    - 1 time   → 0.35
    - 2–5      → 0.60
    - 6–20     → 0.78
    - 21–100   → 0.90
    - 100+     → 0.97
    """
    freq = train_gold_df.groupby("concept_id").size()

    def _score(n: int) -> float:
        if n == 0:
            return 0.05
        elif n == 1:
            return 0.35
        elif n <= 5:
            return 0.60
        elif n <= 20:
            return 0.78
        elif n <= 100:
            return 0.90
        else:
            return 0.97

    return {int(cid): _score(int(n)) for cid, n in freq.items()}


def build_synonym_count_signal(super_dict_path: str) -> dict[int, float]:
    """Per-concept signal from synonym count in the super dictionary.

    Concepts with many synonyms are more ambiguous and FP-prone.
    Mapping (inverse: more synonyms → lower score):
    - 1        → 0.95
    - 2–3      → 0.85
    - 4–7      → 0.72
    - 8–15     → 0.58
    - 16+      → 0.42
    """
    tsv = pd.read_csv(
        super_dict_path, sep="\t", usecols=["snomed_concept_id", "term"]
    )
    syn_count = tsv.groupby("snomed_concept_id").size()

    def _score(n: int) -> float:
        if n <= 1:
            return 0.95
        elif n <= 3:
            return 0.85
        elif n <= 7:
            return 0.72
        elif n <= 15:
            return 0.58
        else:
            return 0.42

    return {int(cid): _score(int(n)) for cid, n in syn_count.items()}


def build_hierarchy_depth_signal(subsumption_path: str) -> dict[int, float]:
    """Per-concept signal from SNOMED hierarchy depth (ancestor count).

    Deeper = more specific = less ambiguous = higher confidence.
    Mapping:
    - 0–3   → 0.20  (root-level abstract: "Body structure", "Clinical finding")
    - 4–7   → 0.45
    - 8–12  → 0.65
    - 13–20 → 0.80
    - 21–30 → 0.90
    - 30+   → 0.95
    """
    import pickle

    sub = pickle.load(open(subsumption_path, "rb"))
    ancestors = sub["ancestors"]

    def _score(n: int) -> float:
        if n <= 3:
            return 0.20
        elif n <= 7:
            return 0.45
        elif n <= 12:
            return 0.65
        elif n <= 20:
            return 0.80
        elif n <= 30:
            return 0.90
        else:
            return 0.95

    return {int(cid): _score(len(ancs)) for cid, ancs in ancestors.items()}


# ---------------------------------------------------------------------------
# Combined concept-level confidence map
# ---------------------------------------------------------------------------

def build_concept_confidence(
    oof_iou: dict[int, float],
    train_freq: dict[int, float],
    syn_count: dict[int, float],
    hier_depth: dict[int, float],
    *,
    w_oof: float = 0.45,
    w_freq: float = 0.30,
    w_depth: float = 0.15,
    w_syn: float = 0.10,
) -> dict[int, float]:
    """Combine all concept-level signals into a single score.

    Weights default to: OOF IoU 45%, training freq 30%, hierarchy 15%, synonyms 10%.
    Missing signals fall back to 0.50 (neutral).
    """
    all_cids = set(oof_iou) | set(train_freq) | set(syn_count) | set(hier_depth)
    out: dict[int, float] = {}
    for cid in all_cids:
        score = (
            w_oof   * oof_iou.get(cid, 0.50)
            + w_freq  * train_freq.get(cid, 0.50)
            + w_depth * hier_depth.get(cid, 0.50)
            + w_syn   * syn_count.get(cid, 0.50)
        )
        out[cid] = score
    return out


# ---------------------------------------------------------------------------
# Core confidence function (applied row-wise after enrichment)
# ---------------------------------------------------------------------------

def compute_confidence(
    row: pd.Series,
    fp_blocklist: set[int],
    concept_conf: dict[int, float] | None = None,
    concept_weight: float = 0.7,
) -> float:
    """Return a [0, 1] confidence score for one annotation row.

    Columns expected: concept_id, span

    Parameters
    ----------
    fp_blocklist    : concept_ids known to be pure FPs (hard zero)
    concept_conf    : combined per-concept score from build_concept_confidence()
    concept_weight  : weight given to concept-level score vs span heuristic (0–1)
    """
    # Hard-zero: known pure-FP concept
    if int(row["concept_id"]) in fp_blocklist:
        return 0.0

    span: str = str(row.get("span", "")).strip()

    # Hard-zero: empty / whitespace / pure punctuation
    if not span or not any(c.isalnum() for c in span):
        return 0.0

    # Hard-zero: digit-only / lab-value pattern
    if _LAB_VALUE_RE.match(span.strip()):
        return 0.0

    # Hard-zero: all tokens are stopwords
    tokens = span.lower().split()
    if tokens and all(t in _STOPWORDS for t in tokens):
        return 0.0

    char_len = len(span)
    token_count = len(tokens)
    span_score = _len_score(char_len) * _token_mult(token_count, char_len)

    if concept_conf is not None:
        cid = int(row["concept_id"])
        if cid in concept_conf:
            return concept_weight * concept_conf[cid] + (1 - concept_weight) * span_score
        # Concept not seen in any signal data: fall back to span heuristic

    return span_score


# ---------------------------------------------------------------------------
# Section detection (re-implemented inline to avoid import side-effects)
# ---------------------------------------------------------------------------
# We reproduce the minimal logic from test_old_challenge_split._find_sections /
# _section_at here so this script is self-contained.

try:
    # Prefer importing from the module if PYTHONPATH is set correctly
    from test_old_challenge_split import _find_sections, _section_at  # type: ignore
except ImportError:
    from engine import COMMON_HEADERS  # type: ignore

    _SECTION_RE = re.compile(
        rf"(?m)^(?P<header>{'|'.join(re.escape(h) for h in COMMON_HEADERS)})\s*:?\s*$",
        re.IGNORECASE,
    )

    def _find_sections(text: str) -> list[tuple[int, str]]:
        breaks = []
        for m in _SECTION_RE.finditer(text):
            breaks.append((m.end(), m.group("header").strip().lower()))
        breaks.sort(key=lambda x: x[0])
        return breaks

    def _section_at(text: str, pos: int, section_breaks: list[tuple[int, str]]) -> str:
        current = "(preamble)"
        for brk_pos, header in section_breaks:
            if brk_pos <= pos:
                current = header
            else:
                break
        return current


# ---------------------------------------------------------------------------
# Enrichment: add span text + section to raw predictions
# ---------------------------------------------------------------------------

def enrich_predictions(pred_df: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
    """Add span, section, span_len, span_tokens columns to pred_df.

    Parameters
    ----------
    pred_df : DataFrame with columns [note_id, start, end, concept_id]
    notes_df : DataFrame with columns [note_id, text]
    """
    note_text: dict[str, str] = dict(zip(notes_df["note_id"], notes_df["text"]))
    note_sections: dict[str, list] = {
        nid: _find_sections(txt) for nid, txt in note_text.items()
    }

    spans = []
    sections = []
    for row in pred_df.itertuples(index=False):
        text = note_text.get(row.note_id, "")
        span = text[int(row.start) : int(row.end)]
        section = _section_at(text, int(row.start), note_sections.get(row.note_id, []))
        spans.append(span)
        sections.append(section)

    enriched = pred_df.copy()
    enriched["span"] = spans
    enriched["section"] = sections
    enriched["span_len"] = enriched["span"].str.len()
    enriched["span_tokens"] = enriched["span"].str.split().str.len().fillna(0).astype(int)
    return enriched


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def threshold_sweep(
    scored_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    thresholds: list[float],
    label: str = "test",
    error_fp_df: Optional[pd.DataFrame] = None,
    error_tp_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Evaluate macro IoU at each confidence threshold.

    Parameters
    ----------
    scored_df : enriched predictions with 'confidence' column
    gold_df : ground-truth annotations with [note_id, start, end, concept_id]
    thresholds : list of float thresholds to evaluate
    label : display label for this dataset
    error_fp_df : optional false_positives.csv (for cross-checking counts)
    error_tp_df : optional correct_matches.csv (for cross-checking counts)
    """
    from runtime_scoring import macro_char_iou  # type: ignore

    gold_cols = gold_df[["note_id", "start", "end", "concept_id"]].copy()
    gold_cols["start"] = gold_cols["start"].astype(int)
    gold_cols["end"] = gold_cols["end"].astype(int)
    gold_cols["concept_id"] = gold_cols["concept_id"].astype(int)

    baseline_iou = macro_char_iou(
        scored_df[["note_id", "start", "end", "concept_id"]].assign(
            start=scored_df["start"].astype(int),
            end=scored_df["end"].astype(int),
            concept_id=scored_df["concept_id"].astype(int),
        ),
        gold_cols,
    )

    rows = []
    n_total = len(scored_df)

    # Build lookup sets from error-analysis files for counting TP/FP drops
    fp_set: set[tuple] = set()
    tp_set: set[tuple] = set()
    if error_fp_df is not None:
        fp_set = set(
            zip(
                error_fp_df["note_id"],
                error_fp_df["pred_start"].astype(int),
                error_fp_df["pred_end"].astype(int),
                error_fp_df["concept_id"].astype(int),
            )
        )
    if error_tp_df is not None:
        tp_set = set(
            zip(
                error_tp_df["note_id"],
                error_tp_df["pred_start"].astype(int),
                error_tp_df["pred_end"].astype(int),
                error_tp_df["concept_id"].astype(int),
            )
        )

    # Total identifiable FPs / TPs in the full prediction set (denominator for %)
    all_keys = set(
        zip(
            scored_df["note_id"],
            scored_df["start"].astype(int),
            scored_df["end"].astype(int),
            scored_df["concept_id"].astype(int),
        )
    )
    total_fp_in_pred = len(all_keys & fp_set) if fp_set else None
    total_tp_in_pred = len(all_keys & tp_set) if tp_set else None

    for t in thresholds:
        kept = scored_df[scored_df["confidence"] >= t].copy()
        n_kept = len(kept)
        n_dropped = n_total - n_kept

        iou = macro_char_iou(
            kept[["note_id", "start", "end", "concept_id"]].assign(
                start=kept["start"].astype(int),
                end=kept["end"].astype(int),
                concept_id=kept["concept_id"].astype(int),
            ),
            gold_cols,
        )

        # Cross-check: how many of the dropped annotations were FPs vs TPs
        fp_dropped = tp_dropped = None
        if fp_set or tp_set:
            dropped = scored_df[scored_df["confidence"] < t]
            dropped_keys = set(
                zip(
                    dropped["note_id"],
                    dropped["start"].astype(int),
                    dropped["end"].astype(int),
                    dropped["concept_id"].astype(int),
                )
            )
            if fp_set:
                fp_dropped = len(dropped_keys & fp_set)
            if tp_set:
                tp_dropped = len(dropped_keys & tp_set)

        rows.append(
            {
                "dataset": label,
                "threshold": t,
                "total_preds": n_kept,
                "n_dropped": n_dropped,
                "fp_dropped": fp_dropped,
                "tp_dropped": tp_dropped,
                "total_fp": total_fp_in_pred,
                "total_tp": total_tp_in_pred,
                "macro_iou": round(iou, 5),
                "delta_iou": round(iou - baseline_iou, 5),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True, help="kiri_super_pred.csv")
    p.add_argument("--notes", required=True, help="test_notes.csv")
    p.add_argument("--gold", required=True, help="test_annotations.csv")
    p.add_argument("--fp-blocklist", required=True, help="fp_blocklist_common.csv")
    p.add_argument("--output", required=True, help="output CSV with confidence column")
    p.add_argument("--oof-pred", required=True, help="kiri_super_oof_pred.csv")
    p.add_argument("--oof-gold", required=True, help="train_annotations.csv (also used for train-freq signal)")
    p.add_argument(
        "--super-dict-tsv",
        default="data/interim/super_dictionary_full.tsv",
        help="super_dictionary_full.tsv for synonym-count signal",
    )
    p.add_argument(
        "--subsumption-pkl",
        default="snomed_index/subsumption.pkl",
        help="subsumption.pkl for hierarchy-depth signal",
    )
    p.add_argument(
        "--concept-weight",
        type=float,
        default=0.7,
        help="Weight given to concept-level score vs span heuristic (default 0.7)",
    )
    p.add_argument(
        "--oof-notes",
        help="train_notes.csv for OOF enrichment (auto-derived from --notes if absent)",
    )
    p.add_argument(
        "--error-fp",
        default=None,
        help="error_analysis/false_positives.csv for FP cross-check (optional)",
    )
    p.add_argument(
        "--error-tp",
        default=None,
        help="error_analysis/correct_matches.csv for TP cross-check (optional)",
    )
    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60],
        help="Confidence thresholds to evaluate (default: 0.0 … 0.60)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("Loading data …")
    pred_df = pd.read_csv(args.pred)
    notes_df = pd.read_csv(args.notes)
    gold_df = pd.read_csv(args.gold)
    fp_blocklist_df = pd.read_csv(args.fp_blocklist)
    fp_blocklist: set[int] = set(fp_blocklist_df["concept_id"].astype(int).tolist())
    oof_pred_df = pd.read_csv(args.oof_pred)
    oof_gold_df = pd.read_csv(args.oof_gold)  # also serves as training gold

    error_fp_df = pd.read_csv(args.error_fp) if args.error_fp else None
    error_tp_df = pd.read_csv(args.error_tp) if args.error_tp else None

    print(f"  Predictions: {len(pred_df):,}")
    print(f"  Notes: {len(notes_df):,}")
    print(f"  Gold annotations: {len(gold_df):,}")
    print(f"  FP blocklist concepts: {len(fp_blocklist):,}")

    # --- Build all concept-level signals ---
    print("\nBuilding concept-level signals …")

    print("  [1/4] OOF IoU signal …")
    oof_iou = build_oof_iou_signal(oof_pred_df, oof_gold_df)
    n_pure_fp = sum(1 for v in oof_iou.values() if v <= 0.05)
    print(f"        {len(oof_iou):,} concepts tracked  ({n_pure_fp} pure-FP in OOF)")

    print("  [2/4] Training gold frequency signal …")
    train_freq = build_train_freq_signal(oof_gold_df)
    print(f"        {len(train_freq):,} concepts with ≥1 training annotation")
    test_cids = set(pred_df["concept_id"].astype(int))
    n_never_seen = len(test_cids - set(train_freq))
    print(f"        {n_never_seen:,} test-predicted concepts never in training gold")

    print("  [3/4] Synonym count signal …")
    syn_count = build_synonym_count_signal(args.super_dict_tsv)
    print(f"        {len(syn_count):,} concepts in super dictionary")

    print("  [4/4] Hierarchy depth signal …")
    hier_depth = build_hierarchy_depth_signal(args.subsumption_pkl)
    print(f"        {len(hier_depth):,} concepts with hierarchy info")

    concept_conf = build_concept_confidence(oof_iou, train_freq, syn_count, hier_depth)
    print(f"  Combined: {len(concept_conf):,} concepts  (concept_weight={args.concept_weight})")

    print("\nEnriching predictions with span text + sections …")
    enriched = enrich_predictions(pred_df, notes_df)

    print("Computing confidence scores …")
    enriched["confidence"] = enriched.apply(
        lambda row: compute_confidence(row, fp_blocklist, concept_conf, args.concept_weight),
        axis=1,
    )
    # Also store span-only score for comparison
    enriched["span_score"] = enriched.apply(
        lambda row: compute_confidence(row, fp_blocklist, None), axis=1
    )

    # Quick summary of score distribution
    for t in [0.0, 0.10, 0.20, 0.30, 0.50]:
        n = (enriched["confidence"] >= t).sum()
        print(f"  conf >= {t:.2f}: {n:,} ({100*n/len(enriched):.1f}%)")

    # Save enriched output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"\nSaved scored predictions → {out_path}")

    # --- Threshold sweep on test set ---
    print("\n" + "=" * 72)
    print("THRESHOLD SWEEP — TEST SET")
    print("=" * 72)
    sweep = threshold_sweep(
        enriched, gold_df, args.thresholds,
        label="test",
        error_fp_df=error_fp_df,
        error_tp_df=error_tp_df,
    )
    _print_sweep(sweep)

    # --- Threshold sweep on OOF (train) set for generalization check ---
    # NOTE: For the OOF sweep we cannot use the same concept_conf map (it was
    # built from OOF data, so it would be train-on-train).  We use span-only
    # scoring here so that the OOF sweep is unbiased.
    print("\n" + "=" * 72)
    print("THRESHOLD SWEEP — OOF (TRAIN) SET  [span-only score, unbiased]")
    print("=" * 72)
    oof_notes_path = args.oof_notes or str(args.notes).replace("test_notes", "train_notes")
    oof_notes_df = pd.read_csv(oof_notes_path)
    print(f"  OOF predictions: {len(oof_pred_df):,}")
    print("  Enriching OOF predictions …")
    oof_enriched = enrich_predictions(oof_pred_df, oof_notes_df)
    oof_enriched["confidence"] = oof_enriched.apply(
        lambda row: compute_confidence(row, fp_blocklist, None), axis=1
    )
    oof_sweep = threshold_sweep(
        oof_enriched, oof_gold_df, args.thresholds, label="oof"
    )
    _print_sweep(oof_sweep)

    # --- Score distribution breakdown by confidence bucket ---
    print("\n" + "=" * 72)
    print("CONFIDENCE DISTRIBUTION (test predictions)")
    print("=" * 72)
    buckets = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    bucket_labels = [
        f"[{buckets[i]:.1f}, {buckets[i+1]:.1f})" for i in range(len(buckets) - 1)
    ]
    enriched["bucket"] = pd.cut(
        enriched["confidence"], bins=buckets, labels=bucket_labels, right=False
    )
    bucket_counts = enriched["bucket"].value_counts().sort_index()
    for label, count in bucket_counts.items():
        print(f"  {label}: {count:,} ({100*count/len(enriched):.1f}%)")


def _print_sweep(df: pd.DataFrame) -> None:
    has_fp = df["fp_dropped"].notna().any()
    has_tp = df["tp_dropped"].notna().any()

    total_fp = int(df["total_fp"].iloc[0]) if has_fp and pd.notna(df["total_fp"].iloc[0]) else None
    total_tp = int(df["total_tp"].iloc[0]) if has_tp and pd.notna(df["total_tp"].iloc[0]) else None

    header = f"{'Thresh':>6} | {'Preds':>6} | {'Drop':>5}"
    if has_fp:
        header += f" | {'FP drop':>10}"
    if has_tp:
        header += f" | {'TP drop':>10}"
    header += f" | {'IoU':>7} | {'Δ IoU':>7}"
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        line = f"  {row['threshold']:>4.2f}  | {row['total_preds']:>6,} | {row['n_dropped']:>5,}"

        if has_fp:
            if pd.notna(row["fp_dropped"]):
                n = int(row["fp_dropped"])
                pct = f"{100*n/total_fp:.0f}%" if total_fp else "n/a"
                line += f" | {n:>5,} ({pct:>4})"
            else:
                line += f" | {'n/a':>10}"

        if has_tp:
            if pd.notna(row["tp_dropped"]):
                n = int(row["tp_dropped"])
                pct = f"{100*n/total_tp:.0f}%" if total_tp else "n/a"
                line += f" | {n:>5,} ({pct:>4})"
            else:
                line += f" | {'n/a':>10}"

        delta = f"+{row['delta_iou']:.4f}" if row["delta_iou"] >= 0 else f"{row['delta_iou']:.4f}"
        line += f" | {row['macro_iou']:.5f} | {delta:>7}"
        print(line)

    if has_fp and total_fp is not None:
        print(f"  (FP % denominator = {total_fp:,} identifiable FPs in pred set)")
    if has_tp and total_tp is not None:
        print(f"  (TP % denominator = {total_tp:,} identifiable TPs in pred set)")


if __name__ == "__main__":
    main()
