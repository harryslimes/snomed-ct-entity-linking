#!/usr/bin/env python3
"""Test the super dictionary on the old challenge train/test split.

Standalone dictionary matcher: loads the super dictionary, matches terms in
clinical notes using n-gram prefiltered regex, resolves overlaps, and scores
predictions against gold annotations.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine import COMMON_HEADERS, generate_lookup_keys
from runtime_scoring import class_char_iou, macro_char_iou

# ---------------------------------------------------------------------------
# Blacklist: terms too generic / ambiguous to match reliably as standalone
# concepts.  Validated against gold annotations -- none of these appear as
# standalone gold spans.
# ---------------------------------------------------------------------------
INTERNAL_BLACKLIST = {
    # KIRI internal blacklist
    "other", "negative", "follow up", "mild", "normal", "inr",
    "changes", "change", "iv",
    # Structural / administrative / modifier terms (verified zero gold)
    "tablet", "patient", "history", "history of", "without", "present",
    "physical", "report", "follow", "probably", "likely", "probable",
    "hospital", "due to", "on admission", "daily", "prior", "entire",
    "finding", "observation", "technique", "method", "product", "substance",
    "agent", "action", "unit", "intake", "structure", "body structure",
    "body", "noted", "type", "form", "order", "level", "dose", "status",
    "condition", "plan", "result", "results", "acute", "positive", "date",
    "state", "onset", "initial", "amount", "single", "event", "primary",
    "complete", "partial", "total", "mixed", "simple", "complex", "major",
    "minor", "open", "closed",
}

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _build_pattern(mention: str) -> re.Pattern[str]:
    """Build a flexible regex from a dictionary mention.

    - Spaces become \\s+
    - Hyphens become [- ]
    - Slashes become [/ ]
    - Trailing optional plural 's'
    """
    parts = []
    escaped = ""
    for ch in mention:
        if ch == " ":
            if escaped:
                parts.append(re.escape(escaped))
                escaped = ""
            parts.append(r"\s+")
        elif ch == "-":
            if escaped:
                parts.append(re.escape(escaped))
                escaped = ""
            parts.append(r"[- ]")
        elif ch == "/":
            if escaped:
                parts.append(re.escape(escaped))
                escaped = ""
            parts.append(r"[/ ]")
        else:
            escaped += ch
    if escaped:
        parts.append(re.escape(escaped))

    pattern_str = "".join(parts) + r"s?"
    return re.compile(pattern_str, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Dictionary loading with n-gram index
# ---------------------------------------------------------------------------
class DictionaryMatcher:
    """Efficient dictionary matcher using bigram prefiltering."""

    def __init__(self):
        # entries: list of (mention, concept_id, pattern)
        self.entries: list[tuple[str, int, re.Pattern[str]]] = []
        # bigram index: (tok1, tok2) -> list of entry indices
        self._bigram_idx: dict[tuple[str, str], list[int]] = defaultdict(list)
        # unigram index: tok -> list of entry indices (for single-token mentions)
        self._unigram_idx: dict[str, list[int]] = defaultdict(list)
        # pattern cache
        self._pattern_cache: dict[str, re.Pattern[str]] = {}

    def _get_pattern(self, mention: str) -> re.Pattern[str]:
        if mention not in self._pattern_cache:
            self._pattern_cache[mention] = _build_pattern(mention)
        return self._pattern_cache[mention]

    def add(self, mention: str, concept_id: int):
        pattern = self._get_pattern(mention)
        idx = len(self.entries)
        self.entries.append((mention, concept_id, pattern))

        tokens = _tokenize(mention)
        if len(tokens) >= 2:
            key = (tokens[0], tokens[1])
            self._bigram_idx[key].append(idx)
            # plural variants
            self._bigram_idx[(tokens[0], tokens[1].rstrip("s"))].append(idx)
            self._bigram_idx[(tokens[0] + "s", tokens[1])].append(idx)
        elif len(tokens) == 1:
            self._unigram_idx[tokens[0]].append(idx)
            self._unigram_idx[tokens[0].rstrip("s")].append(idx)

    def match(self, text: str, note_id: str) -> list[dict]:
        """Find all dictionary matches in text, return list of match dicts."""
        tokens = _tokenize(text)
        token_set = set(tokens)

        # Build bigrams present in this note
        bigrams = set()
        for i in range(len(tokens) - 1):
            bigrams.add((tokens[i], tokens[i + 1]))
            # plural stripped
            bigrams.add((tokens[i].rstrip("s"), tokens[i + 1]))
            bigrams.add((tokens[i], tokens[i + 1].rstrip("s")))
            bigrams.add((tokens[i].rstrip("s"), tokens[i + 1].rstrip("s")))

        # Collect candidate entry indices
        candidates: set[int] = set()
        for bg in bigrams:
            if bg in self._bigram_idx:
                candidates.update(self._bigram_idx[bg])
        for tok in token_set:
            if tok in self._unigram_idx:
                candidates.update(self._unigram_idx[tok])
            stripped = tok.rstrip("s")
            if stripped in self._unigram_idx:
                candidates.update(self._unigram_idx[stripped])

        # Run regex on candidates only
        matches = []
        for idx in candidates:
            mention, concept_id, pattern = self.entries[idx]
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                # Word boundary check
                if start > 0 and text[start - 1].isalnum():
                    continue
                if end < len(text) and text[end].isalnum():
                    continue
                matches.append({
                    "note_id": note_id,
                    "start": start,
                    "end": end,
                    "concept_id": concept_id,
                    "mention": mention,
                })

        return matches


def _remove_overlaps(matches: list[dict]) -> list[dict]:
    """Resolve overlapping matches with two-pass strategy.

    Pass 1: greedily keep shorter (more specific) span on overlap.
    Pass 2: re-check removed spans against final kept set, restoring
    non-overlapping ones that were lost due to greedy replacement.
    """
    if not matches:
        return matches

    # Sort by start, then by span length (prefer shorter)
    matches.sort(key=lambda m: (m["start"], m["end"] - m["start"]))

    # Pass 1: greedy selection
    kept = []
    removed = []
    for m in matches:
        span_len = m["end"] - m["start"]
        replace_idx = None
        overlapping = False

        for i, k in enumerate(kept):
            if m["start"] < k["end"] and m["end"] > k["start"]:
                k_len = k["end"] - k["start"]
                if span_len < k_len:
                    replace_idx = i
                else:
                    overlapping = True
                break

        if replace_idx is not None:
            removed.append(kept[replace_idx])
            kept[replace_idx] = m
        elif not overlapping:
            kept.append(m)
        else:
            removed.append(m)

    # Pass 2: try to restore removed spans that don't conflict
    if removed:
        removed.sort(key=lambda m: (m["start"], m["end"] - m["start"]))
        for m in removed:
            conflicts = False
            for k in kept:
                if m["start"] < k["end"] and m["end"] > k["start"]:
                    conflicts = True
                    break
            if not conflicts:
                kept.append(m)
        kept.sort(key=lambda m: m["start"])

    return kept


# ---------------------------------------------------------------------------
# Trained-dict prediction (KIRI-style two-pass)
# ---------------------------------------------------------------------------
def predict_notes_trained(
    trained_dict_path: Path,
    notes: pd.DataFrame,
) -> pd.DataFrame:
    """Two-pass KIRI-style prediction using a trained dictionary pickle.

    Pass 1: lowercase text + main dict (d)
    Pass 2: original text + uppercase dict (uc_d) + case-sensitive entries
    Then join with overlap removal.
    """
    from train_dictionary import (
        IndexedDict,
        annotate_with_dict,
        remove_overlaps,
        CASE_SENSITIVE_DICT,
    )

    with trained_dict_path.open("rb") as fp:
        blob = pickle.load(fp)
    d = blob["d"]
    uc_d = blob["uc_d"]

    # Merge case-sensitive hardcoded entries into uc_d
    uc_d_full = dict(uc_d)
    uc_d_full.update(CASE_SENSITIVE_DICT)

    headers_lc = [h.lower() for h in COMMON_HEADERS]
    headers_orig = list(COMMON_HEADERS)

    d_indexed = IndexedDict(d, prefilter="bigram")
    uc_indexed = IndexedDict(uc_d_full, prefilter="bigram")

    print(f"  Trained dict: {len(d):,} entries, UC dict: {len(uc_d_full):,} entries")

    all_preds = []
    for _, note_row in notes.iterrows():
        note_id = str(note_row["note_id"])
        text = note_row["text"]
        text_lc = text.lower()

        # Pass 1: lowercase
        ann_lc = annotate_with_dict(text_lc, d_indexed, headers_lc, note_id)

        # Pass 2: original case
        ann_uc = annotate_with_dict(text, uc_indexed, headers_orig, note_id)

        # Join with overlap removal
        combined = pd.concat([ann_lc, ann_uc], ignore_index=True)
        if not combined.empty:
            combined = remove_overlaps(combined)
            all_preds.append(combined)

    if not all_preds:
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id"])

    pred = pd.concat(all_preds, ignore_index=True)
    pred = pred[["note_id", "start", "end", "concept_id"]].copy()
    pred["start"] = pred["start"].astype(int)
    pred["end"] = pred["end"].astype(int)
    pred["concept_id"] = pred["concept_id"].astype(int)
    return pred


# ---------------------------------------------------------------------------
# Skip sections that typically produce false positives
# ---------------------------------------------------------------------------
_SKIP_SECTIONS = {
    "name",
    "admission date",
    "date of birth",
    "attending",
    "service",
    "discharge disposition",
    "discharge condition",
    "discharge instructions",
    "followup instructions",
}

_HEADER_RE = None


def _get_header_re() -> re.Pattern[str]:
    global _HEADER_RE
    if _HEADER_RE is None:
        escaped = [re.escape(h) for h in COMMON_HEADERS]
        _HEADER_RE = re.compile(
            rf"(?m)^(?P<header>{'|'.join(escaped)})\s*:?\s*$", re.IGNORECASE
        )
    return _HEADER_RE


def _section_at(text: str, pos: int, section_breaks: list[tuple[int, str]]) -> str:
    """Return the section header active at character position `pos`."""
    current = "(preamble)"
    for brk_pos, header in section_breaks:
        if brk_pos <= pos:
            current = header
        else:
            break
    return current


def _find_sections(text: str) -> list[tuple[int, str]]:
    """Return sorted list of (char_pos, header_lower) for section breaks."""
    pattern = _get_header_re()
    breaks = []
    for m in pattern.finditer(text):
        breaks.append((m.end(), m.group("header").strip().lower()))
    breaks.sort(key=lambda x: x[0])
    return breaks


# ---------------------------------------------------------------------------
# Load dictionary
# ---------------------------------------------------------------------------
def load_super_dictionary(
    dict_path: Path,
    *,
    train_annotations_path: Optional[Path] = None,
    min_term_len: int = 4,
    single_token_min_len: int = 8,
    max_term_tokens: int = 10,
    blacklist: Optional[set[str]] = None,
) -> DictionaryMatcher:
    """Load super dictionary TSV into an efficient matcher.

    Optionally adds training annotation spans (which teach the matcher
    domain-specific mentions observed in the training set).

    Parameters
    ----------
    dict_path : Path to super_dictionary_full.tsv
    train_annotations_path : optional CSV with concept_id, span columns
    min_term_len : minimum term length in characters (default 3)
    single_token_min_len : minimum length for single-token terms (default 4)
    max_term_tokens : maximum token count per term (default 10)
    blacklist : set of lowercased terms to exclude (default INTERNAL_BLACKLIST)
    """
    matcher = DictionaryMatcher()
    seen: set[tuple[int, str]] = set()
    blacklist = blacklist if blacklist is not None else INTERNAL_BLACKLIST
    skipped_blacklist = 0
    skipped_short = 0

    def _should_add(term: str) -> bool:
        nonlocal skipped_blacklist, skipped_short
        if len(term) < min_term_len:
            skipped_short += 1
            return False
        tokens = term.split()
        if len(tokens) > max_term_tokens:
            return False
        # Single-token terms need a higher character minimum
        if len(tokens) == 1 and len(term) < single_token_min_len:
            skipped_short += 1
            return False
        if term.lower() in blacklist:
            skipped_blacklist += 1
            return False
        return True

    # Load from TSV
    with dict_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        for row in reader:
            concept_code = (row.get("snomed_concept_id") or "").strip()
            term = (row.get("term") or "").strip()
            if not concept_code or not term:
                continue
            if not _should_add(term):
                continue
            try:
                concept_id = int(concept_code)
            except ValueError:
                continue

            key = (concept_id, term.lower())
            if key in seen:
                continue
            seen.add(key)
            matcher.add(term, concept_id)

    # Add training spans if provided
    if train_annotations_path and train_annotations_path.exists():
        with train_annotations_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                concept_id_str = (row.get("concept_id") or "").strip()
                span = (row.get("span") or "").strip()
                if not concept_id_str or not span:
                    continue
                if not _should_add(span):
                    continue
                try:
                    concept_id = int(concept_id_str)
                except ValueError:
                    continue
                key = (concept_id, span.lower())
                if key in seen:
                    continue
                seen.add(key)
                matcher.add(span, concept_id)

    print(f"  Skipped {skipped_blacklist:,} blacklisted, {skipped_short:,} too-short entries")
    return matcher


# ---------------------------------------------------------------------------
# Predict on notes
# ---------------------------------------------------------------------------
def predict_notes(
    matcher: DictionaryMatcher,
    notes: pd.DataFrame,
    *,
    skip_preamble_chars: int = 100,
    skip_sections: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Run dictionary matcher on all notes, return predictions DataFrame."""
    skip_sections = skip_sections or _SKIP_SECTIONS
    all_rows = []

    for _, note_row in notes.iterrows():
        note_id = note_row["note_id"]
        text = note_row["text"]

        # Find section breaks for filtering
        section_breaks = _find_sections(text)

        # Get raw matches
        raw = matcher.match(text, str(note_id))

        # Filter
        filtered = []
        for m in raw:
            # Skip preamble (metadata)
            if m["start"] < skip_preamble_chars:
                continue
            # Skip excluded sections
            section = _section_at(text, m["start"], section_breaks)
            if section in skip_sections:
                continue
            filtered.append(m)

        # Remove overlaps
        deduped = _remove_overlaps(filtered)
        all_rows.extend(deduped)

    if not all_rows:
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id"])

    df = pd.DataFrame(all_rows)[["note_id", "start", "end", "concept_id"]]
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["concept_id"] = df["concept_id"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test the super dictionary on the old challenge train/test split."
    )
    parser.add_argument(
        "--super-dict",
        default="data/interim/super_dictionary_full.tsv",
        help="Path to super_dictionary_full.tsv",
    )
    parser.add_argument(
        "--split-dir",
        default="data/old-challenge-split",
        help="Directory containing train/test split files",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/old_challenge_split",
        help="Directory to write predictions and scores",
    )
    parser.add_argument(
        "--skip-preamble",
        type=int,
        default=100,
        help="Skip matches in the first N characters of each note (metadata)",
    )
    parser.add_argument(
        "--train-boost",
        action="store_true",
        help="Also add training annotation spans to the matcher (domain adaptation)",
    )
    parser.add_argument(
        "--min-term-len",
        type=int,
        default=4,
        help="Minimum term length in characters (default: 4)",
    )
    parser.add_argument(
        "--single-token-min-len",
        type=int,
        default=8,
        help="Minimum length for single-token dictionary terms (default: 8). "
             "Higher values reduce false positives. Use 100 to disable single-token "
             "matching entirely (~0.28 IoU).",
    )
    parser.add_argument(
        "--max-term-tokens",
        type=int,
        default=10,
        help="Maximum token count per dictionary term (default: 10)",
    )
    parser.add_argument(
        "--no-blacklist",
        action="store_true",
        help="Disable internal blacklist filtering",
    )
    parser.add_argument(
        "--trained-dict",
        default=None,
        help="Path to trained dictionary pickle (from train_dictionary.py). "
             "When provided, uses KIRI-style two-pass prediction instead of "
             "standalone matcher.",
    )
    args = parser.parse_args(argv)

    split_dir = Path(args.split_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("Loading test data...", flush=True)
    test_notes = pd.read_csv(split_dir / "test_notes.csv")
    test_gold = pd.read_csv(split_dir / "test_annotations.csv")
    for col in ["start", "end", "concept_id"]:
        if col in test_gold.columns:
            test_gold[col] = test_gold[col].astype(int)

    print(f"Test: {len(test_notes)} notes, {len(test_gold)} annotations")

    # Choose prediction mode
    if args.trained_dict:
        # KIRI-style two-pass prediction with trained dictionary
        trained_path = Path(args.trained_dict)
        if not trained_path.exists():
            print(f"Missing trained dictionary: {trained_path}", file=sys.stderr)
            print("Run train_dictionary.py first.", file=sys.stderr)
            return 2

        print(f"Loading trained dictionary from {trained_path}...", flush=True)
        t0 = time.perf_counter()
        pred = predict_notes_trained(trained_path, test_notes)
        t1 = time.perf_counter()
        print(f"Generated {len(pred):,} predictions in {t1 - t0:.1f}s", flush=True)
    else:
        # Standalone matcher mode (no training)
        dict_path = Path(args.super_dict)
        if not dict_path.exists():
            print(f"Missing super dictionary: {dict_path}", file=sys.stderr)
            print("Run build_super_dictionary.py first.", file=sys.stderr)
            return 2

        train_ann_path = (split_dir / "train_annotations.csv") if args.train_boost else None
        blacklist = set() if args.no_blacklist else None  # None -> use default

        print(f"Loading dictionary from {dict_path}...", flush=True)
        t0 = time.perf_counter()
        matcher = load_super_dictionary(
            dict_path,
            train_annotations_path=train_ann_path,
            min_term_len=args.min_term_len,
            single_token_min_len=args.single_token_min_len,
            max_term_tokens=args.max_term_tokens,
            blacklist=blacklist,
        )
        t1 = time.perf_counter()
        print(f"Loaded {len(matcher.entries):,} dictionary entries in {t1 - t0:.1f}s", flush=True)

        # Predict
        print(f"Running matcher on {len(test_notes)} test notes...", flush=True)
        t2 = time.perf_counter()
        pred = predict_notes(
            matcher,
            test_notes,
            skip_preamble_chars=args.skip_preamble,
        )
        t3 = time.perf_counter()
        print(f"Generated {len(pred):,} predictions in {t3 - t2:.1f}s", flush=True)

    # Save predictions
    pred_path = out_dir / "super_dict_pred.csv"
    pred.to_csv(pred_path, index=False)
    print(f"Saved predictions -> {pred_path}")

    # Score
    print("Scoring...", flush=True)
    gold_cols = test_gold[["note_id", "start", "end", "concept_id"]].copy()
    pred_cols = pred[["note_id", "start", "end", "concept_id"]].copy()

    score = macro_char_iou(pred_cols, gold_cols)

    cls = class_char_iou(pred_cols, gold_cols)
    cls_path = out_dir / "super_dict_class_iou.csv"
    cls.to_csv(cls_path, index=False)

    # Summary stats
    n_concepts_pred = pred["concept_id"].nunique()
    n_concepts_gold = test_gold["concept_id"].nunique()
    n_notes_with_pred = pred["note_id"].nunique()

    valid_cls = cls[cls["union"] > 0]
    n_concepts_scored = len(valid_cls)
    perfect = valid_cls[valid_cls["iou"] >= 0.99]
    zero = valid_cls[valid_cls["iou"] == 0.0]

    # Precision / recall analysis
    pred_only = valid_cls[(valid_cls["intersection"] == 0) & (valid_cls["pred_chars"] > 0)]
    gold_only = valid_cls[(valid_cls["intersection"] == 0) & (valid_cls["pred_chars"] == 0)]

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Macro-averaged character IoU:  {score:.4f}")
    print()
    print(f"Predictions:      {len(pred):>8,}")
    print(f"Gold annotations: {len(test_gold):>8,}")
    print(f"Notes with preds: {n_notes_with_pred:>8,} / {len(test_notes)}")
    print(f"Concepts in pred: {n_concepts_pred:>8,}")
    print(f"Concepts in gold: {n_concepts_gold:>8,}")
    print(f"Concepts scored:  {n_concepts_scored:>8,}")
    print(f"  Perfect (>=0.99 IoU): {len(perfect):>5}")
    print(f"  Zero IoU:             {len(zero):>5}")
    print(f"  Pred-only (FP):       {len(pred_only):>5}")
    print(f"  Gold-only (FN):       {len(gold_only):>5}")
    print()
    print(f"Predictions: {pred_path}")
    print(f"Class IoU:   {cls_path}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
