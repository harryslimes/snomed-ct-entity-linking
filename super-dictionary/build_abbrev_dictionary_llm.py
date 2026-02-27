#!/usr/bin/env python3
"""Build abbreviation dictionary via LLM-based SNOMED disambiguation.

Pipeline:
  1. Extract abbreviation-like spans from data/old-challenge-split annotations
     using heuristic filters (all-caps, short mixed-case, slash forms)
  2. Assign section headers and extract context windows
  3. Deduplicate by (span, section_header) — each unique pair gets one
     representative context example
  4. Run the two-pass vLLM pipeline on each representative:
       a. LLM generates expanded full-form search terms
       b. Hybrid SNOMED retrieval (FAISS + BM25 with RRF fusion)
       c. LLM selects the best matching SNOMED concept
  5. Build a multi-map dictionary: span → [{concept_id, sections, count, …}]
  6. Evaluate predicted concept_ids against gold annotations
     (lookup by (span, section_header) pair)

Outputs (written to --output-dir):
  abbreviation_dictionary_llm.json  — multi-map abbreviation → concept list
  abbrev_eval_results.json          — per-annotation evaluation
  abbrev_eval_summary.txt           — human-readable metrics

Requirements:
  A vLLM server must be running at --vllm-url (default http://localhost:8000)
  serving the model specified by --model.

Example usage:
  python build_abbrev_dictionary_llm.py --split train --limit 200
  python build_abbrev_dictionary_llm.py --split train  # all unique (span, section) pairs
  python build_abbrev_dictionary_llm.py --split test   # evaluate on test split
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))
sys.path.insert(0, str(REPO_ROOT / "rulebook"))
sys.path.insert(0, str(REPO_ROOT))

from engine import get_section_for_pos, segment_sections  # noqa: E402
import rule_testing as rt  # noqa: E402


# ---------------------------------------------------------------------------
# Abbreviation detection heuristics (from build_abbreviation_dictionary.py)
# ---------------------------------------------------------------------------

# Spans that are full English/medical words — not abbreviations.
# These pass the all-caps heuristic but require no expansion (they ARE the term).
_NON_ABBREVIATION_SPANS: frozenset[str] = frozenset({
    # Anatomical body parts / section headers
    "NECK", "CHEST", "LUNGS", "LUNG", "HEART", "ABDOMEN", "PELVIS", "BACK", "BONE",
    "AORTA", "BOWEL", "BOWELS", "GROIN", "SCALP", "TEETH", "BREAST", "BREASTS",
    "AXILLA", "FLANK", "THIGH", "CALF", "SKULL", "ORBIT", "SINUS", "TONSIL",
    "TONSILS", "TONGUE", "PALATE", "THROAT", "WRIST", "ANKLE", "ELBOW",
    "SKIN", "EYES", "LIMBS", "SPINE", "SELLA", "HEAD",
    # Lab analyte / chemical names (the full name, no expansion needed)
    "GLUCOSE", "SODIUM", "CALCIUM", "LIPASE", "AMYLASE", "KETONE", "CHLORIDE",
    "ALBUMIN", "PROTEIN", "BILIRUBIN", "PHOSPHATE", "MAGNESIUM", "POTASSIUM",
    "ETHANOL", "FOLATE", "LITHIUM",
    # Multi-word lab / procedure labels (full phrases, not abbreviations)
    "TOTAL CO2", "UREA N", "ACID FAST", "GRAM STAIN", "WOUND CARE",
    "VITAL SIGNS", "CHEST X-RAY", "BACK PAIN", "BONE MARROW", "CHRONIC PAIN",
    "CT ABDOMEN", "CT PELVIS", "CT HEART", "AORTIC VALVE", "HEART RATE",
    "DRUG ABUSE", "LUNG NODULE", "LEFT HUMERUS",
    # Common section-header / plain words that appear all-caps in notes
    "GENERAL", "CARDIAC", "IMAGING", "VITALS", "URINE", "COLOR", "WOUND",
    "STROKE", "NORMAL", "SPUTUM", "RECTAL", "CULTURE", "BLEEDING", "HERPES",
    "MUCOSA", "PLEURA", "SEPSIS", "UROLOGIC",
    "PULSES", "ABSCESS", "BYPASS", "GOUT", "HYPOXIA", "SYNCOPE", "YEAST",
    "FUNGAL", "ANEMIA", "SURGERY", "WEIGHT", "OBESITY",
})


def _is_abbreviation_candidate(span: str) -> bool:
    """Return True if span looks like a clinical abbreviation.

    Heuristics (any match → True):
      1. All-uppercase alphabetic core, 2-10 chars  (HTN, COPD, CTAB)
      2. Short (<=8 chars) with >=60% uppercase letters  (AOx3, S1S2)
      3. Contains '/' and is short — slash abbreviations  (n/v, NC/AT)

    Exclusions applied before the above:
      - Explicit blocklist of known full words (_NON_ABBREVIATION_SPANS)
      - All-uppercase purely-alphabetic spans ≥ 8 chars (POTASSIUM, EXTREMITIES …
        are always full words; legitimate medical abbreviations are ≤ 7 alpha chars)
    """
    s = span.strip()
    if "\n" in s or len(s) > 12 or len(s) < 2:
        return False
    alpha = "".join(c for c in s if c.isalpha())
    if not alpha:
        return False

    # Reject explicit full-word blocklist (case-sensitive — data is all-caps)
    if s in _NON_ABBREVIATION_SPANS:
        return False

    # Reject long all-uppercase purely-alphabetic spans: real medical
    # abbreviations are short (≤ 7 alpha chars); 8+ char all-caps alpha
    # strings are almost always full spelled-out words (POTASSIUM, EXTREMITIES).
    if alpha.isupper() and alpha == s and len(alpha) >= 8:
        return False

    # Rule 1: all-uppercase core
    if alpha.isupper() and len(alpha) >= 2:
        return True
    # Rule 2: short, mostly uppercase
    upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
    if upper_ratio >= 0.6 and len(s) <= 8:
        return True
    # Rule 3: slash abbreviations
    if "/" in s and len(s) <= 8:
        return True
    return False


# ---------------------------------------------------------------------------
# Abbreviation-specific rules text (injected into the shared prompt builders)
# ---------------------------------------------------------------------------

ABBREV_RULES_SEARCH = """\
=== ABBREVIATION EXPANSION TASK ===

The highlighted text is a clinical abbreviation. Use the section header and
surrounding context to determine the most likely meaning, then generate SNOMED
CT search terms for its full form.

Disambiguation examples:
- MS  in Physical Exam   → "mental status examination"
- MS  in PMH             → "multiple sclerosis"
- SOB                    → "shortness of breath"
- HTN                    → "hypertension"
- COPD                   → "chronic obstructive pulmonary disease"
- CKD                    → "chronic kidney disease"
- NC/AT                  → "normocephalic atraumatic"
- CTAB                   → "clear to auscultation bilaterally"
- A&Ox3                  → "alert and oriented to person place and time"
- c/o                    → use the associated symptom as the search term

Generate the full-form expansion as the primary search term, plus 1-2 alternatives.\
"""

ABBREV_RULES_SELECT = """\
=== ABBREVIATION DISAMBIGUATION TASK ===

The highlighted text is a clinical abbreviation. Select the SNOMED CT concept
that best represents it in this specific clinical context.

Only concepts from these SNOMED top-level categories are annotated in this task:
  "procedure", "body structure", and the "clinical finding" family which includes:
  finding, disorder, morphologic abnormality, situation

When a concept could belong to multiple hierarchies, apply this priority order:
  procedure > finding/disorder > body structure

Use the section header to guide selection within those hierarchies:
- Physical Exam         → prefer finding or disorder
- Past Medical History  → prefer disorder
- Procedures / Surgical → prefer procedure
- Anatomy / body parts  → prefer body structure
- Lab results sections (LABS ON ADMISSION, LABS ON DISCHARGE, PERTINENT RESULTS)
                        → these sections are not annotated; do not select a concept

Where a negated finding has a specific SNOMED concept (e.g. "Apyrexial" for "afebrile"),
prefer that concept over the positive form.

Prefer the most specific concept fully entailed by the span — do not infer detail
not present in the text. Avoid concepts that are too broad (e.g., prefer
"hypertension" over "cardiovascular disorder").\
"""

ABBREV_SEARCH_SYSTEM = """\
You are a clinical NLP agent. Given a clinical abbreviation in context, determine
its most likely expansion and generate SNOMED CT search terms.

The section header and surrounding text are critical for disambiguating overloaded
abbreviations (e.g. "MS" = mental status in Physical Exam, multiple sclerosis in PMH).

Respond with ONLY a JSON object:
{"search_terms": ["primary expansion", "alternative 1", "alternative 2"]}

Generate 2-3 search terms ordered from most to least likely.\
"""

ABBREV_SELECT_SYSTEM = """\
You are a clinical NLP agent. Given a clinical abbreviation in context and
SNOMED CT search results, select the best matching concept.

Prefer the most specific, clinically accurate concept. Use the section and
context to disambiguate when multiple plausible concepts are present.

Respond with ONLY a JSON object:
{"choice": <integer>}

where <integer> is the 1-based index of the best matching candidate from the \
numbered list.\
"""


# ---------------------------------------------------------------------------
# Phase 1: Extract abbreviation annotations with context
# ---------------------------------------------------------------------------

def extract_abbreviation_items(
    ann_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    context_before: int,
    context_after: int,
) -> list[dict]:
    """Extract abbreviation candidates from annotations with section + context.

    Returns one dict per abbreviation annotation instance:
        annotation_id, note_id, start, end, span, gold_concept_id,
        section_header, before, after
    """
    note_texts: dict[str, str] = dict(zip(notes_df.note_id, notes_df.text))
    section_cache: dict[str, list] = {}
    items: list[dict] = []
    skipped_no_text = 0

    for row in tqdm(ann_df.itertuples(), total=len(ann_df), desc="Filtering abbreviations"):
        span = str(row.span).strip() if not pd.isna(row.span) else ""
        if not _is_abbreviation_candidate(span):
            continue

        nid = row.note_id
        note_text = note_texts.get(nid, "")
        if not note_text:
            skipped_no_text += 1
            continue

        # Section header
        if nid not in section_cache:
            section_cache[nid] = segment_sections(note_text)
        sec = get_section_for_pos(int(row.start), section_cache[nid])
        section_header = sec.header if sec else "unknown"

        # Context windows
        before, _, after = rt.get_context(
            note_text, int(row.start), int(row.end),
            context_before, context_after,
        )

        items.append({
            "annotation_id": int(row.annotation_id),
            "note_id": nid,
            "start": int(row.start),
            "end": int(row.end),
            "span": span,
            "gold_concept_id": int(row.concept_id),
            "section_header": section_header,
            "before": before,
            "after": after,
        })

    if skipped_no_text:
        print(f"  Skipped {skipped_no_text} annotations (note text not found)")
    return items


def extract_non_abbreviation_items(
    ann_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    context_before: int,
    context_after: int,
) -> list[dict]:
    """Extract non-abbreviation annotations with section + context.

    Complement of extract_abbreviation_items — keeps everything that
    _is_abbreviation_candidate() rejects. Same return format.
    """
    note_texts: dict[str, str] = dict(zip(notes_df.note_id, notes_df.text))
    section_cache: dict[str, list] = {}
    items: list[dict] = []
    skipped_no_text = 0

    for row in tqdm(ann_df.itertuples(), total=len(ann_df), desc="Filtering non-abbreviations"):
        span = str(row.span).strip() if not pd.isna(row.span) else ""
        if _is_abbreviation_candidate(span):
            continue

        nid = row.note_id
        note_text = note_texts.get(nid, "")
        if not note_text:
            skipped_no_text += 1
            continue

        if nid not in section_cache:
            section_cache[nid] = segment_sections(note_text)
        sec = get_section_for_pos(int(row.start), section_cache[nid])
        section_header = sec.header if sec else "unknown"

        before, _, after = rt.get_context(
            note_text, int(row.start), int(row.end),
            context_before, context_after,
        )

        items.append({
            "annotation_id": int(row.annotation_id),
            "note_id": nid,
            "start": int(row.start),
            "end": int(row.end),
            "span": span,
            "gold_concept_id": int(row.concept_id),
            "section_header": section_header,
            "before": before,
            "after": after,
        })

    if skipped_no_text:
        print(f"  Skipped {skipped_no_text} annotations (note text not found)")
    return items


# ---------------------------------------------------------------------------
# Phase 2: Deduplicate by (span, section_header)
# ---------------------------------------------------------------------------

def build_representative_items(
    all_items: list[dict],
    limit: int = 0,
) -> list[dict]:
    """Select one representative instance per (span, section_header) pair.

    The representative is the instance with the most surrounding context.
    Each representative carries group-level metadata for evaluation:
      - instance_count: how many annotations share this (span, section) pair
      - gold_concept_ids_all: set of gold concept IDs observed in the group
        (often just one, but can be multiple if the abbreviation is truly
        ambiguous even within a section)

    If limit > 0, only the most frequent groups (by instance count) are kept.

    The returned dicts are ready for rt.vllm_pipeline() — they include all
    required keys: before, span, after, section_header, search_rules_text,
    select_rules_text, gold_start, gold_end.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in all_items:
        key = (item["span"], item["section_header"])
        groups[key].append(item)

    # Sort by frequency (most common first) so --limit keeps top abbreviations
    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    if limit > 0:
        sorted_groups = sorted_groups[:limit]

    reps: list[dict] = []
    for (span, section), group in sorted_groups:
        # Pick the instance with the most context text as representative
        best = max(group, key=lambda x: len(x["before"]) + len(x["after"]))
        rep: dict = {
            # --- Required by rt.vllm_pipeline ---
            "before": best["before"],
            "span": span,
            "after": best["after"],
            "section_header": section,
            "search_rules_text": ABBREV_RULES_SEARCH,
            "select_rules_text": ABBREV_RULES_SELECT,
            "gold_start": best["start"],
            "gold_end": best["end"],
            # --- Group metadata for evaluation ---
            "instance_count": len(group),
            "gold_concept_ids_all": sorted({g["gold_concept_id"] for g in group}),
            "representative_note_id": best["note_id"],
        }
        reps.append(rep)

    return reps


# ---------------------------------------------------------------------------
# Phase 3: LLM disambiguation via vllm_pipeline
# ---------------------------------------------------------------------------

def run_abbreviation_pipeline(
    reps: list[dict],
    vllm_url: str,
    model: str,
    max_concurrent: int,
    reasoning_effort: str,
) -> tuple[list[list[str]], list[list[dict]], list[tuple[int, int, int] | None], list[dict]]:
    """Run the two-pass vLLM pipeline with abbreviation-specific system prompts.

    Temporarily monkey-patches rt.SEARCH_SYSTEM and rt.SELECT_SYSTEM so the
    shared vllm_pipeline() uses our abbreviation-focused instructions.
    The original prompts are restored afterwards.
    """
    orig_search = rt.SEARCH_SYSTEM
    orig_select = rt.SELECT_SYSTEM
    try:
        rt.SEARCH_SYSTEM = ABBREV_SEARCH_SYSTEM
        rt.SELECT_SYSTEM = ABBREV_SELECT_SYSTEM
        return rt.vllm_pipeline(
            reps,
            base_url=vllm_url,
            model=model,
            max_concurrent=max_concurrent,
            reasoning_effort=reasoning_effort,
        )
    finally:
        rt.SEARCH_SYSTEM = orig_search
        rt.SELECT_SYSTEM = orig_select


# ---------------------------------------------------------------------------
# Phase 4: Build multi-map dictionary
# ---------------------------------------------------------------------------

def build_dictionary(
    reps: list[dict],
    selections: list[tuple[int, int, int] | None],
    search_terms_list: list[list[str]],
    concept_names: dict[int, tuple[str, str]],
) -> dict[str, list[dict]]:
    """Build the abbreviation → concept multi-map dictionary.

    An abbreviation can have multiple entries if the LLM predicted different
    concepts for different (span, section_header) groups.

    Output structure:
      {
        "HTN": [
          {
            "concept_id": 38341003,
            "concept_name": "Hypertension",
            "hierarchy": "disorder",
            "sections": ["past medical history"],
            "llm_search_terms": ["hypertension", "high blood pressure"],
            "instance_count": 127,
            "gold_concept_ids": [38341003]
          }
        ],
        "MS": [
          {"concept_id": 24700007, "concept_name": "Multiple sclerosis", "sections": ["past medical history"], ...},
          {"concept_id": 36456004, "concept_name": "Mental status", "sections": ["physical exam"], ...}
        ]
      }
    """
    # Accumulate predictions: (span, concept_id) → entry dict
    acc: dict[tuple[str, int], dict] = {}
    for rep, sel, terms in zip(reps, selections, search_terms_list):
        if sel is None:
            continue
        pred_cid, _, _ = sel
        span = rep["span"]
        section = rep["section_header"]
        key = (span, pred_cid)

        if key not in acc:
            cname, hierarchy = concept_names.get(pred_cid, ("Unknown", "unknown"))
            acc[key] = {
                "concept_id": pred_cid,
                "concept_name": cname,
                "hierarchy": hierarchy,
                "sections": [],
                "llm_search_terms": terms,
                "instance_count": 0,
                "gold_concept_ids": [],
            }

        entry = acc[key]
        if section not in entry["sections"]:
            entry["sections"].append(section)
        entry["instance_count"] += rep["instance_count"]
        for gid in rep["gold_concept_ids_all"]:
            if gid not in entry["gold_concept_ids"]:
                entry["gold_concept_ids"].append(gid)

    # Group by span, sort each group by instance count descending
    dictionary: dict[str, list[dict]] = defaultdict(list)
    for (span, _), entry in acc.items():
        dictionary[span].append(entry)
    for span in dictionary:
        dictionary[span].sort(key=lambda x: x["instance_count"], reverse=True)

    return dict(dictionary)


# ---------------------------------------------------------------------------
# Phase 5: Evaluate against gold
# ---------------------------------------------------------------------------

def evaluate(
    all_items: list[dict],
    reps: list[dict],
    selections: list[tuple[int, int, int] | None],
) -> tuple[list[dict], dict]:
    """Evaluate LLM predictions against gold concept_ids.

    For each annotation instance, looks up the prediction for its
    (span, section_header) pair and compares with gold_concept_id.
    Instances whose (span, section) pair was excluded by --limit will have
    pred_concept_id=None and count as "no prediction" (not wrong).
    """
    pred_lookup: dict[tuple[str, str], int | None] = {
        (rep["span"], rep["section_header"]): (sel[0] if sel is not None else None)
        for rep, sel in zip(reps, selections)
    }

    per_ann: list[dict] = []
    for item in all_items:
        key = (item["span"], item["section_header"])
        if key not in pred_lookup:
            # (span, section) was cut by --limit — skip from evaluation
            continue
        pred_cid = pred_lookup[key]
        gold_cid = item["gold_concept_id"]
        correct = pred_cid is not None and pred_cid == gold_cid
        per_ann.append({
            "annotation_id": item["annotation_id"],
            "note_id": item["note_id"],
            "span": item["span"],
            "section_header": item["section_header"],
            "gold_concept_id": gold_cid,
            "pred_concept_id": pred_cid,
            "correct": correct,
        })

    n = len(per_ann)
    n_correct = sum(1 for r in per_ann if r["correct"])
    n_no_pred = sum(1 for r in per_ann if r["pred_concept_id"] is None)
    n_wrong = n - n_correct - n_no_pred

    # Per-section breakdown
    section_stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in per_ann:
        s = section_stats[r["section_header"]]
        s["n"] += 1
        if r["correct"]:
            s["correct"] += 1

    # Per-span stats (ordered by frequency)
    span_stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in per_ann:
        s = span_stats[r["span"]]
        s["n"] += 1
        if r["correct"]:
            s["correct"] += 1

    summary = {
        "n_total": n,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_no_pred": n_no_pred,
        "accuracy": round(n_correct / n, 4) if n > 0 else 0.0,
        "coverage": round((n - n_no_pred) / n, 4) if n > 0 else 0.0,
        "section_breakdown": {
            sec: {"n": s["n"], "acc": round(s["correct"] / s["n"], 3)}
            for sec, s in sorted(section_stats.items())
        },
        # Ordered by frequency (most common span first)
        "span_stats": {
            span: {"n": s["n"], "correct": s["correct"],
                   "acc": round(s["correct"] / s["n"], 3)}
            for span, s in sorted(span_stats.items(),
                                   key=lambda kv: kv[1]["n"], reverse=True)
        },
    }
    return per_ann, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--split-dir", type=Path,
        default=REPO_ROOT / "data" / "old-challenge-split",
        help="Directory containing {train,test}_{annotations,notes}.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "outputs" / "abbreviation_dictionary",
    )
    parser.add_argument(
        "--split", choices=["train", "test", "both"], default="train",
        help="Which data split to extract abbreviations from",
    )
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000",
        help="Base URL of the running vLLM server",
    )
    parser.add_argument(
        "--model", default="/workspaces/snomed-ct-entity-linking/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
        help="Model name served by vLLM",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=0,
        help="Max concurrent LLM requests (0 = unlimited, let vLLM batch)",
    )
    parser.add_argument(
        "--reasoning-effort", default="none",
        choices=["none", "low", "medium", "high"],
    )
    parser.add_argument(
        "--context-before", type=int, default=300,
        help="Characters of context before abbreviation span",
    )
    parser.add_argument(
        "--context-after", type=int, default=100,
        help="Characters of context after abbreviation span",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only the top-N (span, section) pairs by frequency "
             "(0 = all). Useful for quick testing.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading data from {args.split_dir} (split={args.split}) ...")

    splits_to_load = ["train", "test"] if args.split == "both" else [args.split]
    ann_frames, note_frames = [], []
    for s in splits_to_load:
        ann_frames.append(pd.read_csv(args.split_dir / f"{s}_annotations.csv"))
        note_frames.append(pd.read_csv(args.split_dir / f"{s}_notes.csv"))
    ann_df = pd.concat(ann_frames, ignore_index=True)
    notes_df = pd.concat(note_frames, ignore_index=True)

    for col in ("start", "end", "concept_id", "annotation_id"):
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce").fillna(0).astype(int)
    print(f"  {len(ann_df):,} annotations, {len(notes_df):,} notes")

    # ------------------------------------------------------------------
    # Phase 1: Extract abbreviation candidates
    # ------------------------------------------------------------------
    print("\n[Phase 1] Extracting abbreviation candidates ...")
    all_items = extract_abbreviation_items(
        ann_df, notes_df, args.context_before, args.context_after,
    )
    n_unique_spans = len({x["span"] for x in all_items})
    n_unique_pairs = len({(x["span"], x["section_header"]) for x in all_items})
    print(f"  {len(all_items):,} abbreviation annotation instances")
    print(f"  {n_unique_spans:,} unique spans")
    print(f"  {n_unique_pairs:,} unique (span, section) pairs")

    # ------------------------------------------------------------------
    # Phase 2: Deduplicate
    # ------------------------------------------------------------------
    print(f"\n[Phase 2] Building representative set ...")
    reps = build_representative_items(all_items, limit=args.limit)
    suffix = f" (limited from {n_unique_pairs:,})" if args.limit > 0 else ""
    print(f"  {len(reps):,} representative items{suffix}")

    # ------------------------------------------------------------------
    # Phase 3: LLM pipeline
    # ------------------------------------------------------------------
    print(f"\n[Phase 3] Running vLLM pipeline ...")
    print(f"  Model:   {args.model}")
    print(f"  URL:     {args.vllm_url}")
    print(f"  Concurrency: {'unlimited' if args.max_concurrent <= 0 else args.max_concurrent}")
    search_terms_list, candidates_list, selections, timings = run_abbreviation_pipeline(
        reps,
        vllm_url=args.vllm_url,
        model=args.model,
        max_concurrent=args.max_concurrent,
        reasoning_effort=args.reasoning_effort,
    )
    n_successful = sum(1 for s in selections if s is not None)
    print(f"  Successful predictions: {n_successful}/{len(reps)} "
          f"({100 * n_successful / max(len(reps), 1):.1f}%)")

    # ------------------------------------------------------------------
    # Phase 4: Build dictionary
    # ------------------------------------------------------------------
    print("\n[Phase 4] Building abbreviation dictionary ...")
    concept_names = rt.load_concept_names()
    dictionary = build_dictionary(reps, selections, search_terms_list, concept_names)
    total_entries = sum(len(v) for v in dictionary.values())
    n_multi = sum(1 for v in dictionary.values() if len(v) > 1)
    print(f"  {len(dictionary):,} unique abbreviation forms")
    print(f"  {total_entries:,} total (abbrev, concept) entries")
    print(f"  {n_multi:,} abbreviations with multiple meanings (overloaded)")

    # ------------------------------------------------------------------
    # Phase 5: Evaluate
    # ------------------------------------------------------------------
    print("\n[Phase 5] Evaluating against gold annotations ...")
    per_ann, summary = evaluate(all_items, reps, selections)
    print(f"  Instances evaluated: {summary['n_total']:,}")
    print(f"  Correct:             {summary['n_correct']:,} ({100 * summary['accuracy']:.1f}%)")
    print(f"  Wrong:               {summary['n_wrong']:,}")
    print(f"  No prediction:       {summary['n_no_pred']:,}")
    print(f"  Coverage:            {100 * summary['coverage']:.1f}%")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    dict_path = args.output_dir / "abbreviation_dictionary_llm.json"
    dict_path.write_text(json.dumps(dictionary, indent=2))
    print(f"\nDictionary        → {dict_path}")

    eval_path = args.output_dir / "abbrev_eval_results.json"
    eval_path.write_text(json.dumps({
        "model": args.model,
        "split": args.split,
        "context_before": args.context_before,
        "context_after": args.context_after,
        "limit": args.limit,
        "summary": summary,
        "per_representative": [
            {
                "span": r["span"],
                "section_header": r["section_header"],
                "instance_count": r["instance_count"],
                "gold_concept_ids": r["gold_concept_ids_all"],
                "search_terms": search_terms_list[i],
                "n_candidates": len(candidates_list[i]),
                "prediction": list(selections[i]) if selections[i] is not None else None,
                "timing_s": timings[i],
            }
            for i, r in enumerate(reps)
        ],
        "per_annotation": per_ann,
    }, indent=2))
    print(f"Eval results      → {eval_path}")

    # Human-readable summary
    summary_lines = [
        "Abbreviation Dictionary Evaluation",
        "=" * 50,
        f"Model:            {args.model}",
        f"Split:            {args.split}",
        f"Context window:   {args.context_before}B / {args.context_after}A chars",
        f"Limit:            {args.limit if args.limit > 0 else 'all'}",
        "",
        f"Instances:        {summary['n_total']:,}",
        f"Unique spans:     {n_unique_spans:,}",
        f"(span, section) pairs processed: {len(reps):,}",
        "",
        f"Accuracy:         {100 * summary['accuracy']:.1f}%",
        f"Coverage:         {100 * summary['coverage']:.1f}%",
        f"Correct:          {summary['n_correct']:,}",
        f"Wrong:            {summary['n_wrong']:,}",
        f"No prediction:    {summary['n_no_pred']:,}",
        "",
        "Dictionary stats:",
        f"  Unique abbrevs:     {len(dictionary):,}",
        f"  Total entries:      {total_entries:,}",
        f"  Overloaded abbrevs: {n_multi:,}",
        "",
        "Section breakdown:",
    ]
    for sec, stats in sorted(summary["section_breakdown"].items()):
        summary_lines.append(
            f"  {sec:42s}  n={stats['n']:4d}  acc={100 * stats['acc']:.1f}%"
        )
    summary_lines += ["", "Top 40 abbreviations by frequency:"]
    for span, stats in list(summary["span_stats"].items())[:40]:
        summary_lines.append(
            f"  {span:14s}  n={stats['n']:4d}  correct={stats['correct']:4d}"
            f"  acc={100 * stats['acc']:.1f}%"
        )

    summary_text = "\n".join(summary_lines)
    summary_path = args.output_dir / "abbrev_eval_summary.txt"
    summary_path.write_text(summary_text)
    print(f"Summary           → {summary_path}")

    print()
    print(summary_text)


if __name__ == "__main__":
    main()
