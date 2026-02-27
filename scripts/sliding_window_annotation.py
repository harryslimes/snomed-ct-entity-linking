#!/usr/bin/env python3
"""Sliding window span annotation experiment.

Generates candidate spans from word-level n-grams across a clinical note,
runs them through the search+retrieve pipeline with rules, and compares
against gold annotations to identify false positives.

Usage:
  python3 scripts/sliding_window_annotation.py --note-idx 0
  python3 scripts/sliding_window_annotation.py --note-idx 0 --max-ngram 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))
sys.path.insert(0, str(REPO_ROOT))

import rulebook.rule_testing as rt  # noqa: E402
from rulebook.general_rule_loop import (  # noqa: E402
    GENERAL_SEARCH_SYSTEM,
    BASE_SEARCH_RULES,
    load_rules,
    rule_applies,
    _format_rules_block,
)

# Monkey-patch parse_search_terms to allow empty lists (model says "skip")
_orig_parse_search_terms = rt.parse_search_terms

def _patched_parse_search_terms(text: str, fallback_span: str) -> list[str]:
    try:
        match = re.search(r"\{[^{}]*\"search_terms\"[^{}]*\}", text)
        if match:
            result = json.loads(match.group())
            terms = result.get("search_terms", None)
            if terms is not None:  # allow empty list [] as intentional skip
                return terms[:3]
    except (json.JSONDecodeError, KeyError):
        pass
    return [fallback_span.strip()]

rt.parse_search_terms = _patched_parse_search_terms

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
DEFAULT_RULES_FILE = REPO_ROOT / "rulebook" / "general_rules.json"
DEFAULT_MODEL = "/workspaces/snomed-ct-entity-linking/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"

# Words that never form useful standalone annotations
STOPWORDS = frozenset(
    "a an the is was were are of to in for on at by and or not no with he she "
    "his her that this from had has have been will be but if it its all as "
    "which who after some into mr mrs dr ms did do does did than then their "
    "they them there when what where how also very much more most can could "
    "would should may might shall able about above again any because before "
    "between both during each few further here just only own same so still "
    "such too under until up well while without our your we you i my me him "
    "over out".split()
)

# Section header keywords to skip entirely
SECTION_KEYWORDS = frozenset(
    "name unit admission discharge date birth sex service attending allergies "
    "chief complaint major surgical invasive procedure history present illness "
    "past medical social family physical exam pertinent results brief hospital "
    "course medications discharge disposition condition followup instructions "
    "warnings facility".split()
)


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split text into tokens preserving character offsets.

    Returns [(token, start, end), ...] where text[start:end] == token.
    """
    tokens = []
    for m in re.finditer(r"\S+", text):
        tokens.append((m.group(), m.start(), m.end()))
    return tokens


def generate_candidate_spans(
    text: str,
    min_ngram: int = 1,
    max_ngram: int = 4,
) -> list[dict]:
    """Generate candidate spans from word n-grams with character offsets."""
    tokens = tokenize_with_offsets(text)
    seen: set[tuple[int, int]] = set()
    candidates: list[dict] = []

    for n in range(min_ngram, max_ngram + 1):
        for i in range(len(tokens) - n + 1):
            start = tokens[i][1]
            end = tokens[i + n - 1][2]

            if (start, end) in seen:
                continue
            seen.add((start, end))

            span_text = text[start:end]

            # --- Filtering ---
            # Skip if no alphabetic chars
            if not any(c.isalpha() for c in span_text):
                continue

            # Skip redacted fields
            if "___" in span_text:
                continue

            words = [t[0] for t in tokens[i : i + n]]
            words_lower = [w.lower().rstrip(".,;:") for w in words]

            # Skip single-word stopwords or very short
            if n == 1:
                w = words_lower[0]
                if w in STOPWORDS or len(w) <= 2:
                    continue
                # Skip pure numbers / dates
                if re.match(r"^[\d./%:+-]+$", words[0]):
                    continue

            # Skip n-grams starting with a stopword
            if n > 1 and words_lower[0] in STOPWORDS:
                continue

            # Skip lines that are just section headers
            if all(w in SECTION_KEYWORDS for w in words_lower):
                continue

            candidates.append({
                "start": start,
                "end": end,
                "span": span_text,
            })

    return candidates


# ---------------------------------------------------------------------------
# Code-level negative pre-filters (implements negative_rules.json as code)
# ---------------------------------------------------------------------------

# N003 + N010 + N012: Generic narrative words, temporal connectors, demographics
# These single words should almost never be standalone annotations
NEGATIVE_SINGLE_WORDS = frozenset(
    # N003: generic narrative words
    "man woman admitted presented noted seen found developed treated "
    "discussed followed received underwent patient "
    # N010: temporal/narrative connectors
    "resulting initially scheduled prior brief elective "
    # N012: demographics
    "male female".split()
)

# N001: Section header / structural label words (standalone)
SECTION_LABEL_WORDS = frozenset(
    "admission discharge disposition diagnosis condition followup "
    "instructions warnings facility".split()
)

# N002: Field-label words that precede colons
FIELD_LABEL_WORDS = frozenset(
    "birth sex allergies procedure illness name unit date service "
    "attending".split()
)

# N005: Isolated clinical modifiers in administrative phrases
ADMIN_MODIFIERS = frozenset(
    "surgical medical invasive".split()
)

# N009: Service/department names when used as admin assignments
SERVICE_NAMES = frozenset(
    "surgery medicine cardiology neurology oncology orthopedics "
    "urology pediatrics psychiatry radiology pathology".split()
)

# Medication list section names
MED_LIST_SECTIONS = frozenset(
    "medications on admission discharge medications".split("/")
)


def _get_line(text: str, pos: int) -> str:
    """Get the full line containing position `pos`."""
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end]


def _is_in_med_list_section(section: str) -> bool:
    """Check if section header indicates a medication list."""
    sl = section.lower()
    return any(kw in sl for kw in ("medication", "med"))


def apply_negative_prefilters(
    candidates: list[dict],
    text: str,
) -> tuple[list[dict], list[dict]]:
    """Apply code-level negative rules to filter obvious non-annotatable spans.

    Returns (passed, filtered) where each filtered item has a 'filter_rule' key.
    """
    passed: list[dict] = []
    filtered: list[dict] = []

    for cand in candidates:
        span = cand["span"]
        start, end = cand["start"], cand["end"]
        span_lower = span.lower().rstrip(".,;:")
        words = span_lower.split()
        n_words = len(words)
        line = _get_line(text, start)
        line_stripped = line.strip()
        section = find_section_for_pos(start, text)
        # Character right after the span
        char_after = text[end] if end < len(text) else ""

        rule_id = None

        # --- N001: Section headers and structural labels ---
        # Single word that matches section label vocabulary AND appears at
        # line start or is the whole line (after stripping punctuation)
        if n_words == 1 and span_lower in SECTION_LABEL_WORDS:
            # Check if this is a structural position (line start, header-like)
            if line_stripped.rstrip(":").lower() == span_lower or char_after in (":", "\n", ""):
                rule_id = "N001"

        # --- N002: Field labels followed by colons ---
        if not rule_id and n_words == 1 and span_lower in FIELD_LABEL_WORDS:
            if char_after == ":" or span.endswith(":"):
                rule_id = "N002"

        # --- N003 + N010 + N012: Generic narrative words, connectors, demographics ---
        if not rule_id and n_words == 1 and span_lower in NEGATIVE_SINGLE_WORDS:
            rule_id = "N003"

        # --- N004: Compound section header phrases ending with colon ---
        if not rule_id and n_words >= 2:
            if span.rstrip().endswith(":") or char_after == ":":
                # Check if most words are section keywords
                kw_count = sum(1 for w in words if w in SECTION_KEYWORDS or w in SECTION_LABEL_WORDS or w in FIELD_LABEL_WORDS)
                if kw_count >= len(words) * 0.6:
                    rule_id = "N004"

        # --- N005: Isolated clinical modifiers in admin context ---
        if not rule_id and n_words == 1 and span_lower in ADMIN_MODIFIERS:
            # Check if the line looks administrative (contains section keywords)
            line_lower = line.lower()
            if any(kw in line_lower for kw in ("service", "history", "procedure", "or invasive")):
                rule_id = "N005"

        # --- N006: discharge/discharged as admin transition ---
        if not rule_id and span_lower in ("discharge", "discharged", "discharge,"):
            # Allow "discharge plan", "discharge planning" — check if followed by clinical word
            after_text = text[end:end + 20].strip().lower()
            if not after_text.startswith(("plan", "drainage", "summary")):
                rule_id = "N006"

        # --- N007: Isolated drug names in medication list sections ---
        # Only filter single-word drug names in med list sections
        # (multi-word drug names in narrative are allowed)
        if not rule_id and n_words == 1 and _is_in_med_list_section(section):
            # Check if the line looks like a med list entry (starts with number or bullet)
            if re.match(r"^\s*(\d+\.|[-•*]|\d+\))", line_stripped):
                rule_id = "N007"

        # --- N008: follow-up as scheduling term ---
        if not rule_id and span_lower in ("follow-up", "follow", "followup"):
            after_text = text[end:end + 30].strip().lower()
            # "follow-up with your surgeon" is clinical → allow it
            # "follow-up instructions/appointment" is admin → filter
            if any(kw in after_text[:20] for kw in ("instruction", "appointment", ":")):
                rule_id = "N008"

        # --- N009: Service/department names ---
        if not rule_id and n_words == 1 and span_lower in SERVICE_NAMES:
            # Check if preceded by "Service:" pattern
            before_text = text[max(0, start - 15):start].strip().lower()
            if "service" in before_text or line_stripped.lower().startswith("service"):
                rule_id = "N009"

        # --- N011: Sub-spans of multi-word terms ---
        # (handled in a second pass below — needs all candidates)

        if rule_id:
            cand["filter_rule"] = rule_id
            filtered.append(cand)
        else:
            passed.append(cand)

    # N011 (sub-span filtering) is NOT implemented as a code-level filter.
    # In this dataset, sub-spans are often independently annotated gold concepts
    # (e.g., "pancreatitis" within "biliary pancreatitis"). This requires LLM judgment.

    return passed, filtered


def find_section_for_pos(pos: int, text: str) -> str:
    """Simple section detection: find the nearest preceding section header."""
    # Common discharge summary section patterns
    section_pat = re.compile(
        r"^([A-Z][A-Za-z /]+(?:History|Exam|Results|Course|Medications|"
        r"Allergies|Complaint|Procedure|Disposition|Condition|Instructions|"
        r"Warnings|Facility|Service|Attending|Follow-up)):?\s*$",
        re.MULTILINE,
    )
    best_header = "unknown"
    for m in section_pat.finditer(text):
        if m.start() <= pos:
            best_header = m.group(1).strip().rstrip(":")
        else:
            break
    return best_header


def overlap_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Character-level IoU between two spans."""
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--note-idx", type=int, default=0,
                        help="Index of the note in the train split")
    parser.add_argument("--min-ngram", type=int, default=1)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--context-before", type=int, default=100)
    parser.add_argument("--context-after", type=int, default=50)
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--rules-file", type=Path, default=DEFAULT_RULES_FILE)
    parser.add_argument("--negative-rules", type=Path, default=None,
                        help="Path to negative_rules.json (span filtering rules)")
    parser.add_argument("--prefilter", action="store_true",
                        help="Apply code-level negative pre-filters before LLM")
    parser.add_argument("--prefilter-only", action="store_true",
                        help="Only run pre-filters, skip LLM pipeline")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum top-candidate score to report as a detection")
    parser.add_argument("--index-server", default="http://127.0.0.1:8421")
    args = parser.parse_args()

    rt._index_server_url = args.index_server

    # Load data
    notes_df = pd.read_csv(args.split_dir / "train_notes.csv")
    ann_df = pd.read_csv(args.split_dir / "train_annotations.csv")
    for col in ("start", "end", "concept_id", "annotation_id"):
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce").fillna(0).astype(int)

    note_row = notes_df.iloc[args.note_idx]
    note_id = note_row["note_id"]
    note_text = note_row["text"]
    gold_anns = ann_df[ann_df.note_id == note_id].to_dict("records")
    print(f"Note: {note_id}  ({len(note_text)} chars, {len(gold_anns)} gold annotations)")

    # Load rules
    rules = []
    if str(args.rules_file).lower() != "none" and args.rules_file.exists():
        rules = load_rules(args.rules_file)

    # Load negative rules (span filtering)
    negative_rules = []
    negative_rules_block = ""
    if args.negative_rules and args.negative_rules.exists():
        neg_data = json.loads(args.negative_rules.read_text())
        negative_rules = neg_data.get("rules", [])
        print(f"Loaded {len(negative_rules)} negative rules from {args.negative_rules}")
        # Format into prompt block
        neg_lines = [
            "\n=== SPAN FILTERING RULES (when NOT to annotate) ===",
            "If the highlighted span matches any of these patterns, respond with "
            '{"search_terms": []} to SKIP annotation.',
            "",
        ]
        for nr in negative_rules:
            p = nr.get("priority", 2)
            examples = ", ".join(nr.get("examples", [])[:3])
            ex_str = f"  (e.g. {examples})" if examples else ""
            neg_lines.append(f"[P{p}] {nr['id']}: {nr['rule']}{ex_str}")
        negative_rules_block = "\n".join(neg_lines)

    # Load concept names
    concept_names = rt.load_concept_names()

    # Generate candidate spans
    print(f"\nGenerating candidate spans (n-grams {args.min_ngram}-{args.max_ngram}) ...")
    candidates = generate_candidate_spans(
        note_text, args.min_ngram, args.max_ngram,
    )
    print(f"  {len(candidates)} candidate spans generated")

    # Label each candidate: does it overlap a gold annotation?
    for cand in candidates:
        cand["gold_match"] = None
        cand["gold_concept_id"] = None
        best_iou = 0.0
        for ga in gold_anns:
            iou = overlap_iou(cand["start"], cand["end"], int(ga["start"]), int(ga["end"]))
            if iou > best_iou:
                best_iou = iou
                if iou >= 0.5:
                    cand["gold_match"] = ga
                    cand["gold_concept_id"] = int(ga["concept_id"])

    n_gold_overlap = sum(1 for c in candidates if c["gold_match"])
    print(f"  {n_gold_overlap} candidates overlap gold annotations (IoU >= 0.5)")
    print(f"  {len(candidates) - n_gold_overlap} candidates are non-gold (potential FPs)")

    # --- Code-level pre-filtering ---
    prefiltered: list[dict] = []
    if args.prefilter or args.prefilter_only:
        print(f"\nApplying code-level negative pre-filters ...")
        candidates, prefiltered = apply_negative_prefilters(candidates, note_text)

        n_filt_gold = sum(1 for c in prefiltered if c.get("gold_match"))
        n_filt_nongold = len(prefiltered) - n_filt_gold
        print(f"  Filtered out: {len(prefiltered)} spans")
        print(f"    Non-gold (correct filter): {n_filt_nongold}")
        print(f"    Gold (WRONG filter):       {n_filt_gold}")

        # Breakdown by rule
        from collections import Counter
        rule_counts = Counter(c.get("filter_rule") for c in prefiltered)
        for rule_id, count in sorted(rule_counts.items()):
            gold_hit = sum(1 for c in prefiltered
                          if c.get("filter_rule") == rule_id and c.get("gold_match"))
            print(f"    {rule_id}: {count:4d} filtered ({gold_hit} gold)")

        if n_filt_gold > 0:
            print(f"\n  WARNING: Pre-filters incorrectly removed {n_filt_gold} gold spans:")
            for c in prefiltered:
                if c.get("gold_match"):
                    gm = c["gold_match"]
                    print(f"    [{c['start']:4d}:{c['end']:4d}] {c['span']!r:30s}  "
                          f"rule={c.get('filter_rule')}  "
                          f"gold_concept={gm.get('concept_id')}")

        # Recount after filtering
        n_gold_overlap = sum(1 for c in candidates if c["gold_match"])
        print(f"\n  After filtering: {len(candidates)} candidates remain")
        print(f"    Gold-overlapping: {n_gold_overlap}")
        print(f"    Non-gold:         {len(candidates) - n_gold_overlap}")

        if args.prefilter_only:
            print(f"\n  --prefilter-only: skipping LLM pipeline")
            return

    # Build annotation dicts for the pipeline
    print(f"\nBuilding pipeline inputs ...")
    annotations = []
    for cand in candidates:
        before, _, after = rt.get_context(
            note_text, cand["start"], cand["end"],
            args.context_before, args.context_after,
        )
        section = find_section_for_pos(cand["start"], note_text)
        cand["section"] = section

        # Inject rules (use gold_concept_id if available, otherwise no concept scope)
        gold_cids = [cand["gold_concept_id"]] if cand["gold_concept_id"] else []
        applicable = [r for r in rules if rule_applies(r, section, gold_cids)] if gold_cids else [
            r for r in rules
            if not r.get("applies_to", {}).get("ancestor_concept_ids")
            and (not r.get("applies_to", {}).get("sections")
                 or any(s.lower() in section.lower()
                        for s in r["applies_to"]["sections"]))
        ]
        rules_extra = _format_rules_block(applicable)
        search_rules_text = BASE_SEARCH_RULES + (
            "\n\n" + rules_extra if rules_extra else ""
        )
        if negative_rules_block:
            search_rules_text += "\n" + negative_rules_block

        annotations.append({
            "before": before,
            "span": cand["span"],
            "after": after,
            "section_header": section,
            "search_rules_text": search_rules_text,
            "select_rules_text": "",
            "gold_start": cand["start"],
            "gold_end": cand["end"],
        })

    print(f"  {len(annotations)} annotation dicts ready")

    # Run pipeline (search + retrieve, no select)
    print(f"\nRunning search+retrieve pipeline on {len(annotations)} candidates ...")

    # If negative rules present, modify system prompt to allow skipping
    if negative_rules:
        search_system = GENERAL_SEARCH_SYSTEM.replace(
            "Generate 1-3 search terms",
            "Generate 0-3 search terms (0 means SKIP — the span should not be annotated)",
        ) + (
            "\n\nIMPORTANT: If the span should NOT be annotated (per the filtering rules), "
            'respond with {"search_terms": []}.'
        )
    else:
        search_system = GENERAL_SEARCH_SYSTEM

    orig_search = rt.SEARCH_SYSTEM
    t0 = time.time()
    try:
        rt.SEARCH_SYSTEM = search_system
        search_terms_list, candidates_list, _, timings = rt.vllm_pipeline(
            annotations,
            base_url=args.vllm_url,
            model=args.model,
            max_concurrent=args.max_concurrent,
            reasoning_effort=args.reasoning_effort,
            search_only=True,
        )
    finally:
        rt.SEARCH_SYSTEM = orig_search
    elapsed = time.time() - t0
    print(f"  Pipeline done in {elapsed:.1f}s")

    # Debug: show first 20 search terms to check for empty lists
    empty_count = sum(1 for t in search_terms_list if not t)
    print(f"\n  DEBUG: {empty_count}/{len(search_terms_list)} empty search_terms (LLM skips)")
    for i, (cand, terms) in enumerate(zip(candidates, search_terms_list)):
        if i < 10 or not terms:
            print(f"    [{i}] span={cand['span']!r:30s} terms={terms!r}")
        if i >= 10 and terms:
            continue

    # Score results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    detections: list[dict] = []
    for cand, terms, cands in zip(candidates, search_terms_list, candidates_list):
        top_score = cands[0]["score"] if cands else 0.0
        top_cid = cands[0]["concept_id"] if cands else None
        top_name = cands[0]["concept_name"] if cands else None

        is_gold = cand["gold_match"] is not None
        gold_cid = cand["gold_concept_id"]
        gold_retrieved = gold_cid is not None and any(
            c["concept_id"] == gold_cid for c in cands
        )

        detections.append({
            "start": cand["start"],
            "end": cand["end"],
            "span": cand["span"],
            "section": cand["section"],
            "search_terms": terms,
            "top_score": top_score,
            "top_concept_id": top_cid,
            "top_concept_name": top_name,
            "n_candidates": len(cands),
            "is_gold": is_gold,
            "gold_concept_id": gold_cid,
            "gold_retrieved": gold_retrieved,
        })

    # Sort by score descending
    detections.sort(key=lambda d: d["top_score"], reverse=True)

    # Count skipped (empty search terms = LLM said "don't annotate")
    skipped = [d for d in detections if not d["search_terms"]]
    annotated = [d for d in detections if d["search_terms"]]

    n_gold = sum(1 for d in detections if d["is_gold"])
    n_nongold = len(detections) - n_gold

    # Skipped breakdown
    skipped_gold = [d for d in skipped if d["is_gold"]]
    skipped_nongold = [d for d in skipped if not d["is_gold"]]

    # Report: False positives (non-gold that were NOT skipped and have candidates)
    fps = [d for d in annotated if not d["is_gold"] and d["n_candidates"] > 0]
    tps = [d for d in annotated if d["is_gold"] and d["gold_retrieved"]]
    gold_missed = [d for d in detections if d["is_gold"] and not d["gold_retrieved"]]

    print(f"\n  Total candidates:     {len(detections)}")
    print(f"  Gold-overlapping:     {n_gold}")
    print(f"  Non-gold:             {n_nongold}")
    print(f"\n  Skipped by LLM:       {len(skipped)} ({100*len(skipped)/len(detections):.1f}%)")
    print(f"    Correctly skipped:  {len(skipped_nongold)} non-gold (true negatives)")
    print(f"    Wrongly skipped:    {len(skipped_gold)} gold (false negatives from filtering)")
    print(f"\n  Annotated by LLM:     {len(annotated)}")
    print(f"  Gold retrieved (TP):  {len(tps)}")
    print(f"  Gold missed (FN):     {len(gold_missed)}")
    print(f"  Non-gold detections:  {len(fps)} (false positives)")

    if skipped_gold:
        print(f"\n  WARNING: Negative rules incorrectly suppressed {len(skipped_gold)} gold spans:")
        for d in skipped_gold[:15]:
            print(f"    [{d['start']:4d}:{d['end']:4d}] {d['span']!r:30s}  section={d['section']}")

    # Show score distribution
    all_scores = [d["top_score"] for d in detections if d["n_candidates"] > 0]
    if all_scores:
        import statistics
        print(f"\n  Score distribution (all candidates with results):")
        print(f"    min={min(all_scores):.3f}  median={statistics.median(all_scores):.3f}  "
              f"max={max(all_scores):.3f}  mean={statistics.mean(all_scores):.3f}")

    # Top false positives
    print(f"\n{'─'*70}")
    print(f"TOP FALSE POSITIVES (non-gold spans with highest retrieval score)")
    print(f"{'─'*70}")
    for d in fps[:40]:
        gold_tag = ""
        print(
            f"  [{d['start']:4d}:{d['end']:4d}] "
            f"span={d['span']!r:35s}  score={d['top_score']:.3f}  "
            f"section={d['section']}"
        )
        print(
            f"           → {d['top_concept_name']} (cid={d['top_concept_id']})"
        )
        print(
            f"           terms={d['search_terms']}"
        )

    # Gold annotation recall
    print(f"\n{'─'*70}")
    print(f"GOLD ANNOTATIONS: Retrieved vs Missed")
    print(f"{'─'*70}")

    # Group gold annotations and check coverage
    gold_by_span: dict[tuple[int, int], dict] = {}
    for ga in gold_anns:
        key = (int(ga["start"]), int(ga["end"]))
        gold_by_span[key] = ga

    covered = set()
    for d in detections:
        if d["is_gold"] and d["gold_retrieved"]:
            gm = [g for g in gold_anns
                  if overlap_iou(d["start"], d["end"],
                                 int(g["start"]), int(g["end"])) >= 0.5]
            for g in gm:
                covered.add((int(g["start"]), int(g["end"])))

    print(f"\n  Gold annotations:     {len(gold_anns)}")
    print(f"  Covered by candidates: {len(covered)}/{len(gold_anns)}")
    print(f"  Not covered:          {len(gold_anns) - len(covered)}")

    if len(gold_anns) - len(covered) > 0:
        print(f"\n  Uncovered gold annotations:")
        for ga in gold_anns:
            key = (int(ga["start"]), int(ga["end"]))
            if key not in covered:
                cname = concept_names.get(int(ga["concept_id"]), ("?", "?"))
                print(
                    f"    [{ga['start']:4d}:{ga['end']:4d}] "
                    f"span={ga['span']!r:35s}  concept={cname[0]}"
                )


if __name__ == "__main__":
    main()
