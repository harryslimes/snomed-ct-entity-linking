#!/usr/bin/env python3
"""Generate negative annotation rules (what NOT to annotate) from a clinical note.

Creates an inline-annotated note showing TRUE (gold) and FALSE (pipeline FPs)
annotations, sends it to Opus for negative rule generation, then outputs
the rules.

Usage:
  python3 scripts/generate_negative_rules.py --note-idx 0
"""
from __future__ import annotations

import argparse
import json
import pickle
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

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
TERMINOLOGY_CSV = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"
MRCONSO_PATH = REPO_ROOT / "snomed_index" / "mrconso_terms.pkl"


def build_term_to_concepts(
    terminology_csv: Path,
    mrconso_path: Path,
) -> dict[str, set[int]]:
    """Build lowercase term -> set of concept IDs lookup."""
    term_map: dict[str, set[int]] = defaultdict(set)
    ft = pd.read_csv(terminology_csv)
    for _, row in ft.iterrows():
        cid = int(row["concept_id"])
        name = str(row["concept_name"]).strip().lower()
        if name:
            term_map[name].add(cid)
    if mrconso_path.exists():
        with open(mrconso_path, "rb") as f:
            mrconso = pickle.load(f)
        for cid, terms in mrconso.items():
            cid = int(cid)
            for t in terms:
                term_map[t.strip().lower()].add(cid)
    return dict(term_map)


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    tokens = []
    for m in re.finditer(r"\S+", text):
        tokens.append((m.group(), m.start(), m.end()))
    return tokens


STOPWORDS = frozenset(
    "a an the is was were are of to in for on at by and or not no with he she "
    "his her that this from had has have been will be but if it its all as "
    "which who after some into mr mrs dr ms did do does than then their "
    "they them there when what where how also very much more most can could "
    "would should may might shall able about above again any because before "
    "between both during each few further here just only own same so still "
    "such too under until up well while without our your we you i my me him "
    "over out".split()
)


def generate_candidate_spans(
    text: str, min_ngram: int = 1, max_ngram: int = 4,
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
            if not any(c.isalpha() for c in span_text):
                continue
            if "___" in span_text:
                continue
            words = [t[0] for t in tokens[i : i + n]]
            words_lower = [w.lower().rstrip(".,;:") for w in words]
            if n == 1:
                w = words_lower[0]
                if w in STOPWORDS or len(w) <= 2:
                    continue
                if re.match(r"^[\d./%:+-]+$", words[0]):
                    continue
            if n > 1 and words_lower[0] in STOPWORDS:
                continue
            candidates.append({"start": start, "end": end, "span": span_text})
    return candidates


def overlap_iou(a_start, a_end, b_start, b_end):
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union > 0 else 0.0


def find_section_for_pos(pos: int, text: str) -> str:
    section_pat = re.compile(
        r"^([A-Z][A-Za-z /]+(?:History|Exam|Results|Course|Medications|"
        r"Allergies|Complaint|Procedure|Disposition|Condition|Instructions|"
        r"Warnings|Facility|Service|Attending|Follow-up)):?\s*$",
        re.MULTILINE,
    )
    best = "unknown"
    for m in section_pat.finditer(text):
        if m.start() <= pos:
            best = m.group(1).strip().rstrip(":")
        else:
            break
    return best


def build_inline_annotated_note(
    note_text: str,
    gold_anns: list[dict],
    false_positives: list[dict],
    concept_names: dict[int, tuple[str, str]],
) -> str:
    """Build an inline annotated note showing TRUE and FALSE annotations.

    TRUE:  [span | ANNOTATE: Concept Name (hierarchy)]
    FALSE: [span | DO NOT ANNOTATE: reason]
    """
    # Merge all annotations with labels, sort by position
    markers: list[tuple[int, int, str]] = []

    for ga in gold_anns:
        s, e = int(ga["start"]), int(ga["end"])
        cid = int(ga["concept_id"])
        name, hier = concept_names.get(cid, ("Unknown", "unknown"))
        markers.append((s, e, f"ANNOTATE: {name} ({hier})"))

    for fp in false_positives:
        s, e = fp["start"], fp["end"]
        reason = fp.get("reason", "not a clinical concept annotation")
        markers.append((s, e, f"DO NOT ANNOTATE: {reason}"))

    # Sort by start position, then by length (longer first for nesting)
    markers.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Remove overlapping markers (keep the first one at each position)
    used_ranges: list[tuple[int, int]] = []
    clean_markers: list[tuple[int, int, str]] = []
    for s, e, label in markers:
        overlaps = any(
            not (e <= us or s >= ue) for us, ue in used_ranges
        )
        if not overlaps:
            clean_markers.append((s, e, label))
            used_ranges.append((s, e))

    # Build the annotated text
    clean_markers.sort(key=lambda x: x[0])
    result = []
    pos = 0
    for s, e, label in clean_markers:
        if s > pos:
            result.append(note_text[pos:s])
        span_text = note_text[s:e]
        result.append(f"[{span_text} | {label}]")
        pos = e
    if pos < len(note_text):
        result.append(note_text[pos:])

    return "".join(result)


def identify_false_positives(
    note_text: str,
    gold_anns: list[dict],
    term_map: dict[str, set[int]],
    max_ngram: int = 4,
) -> list[dict]:
    """Identify false positive spans: n-grams that match SNOMED dictionary
    but don't overlap any gold annotation. Plus common FP patterns."""

    candidates = generate_candidate_spans(note_text, 1, max_ngram)
    gold_intervals = [(int(g["start"]), int(g["end"])) for g in gold_anns]

    fps: list[dict] = []
    for cand in candidates:
        s, e = cand["start"], cand["end"]
        span = cand["span"]

        # Skip if overlaps gold
        overlaps_gold = any(
            overlap_iou(s, e, gs, ge) >= 0.3
            for gs, ge in gold_intervals
        )
        if overlaps_gold:
            continue

        span_clean = span.strip().rstrip(".,;:").lower()

        # Check dictionary match
        dict_match = term_map.get(span_clean, set())
        if not dict_match:
            continue

        # Categorize the FP
        section = find_section_for_pos(s, note_text)

        # Is it a generic English word that happens to match SNOMED?
        is_generic = len(span_clean.split()) == 1 and span_clean.isalpha()

        # Get best concept name
        best_cid = next(iter(dict_match))
        fps.append({
            "start": s,
            "end": e,
            "span": span,
            "section": section,
            "dict_concept_ids": list(dict_match)[:3],
            "is_generic_word": is_generic,
            "reason": _classify_fp_reason(span_clean, section, note_text, s, e),
        })

    return fps


def _classify_fp_reason(span: str, section: str, text: str, start: int, end: int) -> str:
    """Classify why a span is a false positive."""
    words = span.split()

    # Single generic word
    if len(words) == 1 and span.isalpha() and len(span) <= 10:
        # Check if it's a word that can be clinical in some contexts
        generic_words = {
            "man", "woman", "male", "female", "patient", "mother", "father",
            "passed", "followed", "discuss", "scheduled", "initially",
            "resulting", "currently", "approach", "prior", "clinic",
            "admitted", "alive", "stable", "regular", "floor", "control",
            "stay", "arrived", "tolerating", "controlled", "problem",
            "received", "used", "well", "noted", "night", "improved",
        }
        if span in generic_words:
            return f"generic English word '{span}' used in narrative context, not a clinical finding"

        structural_words = {
            "risks", "benefits", "outcomes", "discussion", "elective",
            "discharge", "admission",
        }
        if span in structural_words:
            return f"structural/administrative word '{span}', not a clinical observation"

    # Sub-span of a larger concept
    context_before = text[max(0, start - 30):start].strip()
    context_after = text[end:min(len(text), end + 30)].strip()
    if len(words) == 1:
        return f"isolated word; clinical meaning requires surrounding context"

    # Multi-word partial match
    return "partial phrase match; not the intended annotation boundary"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--note-idx", type=int, default=0)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--max-fps", type=int, default=60,
                        help="Max false positives to show in inline note")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file for inline annotated note")
    parser.add_argument("--generate-rules", action="store_true",
                        help="Call Opus to generate negative rules")
    parser.add_argument("--existing-rules", type=Path, default=None,
                        help="Path to existing negative_rules.json to extend")
    parser.add_argument("--note-indices", type=str, default=None,
                        help="Comma-separated note indices for multi-note mode (e.g. '0,1,2')")
    parser.add_argument("--rules-output", type=Path,
                        default=REPO_ROOT / "rulebook" / "negative_rules.json")
    args = parser.parse_args()

    # Load data
    notes_df = pd.read_csv(args.split_dir / "train_notes.csv")
    ann_df = pd.read_csv(args.split_dir / "train_annotations.csv")
    for col in ("start", "end", "concept_id", "annotation_id"):
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce").fillna(0).astype(int)

    # Determine which notes to process
    if args.note_indices:
        note_idxs = [int(x.strip()) for x in args.note_indices.split(",")]
    else:
        note_idxs = [args.note_idx]

    # Load concept names
    concept_names = rt.load_concept_names()

    # Build dictionary for FP detection
    print("Building SNOMED dictionary ...")
    term_map = build_term_to_concepts(TERMINOLOGY_CSV, MRCONSO_PATH)
    print(f"  {len(term_map):,} unique terms")

    # Load existing rules if provided
    existing_rules: list[dict] = []
    if args.existing_rules and args.existing_rules.exists():
        existing_data = json.loads(args.existing_rules.read_text())
        existing_rules = existing_data.get("rules", [])
        print(f"\nLoaded {len(existing_rules)} existing negative rules from {args.existing_rules}")

    # Process each note
    all_annotated_notes: list[str] = []
    for note_idx in note_idxs:
        note_row = notes_df.iloc[note_idx]
        note_id = note_row["note_id"]
        note_text = note_row["text"]
        gold_anns = ann_df[ann_df.note_id == note_id].to_dict("records")
        print(f"\nNote {note_idx}: {note_id}  ({len(note_text)} chars, {len(gold_anns)} gold annotations)")

        # Find false positives
        print("  Identifying false positive spans ...")
        fps = identify_false_positives(note_text, gold_anns, term_map, args.max_ngram)
        print(f"  {len(fps)} false positive spans found")

        # Prioritize FPs: prefer single-word generic words and diverse sections
        fps_generic = [f for f in fps if f["is_generic_word"]]
        fps_multi = [f for f in fps if not f["is_generic_word"]]
        per_note_max = args.max_fps // len(note_idxs)
        selected_fps = fps_generic[:per_note_max // 2] + fps_multi[:per_note_max // 2]
        # Deduplicate by span text
        seen_spans: set[str] = set()
        deduped_fps: list[dict] = []
        for fp in selected_fps:
            sp = fp["span"].strip().lower()
            if sp not in seen_spans:
                seen_spans.add(sp)
                deduped_fps.append(fp)
        selected_fps = deduped_fps[:per_note_max]
        print(f"  Selected {len(selected_fps)} FPs for inline annotation")

        # Build inline annotated note
        annotated = build_inline_annotated_note(
            note_text, gold_anns, selected_fps, concept_names,
        )

        out_path = args.output or REPO_ROOT / "scripts" / f"annotated_note_{note_id}.txt"
        out_path.write_text(annotated)
        print(f"  Saved to {out_path}")

        n_true = annotated.count("| ANNOTATE:")
        n_false = annotated.count("| DO NOT ANNOTATE:")
        print(f"  {n_true} TRUE annotations, {n_false} FALSE annotations")

        all_annotated_notes.append(f"=== NOTE {note_idx}: {note_id} ===\n{annotated}")

    combined_annotated = "\n\n".join(all_annotated_notes)

    if not args.generate_rules:
        print("\nDone. Use --generate-rules to call Opus for rule generation.")
        return

    # Call Opus for negative rule generation
    print(f"\n{'='*60}")
    print("Calling Opus for negative annotation rules ...")
    print(f"{'='*60}")

    # Build existing rules block for Opus context
    existing_rules_block = ""
    if existing_rules:
        er_lines = ["=== EXISTING NEGATIVE RULES (already implemented) ==="]
        for r in existing_rules:
            examples = ", ".join(r.get("examples", [])[:3])
            er_lines.append(f"  {r['id']} [P{r.get('priority', 2)}]: {r['rule']}")
            if examples:
                er_lines.append(f"    examples: {examples}")
        existing_rules_block = "\n".join(er_lines)
        next_id = max(int(r["id"][1:]) for r in existing_rules) + 1
    else:
        next_id = 1

    system_prompt = """\
You are a clinical NLP expert creating SPAN FILTERING RULES for a SNOMED CT \
annotation pipeline.

You will see one or more clinical notes with inline annotations:
  [span | ANNOTATE: Concept Name (hierarchy)] = correctly annotated spans
  [span | DO NOT ANNOTATE: reason] = spans that should NOT be annotated

Your task: Write rules that help a downstream LLM decide whether a candidate \
text span should be annotated or skipped. Focus on NEGATIVE rules — patterns \
that identify spans that should NOT be annotated.

The pipeline currently annotates almost everything, including:
- Generic English words that happen to match SNOMED terms (e.g., "man" → Male)
- Narrative/structural words (e.g., "resulting", "initially", "scheduled")
- Isolated qualifiers without their clinical subject
- Sub-spans of multi-word medical terms
- Administrative/documentation terms

Response format — a single JSON object:

{
  "rules": [
    {
      "id": "N0XX",
      "type": "negative",
      "priority": 2,
      "rule": "Concise rule about what NOT to annotate. <100 words.",
      "examples": ["example span 1", "example span 2"]
    }
  ],
  "observations": "Brief notes about NEW patterns you observed beyond existing rules."
}

Rules guidance:
- Each rule MUST be under 100 words. Be telegraphic.
- Focus on GENERAL patterns, not individual words.
- Rules should be ACTIONABLE: a downstream LLM should be able to apply them \
  to decide "skip this span" or "annotate this span".
- DO NOT repeat or rephrase existing rules. Only generate NEW rules for \
  patterns not yet covered.\
"""

    if existing_rules_block:
        user_prompt = f"""\
Below are clinical notes with inline annotations. Spans marked ANNOTATE are \
correct gold-standard annotations. Spans marked DO NOT ANNOTATE are false \
positives that the pipeline incorrectly tried to annotate.

{existing_rules_block}

The existing rules above are already implemented. Your job: study the annotated \
notes below and generate ADDITIONAL negative rules for FP patterns that the \
existing rules do NOT cover. Start IDs from N{next_id:03d}.

INLINE ANNOTATED NOTES:
{combined_annotated}

Return ONLY the JSON object with 'rules' and 'observations' keys. \
Do NOT include any existing rules — only new ones.\
"""
    else:
        user_prompt = f"""\
Analyse these inline-annotated clinical notes. Spans marked ANNOTATE are correct \
gold-standard annotations. Spans marked DO NOT ANNOTATE are false positives \
that the pipeline incorrectly tried to annotate.

Study the patterns of what IS and IS NOT annotated, then generate negative \
rules (what to skip/avoid annotating). Aim for 8-15 rules. Start IDs from N001.

INLINE ANNOTATED NOTES:
{combined_annotated}

Return ONLY the JSON object with 'rules' and 'observations' keys.\
"""

    import asyncio
    import os
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

    async def _opus_call():
        options = ClaudeAgentOptions(
            model="claude-opus-4-6",
            permission_mode="bypassPermissions",
            max_turns=1,
            system_prompt=system_prompt,
            tools=[],
        )
        raw_text = ""
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        raw_text += block.text
        return raw_text

    saved_cc = os.environ.pop("CLAUDECODE", None)
    t0 = time.time()
    try:
        raw = asyncio.run(_opus_call())
    finally:
        if saved_cc is not None:
            os.environ["CLAUDECODE"] = saved_cc
    print(f"  Opus call took {time.time() - t0:.1f}s  ({len(raw)} chars)")

    # Parse rules
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        print(f"  WARNING: Could not parse output:\n{raw[:500]}")
        return

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error: {e}")
        print(raw[:500])
        return

    rules = parsed.get("rules", [])
    observations = parsed.get("observations", "")

    print(f"\n  Generated {len(rules)} negative rules")
    if observations:
        print(f"  Observations: {observations}")

    print(f"\n  Rules:")
    for r in rules:
        examples = ", ".join(r.get("examples", [])[:3])
        print(f"    {r['id']}: {r['rule'][:100]}")
        if examples:
            print(f"         examples: {examples}")

    # Merge with existing rules if provided
    if existing_rules:
        merged_rules = existing_rules + rules
        merged_observations = parsed.get("observations", "")
        output_data = {
            "rules": merged_rules,
            "observations": merged_observations,
        }
        print(f"\n  Merged: {len(existing_rules)} existing + {len(rules)} new = {len(merged_rules)} total")
    else:
        output_data = parsed

    # Save rules
    args.rules_output.write_text(json.dumps(output_data, indent=2))
    print(f"  Saved to {args.rules_output}")


if __name__ == "__main__":
    main()
