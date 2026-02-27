#!/usr/bin/env python3
"""Whole-note span extraction experiment.

Instead of sliding n-gram windows, send the entire note (in 2 halves)
to the LLM and ask it to identify all annotatable clinical spans.

Usage:
  python3 scripts/whole_note_extraction.py --note-idx 0
  python3 scripts/whole_note_extraction.py --note-idx 0 --n-windows 1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import aiohttp
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))
sys.path.insert(0, str(REPO_ROOT))

import rulebook.rule_testing as rt
from rulebook.general_rule_loop import (
    GENERAL_SEARCH_SYSTEM,
    BASE_SEARCH_RULES,
    load_rules,
    _format_rules_block,
)

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
DEFAULT_MODEL = "/workspaces/snomed-ct-entity-linking/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"

EXTRACTION_SYSTEM = """\
You are a clinical NLP span extractor. You will be given a chunk of a clinical \
discharge note bracketed by rules. Extract SNOMED CT concept spans and return \
ONLY a JSON object: {"spans": ["exact span 1", "exact span 2", ...]}\
"""


def split_note_into_windows(text: str, n_windows: int) -> list[tuple[str, int]]:
    """Split note into n roughly equal windows at newline boundaries.

    Returns [(window_text, char_offset), ...]
    """
    if n_windows <= 1:
        return [(text, 0)]

    target_size = len(text) // n_windows
    windows = []
    start = 0

    for i in range(n_windows - 1):
        # Find a newline near the target split point
        target = start + target_size
        best_split = min(target, len(text))
        search_range = min(200, max(target - start, len(text) - target))
        for delta in range(0, search_range):
            if target + delta < len(text) and text[target + delta] == '\n':
                best_split = target + delta + 1
                break
            if target - delta >= start and target - delta < len(text) and text[target - delta] == '\n':
                best_split = target - delta + 1
                break

        # Don't create empty windows
        if best_split <= start:
            best_split = min(start + 1, len(text))
        windows.append((text[start:best_split], start))
        start = best_split
        if start >= len(text):
            break

    # Last window
    if start < len(text):
        windows.append((text[start:], start))
    return windows


def detect_sections(text: str) -> list[tuple[int, str]]:
    """Detect section headers in clinical note text.

    Returns [(char_offset, header_text), ...] sorted by offset.
    Section headers are lines ending with ':' (standard discharge note format).
    """
    sections = []
    pos = 0
    for line in text.split('\n'):
        stripped = line.strip()
        # Section headers: non-empty lines ending with ':'
        # Filter out very short lines and lines that are likely not headers
        if (stripped.endswith(':') and len(stripped) >= 3
                and not stripped[0].isdigit()):
            sections.append((pos, stripped))
        pos += len(line) + 1  # +1 for newline
    return sections


def get_section_for_offset(sections: list[tuple[int, str]], offset: int) -> str | None:
    """Return the section header that contains the given character offset."""
    current = None
    for sec_offset, header in sections:
        if sec_offset > offset:
            break
        current = header
    return current


def find_span_offsets(note_text: str, span_text: str, window_offset: int) -> list[tuple[int, int]]:
    """Find all occurrences of span_text in note_text, return (start, end) pairs.

    First tries exact match. If that fails, builds a regex pattern that treats
    any whitespace in span_text as matching any whitespace sequence (including
    newlines) in note_text, preserving original offsets.
    """
    matches = []
    search_start = 0
    # Try exact match first
    while True:
        idx = note_text.find(span_text, search_start)
        if idx == -1:
            break
        matches.append((idx, idx + len(span_text)))
        search_start = idx + 1
    if matches:
        return matches

    # Fallback: whitespace-normalized regex match
    # Split span on whitespace, escape each part, join with \s+ pattern
    parts = span_text.split()
    if len(parts) <= 1:
        return []  # no whitespace to normalize
    pattern = r'\s+'.join(re.escape(p) for p in parts)
    for m in re.finditer(pattern, note_text):
        matches.append((m.start(), m.end()))
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--note-idx", type=int, default=0)
    parser.add_argument("--n-windows", type=int, default=None,
                        help="Number of windows to split the note into (overrides --window-chars)")
    parser.add_argument("--window-chars", type=int, default=134,
                        help="Target chars per window (used when --n-windows is not set)")
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rules-file", type=Path,
                        default=REPO_ROOT / "rulebook" / "general_rules_v2.json")
    parser.add_argument("--negative-rules", type=Path, default=None)
    parser.add_argument("--index-server", default="http://127.0.0.1:8421")
    parser.add_argument("--reasoning-effort", default=None,
                        help="Reasoning effort for reasoning models (e.g. 'medium'). "
                             "If set, uses extra_body reasoning_effort instead of Qwen3 enable_thinking.")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens for LLM response")
    parser.add_argument("--sandwich-prompt", action="store_true",
                        help="Use sandwich prompt: rules, note, rules reminder, output instructions")
    parser.add_argument("--thinking", action="store_true",
                        help="Enable thinking mode for reasoning models (Qwen3 Thinking variant)")
    args = parser.parse_args()

    rt._index_server_url = args.index_server

    # Load data
    notes_df = pd.read_csv(args.split_dir / "train_notes.csv")
    ann_df = pd.read_csv(args.split_dir / "train_annotations.csv")
    note_id = notes_df.iloc[args.note_idx]["note_id"]
    note_text = notes_df.iloc[args.note_idx]["text"]
    gold_anns = ann_df[ann_df["note_id"] == note_id].to_dict("records")

    print(f"Note: {note_id}  ({len(note_text)} chars, {len(gold_anns)} gold annotations)")

    # Load rules for context in the prompt
    rules = load_rules(args.rules_file)
    print(f"Loaded {len(rules)} rules from {args.rules_file}")

    # Build rules context (universal rules only)
    universal = [r for r in rules
                 if not r.get("applies_to", {}).get("ancestor_concept_ids")]
    rules_block = _format_rules_block(universal)

    # Load negative rules if provided
    neg_block = ""
    if args.negative_rules and args.negative_rules.exists():
        neg_data = json.loads(args.negative_rules.read_text())
        neg_rules = neg_data.get("rules", [])
        print(f"Loaded {len(neg_rules)} negative rules")

    # Detect section headers
    sections = detect_sections(note_text)

    # Split note
    if args.n_windows is not None:
        n_windows = args.n_windows
    else:
        n_windows = max(1, len(note_text) // args.window_chars)
    windows = split_note_into_windows(note_text, n_windows)
    print(f"\nSplit note into {len(windows)} windows:")
    for i, (w, offset) in enumerate(windows):
        sec = get_section_for_offset(sections, offset)
        sec_label = f"  [{sec}]" if sec else ""
        print(f"  Window {i}: {len(w)} chars, offset {offset}{sec_label}")

    # Extract spans from each window
    all_extracted_spans = []  # (span_text, window_idx)

    async def extract_from_window(window_text: str, window_idx: int, section_header: str | None):
        url = f"{args.vllm_url}/v1/chat/completions"

        # section_ctx available but not injected — made model too cautious
        # section_ctx = f"\n[Section: {section_header}]" if section_header else ""
        section_ctx = ""

        if args.sandwich_prompt:
            user_prompt = f"""\
=== ANNOTATION RULES ===
Annotatable concepts include: diagnoses, procedures, findings, body structures, \
medications (in therapeutic context), lab tests, devices, and clinical observations.

DO NOT extract:
- Section headers or structural labels (e.g., 'Admission', 'Discharge Diagnosis:')
- Field labels followed by colons ('Birth:', 'Sex:', 'Allergies:')
- Generic narrative verbs ('admitted', 'presented', 'noted', 'treated', 'followed')
- Demographic words ('man', 'woman', 'male', 'female')
- Temporal connectors ('initially', 'prior', 'resulting', 'scheduled')
- Administrative disposition values ('Home', 'Rehab')
- Drug names in medication lists (just inventory items, not therapeutic references)
- Dosing/route/frequency components ('PO', 'BID', 'Q3H', 'tablet', 'Disp', 'Refills')
- Isolated severity qualifiers ('severe', 'mild', 'moderate')
- Consent/risk vocabulary ('risks', 'benefits', 'outcomes', 'alternatives')
- Workflow status words ('pending', 'collected', 'sent', 'ordered')
- Follow-up as scheduling language (only if it asserts a concrete clinical event)

EXCEPTION: 'discussion' IS a valid SNOMED procedure — always extract it.
=== END RULES ===

=== NOTE TEXT ==={section_ctx}
{window_text}
=== END NOTE TEXT ===

=== RULES REMINDER ===
IMPORTANT: Re-read the rules above before answering. You MUST:
- Copy each span EXACTLY as it appears in the note — character-for-character, including capitalisation
- DO NOT paraphrase, normalise, or combine spans
- DO NOT include section headers, narrative verbs, demographics, admin text, medication list items, dosing components, or severity qualifiers
- DO include: diagnoses, procedures, findings, body structures, lab tests/abbreviations (VS, GEN, CV, PULM, ABD, EXTR, RRR, CTAB, NAD, etc.), devices, observations
- 'discussion' IS always annotatable
=== END REMINDER ===

Respond with ONLY a JSON object: {{"spans": ["exact span 1", "exact span 2", ...]}}"""
        else:
            user_prompt = f"""\
Here is a section of a clinical discharge note. Identify ALL spans that represent \
annotatable SNOMED CT clinical concepts.

=== NOTE TEXT ==={section_ctx}
{window_text}
=== END NOTE TEXT ===

Return the exact text of each annotatable span. Include clinical terms, procedures, \
findings, body structures, lab tests, devices, and observations.
Remember: DO NOT include section headers, narrative verbs, demographics, or administrative text."""

        extra_body = {}
        if args.reasoning_effort:
            extra_body["reasoning_effort"] = args.reasoning_effort
        elif args.thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "extra_body": extra_body,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if "error" in data:
                    print(f"  ERROR window {window_idx}: {data['error'].get('message', data['error'])}")
                    return []
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                # For reasoning models, content may be None — check reasoning_content
                if not content and msg.get("reasoning_content"):
                    # The JSON output should be in content, but if model put
                    # everything in reasoning_content, try to extract from there
                    content = msg["reasoning_content"]
                if not content:
                    print(f"  WARNING: Window {window_idx} returned empty content "
                          f"(finish_reason={data['choices'][0].get('finish_reason', '?')})")
                    return []
                # Parse JSON — handle truncated responses
                spans = []
                try:
                    match = re.search(r'\{[^{}]*"spans"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
                    if match:
                        result = json.loads(match.group())
                        spans = result.get("spans", [])
                    else:
                        result = json.loads(content)
                        spans = result.get("spans", [])
                except (json.JSONDecodeError, KeyError):
                    # Try to salvage truncated JSON — extract quoted strings from array
                    array_match = re.search(r'"spans"\s*:\s*\[', content)
                    if array_match:
                        truncated = content[array_match.end():]
                        spans = re.findall(r'"([^"]+)"', truncated)
                        print(f"  Window {window_idx}: salvaged {len(spans)} spans from truncated JSON")
                    else:
                        print(f"  WARNING: Could not parse window {window_idx} response")
                        print(f"  Raw: {content[:500]}")
                        spans = []

                print(f"  Window {window_idx}: extracted {len(spans)} spans")
                return spans

    async def run_extraction():
        tasks = [extract_from_window(w, i, get_section_for_offset(sections, off))
                 for i, (w, off) in enumerate(windows)]
        return await asyncio.gather(*tasks)

    print(f"\nExtracting spans from {len(windows)} windows...")
    t0 = time.time()
    window_results = asyncio.run(run_extraction())
    elapsed = time.time() - t0
    print(f"  Extraction done in {elapsed:.1f}s")

    # Collect all spans and find their offsets in the original note
    detected_spans = []  # (start, end, span_text, window_idx)
    for win_idx, (spans, (win_text, win_offset)) in enumerate(zip(window_results, windows)):
        for span_text in spans:
            # Find in the window text first
            matches = find_span_offsets(win_text, span_text, win_offset)
            if matches:
                for start_in_win, end_in_win in matches:
                    abs_start = win_offset + start_in_win
                    abs_end = win_offset + end_in_win
                    detected_spans.append((abs_start, abs_end, span_text, win_idx))
            else:
                # Try fuzzy: strip and search in whole note
                matches = find_span_offsets(note_text, span_text, 0)
                if matches:
                    for s, e in matches:
                        detected_spans.append((s, e, span_text, win_idx))
                else:
                    print(f"  WARNING: Could not locate '{span_text[:50]}' in note text")

    # Deduplicate by (start, end)
    seen = set()
    unique_spans = []
    for s, e, text, win in detected_spans:
        if (s, e) not in seen:
            seen.add((s, e))
            unique_spans.append((s, e, text, win))

    print(f"\n  Total extracted: {len(detected_spans)}, unique: {len(unique_spans)}")

    # Match against gold annotations
    def iou(a_start, a_end, b_start, b_end):
        inter = max(0, min(a_end, b_end) - max(a_start, b_start))
        union = (a_end - a_start) + (b_end - b_start) - inter
        return inter / union if union > 0 else 0

    gold_matched = set()
    pred_matched_gold = set()

    for gi, g in enumerate(gold_anns):
        g_start = int(g["start"])
        g_end = int(g["end"])
        for pi, (p_start, p_end, p_text, _) in enumerate(unique_spans):
            if iou(g_start, g_end, p_start, p_end) >= 0.5:
                gold_matched.add(gi)
                pred_matched_gold.add(pi)
                break

    tp = len(gold_matched)
    fn = len(gold_anns) - tp
    fp = len(unique_spans) - len(pred_matched_gold)

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Gold annotations:       {len(gold_anns)}")
    print(f"  Extracted spans:        {len(unique_spans)}")
    print(f"  Gold matched (TP):      {tp}")
    print(f"  Gold missed (FN):       {fn}")
    print(f"  Non-gold extractions:   {fp}")
    print(f"  Recall:                 {tp/len(gold_anns)*100:.1f}%")
    print(f"  Precision:              {tp/len(unique_spans)*100:.1f}%" if unique_spans else "  Precision: N/A")

    # Show matched gold
    print(f"\n--- Matched Gold ({tp}/{len(gold_anns)}) ---")
    for gi, g in enumerate(gold_anns):
        if gi in gold_matched:
            print(f"  ✓ [{int(g['start']):4d}:{int(g['end']):4d}] {note_text[int(g['start']):int(g['end'])]!r:40s} cid={int(g['concept_id'])}")

    # Show missed gold
    print(f"\n--- Missed Gold ({fn}/{len(gold_anns)}) ---")
    for gi, g in enumerate(gold_anns):
        if gi not in gold_matched:
            print(f"  ✗ [{int(g['start']):4d}:{int(g['end']):4d}] {note_text[int(g['start']):int(g['end'])]!r:40s} cid={int(g['concept_id'])}")

    # Show non-gold extractions (false positives)
    print(f"\n--- Non-gold Extractions (FP) (first 30 of {fp}) ---")
    fp_count = 0
    for pi, (p_start, p_end, p_text, win) in enumerate(unique_spans):
        if pi not in pred_matched_gold:
            print(f"  [{p_start:4d}:{p_end:4d}] {p_text!r:40s} (window {win})")
            fp_count += 1
            if fp_count >= 30:
                break

    # Also show all extracted spans for reference
    print(f"\n--- All Extracted Spans ({len(unique_spans)}) ---")
    for s, e, text, win in sorted(unique_spans, key=lambda x: x[0]):
        matched = "GOLD" if any(
            iou(int(g["start"]), int(g["end"]), s, e) >= 0.5 for g in gold_anns
        ) else "    "
        print(f"  [{matched}] [{s:4d}:{e:4d}] {text!r}")


if __name__ == "__main__":
    main()
