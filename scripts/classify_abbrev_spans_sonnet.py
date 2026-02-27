#!/usr/bin/env python3
"""Use Claude Sonnet to classify whether each unique span is a genuine medical
abbreviation or a full English/medical word that slipped through the filter.

Uses the same prompt and context-example approach as classify_abbrev_spans.py
but routes through the claude_code_sdk (same auth as abbrev_rule_loop.py).

Output: scripts/abbrev_classification_sonnet.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from build_abbrev_dictionary_llm import extract_abbreviation_items  # noqa: E402

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
DEFAULT_OUT = REPO_ROOT / "scripts" / "abbrev_classification_sonnet.json"

SYSTEM_PROMPT = """\
You are filtering a clinical NLP dataset to remove spans that are NOT abbreviations.

THE ONLY QUESTION: Is this span a fully spelled-out English or medical word/phrase,
OR is it letters/tokens that stand for (abbreviate) a longer phrase?

CRITICAL RULE — familiarity does NOT matter. An abbreviation is defined by its FORM:
if letters stand for words, it is ALWAYS an abbreviation, even if everyone knows it.

KEEP as ABBREV (letters standing for words — always keep these, even famous ones):
  WBC = white blood cell count        RBC = red blood cell count
  HTN = hypertension                  CAD = coronary artery disease
  COPD = chronic obstructive pulm.    DVT = deep vein thrombosis
  HEENT = head eyes ears nose throat  CTAB = clear to auscultation bilaterally
  EOMI = extraocular movements intact MMM = moist mucous membranes
  RRR = regular rate and rhythm       NAD = no acute distress
  NSTEMI = non-ST-elevation MI        HCO3 = bicarbonate (abbreviated formula)
  WBC, RBC, HCT, HGB, PTT, INR, ALT, AST, BUN, CRP, ESR, TSH — all abbreviations
  HR = heart rate    BP = blood pressure    RR = respiratory rate
  LAD = left anterior descending      RA = right atrium OR rheumatoid arthritis
  NEURO, CV, GEN — physical exam section ABBREVIATIONS (not full words)
  NT = non-tender    ND = non-distended    VS = vital signs    IV = intravenous
  AIDS, HIV, UTI, DVT, PE, SOB, CP — all abbreviations even though well-known

REMOVE as TERM (the span IS the full spelled-out word — no expansion needed):
  GLUCOSE, SODIUM, POTASSIUM, CHLORIDE, CALCIUM, MAGNESIUM  ← element/chemical names
  ALBUMIN, PROTEIN, BILIRUBIN, PHOSPHATE, LIPASE, AMYLASE   ← spelled-out analytes
  ABDOMEN, LUNGS, EXTREMITIES, NECK, CHEST, HEART           ← body part section headers
  HYPERTENSION, GLAUCOMA, ANEMIA, ARTHRITIS, OSTEOPOROSIS   ← full disease names
  SURGERY, ASSESSMENT, MEDICATIONS, ULTRASOUND, CULTURE     ← full procedure/section words
  WEIGHT, ACTIVITY, INSTRUCTIONS, WOUND, BLEEDING           ← plain English words

USE THE CONTEXT EXAMPLES to decide:
  - ">>>GLUCOSE<<< -98 UREA N-15 CREAT-1.0" → lab column header = TERM (remove)
  - ">>>NECK<<< : Supple, no lymphadenopathy" → section header = TERM (remove)
  - ">>>CAD<<< s/p CABG, HTN, DM" → abbreviation inline = ABBREV (keep)
  - ">>>HTN<<< , HLD, DM type 2" → abbreviation in list = ABBREV (keep)
  - ">>>WBC<<< -7.4 RBC-4.2 Hgb-13.1" → abbreviation in lab row = ABBREV (keep)
  - ">>>HEENT<<< : NC/AT, PERRL, EOMI" → abbreviation for exam system = ABBREV (keep)
  - ">>>ABDOMEN<<< : Soft, NT, ND, NABS" → spelled-out section header = TERM (remove)"""

USER_TEMPLATE = """\
Using the context examples, decide for each span: is it a spelled-out TERM to remove, \
or an abbreviation to keep?

Return ONLY valid JSON — no explanation, no preamble, nothing else.
Format: {{"not_abbreviations": ["span1", "span2", ...]}}
Only include spans that are spelled-out TERM (should be removed). Omit all abbreviations.

{entries}"""

ENTRY_TEMPLATE = """\
Span: {span!r}
Usage examples:
{examples}"""


def build_span_examples(
    all_items: list[dict],
    max_examples: int = 3,
    context_chars: int = 60,
    seed: int = 42,
) -> dict[str, list[tuple[str, str]]]:
    rng = random.Random(seed)
    by_span: dict[str, list[dict]] = {}
    for item in all_items:
        by_span.setdefault(item["span"], []).append(item)

    examples: dict[str, list[tuple[str, str]]] = {}
    for span, items in by_span.items():
        sample = rng.sample(items, min(max_examples, len(items)))
        span_examples = []
        for item in sample:
            section = item.get("section_header", "unknown")
            before = item.get("before", "").replace("\n", " ").strip()
            after = item.get("after", "").replace("\n", " ").strip()
            before_tail = before[-context_chars:] if len(before) > context_chars else before
            after_head = after[:context_chars] if len(after) > context_chars else after
            snippet = f"...{before_tail} >>>{span}<<< {after_head}..."
            span_examples.append((section, snippet))
        examples[span] = span_examples
    return examples


async def _sonnet_call(system_prompt: str, user_prompt: str) -> str:
    """Single-turn call to Claude Sonnet via the Claude Agent SDK."""
    from claude_code_sdk import (
        AssistantMessage,
        ClaudeCodeOptions,
        TextBlock,
        query,
    )

    options = ClaudeCodeOptions(
        model="claude-sonnet-4-6",
        permission_mode="bypassPermissions",
        max_turns=1,
        system_prompt=system_prompt,
    )

    saved_cc = os.environ.pop("CLAUDECODE", None)
    try:
        raw_text = ""
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        raw_text += block.text
        return raw_text.strip()
    finally:
        if saved_cc is not None:
            os.environ["CLAUDECODE"] = saved_cc


async def classify_batch(
    batch: list[tuple[str, list[tuple[str, str]]]],
    semaphore: asyncio.Semaphore,
    batch_num: int,
    total_batches: int,
) -> list[str]:
    """Classify one batch of spans. Returns non-abbreviation spans."""
    entries = []
    valid_spans = set()
    for span, span_examples in batch:
        valid_spans.add(span)
        ex_lines = "\n".join(
            f"  [{section}] \"{snippet}\""
            for section, snippet in span_examples
        )
        entries.append(ENTRY_TEMPLATE.format(span=span, examples=ex_lines))

    user_msg = USER_TEMPLATE.format(entries="\n\n".join(entries))

    async with semaphore:
        print(f"  Batch {batch_num}/{total_batches} ...", flush=True)
        try:
            raw = await _sonnet_call(SYSTEM_PROMPT, user_msg)
        except Exception as e:
            print(f"  ERROR in batch {batch_num}: {e}")
            return []

    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        print(f"  WARNING batch {batch_num}: could not parse response:\n{raw[:300]}")
        return []
    try:
        parsed = json.loads(m.group())
        result = [s for s in parsed.get("not_abbreviations", []) if s in valid_spans]
        print(f"  Batch {batch_num}: {len(result)}/{len(valid_spans)} flagged as non-abbrev")
        return result
    except json.JSONDecodeError as e:
        print(f"  WARNING batch {batch_num}: JSON error: {e}\n{raw[:300]}")
        return []


async def main_async(args: argparse.Namespace) -> None:
    print("Loading training data ...")
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    ann_df = pd.read_csv(SPLIT_DIR / "train_annotations.csv")
    all_items = extract_abbreviation_items(ann_df, notes_df, 200, 100)

    unique_spans = sorted(set(it["span"] for it in all_items))
    print(f"  {len(unique_spans)} unique spans pass the current filter")

    print("  Building usage examples ...")
    span_examples = build_span_examples(all_items, max_examples=args.examples_per_span)

    instance_counts: dict[str, int] = {}
    for it in all_items:
        instance_counts[it["span"]] = instance_counts.get(it["span"], 0) + 1

    span_example_pairs = [(s, span_examples[s]) for s in unique_spans]
    batches = [
        span_example_pairs[i:i + args.batch_size]
        for i in range(0, len(span_example_pairs), args.batch_size)
    ]
    print(f"  {len(batches)} batches of ~{args.batch_size} spans\n")

    semaphore = asyncio.Semaphore(args.max_concurrent)
    tasks = [
        classify_batch(batch, semaphore, i + 1, len(batches))
        for i, batch in enumerate(batches)
    ]
    results = await asyncio.gather(*tasks)

    not_abbrevs = sorted(set(s for batch_result in results for s in batch_result))
    abbrevs = sorted(set(unique_spans) - set(not_abbrevs))

    print(f"\nClassification complete:")
    print(f"  Genuine abbreviations : {len(abbrevs)}")
    print(f"  Non-abbreviations     : {len(not_abbrevs)}")
    total_instances = sum(instance_counts.get(s, 0) for s in not_abbrevs)
    print(f"  Instances affected    : {total_instances:,}")

    print(f"\nSpans classified as NOT abbreviations ({len(not_abbrevs)}), sorted by instance count:")
    for s in sorted(not_abbrevs, key=lambda x: -instance_counts.get(x, 0)):
        n = instance_counts.get(s, 0)
        exs = span_examples.get(s, [])
        ex_str = f"  [{exs[0][0]}] \"{exs[0][1][:60]}\"" if exs else ""
        print(f"  {s!r:30s}  n={n:4d}  {ex_str}")

    classification = {
        "not_abbreviations": not_abbrevs,
        "abbreviations": abbrevs,
        "total_spans": len(unique_spans),
        "total_instances_removed": total_instances,
    }
    with open(args.output, "w") as f:
        json.dump(classification, f, indent=2)
    print(f"\nSaved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Spans per LLM call")
    parser.add_argument("--examples-per-span", type=int, default=3,
                        help="Usage examples to show per span")
    parser.add_argument("--max-concurrent", type=int, default=5,
                        help="Concurrent Sonnet calls (keep low to avoid rate limits)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
