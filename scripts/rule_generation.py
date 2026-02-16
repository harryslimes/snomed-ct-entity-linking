#!/usr/bin/env python3
"""Script 1: Use Claude Opus (via Claude Agent SDK) to generate annotation rules.

Reads the first note from the old-challenge-split, builds an inline-annotated version
with bracket notation, and asks Opus to generate a structured set of reusable rules
for SNOMED CT entity linking.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pandas as pd
from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    TextBlock,
    query,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from engine import SectionSpan, segment_sections  # noqa: E402

SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
TERMINOLOGY_CSV = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"
OUTPUT_PATH = Path(__file__).parent / "rules_output.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_first_note() -> tuple[str, str, pd.DataFrame]:
    """Return (note_id, note_text, annotations_df) for the first training note."""
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    ann_df = pd.read_csv(SPLIT_DIR / "train_annotations.csv")

    first_note_id = notes_df.iloc[0]["note_id"]
    note_text = notes_df.iloc[0]["text"]

    note_anns = (
        ann_df[ann_df["note_id"] == first_note_id]
        .copy()
        .sort_values("start")
    )
    note_anns["start"] = note_anns["start"].astype(int)
    note_anns["end"] = note_anns["end"].astype(int)
    note_anns["concept_id"] = note_anns["concept_id"].astype(int)
    note_anns["annotation_id"] = note_anns["annotation_id"].astype(int)

    return first_note_id, note_text, note_anns


def load_concept_info() -> dict[int, tuple[str, str]]:
    """Return mapping concept_id -> (concept_name, hierarchy)."""
    ft = pd.read_csv(TERMINOLOGY_CSV)
    return {
        int(row.concept_id): (row.concept_name, row.hierarchy)
        for row in ft.itertuples()
    }


# ---------------------------------------------------------------------------
# Inline-annotated note (bracket notation)
# ---------------------------------------------------------------------------

def build_inline_annotated_note(
    note_text: str,
    annotations_df: pd.DataFrame,
    concept_info: dict[int, tuple[str, str]],
) -> str:
    """Build a note with bracket-notation annotations.

    Format: [exact span | Concept Name (hierarchy)]{id=N}
    Processes annotations in reverse order to preserve character offsets.
    """
    sorted_anns = annotations_df.sort_values("start", ascending=False)
    annotated = note_text

    for _, row in sorted_anns.iterrows():
        s, e = int(row["start"]), int(row["end"])
        cid = int(row["concept_id"])
        ann_id = int(row["annotation_id"])
        span_text = note_text[s:e]

        cname, _hierarchy = concept_info.get(cid, ("Unknown", "unknown"))
        # concept_name already includes hierarchy tag, e.g. "Gallstone pancreatitis (disorder)"
        tag = f"[{span_text} | {cname}]{{id={ann_id}}}"
        annotated = annotated[:s] + tag + annotated[e:]

    return annotated


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in clinical NLP and SNOMED CT entity linking. You will be shown \
a clinical discharge note with inline annotations marking every span that should \
be linked to a SNOMED CT concept.

Your task is to analyze these annotations and generate a comprehensive, reusable \
set of rules that an automated agent could follow to reproduce these annotations \
on unseen clinical notes.

The agent that will execute these rules has access to:
1. The surrounding text of a candidate annotation region (300 chars before, \
100 chars after) plus a label indicating the section header it falls under \
(e.g. "Physical Exam", "History of Present Illness", "Discharge Instructions").
2. A SNOMED CT semantic search tool that accepts natural-language queries and \
returns ranked candidate concepts. The tool uses dense retrieval \
(SapBERT embeddings + FAISS) combined with sparse retrieval (BM25 keyword \
matching) via Reciprocal Rank Fusion. Each result includes the concept name \
and its hierarchy type (finding, disorder, procedure, body structure, etc.).

The agent works in TWO steps for each potential annotation:
  Step 1 — Search: Generate one or more search terms to query the SNOMED \
terminology index.
  Step 2 — Select: Review the returned SNOMED candidates and select the best \
concept, also confirming the exact character span boundaries in the note text.

OUTPUT FORMAT — respond with a single JSON object (no surrounding text):
{
  "g_rules": [
    {"id": "G1", "rule": "<text of rule>"},
    {"id": "G2", "rule": "<text of rule>"}
  ],
  "numbered_rules": [
    {"id": "R1", "rule": "<text of rule>", "examples": ["<example>"]},
    {"id": "R2", "rule": "<text of rule>", "examples": ["<example>"]}
  ],
  "annotation_rule_map": {
    "<annotation_id>": ["G1", "G2", "R3", "R7"],
    "<annotation_id>": ["G1", "G2", "R1"]
  }
}

RULE GUIDELINES:
- G-rules are UNIVERSAL rules that apply to EVERY annotation. The instruction \
about how to generate search terms for the SNOMED lookup MUST be a G-rule.
- R-rules (numbered) are SPECIFIC rules for particular patterns or situations.
- Rules must be ABSTRACT and GENERIC — they must generalise to unseen notes, \
not just describe this specific note.
- Each rule should be self-contained and actionable.
- Include rules covering: span boundary detection, concept disambiguation, \
section-aware interpretation, abbreviation handling (especially physical exam \
shorthand), implicit/non-literal concept mapping, repeated mentions, and \
multi-word phrase handling.
- The annotation_rule_map MUST cover ALL annotation IDs from the input \
(every {id=N} tag in the annotated note).\
"""


def build_user_prompt(
    annotated_note: str,
    sections: list[SectionSpan],
    n_annotations: int,
) -> str:
    section_listing = "\n".join(
        f"  - {s.header} (chars {s.start}–{s.end})" for s in sections
    )
    return f"""\
Here is a clinical discharge note with inline SNOMED CT annotations.
Each annotation is formatted as:
[exact span text | Concept Name (hierarchy)]{{id=ANNOTATION_ID}}

The note has {n_annotations} annotations across {len(sections)} sections.

SECTION STRUCTURE:
{section_listing}

ANNOTATED NOTE:
{annotated_note}

Please analyse every annotation carefully and generate rules that explain:
1. WHY each span was selected (span boundary rules)
2. HOW to find the correct SNOMED concept for each span (search + selection rules)
3. WHEN section context changes the interpretation (section-aware rules)
4. Special patterns for abbreviations, implicit concepts, and non-obvious mappings

Remember: the rules must be generic enough to work on other clinical notes.\
"""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Extract a JSON object from response, handling optional code fences."""
    # Try code fence first
    match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", text)
    if match:
        return json.loads(match.group(1))
    # Fall back to finding outermost braces
    start = text.index("{")
    end = text.rindex("}") + 1
    return json.loads(text[start:end])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain() -> None:
    print("Loading data ...")
    note_id, note_text, note_anns = load_first_note()
    concept_info = load_concept_info()
    sections = segment_sections(note_text)

    print(f"Note: {note_id}  |  {len(note_text)} chars  |  {len(note_anns)} annotations  |  {len(sections)} sections")

    print("Building inline-annotated note ...")
    annotated_note = build_inline_annotated_note(note_text, note_anns, concept_info)

    user_prompt = build_user_prompt(annotated_note, sections, len(note_anns))
    print(f"Prompt size: ~{len(SYSTEM_PROMPT + user_prompt)} chars")

    print("Calling Claude Opus via Agent SDK ...")
    options = ClaudeCodeOptions(
        model="claude-opus-4-20250514",
        permission_mode="bypassPermissions",
        max_turns=3,
        system_prompt=SYSTEM_PROMPT,
    )

    raw_text = ""
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    raw_text += block.text
        elif isinstance(message, ResultMessage):
            print(f"  Cost: ${message.total_cost_usd:.4f}" if message.total_cost_usd else "  (cost not reported)")

    print(f"Response: {len(raw_text)} chars")
    print(f"Response preview: {raw_text[:500]!r}")

    print("Parsing rules JSON ...")
    rules = extract_json(raw_text)

    # Validate structure
    assert "g_rules" in rules, "Missing g_rules"
    assert "numbered_rules" in rules, "Missing numbered_rules"
    assert "annotation_rule_map" in rules, "Missing annotation_rule_map"

    # Validate annotation coverage
    expected_ids = set(str(aid) for aid in note_anns["annotation_id"])
    mapped_ids = set(rules["annotation_rule_map"].keys())
    missing = expected_ids - mapped_ids
    if missing:
        print(f"WARNING: {len(missing)} annotation IDs not mapped: {sorted(missing)[:10]}...")
    else:
        print(f"All {len(expected_ids)} annotation IDs mapped to rules.")

    print(f"Generated {len(rules['g_rules'])} global rules, {len(rules['numbered_rules'])} numbered rules")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(rules, f, indent=2)
    print(f"Saved rules to {OUTPUT_PATH}")


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":
    main()
