#!/usr/bin/env python3
"""Generate annotation rules using Claude Opus (via Claude Agent SDK).

Supports two modes:
  - Initial: Generate rules from scratch for a single note.
  - Iterative: Extend existing rules with a new note from the greedy sample.

Usage:
  # Initial (first note in greedy order):
  python rule_generation.py --sample scripts/sampled_annotations.json --note-index 0

  # Extend existing rules with next note:
  python rule_generation.py --sample scripts/sampled_annotations.json --note-index 1 \
      --existing-rules scripts/rules/<timestamp>/rules.json

  # Legacy (first note in CSV, no sampling):
  python rule_generation.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
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
RULES_DIR = Path(__file__).parent / "rules"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_note_by_id(note_id: str) -> str:
    """Load a specific note's text by ID."""
    notes_df = pd.read_csv(SPLIT_DIR / "train_notes.csv")
    row = notes_df[notes_df["note_id"] == note_id].iloc[0]
    return row["text"]


def load_sampled_note(
    sample_path: Path,
    note_index: int,
) -> tuple[str, str, pd.DataFrame, dict]:
    """Load a note and its sampled annotations from the greedy sample.

    Returns (note_id, note_text, annotations_df, sample_meta).
    """
    with open(sample_path) as f:
        sample = json.load(f)

    note_ids = sample["selected_note_ids"]
    if note_index >= len(note_ids):
        raise ValueError(f"note_index {note_index} >= {len(note_ids)} available notes")

    note_id = note_ids[note_index]
    note_text = load_note_by_id(note_id)

    # Filter sampled annotations for this note
    note_anns = [a for a in sample["annotations"] if a["note_id"] == note_id]

    # Convert to DataFrame matching the expected format
    df = pd.DataFrame(note_anns)
    df = df.sort_values("start")
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["concept_id"] = df["concept_id"].astype(int)
    df["annotation_id"] = df["annotation_id"].astype(int)

    meta = {
        "n_notes_total": len(note_ids),
        "n_concepts_total": sample["n_concepts_total"],
        "n_concepts_covered": sample["n_concepts_covered"],
    }

    return note_id, note_text, df, meta


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

    Format: [exact span | Concept Name (hierarchy)]{id=N, cid=CONCEPT_ID}
    Processes annotations in reverse order to preserve character offsets.
    """
    sorted_anns = annotations_df.sort_values("start", ascending=False)
    annotated = note_text

    for _, row in sorted_anns.iterrows():
        s, e = int(row["start"]), int(row["end"])
        cid = int(row["concept_id"])
        ann_id = int(row["annotation_id"])
        span_text = note_text[s:e]

        cname, hierarchy = concept_info.get(cid, ("Unknown", "unknown"))
        tag = f"[{span_text} | {cname} ({hierarchy})]{{id={ann_id}, cid={cid}}}"
        annotated = annotated[:s] + tag + annotated[e:]

    return annotated


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_INITIAL = """\
You are an expert in clinical NLP and SNOMED CT entity linking. You will be shown \
a clinical discharge note with inline annotations marking every span that should \
be linked to a SNOMED CT concept. You will also see a CONCEPT ANCESTRY TABLE \
showing the SNOMED hierarchy for every concept in the note.

Your task is to analyze these annotations and generate a comprehensive, reusable \
set of STRUCTURED rules that an automated pipeline could follow to reproduce \
these annotations on unseen clinical notes.

The pipeline has TWO main stages:

  Stage 2 — Search: An LLM generates search terms to query a SNOMED CT \
terminology index. It receives the raw span text plus surrounding context \
(300 chars before, 100 chars after) and the section header. The index uses \
dense retrieval (SapBERT embeddings + FAISS) combined with sparse retrieval \
(BM25 keyword matching) via Reciprocal Rank Fusion. Each result includes the \
concept name and its hierarchy type (finding, disorder, procedure, body \
structure, situation, etc.).

  Stage 3 — Select: An LLM reviews the top SNOMED candidates returned by the \
index and selects the best matching concept. It sees the same clinical context \
and the ranked candidate list with concept names and hierarchy types.

Each stage receives ONLY the rules relevant to its specific task. This means \
rules must be structured with stage-specific fields.

Each structured rule must declare an "applies_to" object that uses SNOMED \
ancestor concept IDs to define which annotations the rule covers. Use the \
CONCEPT ANCESTRY TABLE in the note to find appropriate ancestor IDs.

{output_format}

{rule_guidelines}\
"""

SYSTEM_PROMPT_EXTEND = """\
You are an expert in clinical NLP and SNOMED CT entity linking. You are \
EXTENDING an existing set of annotation rules based on a new clinical note.

You will be shown:
1. The EXISTING rules (JSON) from previous notes
2. A NEW annotated clinical note with a CONCEPT ANCESTRY TABLE

Your task is to produce an UPDATED rule set that incorporates patterns from \
the new note. The rules drive a 3-stage automated pipeline:

  Stage 2 — Search: An LLM generates SNOMED search terms from span text + \
context. Index uses SapBERT + BM25 via Reciprocal Rank Fusion.

  Stage 3 — Select: An LLM picks the best SNOMED concept from ranked candidates.

EXTENSION RULES:
- PRESERVE all existing g_rules and structured_rules unless you find they need \
correction based on new evidence from this note.
- ADD new structured_rules for annotation patterns not covered by existing rules. \
Each new rule MUST have an applies_to with ancestor_concept_ids from the \
CONCEPT ANCESTRY TABLE.
- REFINE existing rules if the new note reveals they are too narrow, too broad, \
or incorrect. You MAY update an existing rule's applies_to.ancestor_concept_ids \
to broaden or narrow its scope.
- ADD new mappings for abbreviations/acronyms seen in this note that are \
context-free (always mean the same thing).
- PRESERVE all existing mappings unless you find one that is incorrect.
- If an existing rule already covers a pattern in the new note (its applies_to \
subsumes the new annotations), DO NOT create a duplicate — just reference the \
existing rule in the annotation_rule_map.
- Keep rule IDs stable: do not renumber existing rules. New g_rules continue \
from the highest existing G-number, new structured_rules continue from the \
highest existing R-number.
- The annotation_rule_map only needs to cover annotations from THIS note \
(the new one), not from previous notes.

{output_format}

{rule_guidelines}\
"""

OUTPUT_FORMAT = """\
OUTPUT FORMAT — respond with a single JSON object (no surrounding text):
{
  "version": "4.0",
  "mappings": {
    "<exact_span_text>": "<search_term_or_expansion>"
  },
  "g_rules": [
    {
      "id": "G1",
      "rule": "<text of universal rule>",
      "stages": ["stage_2_search", "stage_3_select"]
    }
  ],
  "structured_rules": [
    {
      "rule_id": "R1",
      "concept_type": "<semantic category, e.g. Negated_Finding, Medication_Allergy>",
      "applies_to": {
        "ancestor_concept_ids": [<SNOMED concept IDs>],
        "sections": ["<section header patterns>"] or null,
        "span_pattern": "<optional pattern descriptor>" or null
      },
      "stage_2_search": {
        "filtering_logic": "<when to DROP this span before searching, or null if never>",
        "intent_translation": "<how to convert raw span text into effective SNOMED search terms>"
      },
      "stage_3_select": {
        "disambiguation_logic": "<how to choose among candidate concepts for this pattern>",
        "preferred_hierarchy": "<preferred SNOMED hierarchy type, or null>",
        "reject_hierarchies": ["<hierarchy types to avoid>"]
      }
    }
  ],
  "annotation_rule_map": {
    "<annotation_id>": ["G1", "R1"]
  }
}

APPLIES_TO — HOW RULES ARE LINKED TO ANNOTATIONS:
Each structured rule must declare an "applies_to" object that defines WHICH \
annotations the rule is relevant for. This replaces hard-coded annotation-to-rule \
mappings. The fields are:
- "ancestor_concept_ids": A list of SNOMED concept IDs. The rule applies to any \
annotation whose gold concept is a descendant of (or equal to) ANY of these \
concepts in the SNOMED hierarchy. Use the CONCEPT ANCESTRY TABLE provided below \
to find appropriate ancestor concept IDs. Pick the most specific ancestor that \
covers all the annotations you intend the rule for. For example, if a rule applies \
to respiratory findings, use 106048009 (Respiratory finding). If it applies to \
pain-related findings, use 22253000 (Pain).
- "sections": A list of section header substrings the rule is restricted to, or \
null if the rule applies regardless of section. Use case-insensitive matching. \
Example: ["Allergies", "Adverse Drug"] or null.
- "span_pattern": An optional descriptor for span text characteristics, or null. \
Example: "abbreviation" (all-caps, ≤6 chars), "single_letter", "multi_word". \
This is informational and not currently used for matching.

The annotation_rule_map is still required as a VALIDATION artifact. It must map \
every annotation_id from this note to the rules you believe apply. This lets us \
verify that your applies_to criteria actually cover what you intended.

MAPPINGS vs RULES — You must separate knowledge into two categories:
- "mappings" is for DETERMINISTIC, CONTEXT-FREE 1:1 translations. Use it for \
abbreviation expansions, acronym expansions, and direct synonyms where the span \
text ALWAYS maps to the same search term regardless of clinical section or \
surrounding context. Keys are the EXACT span text (case-sensitive). Values are \
the preferred SNOMED search term. At inference time, mapped spans bypass the \
search LLM entirely and use the value directly for retrieval.
- "rules" (G-rules and structured_rules) are for CONTEXTUAL LOGIC that requires \
reading surrounding text, section headers, or clinical context to make a decision. \
If a term's meaning changes based on the document section, neighboring words, \
or grammatical negation, it MUST be a rule, NOT a mapping.
- Example: An abbreviation like "CTAB" always means "clear to auscultation \
bilaterally" — put it in mappings. But "RA" means "room air" in vital signs \
sections and "rheumatoid arthritis" in rheumatology — this ambiguity requires \
a rule, not a mapping.
- For allergy-section medications (e.g. a drug name in the Allergies section), \
the mapping value should be the SNOMED search intent (e.g. "allergy to [drug]"), \
not just the drug name.\
"""

RULE_GUIDELINES = """\
RULE GUIDELINES:
- G-rules are UNIVERSAL rules that apply broadly. Each G-rule has an optional \
"stages" array indicating which stages it applies to. If "stages" is omitted, \
the rule applies to ALL stages.
- Structured rules (R-rules) are SPECIFIC rules for particular patterns. Each \
rule MUST have a "concept_type" categorizing the clinical pattern AND an \
"applies_to" object declaring which annotations the rule covers. Include \
stage_2_search and/or stage_3_select fields depending on which stages the \
rule is relevant to.
- APPLIES_TO is critical: each rule must declare ancestor_concept_ids that \
define the SNOMED subtree the rule applies to. Use the CONCEPT ANCESTRY TABLE \
provided with the note to find appropriate ancestor IDs. Pick the most specific \
ancestor that covers all annotations you intend the rule for. A rule can list \
multiple ancestor IDs (OR logic) if it spans different branches.
- Rules must be ABSTRACT and GENERIC — they must generalise to unseen notes, \
not just describe this specific note.
- CRITICAL: NO SPECIFIC EXAMPLES ANYWHERE. Every rule field must describe an \
abstract STRATEGY or PATTERN, never list specific spans, abbreviations, expansions, \
or concept mappings. The rules will be applied by an LLM that already has medical \
knowledge — your job is to describe WHAT STRATEGY to follow, not to provide a \
lookup table.
- stage_2_search.intent_translation should describe HOW to transform the raw \
span text into effective SNOMED search queries.
- stage_2_search.filtering_logic should describe WHEN to drop a span before \
searching. Use null if the span should never be filtered.
- stage_3_select.disambiguation_logic should describe HOW to choose the correct \
concept from a ranked list of SNOMED candidates.
- stage_3_select.preferred_hierarchy and reject_hierarchies encode structural \
SNOMED preferences (e.g. prefer "procedure" over "finding" for surgical actions).
- Include rules covering: abbreviation handling, section-aware interpretation, \
implicit/non-literal concept mapping, negation handling, multi-word phrase \
handling, and hierarchy disambiguation.
- The annotation_rule_map MUST cover ALL annotation IDs from the input \
(every {id=N} tag in the annotated note). This is used for validation.\
"""


def build_system_prompt(*, extend: bool) -> str:
    """Build the system prompt for initial or extension mode."""
    template = SYSTEM_PROMPT_EXTEND if extend else SYSTEM_PROMPT_INITIAL
    return template.format(
        output_format=OUTPUT_FORMAT,
        rule_guidelines=RULE_GUIDELINES,
    )


def build_concept_ancestry_table(
    annotations_df: pd.DataFrame,
    concept_info: dict[int, tuple[str, str]],
) -> str:
    """Build a SNOMED concept ancestry table for the annotations in this note.

    Groups concepts by hierarchy, then shows each unique concept with its
    immediate parents and key ancestors. This gives the rule agent the
    information needed to pick appropriate ancestor_concept_ids for applies_to.
    """
    from snomed_subsumption import SubsumptionIndex

    idx = SubsumptionIndex.load()

    # Collect unique concept_ids from this note
    unique_cids = sorted(set(int(c) for c in annotations_df["concept_id"]))

    # Group by hierarchy
    by_hierarchy: dict[str, list[int]] = {}
    for cid in unique_cids:
        _, hierarchy = concept_info.get(cid, ("Unknown", "unknown"))
        by_hierarchy.setdefault(hierarchy, []).append(cid)

    lines = ["CONCEPT ANCESTRY TABLE", "=" * 60]
    lines.append("Use these ancestor concept IDs in your applies_to.ancestor_concept_ids fields.")
    lines.append("Pick the most specific ancestor that covers all annotations a rule targets.")
    lines.append("")

    for hierarchy in sorted(by_hierarchy.keys()):
        cids = by_hierarchy[hierarchy]
        lines.append(f"--- {hierarchy.upper()} ({len(cids)} concepts) ---")

        for cid in cids:
            cname, _ = concept_info.get(cid, ("?", "?"))
            parents = idx.get_parents(cid)
            parent_strs = []
            for p in sorted(parents):
                pname = idx.get_name(p)
                parent_strs.append(f"{p} ({pname})")

            lines.append(f"  {cid}: {cname}")
            if parent_strs:
                lines.append(f"    parents: {', '.join(parent_strs[:3])}")

        # Show useful common ancestors for this hierarchy group
        if len(cids) >= 2:
            lca_options = idx.find_lca_with_depth(cids, min_depth=2)
            if lca_options:
                lines.append(f"  Common ancestors for all {hierarchy} concepts:")
                for anc_id, anc_name, depth in lca_options[:5]:
                    lines.append(f"    {anc_id}: {anc_name} (depth={depth})")
        lines.append("")

    return "\n".join(lines)


def build_user_prompt(
    annotated_note: str,
    sections: list[SectionSpan],
    n_annotations: int,
    concept_ancestry_table: str,
    existing_rules: dict | None = None,
) -> str:
    section_listing = "\n".join(
        f"  - {s.header} (chars {s.start}–{s.end})" for s in sections
    )

    parts = []

    if existing_rules is not None:
        parts.append(f"""\
EXISTING RULES (from previous notes):
```json
{json.dumps(existing_rules, indent=2)}
```

--- NEW NOTE BELOW ---
""")

    parts.append(f"""\
Here is a clinical discharge note with inline SNOMED CT annotations.
Each annotation is formatted as:
[exact span text | Concept Name (hierarchy)]{{id=ANNOTATION_ID, cid=CONCEPT_ID}}

The note has {n_annotations} annotations across {len(sections)} sections.

SECTION STRUCTURE:
{section_listing}

{concept_ancestry_table}

ANNOTATED NOTE:
{annotated_note}

Please analyse every annotation carefully and generate rules that:
1. Have correct applies_to.ancestor_concept_ids using the CONCEPT ANCESTRY TABLE above
2. Explain HOW to find the correct SNOMED concept for each span (search + selection rules)
3. Capture WHEN section context changes the interpretation (section-aware rules)
4. Handle abbreviations via mappings, implicit concepts via rules

Remember: the rules must be generic enough to work on other clinical notes. \
The applies_to criteria determine which future annotations each rule will match.\
""")

    return "\n".join(parts)


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
# Validation
# ---------------------------------------------------------------------------

def validate_rules(rules: dict, expected_ann_ids: set[str]) -> None:
    """Validate the generated rules structure."""
    version = rules.get("version")
    assert version in ("2.0", "3.0", "4.0"), f"Missing or wrong version: {version}"
    assert "g_rules" in rules, "Missing g_rules"
    assert "structured_rules" in rules, "Missing structured_rules"
    assert "annotation_rule_map" in rules, "Missing annotation_rule_map"

    # Validate mappings
    mappings = rules.get("mappings", {})
    assert isinstance(mappings, dict), "mappings must be a dict"
    print(f"Mappings: {len(mappings)} entries")

    # Validate structured rules have required fields
    for sr in rules["structured_rules"]:
        assert "rule_id" in sr, f"structured_rule missing rule_id: {sr}"
        assert "concept_type" in sr, f"structured_rule missing concept_type: {sr}"
        has_stage = any(k in sr for k in ("stage_1_span", "stage_2_search", "stage_3_select"))
        assert has_stage, f"structured_rule {sr['rule_id']} has no stage fields"

        # v4.0: validate applies_to
        if version == "4.0":
            applies_to = sr.get("applies_to")
            if applies_to:
                anc_ids = applies_to.get("ancestor_concept_ids", [])
                if anc_ids:
                    print(f"  {sr['rule_id']} ({sr['concept_type']}): applies to ancestors {anc_ids}")
                else:
                    print(f"  WARNING: {sr['rule_id']} has empty ancestor_concept_ids")
            else:
                print(f"  WARNING: {sr['rule_id']} missing applies_to")

    # Validate annotation coverage
    mapped_ids = set(rules["annotation_rule_map"].keys())
    missing = expected_ann_ids - mapped_ids
    if missing:
        print(f"WARNING: {len(missing)} annotation IDs not mapped: {sorted(missing)[:10]}...")
    else:
        print(f"All {len(expected_ann_ids)} annotation IDs mapped to rules.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def amain(args: argparse.Namespace) -> None:
    bench: dict[str, float] = {}
    bench_wall_start = time.monotonic()

    t = time.monotonic()
    concept_info = load_concept_info()
    bench["load_concept_info_s"] = time.monotonic() - t

    # --- Load note and annotations ---
    t = time.monotonic()
    existing_rules = None
    if args.sample:
        sample_path = Path(args.sample)
        note_id, note_text, note_anns, meta = load_sampled_note(sample_path, args.note_index)
        print(f"Sample mode: note [{args.note_index}] of {meta['n_notes_total']}")
        print(f"  Concept coverage: {meta['n_concepts_covered']}/{meta['n_concepts_total']}")
    else:
        note_id, note_text, note_anns = load_first_note()
    bench["load_note_s"] = time.monotonic() - t

    t = time.monotonic()
    if args.existing_rules:
        rules_path = Path(args.existing_rules)
        print(f"Loading existing rules from {rules_path} ...")
        with open(rules_path) as f:
            existing_rules = json.load(f)
        # Strip annotation_rule_map and internal metadata from prompt
        # (annotation_rule_map would be huge, _iteration_meta is internal)
        existing_for_prompt = {
            k: v for k, v in existing_rules.items()
            if k not in ("annotation_rule_map", "_iteration_meta")
        }
        n_g = len(existing_rules.get("g_rules", []))
        n_s = len(existing_rules.get("structured_rules", []))
        n_m = len(existing_rules.get("mappings", {}))
        print(f"  Existing: {n_g} g_rules, {n_s} structured_rules, {n_m} mappings")
    else:
        existing_for_prompt = None
    bench["load_existing_rules_s"] = time.monotonic() - t

    sections = segment_sections(note_text)
    extend_mode = existing_for_prompt is not None

    print(f"Note: {note_id}  |  {len(note_text)} chars  |  {len(note_anns)} annotations  |  {len(sections)} sections")
    print(f"Mode: {'extend' if extend_mode else 'initial'}")

    t = time.monotonic()
    print("Building inline-annotated note ...")
    annotated_note = build_inline_annotated_note(note_text, note_anns, concept_info)
    bench["build_inline_annotated_note_s"] = time.monotonic() - t

    t = time.monotonic()
    print("Building concept ancestry table ...")
    ancestry_table = build_concept_ancestry_table(note_anns, concept_info)
    bench["build_concept_ancestry_table_s"] = time.monotonic() - t
    print(f"  Ancestry table: {len(ancestry_table)} chars")

    t = time.monotonic()
    system_prompt = build_system_prompt(extend=extend_mode)
    user_prompt = build_user_prompt(
        annotated_note, sections, len(note_anns),
        concept_ancestry_table=ancestry_table,
        existing_rules=existing_for_prompt,
    )
    bench["build_prompts_s"] = time.monotonic() - t
    print(f"Prompt size: system ~{len(system_prompt)} chars, user ~{len(user_prompt)} chars")

    model = args.model or "claude-opus-4-20250514"
    print(f"Calling {model} via Agent SDK ...")
    t = time.monotonic()
    options = ClaudeCodeOptions(
        model=model,
        permission_mode="bypassPermissions",
        max_turns=5,
        system_prompt=system_prompt,
    )

    raw_text = ""
    cost_usd = None
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    raw_text += block.text
        elif isinstance(message, ResultMessage):
            cost_usd = message.total_cost_usd
            print(f"  Cost: ${message.total_cost_usd:.4f}" if message.total_cost_usd else "  (cost not reported)")
    bench["llm_inference_s"] = time.monotonic() - t
    if cost_usd is not None:
        bench["llm_cost_usd"] = cost_usd

    print(f"Response: {len(raw_text)} chars")
    print(f"Response preview: {raw_text[:500]!r}")

    t = time.monotonic()
    print("Parsing rules JSON ...")
    rules = extract_json(raw_text)

    expected_ids = set(str(aid) for aid in note_anns["annotation_id"])
    validate_rules(rules, expected_ids)
    bench["parse_and_validate_s"] = time.monotonic() - t

    mappings = rules.get("mappings", {})
    print(f"Generated {len(mappings)} mappings, {len(rules['g_rules'])} global rules, "
          f"{len(rules['structured_rules'])} structured rules")

    # In extend mode, merge annotation_rule_maps from previous iterations
    if existing_rules and "annotation_rule_map" in existing_rules:
        prev_map = existing_rules["annotation_rule_map"]
        new_map = rules.get("annotation_rule_map", {})
        merged_map = {**prev_map, **new_map}
        rules["annotation_rule_map"] = merged_map
        print(f"Merged annotation_rule_map: {len(prev_map)} previous + {len(new_map)} new = {len(merged_map)} total")

    # Save to timestamped subdirectory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = RULES_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    rules_path = session_dir / "rules.json"

    # Also save metadata about this iteration
    iteration_meta = {
        "note_index": args.note_index if args.sample else 0,
        "note_id": note_id,
        "n_annotations": len(note_anns),
        "extend_mode": extend_mode,
        "existing_rules_path": args.existing_rules,
        "sample_path": args.sample,
        "model": model,
    }
    rules["_iteration_meta"] = iteration_meta

    t = time.monotonic()
    with open(rules_path, "w") as f:
        json.dump(rules, f, indent=2)
    bench["save_rules_s"] = time.monotonic() - t
    print(f"Saved rules to {rules_path}")

    # --- Save benchmarks ---
    bench["total_wall_s"] = time.monotonic() - bench_wall_start
    bench["n_annotations"] = len(note_anns)
    bench["n_sections"] = len(sections)
    bench["prompt_system_chars"] = len(system_prompt)
    bench["prompt_user_chars"] = len(user_prompt)
    bench["response_chars"] = len(raw_text)
    bench["ancestry_table_chars"] = len(ancestry_table)
    bench["n_unique_concepts"] = len(set(int(c) for c in note_anns["concept_id"]))
    bench["extend_mode"] = extend_mode
    bench["note_id"] = note_id
    bench["model"] = model

    # Round floats for readability
    bench = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in bench.items()}

    bench_path = session_dir / "benchmark_generation.json"
    with open(bench_path, "w") as f:
        json.dump(bench, f, indent=2)
    print(f"Saved benchmarks to {bench_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SNOMED annotation rules")
    parser.add_argument("--sample", type=str, default=None,
                        help="Path to sampled_annotations.json")
    parser.add_argument("--note-index", type=int, default=0,
                        help="Index into the greedy note order (default: 0)")
    parser.add_argument("--existing-rules", type=str, default=None,
                        help="Path to existing rules.json to extend")
    parser.add_argument("--model", type=str, default=None,
                        help="Model ID (default: claude-opus-4-20250514)")
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
