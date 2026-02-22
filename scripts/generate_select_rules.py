#!/usr/bin/env python3
"""Generate disambiguation (selection) rules from dictionary selection failures.

Sends all 79 failures to Opus in one pass to generate rules that help the
LLM select the correct SNOMED concept from a small candidate list.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

FAILURES_PATH = Path("/tmp/dict_selection_failures.json")
OUTPUT_PATH = REPO / "super-dictionary" / "select_rules.json"

SYSTEM_PROMPT = """\
You are a clinical NLP expert designing CONCEPT SELECTION rules.

CONTEXT: You are writing rules for an LLM that selects the correct SNOMED CT
concept from a short list of 2-6 candidates. The candidates come from an exact
dictionary lookup of a clinical abbreviation. The LLM sees:
  - The abbreviation span (e.g., "CT", "ABD", "AST")
  - Section header (e.g., "physical exam", "pertinent results")
  - Surrounding clinical context (+/- 1 sentence)
  - A numbered list of 2-6 SNOMED concept names

The LLM must pick the correct one. Your rules help it disambiguate.

FAILURE PATTERNS TO ADDRESS:
Below you will see real failures where the LLM picked the wrong concept.
Analyze the patterns and write rules that would fix them.

RULE FORMAT: Return a JSON array of rule objects:
[
  {
    "id": "S001",
    "rule": "Concise disambiguation instruction (max 40 words)",
    "priority": 2,
    "applies_when": {}
  }
]

GUIDELINES:
- Rules should be GENERAL principles, not specific to one abbreviation
- Focus on recurring patterns: e.g., "prefer measurement/procedure over finding
  when span appears in lab results context"
- Use "applies_when": {"sections": ["section name"]} to restrict a rule to
  specific sections. Use {} for universal rules.
- priority: 1 = override (apply first), 2 = standard, 3 = tiebreaker
- Keep rules concise — they are injected into every LLM prompt
- Aim for 5-15 rules that cover the major error patterns
- Do NOT write rules for individual abbreviations — write GENERALIZABLE principles
"""


async def generate_rules(failures: list[dict]) -> list[dict]:
    from claude_code_sdk import query, AssistantMessage, TextBlock, ClaudeCodeOptions

    # Format failures for the prompt
    failure_text = []
    for f in failures:
        cands = ", ".join(f"{c['name']} ({c['hierarchy']})" for c in f["candidates"])
        failure_text.append(
            f"  Span: {f['span']!r}  Section: {f['section']}  "
            f"Context: ...{f['context_before'][-60:]}>>>{f['span']}<<<{f['context_after'][:60]}...\n"
            f"    Candidates: [{cands}]\n"
            f"    LLM chose: {f['chosen_name']} ({f['chosen_id']})\n"
            f"    Gold:       {f['gold_name']} ({f['gold_id']})"
        )

    user_prompt = (
        f"Here are {len(failures)} selection failures. Respond with ONLY a JSON array of rules — no preamble, no explanation.\n\n"
        + "\n\n".join(failure_text)
    )

    options = ClaudeCodeOptions(
        model="claude-opus-4-6",
        max_turns=3,
        system_prompt=SYSTEM_PROMPT,
    )

    print(f"  Calling claude-opus-4-6 with {len(failures)} failures ...")
    t0 = time.time()

    raw_text = ""
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    raw_text += block.text

    elapsed = time.time() - t0
    print(f"  Opus call took {elapsed:.1f}s, output={len(raw_text)} chars")

    # Parse rules from response
    import re

    # Dump raw text for debugging
    Path("/tmp/select_rules_raw.txt").write_text(raw_text)
    print(f"  Raw output saved to /tmp/select_rules_raw.txt")

    # Find JSON array — try code fence first, then bare
    fence_match = re.search(r"```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```", raw_text)
    if fence_match:
        json_text = fence_match.group(1)
    else:
        # Find outermost [ ... ] being careful with nesting
        match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", raw_text)
        if not match:
            print(f"  ERROR: No JSON array found in response")
            print(raw_text[:500])
            return []
        json_text = match.group()

    rules = json.loads(json_text)
    return rules


def main():
    # Clear CLAUDECODE env to avoid SDK routing issues
    saved_cc = os.environ.pop("CLAUDECODE", None)

    failures = json.loads(FAILURES_PATH.read_text())
    print(f"Loaded {len(failures)} failures")

    rules = asyncio.run(generate_rules(failures))

    if saved_cc is not None:
        os.environ["CLAUDECODE"] = saved_cc

    print(f"\nGenerated {len(rules)} rules:")
    for r in rules:
        print(f"  {r['id']}: {r['rule'][:80]}...")

    # Save
    output = {"rules": rules}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
