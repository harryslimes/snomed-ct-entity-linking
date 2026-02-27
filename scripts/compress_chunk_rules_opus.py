#!/usr/bin/env python3
"""Compress 366 chunk-level extraction rules into a concise set using Opus.

Reads the large-scale rules file, sends all rules to Claude Opus for
merging/compression, and writes the compressed rules to a new file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rulebook.general_rule_loop import _run_sonnet, _parse_sonnet_rules

COMPRESSION_SYSTEM = """\
You are deduplicating a set of clinical NLP extraction rules.

You will receive ~366 rules. Many are near-duplicates (same span, same logic, \
slightly different wording). Your job is to REMOVE DUPLICATES and lightly merge \
rules that say the exact same thing. Do NOT generalise or abstract.

## Critical constraints

1. **Only merge true duplicates.** Two rules are duplicates if they target the \
same span text AND give the same instruction. If two rules target the same span \
but give DIFFERENT advice (e.g., one says extract in lab context, another says \
skip in medication lists), keep BOTH as separate rules.

2. **Do NOT merge across different span texts.** A rule about 'wbc' and a rule \
about 'mcv' must remain separate even if the logic is similar. The extraction \
model needs span-specific guidance. The only exception: if 3+ rules for \
different spans are TRULY identical in logic (e.g., multiple rules all saying \
"skip X when in a line containing 'Sig:', 'Disp:', 'Refills:'"), you may merge \
them into one rule that lists all affected spans explicitly.

3. **Preserve ALL literal patterns.** Keep every quoted string, field name, \
delimiter, and format pattern exactly as-is. Never replace specific patterns \
with generic descriptions.

4. **Preserve extract vs skip as separate rules.** Do NOT combine "extract when \
X" and "skip when Y" into a single conditional rule. Keep them as two rules.

5. **Do NOT drop rules.** Every piece of logic from the input must appear in \
the output. You are deduplicating, not filtering.

6. **Keep rules under 100 words each.**

7. **Expected output size: 80-150 rules.** If your output is under 60 rules, \
you are merging too aggressively. If over 200, you are not deduplicating enough.

## Output format

Return a JSON object:
```json
{
  "rules": [
    {
      "id": "CR001",
      "rule": "The rule text...",
      "priority": 2,
      "source_rules": ["CE019", "CE020"]
    }
  ],
  "compression_notes": "Brief summary of what was deduplicated"
}
```

The `source_rules` field lists original rule IDs that were merged. For rules \
kept as-is, list just the single source ID.\
"""


def build_user_prompt(rules: list[dict], rejected_spans: list[str]) -> str:
    """Build the user prompt with all rules and context."""
    # Format rules grouped by span
    rules_text = ""
    for i, r in enumerate(rules):
        rules_text += f"[{r['id']}] (P{r.get('priority', 2)}) {r['rule']}\n"

    rejected_text = ", ".join(f'"{s}"' for s in rejected_spans[:30])

    return f"""\
Here are {len(rules)} extraction rules generated from contrastive chunk-level \
analysis of 30 clinical notes (2,902 chunks, 11,219 gold annotations).

These rules were individually tested and accepted (net positive improvement). \
However, many are near-duplicates. Deduplicate them — remove rules that say \
the same thing as another rule for the same span. Lightly merge where two \
rules for the same span are clearly redundant. Do NOT generalise across \
different span texts. Do NOT drop unique logic.

=== ALL RULES ({len(rules)}) ===
{rules_text}
=== END RULES ===

Deduplicate these {len(rules)} rules. Target 80-150 rules. Return JSON only."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=REPO_ROOT / "rulebook" / "chunk_rules_largescale.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "rulebook" / "chunk_rules_compressed.json",
    )
    parser.add_argument(
        "--changelog", type=Path,
        default=REPO_ROOT / "scripts" / "chunk_rule_runs" / "20260224T115559Z" / "rule_changelog.jsonl",
    )
    parser.add_argument(
        "--groups", type=Path,
        default=REPO_ROOT / "scripts" / "chunk_rule_runs" / "20260224T115559Z" / "groups.json",
    )
    parser.add_argument("--model", default="claude-sonnet-4-6")
    args = parser.parse_args()

    # Load rules
    rules = json.load(open(args.input))["rules"]
    print(f"Loaded {len(rules)} rules from {args.input}")

    # Get rejected spans for context
    rejected_spans = []
    if args.changelog.exists() and args.groups.exists():
        changelog = [json.loads(l) for l in open(args.changelog)]
        groups = json.load(open(args.groups))
        group_span = {i: g["span_text"] for i, g in enumerate(groups)}

        accepted_groups = set()
        rejected_groups = set()
        for e in changelog:
            if e["action"] == "accept":
                accepted_groups.add(e["group_idx"])
            else:
                rejected_groups.add(e["group_idx"])
        pure_rejected = rejected_groups - accepted_groups
        rejected_spans = [group_span[i] for i in sorted(pure_rejected) if i in group_span]
        print(f"Rejected spans for context: {len(rejected_spans)}")

    # Build prompt
    user_prompt = build_user_prompt(rules, rejected_spans)
    print(f"Prompt length: {len(user_prompt):,} chars (~{len(user_prompt)//4:,} tokens)")
    print(f"Using model: {args.model}")

    # Call Opus
    print(f"\nCalling {args.model} for rule compression...")
    t0 = time.time()
    raw = _run_sonnet(COMPRESSION_SYSTEM, user_prompt, model=args.model)
    elapsed = time.time() - t0
    print(f"Response received in {elapsed:.1f}s ({len(raw):,} chars)")

    # Parse response
    parsed_rules, _ = _parse_sonnet_rules(raw)
    if not parsed_rules:
        # Try to find JSON in the raw output more aggressively
        import re
        match = re.search(r'\{[\s\S]*"rules"\s*:\s*\[[\s\S]*\]\s*[\s\S]*\}', raw)
        if match:
            try:
                result = json.loads(match.group())
                parsed_rules = result.get("rules", [])
                compression_notes = result.get("compression_notes", "")
            except json.JSONDecodeError:
                print(f"ERROR: Could not parse JSON from response")
                print(f"Raw output (first 2000 chars):\n{raw[:2000]}")
                # Save raw output for inspection
                raw_path = args.output.with_suffix(".raw.txt")
                raw_path.write_text(raw)
                print(f"Raw output saved to {raw_path}")
                sys.exit(1)
    else:
        compression_notes = ""
        # Try to extract compression_notes from raw
        import re
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                result = json.loads(match.group())
                compression_notes = result.get("compression_notes", "")
            except json.JSONDecodeError:
                pass

    print(f"\nCompressed: {len(rules)} → {len(parsed_rules)} rules")

    # Calculate token estimate
    compressed_text = "\n".join(
        f"- [{r['id']}] (P{r.get('priority', 2)}) {r['rule']}"
        for r in parsed_rules
    )
    print(f"Compressed rules: {len(compressed_text):,} chars (~{len(compressed_text)//4:,} tokens)")

    # Save
    output_data = {
        "rules": parsed_rules,
        "metadata": {
            "source_file": str(args.input),
            "source_rules_count": len(rules),
            "compressed_rules_count": len(parsed_rules),
            "model": args.model,
            "compression_notes": compression_notes,
        }
    }
    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"Saved to {args.output}")

    # Print summary
    print(f"\n{'='*70}")
    print("COMPRESSED RULES")
    print(f"{'='*70}")
    for r in parsed_rules:
        cat = r.get("category", "?")
        sources = r.get("source_rules", [])
        print(f"  [{r['id']}] P{r.get('priority', 2)} ({cat}, from {len(sources)} rules)")
        print(f"    {r['rule'][:150]}...")
        print()

    if compression_notes:
        print(f"\nCompression notes: {compression_notes}")


if __name__ == "__main__":
    main()
