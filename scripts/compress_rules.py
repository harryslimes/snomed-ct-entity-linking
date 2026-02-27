#!/usr/bin/env python3
"""Compress verbose rules into concise form using Claude Sonnet.

Reads a rules JSON file, sends batches of rules to Sonnet for compression,
and writes a new file with compressed rules.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

BATCH_SIZE = 10  # rules per Sonnet call


COMPRESS_SYSTEM = """\
You are compressing disambiguation rules for a clinical NLP pipeline.

Each rule is injected into EVERY prompt sent to a small LLM. Verbose rules waste
tokens and degrade performance. Your job: rewrite each rule to be as short as
possible while preserving its actionable logic.

Guidelines:
- Target: each rule under 200 words, ideally under 150.
- Be telegraphic. Use imperative voice. Drop filler phrases.
- REMOVE exhaustive example lists. Keep at most 1-2 examples if the pattern
  is genuinely ambiguous. Convert long lists to a general principle.
  Bad:  "for 'HTN','COPD','DM','CKD','CHF','AF','CAD','UTI','MI','MRSA','DVT' ..."
  Good: "for common disease abbreviations"
- REMOVE redundant restatements of the same idea.
- Preserve the core disambiguation logic, priority interactions, and any
  mandatory overrides for specific abbreviations that have no generic equivalent.
- Keep the same rule ID, stage, priority, and applies_when fields unchanged.
- If a rule is already concise (<150 words), return it unchanged.

Return ONLY a JSON array of the compressed rules (same schema as input).\
"""


async def _sonnet_call(system_prompt: str, user_prompt: str) -> str:
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
    finally:
        if saved_cc is not None:
            os.environ["CLAUDECODE"] = saved_cc
    return raw_text


async def compress_batch(rules: list[dict]) -> list[dict]:
    user_prompt = (
        f"Compress these {len(rules)} rules. Return the JSON array.\n\n"
        + json.dumps(rules, indent=2)
    )
    raw = await _sonnet_call(COMPRESS_SYSTEM, user_prompt)
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        print(f"  WARNING: Could not parse response, returning originals")
        return rules
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON error: {e}, returning originals")
        return rules


async def main_async(input_file: Path, output_file: Path) -> None:
    with open(input_file) as f:
        rules = json.load(f)

    print(f"Input: {len(rules)} rules, {sum(len(r['rule']) for r in rules):,} chars")

    compressed: list[dict] = []
    for i in range(0, len(rules), BATCH_SIZE):
        batch = rules[i : i + BATCH_SIZE]
        print(f"  Compressing rules {i+1}-{i+len(batch)} ...")
        result = await compress_batch(batch)
        compressed.extend(result)
        print(f"    {sum(len(r['rule']) for r in batch):,} -> {sum(len(r['rule']) for r in result):,} chars")

    print(f"\nOutput: {len(compressed)} rules, {sum(len(r['rule']) for r in compressed):,} chars")

    with open(output_file, "w") as f:
        json.dump(compressed, f, indent=2)
    print(f"Written to {output_file}")


def main() -> None:
    input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("super-dictionary/abbrev_rules_cleaned_v2.json")
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file.with_stem(input_file.stem + "_compressed")
    asyncio.run(main_async(input_file, output_file))


if __name__ == "__main__":
    main()
