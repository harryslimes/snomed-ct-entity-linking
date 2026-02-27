#!/usr/bin/env python3
"""Send all positive + negative rules to Sonnet for complete reassessment.

Asks Sonnet to:
1. Merge positive and negative rules into a single coherent ruleset
2. Rephrase positive rules to make what should be EXCLUDED clearer
3. Remove redundant or overlapping rules
4. Fix problematic rules (e.g. N016 killing 'discussion' as gold procedure)
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))


async def _sonnet_call(system_prompt: str, user_prompt: str, model: str = "claude-sonnet-4-6") -> str:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
        query,
    )

    t_start = time.time()
    options = ClaudeAgentOptions(
        model=model,
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
    elapsed = time.time() - t_start
    print(f"  Sonnet call: {elapsed:.1f}s, {len(raw_text)} chars output")
    return raw_text


def main():
    # Load current rules
    pos_rules_path = REPO_ROOT / "rulebook" / "general_rules.json"
    neg_rules_path = REPO_ROOT / "rulebook" / "negative_rules_v2.json"

    pos_data = json.loads(pos_rules_path.read_text())
    neg_data = json.loads(neg_rules_path.read_text())

    pos_rules = pos_data.get("rules", [])
    neg_rules = neg_data.get("rules", [])

    print(f"Loaded {len(pos_rules)} positive rules, {len(neg_rules)} negative rules")

    # Format rules for Sonnet
    pos_block = "=== CURRENT POSITIVE RULES (search term generation guidance) ===\n"
    for r in pos_rules:
        scope_parts = []
        applies_to = r.get("applies_to", {})
        if applies_to.get("sections"):
            scope_parts.append(f"sections: {', '.join(applies_to['sections'])}")
        if applies_to.get("ancestor_concept_ids"):
            scope_parts.append(f"concept_ids: {applies_to['ancestor_concept_ids']}")
        scope = f" [scope: {'; '.join(scope_parts)}]" if scope_parts else " [universal]"
        pos_block += f"\n{r['id']} (P{r.get('priority', 2)}){scope}:\n  {r['rule']}\n"

    neg_block = "\n=== CURRENT NEGATIVE RULES (span filtering — what NOT to annotate) ===\n"
    for r in neg_rules:
        examples = ", ".join(r.get("examples", [])[:4])
        neg_block += f"\n{r['id']} (P{r.get('priority', 2)}):\n  {r['rule']}\n  Examples: {examples}\n"

    system_prompt = """\
You are a clinical NLP rules engineer. You are redesigning the annotation ruleset \
for a SNOMED CT entity linking pipeline that processes discharge summary notes.

The pipeline works as follows:
1. A sliding window generates candidate spans (1-4 word n-grams) from clinical notes
2. Code-level pre-filters remove obvious non-annotatable spans (section headers, demographics, etc.)
3. An LLM (Qwen3-30B) receives each candidate span with surrounding context + rules
4. The LLM either generates search terms to find the SNOMED concept, or returns [] to skip
5. Search terms are matched against a SNOMED CT index to retrieve concept candidates

The rules are injected into the LLM prompt. There are two types:
- POSITIVE rules: Guide the LLM on HOW to generate search terms for specific types of spans
- NEGATIVE rules: Tell the LLM WHEN to skip annotation (return empty search_terms)

CRITICAL PROBLEM: The current positive rules focus entirely on "what to search for" but \
give NO guidance on what should be EXCLUDED. The LLM tends to over-annotate because \
positive rules never say "but don't annotate X in this context". Meanwhile, negative \
rules are in a separate block and are often ignored or under-weighted.

YOUR TASK: Produce a single unified ruleset where:
1. Each rule clearly states BOTH what to annotate AND what to exclude
2. Positive rules include explicit exclusion clauses (e.g., "Annotate X, but NOT when Y")
3. Negative patterns are folded into the relevant positive rules where possible
4. Standalone negative rules are kept only for patterns that don't fit any positive rule
5. Rules are concise (<100 words each) and actionable
6. Remove redundancy — merge overlapping rules
7. Fix N016: "discussion" IS a valid gold SNOMED procedure (Discussion (procedure)). \
   The rule about workflow verbs must explicitly EXCLUDE "discussion" from its skip list.
8. Fix N003/N012 overlap: these cover similar ground (demographic/narrative words). Merge.

IMPORTANT CONSTRAINTS:
- The applies_to.ancestor_concept_ids and applies_to.sections fields MUST be preserved \
  exactly as they are — these control which annotations get which rules via subsumption matching
- Rule IDs can be renumbered but the structure must remain: {id, rule, priority, applies_to}
- Negative rules need: {id, type: "negative", rule, priority, examples}
- Keep total rule count manageable (aim for 20-30 total, was 46 before)
- Priority: P1 = override, P2 = standard, P3 = guideline

Response format: Return a single JSON object with:
{
  "positive_rules": [
    {"id": "R001", "rule": "...", "priority": N, "applies_to": {...}}
  ],
  "negative_rules": [
    {"id": "N001", "type": "negative", "rule": "...", "priority": N, "examples": [...]}
  ],
  "changelog": "Brief description of what changed and why"
}\
"""

    user_prompt = f"""\
Here are all the current rules. Completely reassess and rewrite them per the instructions.

{pos_block}

{neg_block}

=== KNOWN ISSUES ===

1. N016 incorrectly catches "discussion" — it's a real SNOMED procedure (Discussion (procedure),
   concept 223482009). The rule about workflow verbs must whitelist "discussion".

2. N003 and N012 overlap heavily (both about narrative words and demographics). Merge them.

3. Positive rules R001-R028 NEVER mention what to exclude. When the LLM sees a candidate
   span like "admission" with positive rules but no clear exclusion guidance, it tries to
   annotate it. Positive rules need explicit "DO NOT annotate when..." clauses.

4. N011 (sub-span filtering) is dangerous in practice because sub-spans are often
   independently annotated gold concepts (e.g., "pancreatitis" within "biliary pancreatitis"
   is annotated as its own concept). This rule needs careful rewording or removal.

5. The LLM receives BOTH positive and negative rules. When positive rules are enthusiastic
   about annotating ("generate search terms for...") and negative rules say "skip this",
   the positive rules tend to win. The rephrasing should make positive rules more balanced.

6. N007 (drug names in med lists) is too blunt — some drug names in med lists ARE annotated
   in the gold standard when they represent the therapeutic concept. The rule should be
   more nuanced.

Return ONLY the JSON object. No commentary outside JSON.\
"""

    print(f"\nCalling Sonnet for rule reassessment...")
    saved_cc = os.environ.pop("CLAUDECODE", None)
    try:
        raw = asyncio.run(_sonnet_call(system_prompt, user_prompt))
    finally:
        if saved_cc is not None:
            os.environ["CLAUDECODE"] = saved_cc

    # Parse JSON
    # Try to find JSON in the response
    json_match = None
    # Look for the outermost { ... }
    brace_depth = 0
    start_idx = None
    for i, ch in enumerate(raw):
        if ch == '{':
            if brace_depth == 0:
                start_idx = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start_idx is not None:
                json_match = raw[start_idx:i+1]
                break

    if not json_match:
        print("ERROR: Could not extract JSON from Sonnet response")
        print("Raw response:")
        print(raw[:3000])
        # Save raw anyway
        raw_path = REPO_ROOT / "rulebook" / "reassessment_raw.txt"
        raw_path.write_text(raw)
        print(f"Saved raw to {raw_path}")
        return

    try:
        parsed = json.loads(json_match)
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parse failed: {e}")
        raw_path = REPO_ROOT / "rulebook" / "reassessment_raw.txt"
        raw_path.write_text(raw)
        print(f"Saved raw to {raw_path}")
        return

    pos_rules_new = parsed.get("positive_rules", [])
    neg_rules_new = parsed.get("negative_rules", [])
    changelog = parsed.get("changelog", "")

    print(f"\n{'='*60}")
    print(f"REASSESSMENT RESULTS")
    print(f"{'='*60}")
    print(f"  Positive rules: {len(pos_rules)} → {len(pos_rules_new)}")
    print(f"  Negative rules: {len(neg_rules)} → {len(neg_rules_new)}")
    print(f"  Total:          {len(pos_rules)+len(neg_rules)} → {len(pos_rules_new)+len(neg_rules_new)}")
    print(f"\nChangelog:\n  {changelog}")

    print(f"\n--- Positive Rules ---")
    for r in pos_rules_new:
        print(f"  {r['id']} (P{r.get('priority', 2)}): {r['rule'][:120]}...")

    print(f"\n--- Negative Rules ---")
    for r in neg_rules_new:
        examples = ", ".join(r.get("examples", [])[:3])
        print(f"  {r['id']} (P{r.get('priority', 2)}): {r['rule'][:120]}...")
        if examples:
            print(f"    ex: {examples}")

    # Save outputs
    pos_out = REPO_ROOT / "rulebook" / "general_rules_v2.json"
    neg_out = REPO_ROOT / "rulebook" / "negative_rules_v3.json"

    pos_out.write_text(json.dumps({"rules": pos_rules_new}, indent=2))
    neg_out.write_text(json.dumps({"rules": neg_rules_new}, indent=2))

    print(f"\nSaved positive rules → {pos_out}")
    print(f"Saved negative rules → {neg_out}")

    # Also save combined raw
    combined_path = REPO_ROOT / "rulebook" / "reassessment_full.json"
    combined_path.write_text(json.dumps(parsed, indent=2))
    print(f"Saved full reassessment → {combined_path}")


if __name__ == "__main__":
    main()
