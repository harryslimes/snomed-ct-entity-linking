#!/usr/bin/env python3
"""Generate negative-only (skip) rules using Qwen via vLLM.

Analyzes FP extraction patterns from the baseline extraction index and asks
Qwen to generate concise "do not annotate" rules based on those patterns.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import aiohttp

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_fp_examples(index_path: Path) -> list[dict]:
    """Load all FP examples with their chunk context."""
    data = json.loads(index_path.read_text())
    chunks = data["chunks"]

    fps = []
    for c in chunks:
        for fp in c["fp_spans"]:
            fps.append({
                "text": fp["text"],
                "note_id": c["note_id"],
                "section": c.get("section_header", ""),
                "chunk_text": c["chunk_text"],
                "leading_context": c.get("leading_context", ""),
                "trailing_context": c.get("trailing_context", ""),
            })
    return fps


def categorize_fps(fps: list[dict]) -> dict[str, list[dict]]:
    """Group FPs into thematic categories for batched rule generation."""
    categories = defaultdict(list)

    for fp in fps:
        text = fp["text"].strip()
        chunk = fp["chunk_text"]
        section = (fp["section"] or "").lower()

        # Standalone numbers
        if text.isdigit() or (len(text) <= 3 and text.replace(".", "").isdigit()):
            categories["standalone_numbers"].append(fp)
        # Lab values with delimiters (e.g., "K-4.4", "Mg-2.0")
        elif "-" in text and any(c.isdigit() for c in text):
            categories["lab_value_fragments"].append(fp)
        # Placeholder patterns
        elif "___" in text:
            categories["placeholder_patterns"].append(fp)
        # Physical exam boilerplate
        elif any(p in text.lower() for p in [
            "clear and coherent", "alert and interactive", "no rebound",
            "normal", "intact", "symmetric", "nontender", "soft",
        ]):
            categories["physical_exam_boilerplate"].append(fp)
        # Medication names in med list context
        elif "medication" in section or "discharge" in section:
            categories["medication_list_items"].append(fp)
        # Vital sign values (e.g., "rr 16", "bp 120/80")
        elif any(text.lower().startswith(v) for v in ["rr ", "hr ", "bp ", "t ", "o2 "]):
            categories["vital_sign_values"].append(fp)
        # Imaging/radiology context
        elif any(p in section.lower() for p in ["radiology", "imaging", "pertinent"]):
            categories["imaging_context"].append(fp)
        # Everything else
        else:
            categories["general"].append(fp)

    return dict(categories)


def sample_diverse_fps(fps: list[dict], max_per_category: int = 30) -> list[dict]:
    """Sample diverse FP examples, preferring high-frequency and diverse contexts."""
    # Count by text
    text_counts = Counter(fp["text"].lower() for fp in fps)

    # Sample: prefer diverse texts, sections, notes
    sampled = []
    seen_texts = set()
    seen_sections = set()

    # First pass: one example per unique text (prioritized by frequency)
    by_text = defaultdict(list)
    for fp in fps:
        by_text[fp["text"].lower()].append(fp)

    for text, count in text_counts.most_common():
        if len(sampled) >= max_per_category:
            break
        examples = by_text[text]
        # Pick one from a section we haven't seen yet if possible
        picked = None
        for ex in examples:
            if ex["section"] not in seen_sections:
                picked = ex
                break
        if not picked:
            picked = examples[0]
        sampled.append(picked)
        seen_texts.add(text)
        seen_sections.add(picked["section"])

    return sampled


def build_negative_rule_prompt(category: str, examples: list[dict]) -> tuple[str, str]:
    """Build system + user prompt for Qwen to generate negative rules."""
    system = """\
You are writing annotation rules for a clinical NLP system that extracts SNOMED CT entities from clinical notes.

Your task: write rules about what should NOT be annotated. You will see examples of text spans that were incorrectly extracted (false positives). Write concise, general rules that would prevent these false positives.

Rules should be:
- Phrased as "Do not annotate..." or "Skip..."
- General enough to cover many instances, not just the specific examples shown
- Under 60 words each
- Focused on observable patterns in the text (section type, formatting, surrounding context)

Return a JSON object:
```json
{
  "rules": [
    {"id": "NR001", "rule": "Do not annotate...", "priority": 2},
    {"id": "NR002", "rule": "Skip...", "priority": 2}
  ]
}
```

Generate 3-8 rules. Be specific about patterns, not vague."""

    # Format examples
    examples_text = ""
    for i, ex in enumerate(examples[:25]):
        ctx_before = ex["leading_context"][-50:] if ex["leading_context"] else ""
        ctx_after = ex["trailing_context"][:50] if ex["trailing_context"] else ""
        examples_text += f"""
[FP-{i+1}] Span: "{ex['text']}"
  Section: {ex['section'] or '(unknown)'}
  Context: ...{ctx_before}[{ex['chunk_text']}]{ctx_after}...
"""

    user = f"""Here are {len(examples[:25])} false positive extractions from the category "{category}".
These spans were incorrectly extracted by the model and should NOT have been annotated.

{examples_text}

Based on these patterns, write 3-8 concise negative rules (what NOT to annotate).
Focus on general patterns, not individual spans. Return JSON only."""

    return system, user


async def call_qwen(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> str:
    """Call Qwen via vLLM."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }

    async with session.post(f"{url}/v1/chat/completions", json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            print(f"  ERROR: vLLM returned {resp.status}: {text[:200]}")
            return ""
        result = await resp.json()
        return result["choices"][0]["message"]["content"]


def parse_rules_from_response(raw: str) -> list[dict]:
    """Extract rules from Qwen's response."""
    import re

    # Try to find JSON
    # First try: find the JSON block
    match = re.search(r'\{[\s\S]*"rules"\s*:\s*\[[\s\S]*\]\s*\}', raw)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed.get("rules", [])
        except json.JSONDecodeError:
            pass

    # Second try: find just the array
    match = re.search(r'\[[\s\S]*\]', raw)
    if match:
        try:
            rules = json.loads(match.group())
            if isinstance(rules, list):
                return rules
        except json.JSONDecodeError:
            pass

    print(f"  WARNING: Could not parse rules from response:\n{raw[:300]}")
    return []


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path,
        default=REPO_ROOT / "scripts" / "chunk_rule_runs" / "20260224T115559Z" / "extraction_index.json",
        help="Baseline extraction index with FP data",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument(
        "--model",
        default="/workspaces/snomed-ct-entity-linking/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "rulebook" / "chunk_rules_negative_only.json",
    )
    parser.add_argument("--max-examples-per-cat", type=int, default=25)
    parser.add_argument("--n-samples", type=int, default=4,
                       help="Generate N rule sets per category, keep all unique")
    args = parser.parse_args()

    # Load FP examples
    print(f"Loading FP examples from {args.index}...")
    fps = load_fp_examples(args.index)
    print(f"  Total FP instances: {len(fps)}")

    # Categorize
    categories = categorize_fps(fps)
    print(f"\nFP categories:")
    for cat, examples in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(examples)} FPs")

    # Generate rules per category
    all_rules = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        for cat, cat_fps in sorted(categories.items(), key=lambda x: -len(x[1])):
            if len(cat_fps) < 3:
                print(f"\n  Skipping {cat} (only {len(cat_fps)} FPs)")
                continue

            print(f"\n{'='*60}")
            print(f"Category: {cat} ({len(cat_fps)} FPs)")
            print(f"{'='*60}")

            sampled = sample_diverse_fps(cat_fps, max_per_category=args.max_examples_per_cat)
            system, user = build_negative_rule_prompt(cat, sampled)

            print(f"  Prompt: {len(system)+len(user):,} chars")

            for attempt in range(args.n_samples):
                print(f"  Sample {attempt+1}/{args.n_samples}...")
                raw = await call_qwen(
                    session, args.vllm_url, args.model,
                    system, user,
                    temperature=0.5 if attempt > 0 else 0.3,
                )
                if not raw:
                    continue

                rules = parse_rules_from_response(raw)
                if rules:
                    # Tag rules with category
                    for r in rules:
                        r["category"] = cat
                        r["source"] = "qwen_negative_gen"
                    all_rules.extend(rules)
                    print(f"    Got {len(rules)} rules")
                else:
                    print(f"    Failed to parse rules")

    # Deduplicate by rule text (exact match)
    seen = set()
    unique_rules = []
    for r in all_rules:
        text_norm = r.get("rule", "").strip().lower()
        if text_norm and text_norm not in seen:
            seen.add(text_norm)
            unique_rules.append(r)

    # Re-number
    for i, r in enumerate(unique_rules):
        r["id"] = f"NR{i+1:03d}"

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Total rules generated: {len(all_rules)}")
    print(f"  Unique rules: {len(unique_rules)}")

    # Print all rules
    for r in unique_rules:
        print(f"  [{r['id']}] ({r['category']}) {r['rule'][:120]}")

    # Save
    output_data = {
        "rules": unique_rules,
        "metadata": {
            "source_index": str(args.index),
            "total_fps": len(fps),
            "categories": {k: len(v) for k, v in categories.items()},
            "model": args.model,
            "n_samples": args.n_samples,
        },
    }
    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"\nSaved {len(unique_rules)} rules to {args.output}")

    # Token estimate
    rules_text = "\n".join(f"- {r['rule']}" for r in unique_rules)
    print(f"Rules text: {len(rules_text):,} chars (~{len(rules_text)//4:,} tokens)")


if __name__ == "__main__":
    asyncio.run(main())
