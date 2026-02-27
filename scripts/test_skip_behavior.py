#!/usr/bin/env python3
"""Quick test: does Qwen3-30B actually return empty search_terms for should-skip spans?"""
import sys, json, asyncio, aiohttp
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "super-dictionary"))

import rulebook.rule_testing as rt
from rulebook.general_rule_loop import (
    GENERAL_SEARCH_SYSTEM, BASE_SEARCH_RULES, load_rules, _format_rules_block,
)

REPO = Path(__file__).resolve().parents[1]

def main():
    rules = load_rules(REPO / "rulebook" / "general_rules_v2.json")
    neg_data = json.loads((REPO / "rulebook" / "negative_rules_v3.json").read_text())
    neg_rules = neg_data.get("rules", [])

    # Build negative rules block
    neg_lines = [
        "\n=== SPAN FILTERING RULES (when NOT to annotate) ===",
        'If the highlighted span matches any of these patterns, respond with '
        '{"search_terms": []} to SKIP annotation.',
        "",
    ]
    for nr in neg_rules:
        p = nr.get("priority", 2)
        examples = ", ".join(nr.get("examples", [])[:3])
        ex_str = f"  (e.g. {examples})" if examples else ""
        neg_lines.append(f"[P{p}] {nr['id']}: {nr['rule']}{ex_str}")
    neg_block = "\n".join(neg_lines)

    # System prompt with skip instruction
    search_system = GENERAL_SEARCH_SYSTEM.replace(
        "Generate 1-3 search terms",
        "Generate 0-3 search terms (0 means SKIP — the span should not be annotated)",
    ) + (
        "\n\nIMPORTANT: If the span should NOT be annotated (per the filtering rules), "
        'respond with {"search_terms": []}.'
    )

    # Universal rules only (no concept scope)
    universal = [r for r in rules if not r.get("applies_to", {}).get("ancestor_concept_ids")]
    rules_extra = _format_rules_block(universal)

    test_cases = [
        ("admitted", "History of Present Illness",
         "...is a ___ year old woman who was ", " to the hospital for..."),
        ("severe", "Brief Hospital Course",
         "...developed ", " biliary pancreatitis with..."),
        ("Home", "Discharge Disposition",
         "Discharge Disposition: ", ""),
        ("Discharge", "Discharge Diagnosis",
         "", " Diagnosis:\n1. Biliary pancreatitis"),
        ("man", "History of Present Illness",
         "...is a 65 year old ", " who presented with..."),
        ("treated", "Brief Hospital Course",
         "...patient was ", " with IV fluids..."),
        ("pancreatitis", "Brief Hospital Course",
         "...developed biliary ", " requiring hospitalization..."),
    ]

    async def run():
        url = "http://localhost:8000/v1/chat/completions"
        for span, section, before, after in test_cases:
            search_rules_text = (
                BASE_SEARCH_RULES
                + ("\n\n" + rules_extra if rules_extra else "")
                + "\n" + neg_block
            )
            prompt = rt.build_search_prompt(before, span, after, section, search_rules_text)
            payload = {
                "model": "models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
                "messages": [
                    {"role": "system", "content": search_system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.0,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            async with aiohttp.ClientSession() as s:
                async with s.post(url, json=payload) as r:
                    data = await r.json()
                    content = data["choices"][0]["message"]["content"]
                    terms = rt.parse_search_terms(content, span)
                    tag = "SKIP" if not terms else "KEEP"
                    raw = repr(content)[:120]
                    print(f"[{tag:4s}] span={span!r:20s} section={section!r:35s} -> {terms!r}")
                    print(f"       raw: {raw}")

    asyncio.run(run())

if __name__ == "__main__":
    main()
