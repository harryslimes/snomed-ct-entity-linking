#!/usr/bin/env python3
"""Quick test: generate initial rules from the annotation guide via Opus."""
import sys, json
sys.path.insert(0, "super-dictionary")
sys.path.insert(0, "rulebook")

from abbrev_rule_loop import generate_initial_rules_from_guide
from pathlib import Path

print("Starting initial rules generation ...", flush=True)
rules = generate_initial_rules_from_guide(
    Path("docs/official_annotation_guide.md"),
    model="claude-opus-4-6",
)
print(f"\n=== GENERATED {len(rules)} RULES ===", flush=True)
print(json.dumps(rules, indent=2), flush=True)
