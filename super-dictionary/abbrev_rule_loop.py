#!/usr/bin/env python3
"""Iterative rule improvement loop for search-term generation rules.

Optimises the search-term LLM prompt so the correct SNOMED concept appears
in the top K retrieved candidates (retrieval recall). No select phase.

Workflow per batch:
  1. Select a batch of (span, section) pairs from the training set
  2. Run search LLM + hybrid retrieval with current rules
  3. Show retrieval misses (gold concept not in top K candidates)
  4. Generate new search-term rules from misses via Claude Sonnet
  5. Test each rule by retrieval recall delta (fixed - broken)
  6. If net gain >= 1 → keep rule; else discard
  7. Advance to the next batch (interactive or --auto-advance)

Rules file format (JSON):
  {
    "rules": [
      {
        "id": "R001",
        "stage": "search",
        "rule": "Rule text shown to the search-term LLM.",
        "applies_when": {
          "sections": ["physical exam"]   // trigger if annotation is in this section
        }
      }
    ]
  }

  Rules with no "applies_when" (or empty dict) are universal — always included.
  Rules are filtered by section only.

Run directories:
  Each run creates a timestamped directory containing state.json (batch index +
  current rules). Pass --resume <dir> to continue from where you left off.

Usage:
  python abbrev_rule_loop.py --batch-size 50
  python abbrev_rule_loop.py --resume rulebook/runs/20260219T123456Z
  python abbrev_rule_loop.py --auto-advance --rounds 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))
sys.path.insert(0, str(REPO_ROOT / "rulebook"))
sys.path.insert(0, str(REPO_ROOT))

from engine import get_section_for_pos, segment_sections  # noqa: E402
import rule_testing as rt  # noqa: E402
from build_abbrev_dictionary_llm import (  # noqa: E402
    ABBREV_SEARCH_SYSTEM,
    ABBREV_RULES_SEARCH as BASE_SEARCH_RULES,
    extract_abbreviation_items,
)

DEFAULT_RULES_FILE = REPO_ROOT / "super-dictionary" / "abbrev_rules.json"
DEFAULT_MODEL = "/workspaces/snomed-ct-entity-linking/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
DEFAULT_RUNS_DIR = REPO_ROOT / "rulebook" / "runs"
SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
INDEX_DIR = REPO_ROOT / "snomed_index"

# Lazy-loaded: sctid -> list of indexed description texts
_index_terms_by_sctid: dict[int, list[str]] | None = None

def _get_indexed_terms(sctid: int | str, max_terms: int = 10) -> list[str]:
    """Return the actual indexed description texts for a SNOMED concept.

    These are the terms in the BM25/FAISS index — showing them helps the rule
    generator understand what search terms would have retrieved the concept.
    """
    global _index_terms_by_sctid
    if _index_terms_by_sctid is None:
        import pickle
        import bm25s
        bm25 = bm25s.BM25.load(str(INDEX_DIR / "bm25s"), load_corpus=True)
        with open(INDEX_DIR / "bm25_sctids.pkl", "rb") as f:
            bm25_sctids = pickle.load(f)
        _index_terms_by_sctid = {}
        for i, sid in enumerate(bm25_sctids):
            _index_terms_by_sctid.setdefault(sid, []).append(bm25.corpus[i]["text"])
    return _index_terms_by_sctid.get(int(sctid), [])[:max_terms]


# ---------------------------------------------------------------------------
# Rules management
# ---------------------------------------------------------------------------

def load_rules(rules_file: Path) -> list[dict]:
    """Load rules from JSON file; create an empty one if it doesn't exist."""
    if not rules_file.exists():
        rules_file.write_text(json.dumps({"rules": []}, indent=2))
        print(f"Created empty rules file: {rules_file}")
        return []
    data = json.loads(rules_file.read_text())
    rules = data.get("rules", [])
    print(f"Loaded {len(rules)} rules from {rules_file}")
    return rules


def save_rules(rules: list[dict], rules_file: Path) -> None:
    """Persist rules to JSON file."""
    rules_file.write_text(json.dumps({"rules": rules}, indent=2))
    print(f"Saved {len(rules)} rules → {rules_file}")


def _next_rule_id(existing_rules: list[dict]) -> str:
    """Return the next available rule ID like R042."""
    ids = [re.search(r"\d+", r["id"]).group() for r in existing_rules if re.search(r"\d+", r["id"])]
    next_n = max((int(n) for n in ids), default=0) + 1
    return f"R{next_n:03d}"


def _rule_in_stage(rule: dict, stage: str) -> bool:
    """True if this rule belongs to the given pipeline stage.

    A rule's 'stage' field may be "search" or "both" (default).
    """
    return rule.get("stage", "both") in (stage, "both")


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def create_run_dir(runs_dir: Path, args: argparse.Namespace) -> Path:
    """Create a timestamped run directory and save the initial config."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("resume",)
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    return run_dir


def save_state(
    run_dir: Path,
    batch_idx: int,
    rules: list[dict],
    holdout_pairs: list | None = None,
    holdout_history: list | None = None,
) -> None:
    """Checkpoint batch index and current rules to state.json.

    Preserves existing holdout_pairs / holdout_history when the caller does not
    supply them, so batch-progress saves don't accidentally clobber holdout state.
    """
    path = run_dir / "state.json"
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    state = {
        "batch_idx": batch_idx,
        "rules": rules,
        "holdout_pairs": (
            holdout_pairs if holdout_pairs is not None
            else existing.get("holdout_pairs", [])
        ),
        "holdout_history": (
            holdout_history if holdout_history is not None
            else existing.get("holdout_history", [])
        ),
    }
    path.write_text(json.dumps(state, indent=2))


def log_rule_change(
    run_dir: Path,
    batch_idx: int,
    rule: dict,
    action: str,
    n_fixed: int,
    n_broken: int,
    fixed_details: list[dict] | None = None,
    broken_details: list[dict] | None = None,
) -> None:
    """Append a rule test result to rule_changelog.jsonl."""
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "batch_idx": batch_idx,
        "action": action,  # "tested_keep", "tested_drop", "combined_accept", "combined_reject"
        "rule_id": rule["id"],
        "replaces": rule.get("replaces"),
        "rule_text": rule["rule"],
        "priority": rule.get("priority", 2),
        "applies_when": rule.get("applies_when", {}),
        "n_fixed": n_fixed,
        "n_broken": n_broken,
        "net": n_fixed - n_broken,
    }
    if fixed_details:
        entry["fixed"] = fixed_details[:10]
    if broken_details:
        entry["broken"] = broken_details[:10]
    path = run_dir / "rule_changelog.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_state(run_dir: Path) -> tuple[int, list[dict], list, list]:
    """Return (batch_idx, rules, holdout_pairs, holdout_history) from state.json."""
    state = json.loads((run_dir / "state.json").read_text())
    holdout_pairs = [tuple(p) for p in state.get("holdout_pairs", [])]
    holdout_history = state.get("holdout_history", [])
    return state["batch_idx"], state.get("rules", []), holdout_pairs, holdout_history


# ---------------------------------------------------------------------------
# Holdout split
# ---------------------------------------------------------------------------

def _split_holdout(
    all_pair_keys: list[tuple[str, str]],
    frac: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (train_keys, holdout_keys) by random split.

    The holdout is a fixed random sample used as a regression set across all
    batches — it is never included in batch rotation or rule optimisation.
    Using the same seed on resume gives exactly the same holdout.
    """
    import random as _random
    if frac <= 0:
        return list(all_pair_keys), []
    rng = _random.Random(seed)
    keys = list(all_pair_keys)
    rng.shuffle(keys)
    n_holdout = max(1, int(len(keys) * frac))
    return keys[n_holdout:], keys[:n_holdout]  # (train, holdout)


# ---------------------------------------------------------------------------
# Per-annotation rule filtering
# ---------------------------------------------------------------------------

def rule_applies_search(rule: dict, section: str) -> bool:
    """True if the rule should appear in the search-phase prompt for this section."""
    if not _rule_in_stage(rule, "search"):
        return False
    aw = rule.get("applies_when", {})
    sections = aw.get("sections", [])
    if not sections:
        return True
    return section.lower() in {s.lower() for s in sections}



def _format_rules_block(applicable_rules: list[dict], stage: str) -> str:
    """Format a filtered rule list into a prompt section.

    Rules are sorted by priority (1 = override, 2 = standard, 3 = guideline) and
    labelled [P1]/[P2]/[P3] so the LLM applies them in the correct order. When two
    rules conflict, the lower-numbered priority wins.
    """
    if not applicable_rules:
        return ""
    sorted_rules = sorted(applicable_rules, key=lambda r: (r.get("priority", 2), r["id"]))
    lines = [
        f"=== DISAMBIGUATION RULES ({stage}) ===",
        "Priority: [P1] override (apply first) → [P2] standard → [P3] guideline (yields to P1/P2).",
    ]
    for r in sorted_rules:
        aw = r.get("applies_when", {})
        cond_parts = []
        if aw.get("sections"):
            cond_parts.append(f"section: {', '.join(aw['sections'])}")
        if aw.get("candidate_hierarchies"):
            cond_parts.append(f"candidates include: {', '.join(aw['candidate_hierarchies'])}")
        cond = f" [when {'; '.join(cond_parts)}]" if cond_parts else ""
        p = r.get("priority", 2)
        lines.append(f"[P{p}] {r['id']}{cond}: {r['rule']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------

def build_reps_for_batch(
    all_items: list[dict],
    batch_pairs: list[tuple[str, str]],
    rules: list[dict],
) -> list[dict]:
    """Build representative items for a specific batch with per-rep rule injection."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in all_items:
        key = (item["span"], item["section_header"])
        groups[key].append(item)

    reps: list[dict] = []
    for span, section in batch_pairs:
        group = groups.get((span, section), [])
        if not group:
            continue
        best = max(group, key=lambda x: len(x["before"]) + len(x["after"]))

        # --- Search rules: filter by section only ---
        search_rules = [r for r in rules if rule_applies_search(r, section)]
        search_extra = _format_rules_block(search_rules, "SEARCH")
        search_rules_text = BASE_SEARCH_RULES + ("\n\n" + search_extra if search_extra else "")

        reps.append({
            "before": best["before"],
            "span": span,
            "after": best["after"],
            "section_header": section,
            "search_rules_text": search_rules_text,
            "gold_start": best["start"],
            "gold_end": best["end"],
            "instance_count": len(group),
            "gold_concept_ids_all": sorted({g["gold_concept_id"] for g in group}),
            "representative_note_id": best["note_id"],
        })
    return reps


def run_batch(
    reps: list[dict],
    vllm_url: str,
    model: str,
    max_concurrent: int,
    reasoning_effort: str,
    *,
    search_terms_only: bool = False,
) -> tuple[list[list[str]], list[list[dict]], list[dict], float]:
    """Run search + retrieval on a batch of reps (no select phase).

    If search_terms_only=True, runs only LLM calls (no retrieval).
    candidates_list will be empty lists — caller must do retrieval.

    Returns (search_terms, candidates_list, timings, elapsed).
    """
    orig_search = rt.SEARCH_SYSTEM
    t0 = time.time()
    try:
        rt.SEARCH_SYSTEM = ABBREV_SEARCH_SYSTEM
        search_terms, candidates_list, _selections, timings = rt.vllm_pipeline(
            reps,
            base_url=vllm_url,
            model=model,
            max_concurrent=max_concurrent,
            reasoning_effort=reasoning_effort,
            search_only=True,
            search_terms_only=search_terms_only,
        )
    finally:
        rt.SEARCH_SYSTEM = orig_search
    elapsed = time.time() - t0
    return search_terms, candidates_list, timings, elapsed


# ---------------------------------------------------------------------------
# Timing display
# ---------------------------------------------------------------------------

def print_timing_summary(timings: list[dict], elapsed: float) -> None:
    """Print aggregate timing metrics from a completed batch run."""
    n = len(timings)
    if n == 0:
        return
    bypass = sum(1 for t in timings if t.get("mapping_bypass"))
    non_bypass = [t for t in timings if not t.get("mapping_bypass")]
    avg_search = sum(t["search_s"] for t in non_bypass) / max(len(non_bypass), 1)
    avg_retr = sum(t["retrieval_s"] for t in timings) / n
    avg_total = sum(t["total_s"] for t in timings) / n
    throughput = n / elapsed if elapsed > 0 else 0

    print(f"\n  Timing ({n} reps):")
    print(f"    Wall time:         {elapsed:.1f}s  ({throughput:.1f} reps/s)")
    print(f"    Avg per-rep:       {avg_total:.2f}s")
    print(f"      ├─ Search (LLM): {avg_search:.2f}s")
    print(f"      └─ Retrieval:    {avg_retr:.2f}s")
    if bypass:
        print(f"    Mapping bypasses:  {bypass}")


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def build_failure_report(
    reps: list[dict],
    search_terms_list: list[list[str]],
    candidates_list: list[list[dict]],
    concept_names: dict[int, tuple[str, str]],
) -> list[dict]:
    """Build a structured failure report for retrieval misses.

    A retrieval miss = the gold concept does NOT appear in the top K candidates.
    """
    failures: list[dict] = []
    for rep, terms, cands in zip(reps, search_terms_list, candidates_list):
        gold_ids = set(rep["gold_concept_ids_all"])
        if any(c["concept_id"] in gold_ids for c in cands):
            continue  # retrieval hit — not a failure

        gold_names = {
            gid: concept_names.get(gid, ("Unknown", "unknown"))
            for gid in gold_ids
        }

        # Include the clinical context around the span (already truncated
        # by --context-before / --context-after, typically 200/100 chars)
        before_text = rep.get("before", "")
        after_text = rep.get("after", "")
        context_excerpt = f"...{before_text}>>>{rep['span']}<<<{after_text}..."

        failures.append({
            "span": rep["span"],
            "section": rep["section_header"],
            "instance_count": rep["instance_count"],
            "gold_ids": list(gold_ids),
            "gold_names": {str(k): v for k, v in gold_names.items()},
            "search_terms": terms,
            "context": context_excerpt,
            "n_candidates": len(cands),
            "top_candidates": [
                {
                    "concept_id": c["concept_id"],
                    "concept_name": c["concept_name"],
                    "hierarchy": c["hierarchy"],
                    "score": c["score"],
                }
                for c in cands[:5]
            ],
            "failure_mode": "retrieval_miss",
        })
    return failures


def print_failure_report(failures: list[dict], max_show: int = 15) -> None:
    """Print a human-readable failure analysis (retrieval misses only)."""
    n = len(failures)
    print(f"\n  Retrieval misses: {n}")

    if not failures:
        return

    print(f"\n  Top retrieval misses (up to {max_show}, by instance count):")
    sorted_f = sorted(failures, key=lambda f: f["instance_count"], reverse=True)
    for f in sorted_f[:max_show]:
        gold_str = ", ".join(
            name for gid, (name, hier) in f["gold_names"].items()
        )
        cands_str = " | ".join(
            c['concept_name'] for c in f["top_candidates"][:3]
        )
        print(
            f"\n  ── {f['span']!r} | {f['section']} (n={f['instance_count']})"
        )
        print(f"     Gold:     {gold_str}")
        print(f"     Search:   {f['search_terms']}")
        print(f"     Top cands: {cands_str}")


# ---------------------------------------------------------------------------
# LLM rule generation
# ---------------------------------------------------------------------------

RULE_GEN_SYSTEM = """\
You are a clinical NLP expert improving search term generation for a SNOMED CT
abbreviation disambiguation pipeline.

Pipeline overview:
  1. Search: an LLM generates search terms from the abbreviation + clinical context.
  2. Retrieval: a hybrid FAISS+BM25 index returns the top candidate concepts.
  3. Scoring: check if the gold concept appears in the top K candidates.

You are ONLY working on step 1 — improving the search terms so the correct
(gold) concept appears in the retrieved candidates. Every failure shown is a
RETRIEVAL MISS: the gold concept was NOT in the top candidates.

For each failure you will see:
  - The abbreviation span, section header, and surrounding clinical context
  - The gold concept and its index synonyms (terms that WOULD match in the index)
  - The search terms the LLM actually generated (which failed to retrieve gold)
  - What was retrieved instead (top candidates)

The context excerpt shows the span in its original clinical note with >>>markers<<<.
Use it to understand WHY the LLM generated the wrong search terms — was it
misled by surrounding text, wrong section interpretation, or wrong expansion?

Your job: write rules that guide the search-term LLM to generate better terms.
Think about WHY the generated search terms missed — wrong expansion, missing
context, wrong hierarchy focus, etc. — and write rules to fix the pattern.

Response format — a single JSON object:

{
  "rules": [
    {
      "id": "R###",
      "replaces": null,    // or "R###" to replace an existing rule
      "stage": "search",   // MUST be "search" — these are search-term rules
      "priority": 2,       // 1=override, 2=standard, 3=guideline
      "rule": "Concise rule text for the search-term LLM.",
      "applies_when": {
        "sections": ["section name"]   // omit if universal
      }
    }
  ],
  "feedback": {
    "index_gaps": null,  // concepts genuinely missing from the index
    "context": null,     // whether more context would help
    "other": null
  }
}

Rules guidance:
  - CRITICAL: Each rule MUST be under 100 words. Be telegraphic.
  - All rules MUST have "stage": "search".
  - Write GENERIC principles for CLASSES of abbreviations, not specific ones.
  - NEVER reference specific abbreviations or SNOMED concept names in a rule.
  - Focus on: abbreviation expansion strategies, section-aware disambiguation,
    synonym generation patterns, hierarchy-appropriate search terms.
  - Prefer replacing/improving existing rules over adding new ones.\
"""


async def _sonnet_call(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Single-turn call via the Claude Agent SDK.

    Uses Claude Code credentials (ANTHROPIC_API_KEY or session OAuth).
    Caller must ensure CLAUDECODE env var is unset before calling this
    (it blocks nested CLI launches).
    """
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
        query,
    )

    t_sdk_start = time.time()

    options = ClaudeAgentOptions(
        model=model,
        permission_mode="bypassPermissions",
        max_turns=1,
        system_prompt=system_prompt,
        tools=[],  # no tools — pure text generation, avoids Claude Code system prompt overhead
    )

    t_first_token = None
    raw_text = ""
    async for message in query(prompt=user_prompt, options=options):
        if t_first_token is None:
            t_first_token = time.time()
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    raw_text += block.text

    t_done = time.time()
    startup = (t_first_token or t_done) - t_sdk_start
    generation = t_done - (t_first_token or t_done)
    print(
        f"    [SDK timing] startup={startup:.1f}s  generation={generation:.1f}s  "
        f"total={t_done - t_sdk_start:.1f}s  output={len(raw_text)} chars"
    )
    return raw_text


def _run_sonnet(system_prompt: str, user_prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """Synchronous wrapper: clears CLAUDECODE env var, runs _sonnet_call, restores."""
    import asyncio
    import os
    saved_cc = os.environ.pop("CLAUDECODE", None)
    try:
        return asyncio.run(_sonnet_call(system_prompt, user_prompt, model=model))
    finally:
        if saved_cc is not None:
            os.environ["CLAUDECODE"] = saved_cc


def _parse_sonnet_rules(raw: str) -> tuple[list[dict], dict]:
    """Parse a Sonnet response into (rules, feedback). Returns ([], {}) on failure."""
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        print(f"  WARNING: Could not parse Sonnet output:\n{raw[:400]}")
        return [], {}
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error: {e}\n{raw[:400]}")
        return [], {}
    return parsed.get("rules", []), parsed.get("feedback", {})


def generate_rules_via_llm(
    failures: list[dict],
    existing_rules: list[dict],
    max_new_rules: int = 5,
    rule_effectiveness: dict | None = None,
    n_parallel: int = 1,
    model: str = "claude-sonnet-4-6",
) -> list[dict]:
    """Use an LLM to propose new rules from the observed failure cases.

    If n_parallel > 1, fires N concurrent LLM calls and pools all proposals,
    renumbering IDs to avoid collisions. Individual rule testing downstream picks
    the best ones (best-of-N).

    If rule_effectiveness is provided, it contains the diff from the previous round:
    which predictions the last batch of rules fixed and broke. This helps the model
    understand what's actually working and generate more targeted rules.
    """
    import asyncio

    sorted_failures = sorted(failures, key=lambda f: f["instance_count"], reverse=True)

    if not sorted_failures:
        print("  No retrieval misses to address.")
        return []

    failure_lines: list[str] = []
    for f in sorted_failures[:12]:
        gold_str = ", ".join(
            name for gid, (name, hier) in f["gold_names"].items()
        )
        # Get the actual indexed terms for the gold concept(s)
        indexed_terms: list[str] = []
        for gid in f["gold_ids"]:
            indexed_terms.extend(_get_indexed_terms(gid, max_terms=6))
        indexed_str = ", ".join(indexed_terms[:8]) if indexed_terms else "(not in index)"

        cands_str = " | ".join(
            c["concept_name"] for c in f["top_candidates"][:4]
        )
        context_line = f"  Context: {f['context']}\n" if f.get("context") else ""
        failure_lines.append(
            f"Span={f['span']!r}  Section={f['section']!r}  n={f['instance_count']}\n"
            f"{context_line}"
            f"  Gold: {gold_str}\n"
            f"  Gold indexed terms: {indexed_str}\n"
            f"  Search terms generated: {f['search_terms']}\n"
            f"  Retrieved instead: {cands_str}"
        )

    existing_text = ""
    if existing_rules:
        # Truncate each rule to keep the prompt within OS argument limits.
        # The model only needs enough context to avoid duplicating existing rules.
        summaries = []
        for r in existing_rules:
            rule_text = r["rule"]
            if len(rule_text) > 150:
                rule_text = rule_text[:150] + "..."
            summaries.append(f"  {r['id']}: {rule_text}")
        existing_text = "\n\nEXISTING RULES (do not duplicate):\n" + "\n".join(summaries)

    # Build effectiveness feedback from the previous round
    effectiveness_text = ""
    if rule_effectiveness:
        fixed = rule_effectiveness.get("fixed", [])
        broken = rule_effectiveness.get("broken", [])
        accepted = rule_effectiveness.get("accepted_rules", [])
        net = rule_effectiveness.get("net_gain", 0)
        parts = [
            f"\n\nPREVIOUS ROUND FEEDBACK (rules {', '.join(accepted)}):",
            f"  Net gain: {net} predictions (fixed {len(fixed)}, broke {len(broken)})",
        ]
        def _dedup_counts(items: list[dict]) -> list[str]:
            from collections import Counter
            counts = Counter((item["span"], item["section"]) for item in items)
            return [
                f"    - span={span!r}  section={sec!r}  (×{n})" if n > 1
                else f"    - span={span!r}  section={sec!r}"
                for (span, sec), n in counts.most_common()
            ]

        if fixed:
            parts.append("  Predictions FIXED by the last rules:")
            parts.extend(_dedup_counts(fixed)[:10])
        if broken:
            parts.append("  Predictions BROKEN by the last rules (avoid repeating these patterns):")
            parts.extend(_dedup_counts(broken)[:10])
        parts.append(
            "  Use this feedback to understand what types of rules are effective. "
            "Avoid generating rules that would repeat the broken patterns."
        )
        effectiveness_text = "\n".join(parts)

    next_id_base = max(
        (int(re.search(r"\d+", r["id"]).group()) for r in existing_rules
         if re.search(r"\d+", r["id"])),
        default=0
    ) + 1

    limit_text = (
        f"up to {max_new_rules} new rules"
        if max_new_rules < 99
        else "as many new rules as needed"
    )
    n_shown = len(sorted_failures[:12])
    user_prompt = (
        f"Analyse these {n_shown} RETRIEVAL MISSES (gold concept not in top candidates) "
        f"and propose {limit_text} search-term rules (start IDs from R{next_id_base:03d}). "
        f"You may also REPLACE existing rules by setting \"replaces\": \"R###\" — "
        f"replacements don't count toward the limit.\n"
        f"For each failure, compare the search terms the LLM generated vs the gold "
        f"indexed terms — the gap shows what the LLM should have generated.\n\n"
        "RETRIEVAL MISSES:\n" + "\n\n".join(failure_lines)
        + existing_text
        + effectiveness_text
        + "\n\nReturn ONLY the JSON object with 'rules' and 'feedback' keys."
    )

    n_parallel = max(1, n_parallel)

    model_label = model.split("/")[-1] if "/" in model else model

    if n_parallel == 1:
        # Single call (original path)
        t0 = time.time()
        try:
            raw = _run_sonnet(RULE_GEN_SYSTEM, user_prompt, model=model)
        except Exception as e:
            print(f"  ERROR calling {model_label} for rule generation: {e}")
            return []
        print(f"  {model_label} call took {time.time() - t0:.1f}s")
        new_rules, feedback = _parse_sonnet_rules(raw)
    else:
        # Parallel best-of-N calls
        print(f"  Firing {n_parallel} parallel {model_label} calls ...")

        async def _parallel_calls():
            import os
            saved_cc = os.environ.pop("CLAUDECODE", None)
            try:
                async def _timed_call(idx: int):
                    t0 = time.time()
                    result = await _sonnet_call(RULE_GEN_SYSTEM, user_prompt, model=model)
                    print(f"  {model_label} call {idx + 1}/{n_parallel} took {time.time() - t0:.1f}s")
                    return result
                tasks = [_timed_call(i) for i in range(n_parallel)]
                return await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                if saved_cc is not None:
                    os.environ["CLAUDECODE"] = saved_cc

        t0 = time.time()
        results = asyncio.run(_parallel_calls())
        print(f"  All {n_parallel} calls completed in {time.time() - t0:.1f}s")

        # Collect rules from all successful calls
        all_rules: list[dict] = []
        all_feedback: dict = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  {model_label} call {i + 1}/{n_parallel} failed: {result}")
                continue
            rules_i, feedback_i = _parse_sonnet_rules(result)
            if rules_i:
                print(f"  {model_label} call {i + 1}/{n_parallel}: {len(rules_i)} rule(s)")
                all_rules.extend(rules_i)
            else:
                print(f"  {model_label} call {i + 1}/{n_parallel}: 0 rules")
            if feedback_i and not all_feedback:
                all_feedback = feedback_i  # keep first non-empty feedback

        # Renumber all pooled rules with unique sequential IDs
        for idx, rule in enumerate(all_rules):
            rule["id"] = f"R{next_id_base + idx:03d}"

        new_rules = all_rules
        feedback = all_feedback

    if not new_rules:
        print("  No rules generated.")
        return []

    # Separate replacements from genuinely new rules for reporting
    replacements = [r for r in new_rules if r.get("replaces")]
    additions = [r for r in new_rules if not r.get("replaces")]
    rpt_parts = []
    if additions:
        rpt_parts.append(f"{len(additions)} new")
    if replacements:
        rpt_parts.append(f"{len(replacements)} replacement(s)")
    print(f"  {model_label} proposed {' + '.join(rpt_parts) or '0'} rule(s)")

    fb_items = {k: v for k, v in (feedback or {}).items() if v}
    if fb_items:
        print("\n  Sonnet feedback:")
        for key, text in fb_items.items():
            print(f"    [{key}] {text}")

    return new_rules


# ---------------------------------------------------------------------------
# Initial rules from annotation guide
# ---------------------------------------------------------------------------

INIT_RULES_SYSTEM = """\
You are a clinical NLP expert designing search-term generation rules.

This is a SUBTASK within a larger SNOMED CT entity linking pipeline. The full
pipeline has multiple stages; you are writing rules for ONE stage only:

  SEARCH-TERM GENERATION (your stage):
    Input:  A clinical text span (often an abbreviation like "HTN", "CABG", "SOB")
            plus its surrounding sentence context and section header.
    Output: A list of natural-language search terms (e.g. "hypertension",
            "coronary artery bypass graft") that will be used to query an index.

  What happens AFTER your stage (not your concern):
    - The search terms are sent to a hybrid index (BM25 keyword + SapBERT dense
      embedding) over SNOMED CT descriptions (synonyms, FSNs, common names).
    - The index returns the top K candidate SNOMED concepts.
    - A separate selection model picks the best candidate.

Your rules guide the search-term LLM. Good search terms are ones that closely
match SNOMED description texts in the index — think official clinical names,
common synonyms, and expanded forms of abbreviations.

You will be given official annotation guidelines. Distil them into rules that
help the search-term LLM generate better search terms. Many guidelines are about
concept SELECTION (choosing between candidates) — OMIT those entirely. Focus on
what affects RETRIEVABILITY: how to expand abbreviations, when to use context,
what terms to generate vs skip, etc.

Rules guidance:
  - Each rule MUST be under 100 words. Be telegraphic.
  - All rules MUST have "stage": "search".
  - Write GENERIC principles for CLASSES of patterns, not specific abbreviations.
  - NEVER reference specific abbreviations or SNOMED concept names in a rule.
  - Priority: 1=override, 2=standard, 3=guideline. Most rules should be P2.
    Reserve P1 only for hard scope rules (what to skip entirely).
  - "applies_when": use {"sections": ["section name"]} to restrict a rule to
    specific clinical note sections. Use {} (empty) for universal rules.
    The ONLY supported key is "sections" — do not invent other condition keys.

Return ONLY a JSON object:
{
  "rules": [
    {
      "id": "R001",
      "stage": "search",
      "priority": 2,
      "rule": "Concise rule text for the search-term LLM.",
      "applies_when": {}
    }
  ]
}\
"""


def generate_initial_rules_from_guide(
    guide_path: Path,
    model: str = "claude-opus-4-6",
) -> list[dict]:
    """Read the annotation guide and generate foundational search-term rules.

    Uses a strong model (default Opus) to produce high-quality initial rules
    that encode the annotation guidelines as search-term generation principles.
    """
    if not guide_path.exists():
        print(f"  WARNING: Annotation guide not found: {guide_path}")
        return []

    guide_text = guide_path.read_text()
    model_label = model.split("/")[-1] if "/" in model else model

    user_prompt = (
        "Read the following SNOMED CT annotation guidelines. These guidelines cover "
        "the FULL annotation process, but you are writing rules for ONE subtask only: "
        "generating search terms that will be sent to a hybrid BM25+SapBERT index "
        "over SNOMED CT descriptions.\n\n"
        "The index contains SNOMED description texts (fully specified names, synonyms, "
        "common clinical names). A search term is 'good' if it lexically or semantically "
        "matches one of these description texts closely enough to retrieve the correct "
        "concept in the top K results.\n\n"
        "Extract rules that affect HOW search terms should be generated:\n"
        "  - Abbreviation expansion strategies (the most common input type)\n"
        "  - When/how to use context from neighbouring sentences (+/- 1 sentence rule)\n"
        "  - Section header as a disambiguation signal\n"
        "  - When to generate multiple search terms vs a single compound term\n"
        "  - What NOT to generate search terms for (headings, lab results, incidental mentions)\n"
        "  - Negative findings: when to search for the negative form vs positive form\n"
        "  - Physical objects: search for 'in situ' finding or procedure instead\n\n"
        "OMIT guidelines that are purely about choosing between retrieved candidates "
        "(e.g. 'prefer Procedure over Finding') — that's a separate selection phase.\n\n"
        "Number rules sequentially from R001.\n\n"
        f"ANNOTATION GUIDELINES:\n\n{guide_text}\n\n"
        "Return ONLY the JSON object with the 'rules' key."
    )

    print(f"\n  Generating initial rules from annotation guide via {model_label} ...")
    t0 = time.time()
    try:
        raw = _run_sonnet(INIT_RULES_SYSTEM, user_prompt, model=model)
    except Exception as e:
        print(f"  ERROR calling {model_label} for initial rules: {e}")
        return []
    print(f"  {model_label} call took {time.time() - t0:.1f}s")

    rules, _ = _parse_sonnet_rules(raw)
    if not rules:
        print("  WARNING: No rules parsed from initial generation.")
        return []

    # Ensure all rules have stage=search and valid structure.
    # Strip applies_when keys the system doesn't support (only "sections" is
    # used by rule_applies_search); keep the rest as universal rules.
    valid_aw_keys = {"sections"}
    for r in rules:
        r["stage"] = "search"
        r.setdefault("priority", 2)
        aw = r.get("applies_when", {})
        r["applies_when"] = {k: v for k, v in aw.items() if k in valid_aw_keys}

    print(f"  Generated {len(rules)} initial rules from annotation guide.")
    for r in rules:
        print(f"    {r['id']}: {r['rule'][:80]}...")

    return rules


RULE_MERGE_SYSTEM = """\
You are a clinical NLP expert maintaining a rule set for abbreviation disambiguation.

You will be given a list of rules that have accumulated over multiple improvement rounds.
Many rules may be redundant, overlapping, or express the same principle in different words.

Your task: merge and compress the rule list into a smaller, non-redundant set that
preserves all distinct disambiguation logic.

Merging guidelines:
  - Combine rules that address the same root pattern into one concise rule.
  - When merging, keep the most general formulation. Prefer class-level language
    ("anatomical abbreviations", "measurement abbreviations") over listing specifics.
  - NEVER reference specific abbreviations, SNOMED concept names, or concept IDs.
    Rules must describe CLASSES and PATTERNS, not individual cases.
  - Each merged rule MUST be under 100 words. If a merge would exceed this, split
    into two focused rules instead of one bloated rule.
  - Preserve all distinct "applies_when" conditions; if merged rules have different
    conditions, either generalise the condition or split into two rules.
  - Do NOT drop logic that addresses a genuinely distinct failure pattern.
  - DEMOTE priority aggressively: P1 (override) should be reserved ONLY for rules
    that must absolutely override all others (e.g., fundamental annotation scope
    rules). Most rules should be P2 (standard). When merging, demote P1 → P2
    unless the rule genuinely needs override semantics. A rule that gives general
    guidance or a preference (not a hard override) should be P2 or P3.
  - Remove duplicate or near-duplicate rules entirely (don't merge, just delete one).
  - Renumber IDs sequentially from R001.
  - Aim to reduce count by 20-30%.

Return ONLY a JSON array of the merged rules (same schema as input):
[
  {
    "id": "R###",
    "rule": "Merged rule text.",
    "priority": 2,
    "stage": "search",
    "applies_when": { ... }   // omit or use {} for universal rules
  }
]\
"""


def merge_rules_via_llm(rules: list[dict], rules_file: Path | None = None) -> list[dict]:
    """Use Claude Opus to merge redundant rules into a smaller, non-redundant set.

    Opus is used as the primary model for merging because it better understands
    rule semantics and produces more conservative, higher-quality consolidations
    than Sonnet. Falls back to Sonnet if Opus fails.
    """
    import asyncio
    from datetime import datetime, timezone

    if len(rules) < 2:
        print("  Nothing to merge (fewer than 2 rules).")
        return rules

    if rules_file is not None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = rules_file.with_stem(f"{rules_file.stem}_backup_{ts}")
        backup.write_text(json.dumps({"rules": rules}, indent=2))
        print(f"  Backup saved → {backup}")

    rules_text = json.dumps(rules, indent=2)
    target_n = max(int(len(rules) * 0.75), len(rules) - 8)
    user_prompt = (
        f"IMPORTANT: Your entire response must be ONLY a JSON array — no analysis, "
        f"no explanation, no preamble. Start your response with [ and end with ].\n\n"
        f"Merge and compress these {len(rules)} rules into a smaller non-redundant set. "
        f"Target approximately {target_n} rules (20-25% reduction). "
        f"Do NOT reduce by more than 35% — over-aggressive merges regress accuracy.\n\n"
        f"CURRENT RULES:\n{rules_text}\n\n"
        "OUTPUT: A JSON array of the merged rules. Nothing else."
    )

    def _extract_json_array(text: str) -> re.Match | None:
        """Find the last valid JSON array in text (handles prose preambles)."""
        # Try to find ```json ... ``` code block first
        code_block = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if code_block:
            return code_block  # group(1) has the array

        # Find all [...] spans and return the last one that looks like a rule array
        # (i.e. contains at least one JSON object with an "id" key)
        for m in reversed(list(re.finditer(r"\[[\s\S]*?\](?=\s*$|\s*```)", text))):
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return m
            except json.JSONDecodeError:
                continue

        # Final fallback: first greedy match
        return re.search(r"\[[\s\S]*\]", text)

    print("  Calling Claude Sonnet for merge ...")
    t0 = time.time()
    try:
        raw = _run_sonnet(RULE_MERGE_SYSTEM, user_prompt)
    except Exception as e:
        print(f"  ERROR calling Sonnet for rule merging: {e}")
        return rules
    print(f"  Sonnet merge call took {time.time() - t0:.1f}s")

    match = _extract_json_array(raw)
    if not match:
        print(f"  WARNING: Could not parse Sonnet merge output:\n{raw[:200]}")
        return rules

    # Handle both full-match and group(1) from code-block extraction
    raw_json = match.group(1) if match.lastindex else match.group()
    try:
        merged = json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse error during merge: {e}\n{raw[:400]}")
        return rules

    print(f"  Merged {len(rules)} rules → {len(merged)} rules "
          f"({100 * (1 - len(merged) / len(rules)):.0f}% reduction)")
    return merged


def _merge_with_validation(
    rules: list[dict],
    all_pair_keys: list[tuple[str, str]],
    all_items: list[dict],
    concept_names: dict[int, tuple[str, str]],
    vllm_url: str,
    model: str,
    max_concurrent: int,
    reasoning_effort: str,
    rules_file: Path | None = None,
    sample_size: int = 30,
    threshold: float = -0.02,
    holdout_pairs: list | None = None,
    holdout_threshold: float = -0.05,
) -> list[dict]:
    """Merge rules via LLM, then validate on a random sample.

    Runs the pipeline on a sample of pairs with both the original and merged
    rules. If the merged retrieval recall is within `threshold` of the
    original, the merged rules are accepted. Otherwise the original rules
    are returned unchanged.
    """
    import random

    merged = merge_rules_via_llm(rules, rules_file)
    if merged is rules:
        return rules  # merge failed or nothing to merge (merge_rules_via_llm returned same obj)

    sample_n = min(sample_size, len(all_pair_keys))
    sample_pairs = random.sample(all_pair_keys, sample_n)

    print(f"\n  Validating merge on {sample_n} sample pairs ...")

    print(f"\n  [Merge validation] Original rules ({len(rules)}) ...")
    orig_recall, *_ = _run_and_evaluate(
        all_items, sample_pairs, rules, concept_names,
        vllm_url, model, max_concurrent, reasoning_effort,
    )

    print(f"\n  [Merge validation] Merged rules ({len(merged)}) ...")
    merged_recall, *_ = _run_and_evaluate(
        all_items, sample_pairs, merged, concept_names,
        vllm_url, model, max_concurrent, reasoning_effort,
    )

    delta = merged_recall - orig_recall
    sign = "+" if delta >= 0 else ""
    print(
        f"\n  Merge validation result: {100 * orig_recall:.1f}% → {100 * merged_recall:.1f}%  "
        f"({sign}{100 * delta:.1f}%)  threshold: {100 * threshold:+.1f}%"
    )

    if delta < threshold:
        print(
            f"  Merge REJECTED — recall drop {100 * delta:.1f}% exceeds threshold "
            f"{100 * threshold:.1f}%. Keeping original {len(rules)} rules."
        )
        return rules

    # Batch sample passed — now check holdout if available.
    if holdout_pairs:
        holdout_n = min(sample_size, len(holdout_pairs))
        holdout_sample = random.sample(holdout_pairs, holdout_n)
        print(f"\n  [Merge validation] Holdout check ({holdout_n} pairs) ...")
        h_orig_recall, *_ = _run_and_evaluate(
            all_items, holdout_sample, rules, concept_names,
            vllm_url, model, max_concurrent, reasoning_effort,
        )
        h_merged_recall, *_ = _run_and_evaluate(
            all_items, holdout_sample, merged, concept_names,
            vllm_url, model, max_concurrent, reasoning_effort,
        )
        h_delta = h_merged_recall - h_orig_recall
        h_sign = "+" if h_delta >= 0 else ""
        print(
            f"  Holdout: {100 * h_orig_recall:.1f}% → "
            f"{100 * h_merged_recall:.1f}%  "
            f"({h_sign}{100 * h_delta:.1f}%)  threshold: {100 * holdout_threshold:+.1f}%"
        )
        if h_delta < holdout_threshold:
            print(
                f"  Merge REJECTED on holdout — {100 * h_delta:.1f}% drop exceeds "
                f"threshold {100 * holdout_threshold:.1f}%. Keeping original {len(rules)} rules."
            )
            return rules

    print(f"  Merge ACCEPTED ({len(rules)} → {len(merged)} rules).")
    return merged




# ---------------------------------------------------------------------------
# Rule coverage audit
# ---------------------------------------------------------------------------

def _audit_rule_coverage(reps: list[dict], rules: list[dict]) -> None:
    """Print a rule-injection coverage report for the current batch.

    Shows how many batch pairs each rule fires for. Rules that fire for <10%
    of pairs are flagged as potentially over-specific.
    """
    n = len(reps)
    if n == 0:
        return

    search_counts: dict[str, int] = defaultdict(int)

    for rep in reps:
        section = rep["section_header"]
        for r in rules:
            if rule_applies_search(r, section):
                search_counts[r["id"]] += 1

    narrow_threshold = 0.10
    narrow: list[str] = []

    print(f"\n  Rule coverage audit ({n} pairs):")
    print(f"  {'ID':6s}  {'fires':>6}  {'%':>6}  {'P':>2}  note")
    for r in rules:
        rid = r["id"]
        sc = search_counts.get(rid, 0)
        pct = sc / n
        p = r.get("priority", 2)
        note = ""
        if sc == 0:
            note = "NEVER FIRED"
        elif pct < narrow_threshold:
            note = f"narrow ({100 * pct:.0f}%)"
            narrow.append(rid)
        print(f"  {rid:6s}  {sc:6d}  {100 * pct:5.1f}%  P{p}  {note}")

    if narrow:
        print(
            f"\n  Narrow rules ({len(narrow)}): {', '.join(narrow)}\n"
            "  These fire for <10% of pairs — likely memorised special cases.\n"
            "  Consider moving named-abbreviation logic to a lookup table."
        )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_retrieval_recall(
    reps: list[dict],
    candidates_list: list[list[dict]],
) -> tuple[float, int, int, dict[str, dict]]:
    """Compute retrieval recall: fraction of reps where gold is in candidates.

    Returns (recall, n_hits, n_total, section_breakdown).
    """
    section_stats: dict[str, dict] = defaultdict(lambda: {"hits": 0, "n": 0})
    n_hits = 0
    for rep, cands in zip(reps, candidates_list):
        gold_ids = set(rep["gold_concept_ids_all"])
        hit = any(c["concept_id"] in gold_ids for c in cands)
        sec = rep["section_header"]
        section_stats[sec]["n"] += 1
        if hit:
            n_hits += 1
            section_stats[sec]["hits"] += 1
    n_total = len(reps)
    recall = n_hits / n_total if n_total else 0.0
    return recall, n_hits, n_total, dict(section_stats)


def _run_and_evaluate(
    all_items: list[dict],
    batch_pairs: list[tuple[str, str]],
    rules: list[dict],
    concept_names: dict[int, tuple[str, str]],
    vllm_url: str,
    model: str,
    max_concurrent: int,
    reasoning_effort: str,
    reps: list[dict] | None = None,
    precomputed_search_terms: list[list[str]] | None = None,
    precomputed_candidates: list[list[dict]] | None = None,
) -> tuple[float, list[dict], list[list[str]], list[list[dict]]]:
    """Build reps → run search+retrieve → evaluate retrieval recall.

    Returns (recall, failures, search_terms, candidates_list).

    If precomputed_search_terms and precomputed_candidates are provided,
    skips the pipeline entirely and scores from the cached results.
    """
    if reps is None:
        reps = build_reps_for_batch(all_items, batch_pairs, rules)

    if precomputed_search_terms is not None and precomputed_candidates is not None:
        search_terms = precomputed_search_terms
        candidates_list = precomputed_candidates
    else:
        search_terms, candidates_list, timings, elapsed = run_batch(
            reps, vllm_url, model, max_concurrent, reasoning_effort,
        )
        print_timing_summary(timings, elapsed)

    recall, n_hits, n_total, section_breakdown = compute_retrieval_recall(
        reps, candidates_list,
    )
    print(
        f"\n  Retrieval recall: {100 * recall:.1f}%  "
        f"({n_hits}/{n_total})"
    )
    print("\n  Section breakdown:")
    for sec, stats in sorted(section_breakdown.items()):
        sec_recall = stats["hits"] / stats["n"] if stats["n"] else 0
        print(f"    {sec:40s}  n={stats['n']:4d}  recall={100 * sec_recall:.1f}%")

    failures = build_failure_report(reps, search_terms, candidates_list, concept_names)
    print_failure_report(failures)

    return recall, failures, search_terms, candidates_list


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--split", default="train", choices=["train", "test", "both"])
    parser.add_argument(
        "--rules-file", type=Path, default=DEFAULT_RULES_FILE,
        help="Path to JSON rules file (ignored when --resume is used)",
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-concurrent", type=int, default=32)
    parser.add_argument("--reasoning-effort", default="none",
                        choices=["none", "low", "medium", "high"])
    parser.add_argument("--context-before", type=int, default=200)
    parser.add_argument("--context-after", type=int, default=100)
    parser.add_argument(
        "--holdout-frac", type=float, default=0.1, metavar="FRAC",
        help="Fraction of (span, section) pairs to hold out as a fixed regression set "
             "(default: 0.1). Set to 0 to disable. Held-out pairs are never used for "
             "rule optimisation; retrieval recall on them is reported after each improvement.",
    )
    parser.add_argument(
        "--holdout-seed", type=int, default=42, metavar="SEED",
        help="Random seed for the holdout split (default: 42). "
             "Use the same seed to reproduce the exact same holdout set.",
    )
    parser.add_argument(
        "--audit-rules", action="store_true",
        help="Print a rule-coverage audit after each batch baseline run",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Number of unique (span, section) pairs per batch",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Max rule-improvement rounds per batch before advancing",
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip LLM rule generation (evaluate only)",
    )
    parser.add_argument(
        "--auto-advance", action="store_true",
        help="Automatically advance to the next batch without prompting",
    )
    parser.add_argument(
        "--max-new-rules", type=int, default=5,
        help="Max new rules to request from LLM per round",
    )
    parser.add_argument(
        "--max-rules", type=int, default=25,
        help="Auto-merge rules when count exceeds this threshold (0 = never auto-merge)",
    )
    parser.add_argument(
        "--holdout-reject-threshold", type=float, default=-0.03,
        help="Reject new rules if holdout regresses by more than this (default: -0.03 = -3%%)",
    )
    parser.add_argument(
        "--batch-target", type=float, default=0.65,
        help="Skip rule generation if batch retrieval recall is at or above this target (default: 0.65 = 65%%)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Run a merge pass on the current rules file and exit",
    )
    parser.add_argument(
        "--merge-sample", type=int, default=30, metavar="N",
        help="Number of (span, section) pairs to sample when validating a merge (default: 30)",
    )
    parser.add_argument(
        "--merge-threshold", type=float, default=-0.02, metavar="DELTA",
        help="Minimum allowed recall change after merging (default: -0.02 = allow up to 2pp drop)",
    )
    parser.add_argument(
        "--context-length", type=int, default=8192,  # kept for vLLM search prompt budget
    )
    parser.add_argument(
        "--index-server", type=str, default="http://127.0.0.1:8421", metavar="URL",
        help="URL of a running snomed_index_server.py (default: http://127.0.0.1:8421). "
             "The index server is REQUIRED — in-process retrieval is too slow.",
    )
    parser.add_argument(
        "--sonnet-n", type=int, default=1, metavar="N",
        help="Number of parallel LLM calls per rule-generation step. "
             "All proposals are pooled and tested individually (best-of-N). Default: 1",
    )
    parser.add_argument(
        "--init-rules-model", default="claude-opus-4-6", metavar="MODEL",
        help="Claude model for generating initial rules from the annotation guide "
             "(default: claude-opus-4-6). Set to 'skip' to disable.",
    )
    parser.add_argument(
        "--first-stage-model", default="claude-opus-4-6", metavar="MODEL",
        help="Claude model for the first N batches of rule generation "
             "(default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--first-stage-batches", type=int, default=3, metavar="N",
        help="Number of initial batches that use --first-stage-model (default: 3). "
             "After this many batches, switches to --remaining-model.",
    )
    parser.add_argument(
        "--remaining-model", default="claude-sonnet-4-6", metavar="MODEL",
        help="Claude model for rule generation after the first stage "
             "(default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--annotation-guide", type=Path,
        default=REPO_ROOT / "docs" / "official_annotation_guide.md",
        metavar="PATH",
        help="Path to the annotation guide markdown file for initial rules generation",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, metavar="RUN_DIR",
        help="Resume a previous run from its directory (e.g. rulebook/runs/20260219T123456Z)",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
        help="Where to create new run directories (default: rulebook/runs/)",
    )
    args = parser.parse_args()

    rt._index_server_url = args.index_server

    # ------------------------------------------------------------------
    # Pre-flight: verify vLLM server is reachable
    # ------------------------------------------------------------------
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(f"{args.vllm_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            model_ids = [m["id"] for m in data.get("data", [])]
            print(f"vLLM server OK at {args.vllm_url}  model(s): {model_ids}")
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        print(f"ERROR: Cannot reach vLLM server at {args.vllm_url}")
        print(f"  {e}")
        print("  Start the vLLM server first, then re-run this script.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Pre-flight: verify SNOMED index server is reachable
    # ------------------------------------------------------------------
    try:
        req = urllib.request.Request(f"{args.index_server}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5):
            print(f"Index server OK at {args.index_server}")
    except (urllib.error.URLError, OSError) as e:
        print(f"ERROR: Cannot reach SNOMED index server at {args.index_server}")
        print(f"  {e}")
        print("  Start snomed_index_server.py first, then re-run this script.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"Loading data from {args.split_dir} ...")
    splits = ["train", "test"] if args.split == "both" else [args.split]
    ann_frames, note_frames = [], []
    for s in splits:
        ann_frames.append(pd.read_csv(args.split_dir / f"{s}_annotations.csv"))
        note_frames.append(pd.read_csv(args.split_dir / f"{s}_notes.csv"))
    ann_df = pd.concat(ann_frames, ignore_index=True)
    notes_df = pd.concat(note_frames, ignore_index=True)
    for col in ("start", "end", "concept_id", "annotation_id"):
        ann_df[col] = pd.to_numeric(ann_df[col], errors="coerce").fillna(0).astype(int)
    print(f"  {len(ann_df):,} annotations, {len(notes_df):,} notes")

    # ------------------------------------------------------------------
    # Extract abbreviation items
    # ------------------------------------------------------------------
    print("\nExtracting abbreviation candidates ...")
    all_items = extract_abbreviation_items(
        ann_df, notes_df, args.context_before, args.context_after,
    )
    print(f"  {len(all_items):,} abbreviation instances extracted")

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for item in all_items:
        pair_counts[(item["span"], item["section_header"])] += 1
    all_pairs_sorted = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    all_pair_keys = [k for k, _ in all_pairs_sorted]

    n_pairs = len(all_pair_keys)
    print(f"  {n_pairs:,} unique (span, section) pairs")

    # Holdout split — computed now for fresh runs; loaded from state on resume.
    # train_pair_keys drives the batch rotation; holdout_pairs are only used for
    # regression evaluation and are never seen by the rule-optimisation loop.
    holdout_pairs: list[tuple[str, str]] = []
    holdout_history: list[dict] = []
    if not args.resume and args.holdout_frac > 0:
        train_pair_keys, holdout_pairs = _split_holdout(
            all_pair_keys, args.holdout_frac, args.holdout_seed,
        )
    else:
        train_pair_keys = list(all_pair_keys)  # may be replaced after state load below

    n_train = len(train_pair_keys)
    n_batches = (n_train + args.batch_size - 1) // args.batch_size
    if holdout_pairs:
        print(f"  Holdout: {len(holdout_pairs)} pairs reserved  |  "
              f"Train: {n_train} pairs  |  {n_batches} batches of {args.batch_size}")
    else:
        print(f"  {n_train} pairs → {n_batches} batches of {args.batch_size}")

    # ------------------------------------------------------------------
    # Load concept names
    # ------------------------------------------------------------------
    print("\nLoading concept names ...")
    concept_names = rt.load_concept_names()

    # ------------------------------------------------------------------
    # Set up run directory and load rules
    # ------------------------------------------------------------------
    if args.resume:
        run_dir = args.resume.resolve()
        if not run_dir.exists():
            print(f"ERROR: Run directory not found: {run_dir}")
            return
        batch_idx, rules, saved_holdout, holdout_history = load_state(run_dir)
        if saved_holdout:
            # Use the exact holdout set from the original run so regression numbers
            # are comparable across sessions.
            holdout_pairs = saved_holdout
            holdout_set = set(holdout_pairs)
            train_pair_keys = [k for k in all_pair_keys if k not in holdout_set]
            n_train = len(train_pair_keys)
            n_batches = (n_train + args.batch_size - 1) // args.batch_size
        print(f"\nResuming from {run_dir}")
        print(f"  Batch: {batch_idx}  Rules: {len(rules)}  Holdout: {len(holdout_pairs)}")
    else:
        rules = load_rules(args.rules_file)
        if args.merge:
            if not rules:
                print("No rules to merge.")
                return
            print(f"\nMerging {len(rules)} rules via Sonnet (with validation sample={args.merge_sample}) ...")
            merged = _merge_with_validation(
                rules, all_pair_keys, all_items, concept_names,
                args.vllm_url, args.model, args.max_concurrent, args.reasoning_effort,
                rules_file=args.rules_file,
                sample_size=args.merge_sample,
                threshold=args.merge_threshold,
                holdout_pairs=holdout_pairs or None,
            )
            if merged is not rules:
                save_rules(merged, args.rules_file)
            print("Done.")
            return
        run_dir = create_run_dir(args.runs_dir, args)
        batch_idx = 0

        # Generate initial rules from annotation guide if starting from scratch
        if not rules and args.init_rules_model != "skip":
            initial_rules = generate_initial_rules_from_guide(
                args.annotation_guide,
                model=args.init_rules_model,
            )
            if initial_rules:
                rules = initial_rules
                save_rules(rules, args.rules_file)
                print(f"  Bootstrapped {len(rules)} rules from annotation guide.")

        save_state(run_dir, batch_idx, rules,
                   holdout_pairs=holdout_pairs, holdout_history=holdout_history)
        print(f"\nRun directory: {run_dir}")

    # ------------------------------------------------------------------
    # Model selection helper
    # ------------------------------------------------------------------
    def _get_rule_gen_model(batch_idx: int) -> str:
        """Return the appropriate LLM model for the current batch index."""
        if batch_idx < args.first_stage_batches:
            return args.first_stage_model
        return args.remaining_model

    # ------------------------------------------------------------------
    # Interactive batch loop
    # ------------------------------------------------------------------
    last_rule_effectiveness: dict | None = None  # feedback from previous round

    # Next-batch prefetch cache.  During Sonnet calls the GPU is idle, so we
    # run the next batch's search+retrieve phase in a background thread.
    # Keyed by (batch_idx, rules_hash) to detect invalidation from rule changes.
    import concurrent.futures as _cf
    _prefetch_executor = _cf.ThreadPoolExecutor(max_workers=1)
    _prefetch_future: _cf.Future | None = None
    _prefetch_key: tuple[int, str] | None = None  # (batch_idx, rules_fingerprint)

    def _rules_fingerprint(rules: list[dict]) -> str:
        """Hash of search-affecting rules to detect cache invalidation."""
        return json.dumps(rules, sort_keys=True)

    def _prefetch_next_batch(
        next_batch_idx: int, rules: list[dict],
    ) -> tuple[list[list[str]], list[list[dict]]]:
        """Run search+retrieve for the next batch (called in bg thread)."""
        next_start = next_batch_idx * args.batch_size
        next_pairs = train_pair_keys[next_start: next_start + args.batch_size]
        if not next_pairs:
            return [], []
        next_reps = build_reps_for_batch(all_items, next_pairs, rules)
        st, cands, _, _ = run_batch(
            next_reps, args.vllm_url, args.model,
            args.max_concurrent, args.reasoning_effort,
        )
        return st, cands

    def _start_prefetch(next_idx: int, rules: list[dict]) -> None:
        nonlocal _prefetch_future, _prefetch_key
        if next_idx >= n_batches:
            return
        fp = _rules_fingerprint(rules)
        new_key = (next_idx, fp)
        if _prefetch_key == new_key and _prefetch_future is not None:
            return
        if _prefetch_future is not None:
            _prefetch_future.cancel()
        _prefetch_key = new_key
        _prefetch_future = _prefetch_executor.submit(
            _prefetch_next_batch, next_idx, rules,
        )
        print(f"  [Prefetch] Started search+retrieve for batch {next_idx + 1} in background")

    def _get_prefetch(batch_idx: int, rules: list[dict]) -> tuple[list[list[str]], list[list[dict]]] | None:
        nonlocal _prefetch_future, _prefetch_key
        if _prefetch_future is None or _prefetch_key is None:
            return None
        expected_key = (batch_idx, _rules_fingerprint(rules))
        if _prefetch_key != expected_key:
            print(f"  [Prefetch] Cache invalidated (rules changed)")
            _prefetch_future.cancel()
            _prefetch_future = None
            _prefetch_key = None
            return None
        try:
            st, cands = _prefetch_future.result(timeout=0.1)
            print(f"  [Prefetch] Using cached search+retrieve for batch {batch_idx + 1}")
            _prefetch_future = None
            _prefetch_key = None
            return st, cands
        except _cf.TimeoutError:
            print(f"  [Prefetch] Waiting for background search+retrieve ...")
            st, cands = _prefetch_future.result()
            print(f"  [Prefetch] Using cached search+retrieve for batch {batch_idx + 1}")
            _prefetch_future = None
            _prefetch_key = None
            return st, cands
        except Exception as e:
            print(f"  [Prefetch] Failed: {e}")
            _prefetch_future = None
            _prefetch_key = None
            return None

    while batch_idx < n_batches:
        batch_start = batch_idx * args.batch_size
        batch_pairs = train_pair_keys[batch_start: batch_start + args.batch_size]

        cur_model = _get_rule_gen_model(batch_idx)
        cur_model_label = cur_model.split("/")[-1] if "/" in cur_model else cur_model
        stage_label = (
            f"first-stage ({cur_model_label})"
            if batch_idx < args.first_stage_batches
            else f"remaining ({cur_model_label})"
        )

        print(f"\n{'=' * 70}")
        print(
            f"BATCH {batch_idx + 1}/{n_batches}  "
            f"({len(batch_pairs)} pairs, "
            f"instances {batch_start + 1}–{batch_start + len(batch_pairs)})  "
            f"Rules: {len(rules)}  |  {stage_label}  |  {run_dir.name}"
        )
        print(f"{'=' * 70}")

        reps = build_reps_for_batch(all_items, batch_pairs, rules)

        # ── Auto-merge if too many rules ──────────────────────────────────
        if args.max_rules > 0 and len(rules) > args.max_rules:
            print(f"\n  Rule count ({len(rules)}) exceeds --max-rules {args.max_rules}; merging ...")
            merged = _merge_with_validation(
                rules, all_pair_keys, all_items, concept_names,
                args.vllm_url, args.model, args.max_concurrent, args.reasoning_effort,
                rules_file=args.rules_file,
                sample_size=args.merge_sample,
                threshold=args.merge_threshold,
                holdout_pairs=holdout_pairs or None,
            )
            if merged is not rules:
                rules = merged
                save_rules(rules, args.rules_file)
                save_state(run_dir, batch_idx, rules)
            reps = build_reps_for_batch(all_items, batch_pairs, rules)

        # ── Baseline run (search + retrieve only) ─────────────────────────
        print("\n[Baseline — current rules]")
        pf_kwargs: dict = {}
        prefetched = _get_prefetch(batch_idx, rules)
        if prefetched is not None:
            pf_kwargs["precomputed_search_terms"] = prefetched[0]
            pf_kwargs["precomputed_candidates"] = prefetched[1]
        baseline_recall, failures, search_terms, candidates_list = _run_and_evaluate(
            all_items, batch_pairs, rules, concept_names,
            args.vllm_url, args.model, args.max_concurrent, args.reasoning_effort,
            reps=reps,
            **pf_kwargs,
        )
        best_recall = baseline_recall
        best_rules = list(rules)

        if args.audit_rules:
            _audit_rule_coverage(reps, rules)

        if args.skip_generate or not failures or best_recall >= args.batch_target:
            if not failures:
                print("\n  All gold concepts retrieved on this batch!")
            elif best_recall >= args.batch_target:
                print(
                    f"\n  Retrieval recall {100 * best_recall:.1f}% meets target "
                    f"{100 * args.batch_target:.1f}% — skipping rule generation."
                )
            _start_prefetch(batch_idx + 1, rules)
            save_state(run_dir, batch_idx + 1, rules)
            batch_idx += 1
            continue

        # ── Rule improvement rounds ───────────────────────────────────────
        current_failures = failures
        for round_idx in range(args.rounds):
            print(f"\n{'─' * 50}")
            print(
                f"Rule improvement round {round_idx + 1}/{args.rounds}  |  "
                f"Retrieval recall: {100 * best_recall:.1f}%  |  Rules: {len(best_rules)}"
            )

            if args.max_rules > 0 and len(best_rules) < args.max_rules:
                effective_limit = 99
            else:
                effective_limit = args.max_new_rules
            rule_gen_model = _get_rule_gen_model(batch_idx)
            rule_gen_label = rule_gen_model.split("/")[-1] if "/" in rule_gen_model else rule_gen_model
            print(
                f"\n  Generating rules from failures via {rule_gen_label} "
                f"(limit: {'unlimited' if effective_limit >= 99 else effective_limit}) ..."
            )
            _start_prefetch(batch_idx + 1, best_rules)
            new_rules = generate_rules_via_llm(
                current_failures, best_rules, effective_limit,
                rule_effectiveness=last_rule_effectiveness,
                n_parallel=args.sonnet_n,
                model=rule_gen_model,
            )

            if not new_rules:
                print("  No new rules generated; stopping improvement for this batch.")
                break

            print("\n  Proposed rules:")
            for r in new_rules:
                aw = r.get("applies_when", {})
                cond_parts = []
                if aw.get("sections"):
                    cond_parts.append(f"sections={aw['sections']}")
                cond = f" [{', '.join(cond_parts)}]" if cond_parts else ""
                print(f"    {r['id']}{cond}: {r['rule']}")

            # ── Evaluate all proposed rules in parallel ────────────────────
            if _prefetch_future is not None and not _prefetch_future.done():
                print("  [Prefetch] Waiting for background prefetch to finish ...")
                try:
                    _prefetch_future.result(timeout=600)
                    print("  [Prefetch] Background prefetch complete")
                except Exception as e:
                    print(f"  [Prefetch] Background prefetch failed: {e}")

            accepted_this_round: list[dict] = []

            def _apply_rule(base_rules: list[dict], rule: dict) -> list[dict]:
                """Build a test rule set: replace if 'replaces' is set, else append."""
                replaces_id = rule.get("replaces")
                if replaces_id:
                    clean_rule = {k: v for k, v in rule.items() if k != "replaces"}
                    result = [clean_rule if r["id"] == replaces_id else r for r in base_rules]
                    if all(r["id"] != clean_rule["id"] for r in result):
                        result.append(clean_rule)
                    return result
                return base_rules + [rule]

            # Build baseline retrieval recall lookup
            baseline_retrieval: dict[tuple[str, str], bool] = {}
            for rep, cands in zip(reps, candidates_list):
                key = (rep["span"], rep["section_header"])
                gold_ids = set(rep["gold_concept_ids_all"])
                hit = any(c["concept_id"] in gold_ids for c in cands)
                baseline_retrieval[key] = hit

            # Build all rule-variant reps and concatenate into one mega-batch
            n_reps = len(reps)
            all_test_reps: list[dict] = []
            rule_variant_rules: list[list[dict]] = []
            for rule in new_rules:
                test_rules = _apply_rule(best_rules, rule)
                rule_variant_rules.append(test_rules)
                variant_reps = build_reps_for_batch(all_items, batch_pairs, test_rules)
                all_test_reps.extend(variant_reps)

            n_total_prompts = len(all_test_reps)
            print(
                f"\n  Testing {len(new_rules)} rules in parallel "
                f"({n_total_prompts} total prompts) ..."
            )

            # Phase 1: All LLM calls in one batch (no retrieval)
            all_search_terms, _, _, _ = run_batch(
                all_test_reps, args.vllm_url, args.model,
                args.max_concurrent, args.reasoning_effort,
                search_terms_only=True,
            )

            # Phase 2: Deduplicated batch retrieval
            # Many rule variants produce identical search terms for the same
            # annotation (rule only applies to a subset), so we deduplicate
            # by the frozen set of search terms to avoid redundant encoding.
            t_retr = time.time()
            unique_items: dict[tuple[str, ...], str] = {}  # terms_key → dedup_id
            idx_to_dedup: dict[int, str] = {}  # original_idx → dedup_id
            for i, terms in enumerate(all_search_terms):
                if not terms:
                    continue
                key = tuple(sorted(terms))
                if key not in unique_items:
                    dedup_id = str(len(unique_items))
                    unique_items[key] = dedup_id
                idx_to_dedup[i] = unique_items[key]

            dedup_batch = [
                (dedup_id, list(key))
                for key, dedup_id in unique_items.items()
            ]
            n_total_queries = sum(len(t) for t in all_search_terms if t)
            n_dedup_items = len(dedup_batch)
            n_dedup_queries = sum(len(terms) for _, terms in dedup_batch)
            result_map = rt.snomed_search_batch_items(dedup_batch, top_k=10)

            all_test_candidates = [
                result_map.get(idx_to_dedup.get(i, ""), [])[:15]
                for i in range(n_total_prompts)
            ]
            print(
                f"  Retrieval: {n_total_queries} queries → "
                f"{n_dedup_items} unique ({n_dedup_queries} queries), "
                f"1 batch call ({time.time() - t_retr:.1f}s)"
            )

            # Slice results back per rule variant and score
            for rule_idx, rule in enumerate(new_rules):
                start = rule_idx * n_reps
                end = start + n_reps
                test_reps_slice = all_test_reps[start:end]
                test_cands_slice = all_test_candidates[start:end]
                replaces_tag = f" (replaces {rule['replaces']})" if rule.get("replaces") else ""

                n_fixed = n_broken = 0
                fixed_details: list[dict] = []
                broken_details: list[dict] = []
                for rep, cands in zip(test_reps_slice, test_cands_slice):
                    key = (rep["span"], rep["section_header"])
                    gold_ids = set(rep["gold_concept_ids_all"])
                    hit = any(c["concept_id"] in gold_ids for c in cands)
                    was_hit = baseline_retrieval.get(key, False)
                    if hit and not was_hit:
                        n_fixed += 1
                        fixed_details.append({"span": rep["span"], "section": rep["section_header"]})
                    elif was_hit and not hit:
                        n_broken += 1
                        broken_details.append({"span": rep["span"], "section": rep["section_header"]})

                net = n_fixed - n_broken
                kept = net >= 1
                print(
                    f"  [{rule_idx + 1}/{len(new_rules)}] {rule['id']}{replaces_tag}: "
                    f"+{n_fixed} fixed, -{n_broken} broken, net={net}  "
                    f"{'KEEP' if kept else 'DROP'}  "
                    f"{rule['rule'][:60]}..."
                )

                log_rule_change(
                    run_dir, batch_idx, rule,
                    action="tested_keep" if kept else "tested_drop",
                    n_fixed=n_fixed, n_broken=n_broken,
                    fixed_details=fixed_details, broken_details=broken_details,
                )

                if kept:
                    accepted_this_round.append(rule)

            if not accepted_this_round:
                print(
                    f"\n  All {len(new_rules)} rules tested individually — "
                    f"none had net gain >= 1."
                )
            else:
                # Build combined rule set
                candidate_rules = best_rules[:]
                for rule in accepted_this_round:
                    candidate_rules = _apply_rule(candidate_rules, rule)

                print(
                    f"\n  {len(accepted_this_round)}/{len(new_rules)} rules passed. "
                    f"Evaluating combined recall ..."
                )
                new_recall, new_failures, new_st, new_cands = _run_and_evaluate(
                    all_items, batch_pairs, candidate_rules, concept_names,
                    args.vllm_url, args.model, args.max_concurrent, args.reasoning_effort,
                )
                delta = new_recall - best_recall
                sign = "+" if delta >= 0 else ""

                # Count net fixed/broken vs baseline
                combined_fixed: list[dict] = []
                combined_broken: list[dict] = []
                combined_reps = build_reps_for_batch(all_items, batch_pairs, candidate_rules)
                for rep, cands in zip(combined_reps, new_cands):
                    key = (rep["span"], rep["section_header"])
                    gold_ids = set(rep["gold_concept_ids_all"])
                    hit = any(c["concept_id"] in gold_ids for c in cands)
                    was_hit = baseline_retrieval.get(key, False)
                    if hit and not was_hit:
                        combined_fixed.append({"span": rep["span"], "section": rep["section_header"]})
                    elif was_hit and not hit:
                        combined_broken.append({"span": rep["span"], "section": rep["section_header"]})
                combined_net = len(combined_fixed) - len(combined_broken)

                print(
                    f"  Combined: {100 * best_recall:.1f}% → {100 * new_recall:.1f}%  "
                    f"({sign}{100 * delta:.1f}%)  "
                    f"net={combined_net}"
                )

                if combined_net < 1:
                    print("  Combined net gain < 1 — discarding.")
                    for r in accepted_this_round:
                        log_rule_change(
                            run_dir, batch_idx, r,
                            action="combined_reject",
                            n_fixed=len(combined_fixed), n_broken=len(combined_broken),
                        )
                else:
                    # ── Holdout regression check ──────────────────────
                    holdout_rejected = False
                    if holdout_pairs:
                        print("\n  [Holdout] Evaluating retrieval recall regression ...")
                        h_recall, _, h_st, h_cands = _run_and_evaluate(
                            all_items, holdout_pairs, candidate_rules, concept_names,
                            args.vllm_url, args.model, args.max_concurrent,
                            args.reasoning_effort,
                        )

                        accepted_holdouts = [
                            h for h in holdout_history if h.get("accepted", True)
                        ]
                        if accepted_holdouts:
                            recent = accepted_holdouts[-3:]
                            prev_holdout = min(h["recall"] for h in recent)
                        else:
                            prev_holdout = h_recall
                        holdout_delta = h_recall - prev_holdout

                        holdout_entry = {
                            "batch_idx": batch_idx,
                            "round": round_idx + 1,
                            "recall": round(h_recall, 4),
                        }

                        if holdout_delta < args.holdout_reject_threshold:
                            print(
                                f"  Holdout: {100 * h_recall:.1f}% (prev accepted: "
                                f"{100 * prev_holdout:.1f}%, delta: "
                                f"{100 * holdout_delta:+.1f}%)  — REJECTING "
                                f"(threshold: {100 * args.holdout_reject_threshold:.1f}%)"
                            )
                            holdout_entry["accepted"] = False
                            holdout_history.append(holdout_entry)
                            holdout_rejected = True
                            for r in accepted_this_round:
                                log_rule_change(
                                    run_dir, batch_idx, r,
                                    action="holdout_reject",
                                    n_fixed=len(combined_fixed), n_broken=len(combined_broken),
                                )
                        else:
                            holdout_entry["accepted"] = True
                            holdout_history.append(holdout_entry)
                            trend = " → ".join(
                                f"{100 * h['recall']:.1f}%"
                                for h in holdout_history[-5:]
                                if h.get("accepted", True)
                            )
                            print(f"  Holdout: {100 * h_recall:.1f}%  (trend: {trend})")

                        save_state(run_dir, batch_idx, rules, holdout_history=holdout_history)

                    if not holdout_rejected:
                        last_rule_effectiveness = {
                            "fixed": combined_fixed,
                            "broken": combined_broken,
                            "accepted_rules": [r["id"] for r in accepted_this_round],
                            "net_gain": combined_net,
                        }
                        accepted_labels = []
                        for r in accepted_this_round:
                            lbl = r["id"]
                            if r.get("replaces"):
                                lbl += f"→{r['replaces']}"
                            accepted_labels.append(lbl)
                        print(f"  ACCEPTED [{', '.join(accepted_labels)}] — saving.")
                        for r in accepted_this_round:
                            log_rule_change(
                                run_dir, batch_idx, r,
                                action="combined_accept",
                                n_fixed=len(combined_fixed), n_broken=len(combined_broken),
                                fixed_details=combined_fixed, broken_details=combined_broken,
                            )
                        best_recall = new_recall
                        best_rules = candidate_rules
                        current_failures = new_failures
                        rules = best_rules
                        save_rules(rules, args.rules_file)
                        save_state(run_dir, batch_idx, rules, holdout_history=holdout_history)

                        if args.max_rules > 0 and len(rules) > args.max_rules:
                            print(f"\n  Rule count ({len(rules)}) exceeds --max-rules {args.max_rules}; merging ...")
                            merged = _merge_with_validation(
                                rules, all_pair_keys, all_items, concept_names,
                                args.vllm_url, args.model, args.max_concurrent, args.reasoning_effort,
                                rules_file=args.rules_file,
                                sample_size=args.merge_sample,
                                threshold=args.merge_threshold,
                                holdout_pairs=holdout_pairs or None,
                            )
                            if merged is not rules:
                                rules = merged
                                best_rules = merged
                                save_rules(rules, args.rules_file)
                                save_state(run_dir, batch_idx, rules)

            if not current_failures or best_recall >= 0.95:
                print("  Stopping rounds early (no failures or high recall).")
                break

        # ── Advance decision ──────────────────────────────────────────────
        print(
            f"\n  Batch {batch_idx + 1} complete.  "
            f"Baseline: {100 * baseline_recall:.1f}% → Best: {100 * best_recall:.1f}%  "
            f"Rules: {len(rules)}"
        )
        if args.auto_advance:
            save_state(run_dir, batch_idx + 1, rules)
            batch_idx += 1
        else:
            while True:
                choice = input(
                    f"\n  Advance to batch {batch_idx + 2}? [y = yes / r = retry this batch / q = quit]  "
                ).strip().lower()
                if choice in ("y", "r", "q"):
                    break
            if choice == "q":
                save_state(run_dir, batch_idx + 1, rules)
                print(f"Exiting. Resume with: --resume {run_dir}")
                break
            elif choice == "y":
                save_state(run_dir, batch_idx + 1, rules)
                batch_idx += 1

    print(f"\nDone. Final rule count: {len(rules)}")
    print(f"Resume with: --resume {run_dir}")


if __name__ == "__main__":
    main()
