#!/usr/bin/env python3
"""Iterative rule optimisation loop for NON-ABBREVIATION annotations.

Port of super-dictionary/abbrev_rule_loop.py with key differences:
  - Uses v4.0 structured rules with applies_to.ancestor_concept_ids
  - Subsumption-based rule filtering via SubsumptionIndex.match_rule_applies_to()
  - Failure reports enriched with concept ancestry context
  - Merge/compression uses SNOMED hierarchy LCA for rule abstraction
  - Processes non-abbreviation annotations (complement of abbreviation candidates)

Pipeline per batch:
  1. Build representative items for the batch (span, section pairs)
  2. Inject applicable rules via subsumption matching
  3. Run search+retrieve via vLLM
  4. Compute retrieval recall; build failure report
  5. Send failures to Claude for rule generation
  6. Test each proposed rule individually (fixed - broken)
  7. Accept rules with net >= 1; validate on holdout
  8. Auto-merge when rule count exceeds threshold

Requires:
  - vLLM server running on --vllm-url
  - SNOMED index server running on --index-server
  - snomed_index/subsumption.pkl for subsumption queries
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
sys.path.insert(0, str(REPO_ROOT))

import rulebook.rule_testing as rt  # noqa: E402
from build_abbrev_dictionary_llm import extract_non_abbreviation_items  # noqa: E402
from snomed_subsumption import SubsumptionIndex  # noqa: E402

DEFAULT_RULES_FILE = Path(__file__).parent / "general_rules.json"
DEFAULT_MODEL = "/workspaces/snomed-ct-entity-linking/models/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
DEFAULT_RUNS_DIR = REPO_ROOT / "rulebook" / "runs"
SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
INDEX_DIR = REPO_ROOT / "snomed_index"

# Lazy-loaded: sctid -> list of indexed description texts
_index_terms_by_sctid: dict[int, list[str]] | None = None


def _get_indexed_terms(sctid: int | str, max_terms: int = 10) -> list[str]:
    """Return the indexed description texts for a SNOMED concept."""
    global _index_terms_by_sctid
    if _index_terms_by_sctid is None:
        descs_path = INDEX_DIR / "descriptions.csv"
        if descs_path.exists():
            desc_df = pd.read_csv(descs_path)
            _index_terms_by_sctid = {}
            for _, row in desc_df.iterrows():
                sid = int(row["sctid"])
                _index_terms_by_sctid.setdefault(sid, []).append(str(row["term"]))
        else:
            _index_terms_by_sctid = {}
    return _index_terms_by_sctid.get(int(sctid), [])[:max_terms]


# ---------------------------------------------------------------------------
# Rules I/O
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


# ---------------------------------------------------------------------------
# Run directory management
# ---------------------------------------------------------------------------

def create_run_dir(runs_dir: Path, args: argparse.Namespace) -> Path:
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
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "batch_idx": batch_idx,
        "action": action,
        "rule_id": rule["id"],
        "replaces": rule.get("replaces"),
        "rule_text": rule["rule"],
        "priority": rule.get("priority", 2),
        "applies_to": rule.get("applies_to", {}),
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
    import random as _random
    if frac <= 0:
        return list(all_pair_keys), []
    rng = _random.Random(seed)
    keys = list(all_pair_keys)
    rng.shuffle(keys)
    n_holdout = max(1, int(len(keys) * frac))
    return keys[n_holdout:], keys[:n_holdout]


# ---------------------------------------------------------------------------
# Subsumption-based rule filtering
# ---------------------------------------------------------------------------

_subsumption_idx: SubsumptionIndex | None = None


def _get_subsumption_idx() -> SubsumptionIndex:
    global _subsumption_idx
    if _subsumption_idx is None:
        _subsumption_idx = SubsumptionIndex.load()
    return _subsumption_idx


def rule_applies(
    rule: dict,
    section: str,
    gold_concept_ids: list[int],
) -> bool:
    """Check if a rule applies to an annotation via subsumption.

    A rule without applies_to.ancestor_concept_ids is universal (applies to all).
    A rule with ancestor_concept_ids applies only when at least one gold concept
    is a descendant of at least one of the ancestor IDs.
    Section filtering is also respected if applies_to.sections is set.
    """
    applies_to = rule.get("applies_to", {})

    # Universal rule: no ancestor constraint
    ancestor_ids = applies_to.get("ancestor_concept_ids", [])
    if not ancestor_ids:
        # Still check section filter if present
        rule_sections = applies_to.get("sections")
        if rule_sections and section:
            section_lower = section.lower()
            if not any(s.lower() in section_lower for s in rule_sections):
                return False
        elif rule_sections and not section:
            return False
        return True

    # Concept-scoped rule: check subsumption
    idx = _get_subsumption_idx()
    for gid in gold_concept_ids:
        if idx.match_rule_applies_to(gid, section, applies_to):
            return True
    return False


# ---------------------------------------------------------------------------
# Format rules for prompt injection
# ---------------------------------------------------------------------------

def _format_rules_block(applicable_rules: list[dict]) -> str:
    """Format rules into the search prompt.

    Uses v4.0 structured rule format with concept scope info.
    """
    if not applicable_rules:
        return ""
    sorted_rules = sorted(
        applicable_rules,
        key=lambda r: (r.get("priority", 2), r.get("id", "")),
    )
    lines = [
        "=== ANNOTATION RULES ===",
        "Priority: [P1] override → [P2] standard → [P3] guideline.",
    ]
    for r in sorted_rules:
        p = r.get("priority", 2)
        scope_parts = []
        applies_to = r.get("applies_to", {})
        if applies_to.get("sections"):
            scope_parts.append(f"section: {', '.join(applies_to['sections'])}")
        if applies_to.get("ancestor_concept_ids"):
            scope_parts.append(f"concept scope: {applies_to['ancestor_concept_ids']}")
        scope = f" [when {'; '.join(scope_parts)}]" if scope_parts else ""
        lines.append(f"[P{p}] {r['id']}{scope}: {r['rule']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Search system prompts for non-abbreviation annotations
# ---------------------------------------------------------------------------

GENERAL_SEARCH_SYSTEM = """\
You are a clinical NLP agent. Given a text excerpt from a clinical note with a \
highlighted region, generate search terms for querying a SNOMED CT terminology index.

The highlighted region may be a medical term, procedure name, body part, \
symptom description, finding, or any clinical concept. Generate search terms \
that would retrieve the correct SNOMED CT concept.

Respond with ONLY a JSON object:
{"search_terms": ["term1", "term2", "term3"]}

Generate 1-3 search terms, ordered from most to least likely to match. \
Include the exact span text as one term, plus expanded/alternative forms.\
"""

BASE_SEARCH_RULES = """\
=== ANNOTATION TASK ===

The highlighted text is a clinical concept span. Use the section header and
surrounding context to determine the correct SNOMED CT concept, then generate
search terms that would retrieve it from the SNOMED index.

Key strategies:
- Use the EXACT span text as one search term
- Add the full clinical term if the span is a partial word or shorthand
- Consider the section context (e.g. "Medications" section means drug names)
- For implicit concepts (e.g. a drug name implying its therapeutic use),
  generate terms for what the annotation is actually about
- For multi-word spans, also try component terms and synonyms\
"""


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------

def build_reps_for_batch(
    all_items: list[dict],
    batch_pairs: list[tuple[str, str]],
    rules: list[dict],
) -> list[dict]:
    """Build representative items for a batch with concept-scoped rule injection.

    For each (span, section) pair, picks the instance with the most context,
    then injects only the rules that match via subsumption against the gold
    concept IDs for that group.
    """
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

        # Collect all gold concept IDs for this group
        gold_cids = sorted({g["gold_concept_id"] for g in group})

        # Filter rules by subsumption: only inject rules whose
        # applies_to.ancestor_concept_ids match the gold concepts
        applicable = [r for r in rules if rule_applies(r, section, gold_cids)]
        rules_extra = _format_rules_block(applicable)
        search_rules_text = BASE_SEARCH_RULES + (
            "\n\n" + rules_extra if rules_extra else ""
        )

        reps.append({
            "before": best["before"],
            "span": span,
            "after": best["after"],
            "section_header": section,
            "search_rules_text": search_rules_text,
            "gold_start": best["start"],
            "gold_end": best["end"],
            "instance_count": len(group),
            "gold_concept_ids_all": gold_cids,
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

    Returns (search_terms, candidates_list, timings, elapsed).
    """
    orig_search = rt.SEARCH_SYSTEM
    t0 = time.time()
    try:
        rt.SEARCH_SYSTEM = GENERAL_SEARCH_SYSTEM
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


def print_timing_summary(timings: list[dict], elapsed: float) -> None:
    if not timings:
        return
    search_times = [t.get("search_s", 0) for t in timings]
    retrieve_times = [t.get("retrieve_s", 0) for t in timings]
    avg_search = sum(search_times) / len(search_times)
    avg_retrieve = sum(retrieve_times) / len(retrieve_times)
    print(
        f"  Timing: {len(timings)} items, elapsed={elapsed:.1f}s, "
        f"avg_search={avg_search:.2f}s, avg_retrieve={avg_retrieve:.2f}s"
    )


# ---------------------------------------------------------------------------
# Retrieval recall computation
# ---------------------------------------------------------------------------

def compute_retrieval_recall(
    reps: list[dict],
    candidates_list: list[list[dict]],
) -> tuple[float, int, int, dict[str, dict]]:
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


# ---------------------------------------------------------------------------
# Failure reporting with concept ancestry
# ---------------------------------------------------------------------------

def build_failure_report(
    reps: list[dict],
    search_terms_list: list[list[str]],
    candidates_list: list[list[dict]],
    concept_names: dict[int, tuple[str, str]],
) -> list[dict]:
    """Build a structured failure report enriched with concept ancestry.

    Each failure includes the gold concept's parent concepts and hierarchy
    to help the rule generator understand the concept scope.
    """
    idx = _get_subsumption_idx()
    failures: list[dict] = []

    for rep, terms, cands in zip(reps, search_terms_list, candidates_list):
        gold_ids = set(rep["gold_concept_ids_all"])
        if any(c["concept_id"] in gold_ids for c in cands):
            continue

        gold_names = {
            gid: concept_names.get(gid, ("Unknown", "unknown"))
            for gid in gold_ids
        }

        # Build ancestry info per gold concept
        ancestry_info: list[dict] = []
        for gid in gold_ids:
            name, hierarchy = concept_names.get(gid, ("?", "?"))
            parents = idx.get_parents(gid)
            parent_strs = []
            for p in sorted(parents):
                pname = idx.get_name(p)
                parent_strs.append(f"{p} ({pname})")
            ancestry_info.append({
                "concept_id": gid,
                "name": name,
                "hierarchy": hierarchy,
                "parents": parent_strs[:3],
            })

        before_text = rep.get("before", "")
        after_text = rep.get("after", "")
        context_excerpt = f"...{before_text}>>>{rep['span']}<<<{after_text}..."

        failures.append({
            "span": rep["span"],
            "section": rep["section_header"],
            "instance_count": rep["instance_count"],
            "gold_ids": list(gold_ids),
            "gold_names": {str(k): v for k, v in gold_names.items()},
            "ancestry": ancestry_info,
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
            c["concept_name"] for c in f["top_candidates"][:3]
        )
        print(f"\n  ── {f['span']!r} | {f['section']} (n={f['instance_count']})")
        print(f"     Gold:     {gold_str}")
        if f.get("ancestry"):
            for a in f["ancestry"][:2]:
                parent_str = ", ".join(a["parents"][:2]) if a["parents"] else "none"
                print(f"     Ancestry: {a['hierarchy']} → parents: {parent_str}")
        print(f"     Search:   {f['search_terms']}")
        print(f"     Top cands: {cands_str}")


# ---------------------------------------------------------------------------
# Rule coverage audit
# ---------------------------------------------------------------------------

def _audit_rule_coverage(reps: list[dict], rules: list[dict]) -> None:
    n = len(reps)
    if n == 0:
        return

    counts: dict[str, int] = defaultdict(int)
    for rep in reps:
        section = rep["section_header"]
        gold_cids = rep["gold_concept_ids_all"]
        for r in rules:
            if rule_applies(r, section, gold_cids):
                counts[r["id"]] += 1

    narrow_threshold = 0.10
    narrow: list[str] = []
    print(f"\n  Rule coverage audit ({n} pairs):")
    print(f"  {'ID':6s}  {'fires':>6}  {'%':>6}  {'P':>2}  note")
    for r in rules:
        rid = r["id"]
        sc = counts.get(rid, 0)
        pct = sc / n
        p = r.get("priority", 2)
        scope = r.get("applies_to", {}).get("ancestor_concept_ids", [])
        scope_str = f" [scope: {scope}]" if scope else ""
        note = ""
        if sc == 0:
            note = f"NEVER FIRED{scope_str}"
        elif pct < narrow_threshold:
            note = f"narrow ({100 * pct:.0f}%){scope_str}"
            narrow.append(rid)
        else:
            note = scope_str.strip()
        print(f"  {rid:6s}  {sc:6d}  {100 * pct:5.1f}%  P{p}  {note}")

    if narrow:
        print(
            f"\n  Narrow rules ({len(narrow)}): {', '.join(narrow)}\n"
            "  These fire for <10% of pairs — concept-scoped rules are expected to be narrow."
        )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

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
# LLM rule generation
# ---------------------------------------------------------------------------

RULE_GEN_SYSTEM = """\
You are a clinical NLP expert improving search term generation for a SNOMED CT
annotation pipeline that handles non-abbreviation clinical text spans.

Pipeline overview:
  1. Search: an LLM generates search terms from the span + clinical context.
  2. Retrieval: a hybrid FAISS+BM25 index returns the top candidate concepts.
  3. Scoring: check if the gold concept appears in the top K candidates.

You are ONLY working on step 1 — improving search terms so the correct (gold)
concept appears in the retrieved candidates. Every failure is a RETRIEVAL MISS.

For each failure you will see:
  - The span text, section header, and surrounding clinical context
  - The gold concept, its hierarchy, and its parent concepts in SNOMED
  - The search terms the LLM actually generated (which failed)
  - What was retrieved instead (top candidates)

The context excerpt shows the span in its original clinical note with >>>markers<<<.

Your job: write rules that guide the search-term LLM to generate better terms.
Think about WHY the generated search terms missed and write rules to fix the pattern.

Response format — a single JSON object:

{
  "rules": [
    {
      "id": "R###",
      "replaces": null,
      "priority": 2,
      "rule": "Concise rule text for the search-term LLM.",
      "applies_to": {
        "ancestor_concept_ids": [12345],
        "sections": ["section name"]
      }
    }
  ],
  "feedback": {
    "index_gaps": null,
    "context": null,
    "other": null
  }
}

Rules guidance:
  - CRITICAL: Each rule MUST be under 100 words. Be telegraphic.
  - Write GENERIC principles for CLASSES of annotations when possible.
  - You MAY scope rules to specific concept hierarchies using
    ancestor_concept_ids — use the SNOMED parent/ancestor IDs shown in the
    failure report. This is useful when a rule only applies to a specific
    category (e.g. body structures, procedures, medications).
  - Prefer GENERAL rules that apply broadly. Only use ancestor_concept_ids
    when the pattern is genuinely specific to that concept hierarchy.
  - Omit applies_to entirely (or use {}) for universal rules.
  - Focus on: section-aware disambiguation, synonym generation, hierarchy
    selection, implicit concept mapping, multi-word phrase strategies.
  - Prefer replacing/improving existing rules over adding new ones.\
"""


async def _sonnet_call(
    system_prompt: str,
    user_prompt: str,
    model: str = "claude-sonnet-4-6",
) -> str:
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
    import asyncio
    import os
    saved_cc = os.environ.pop("CLAUDECODE", None)
    try:
        return asyncio.run(_sonnet_call(system_prompt, user_prompt, model=model))
    finally:
        if saved_cc is not None:
            os.environ["CLAUDECODE"] = saved_cc


def _parse_sonnet_rules(raw: str) -> tuple[list[dict], dict]:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        print(f"  WARNING: Could not parse LLM output:\n{raw[:400]}")
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
    """Use an LLM to propose new rules from observed failures.

    Enriches the failure context with concept ancestry information so
    the rule generator can propose concept-scoped rules with appropriate
    ancestor_concept_ids.
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
        indexed_terms: list[str] = []
        for gid in f["gold_ids"]:
            indexed_terms.extend(_get_indexed_terms(gid, max_terms=6))
        indexed_str = ", ".join(indexed_terms[:8]) if indexed_terms else "(not in index)"

        cands_str = " | ".join(
            c["concept_name"] for c in f["top_candidates"][:4]
        )
        context_line = f"  Context: {f['context']}\n" if f.get("context") else ""

        # Ancestry info for concept-scoped rule generation
        ancestry_lines = ""
        if f.get("ancestry"):
            ancestry_parts = []
            for a in f["ancestry"]:
                parent_str = ", ".join(a["parents"][:3]) if a["parents"] else "none"
                ancestry_parts.append(
                    f"    Hierarchy: {a['hierarchy']}  Parents: {parent_str}"
                )
            ancestry_lines = "\n".join(ancestry_parts) + "\n"

        failure_lines.append(
            f"Span={f['span']!r}  Section={f['section']!r}  n={f['instance_count']}\n"
            f"{context_line}"
            f"  Gold: {gold_str}\n"
            f"{ancestry_lines}"
            f"  Gold indexed terms: {indexed_str}\n"
            f"  Search terms generated: {f['search_terms']}\n"
            f"  Retrieved instead: {cands_str}"
        )

    existing_text = ""
    if existing_rules:
        summaries = []
        for r in existing_rules:
            rule_text = r["rule"]
            if len(rule_text) > 150:
                rule_text = rule_text[:150] + "..."
            scope = r.get("applies_to", {}).get("ancestor_concept_ids", [])
            scope_str = f" [scope: {scope}]" if scope else ""
            summaries.append(f"  {r['id']}{scope_str}: {rule_text}")
        existing_text = "\n\nEXISTING RULES (do not duplicate):\n" + "\n".join(summaries)

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
                f"    - span={span!r}  section={sec!r}  (x{n})" if n > 1
                else f"    - span={span!r}  section={sec!r}"
                for (span, sec), n in counts.most_common()
            ]

        if fixed:
            parts.append("  Predictions FIXED by the last rules:")
            parts.extend(_dedup_counts(fixed)[:10])
        if broken:
            parts.append("  Predictions BROKEN by the last rules (avoid repeating):")
            parts.extend(_dedup_counts(broken)[:10])
        parts.append(
            "  Use this feedback to understand what types of rules are effective."
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
        f"and propose {limit_text} (start IDs from R{next_id_base:03d}). "
        f"You may also REPLACE existing rules by setting \"replaces\": \"R###\" — "
        f"replacements don't count toward the limit.\n"
        f"For each failure, compare the search terms generated vs the gold "
        f"indexed terms — the gap shows what the LLM should have generated.\n"
        f"Use the ancestry info (hierarchy + parents) to decide if a rule should be "
        f"concept-scoped (via ancestor_concept_ids) or universal.\n\n"
        "RETRIEVAL MISSES:\n" + "\n\n".join(failure_lines)
        + existing_text
        + effectiveness_text
        + "\n\nReturn ONLY the JSON object with 'rules' and 'feedback' keys."
    )

    n_parallel = max(1, n_parallel)
    model_label = model.split("/")[-1] if "/" in model else model

    if n_parallel == 1:
        t0 = time.time()
        try:
            raw = _run_sonnet(RULE_GEN_SYSTEM, user_prompt, model=model)
        except Exception as e:
            print(f"  ERROR calling {model_label} for rule generation: {e}")
            return []
        print(f"  {model_label} call took {time.time() - t0:.1f}s")
        new_rules, feedback = _parse_sonnet_rules(raw)
    else:
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
                all_feedback = feedback_i

        for idx, rule in enumerate(all_rules):
            rule["id"] = f"R{next_id_base + idx:03d}"
        new_rules = all_rules
        feedback = all_feedback

    if not new_rules:
        print("  No rules generated.")
        return []

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
        print("\n  Feedback:")
        for key, text in fb_items.items():
            print(f"    [{key}] {text}")

    return new_rules


# ---------------------------------------------------------------------------
# Rule merge/compression with hierarchy-based abstraction
# ---------------------------------------------------------------------------

RULE_MERGE_SYSTEM = """\
You are a clinical NLP expert maintaining a rule set for SNOMED CT annotation.

You will be given a list of rules that have accumulated over multiple improvement rounds.
Many rules may be redundant, overlapping, or express the same principle differently.
Some rules are concept-scoped (applies_to.ancestor_concept_ids) while others are universal.

Your task: merge and compress the rule list into a smaller, non-redundant set.

Merging guidelines:
  - Combine rules that address the same root pattern into one concise rule.
  - HIERARCHY-BASED MERGING: If multiple rules target different concepts that share
    a common ancestor in the SNOMED hierarchy, merge them with ancestor_concept_ids
    set to the common ancestor. The HIERARCHY CONTEXT below shows these relationships.
  - A rule scoped to a very high-level hierarchy node (depth 1-2) is effectively
    universal — consider dropping ancestor_concept_ids entirely.
  - Each merged rule MUST be under 100 words.
  - Preserve all distinct applies_to conditions; if merged rules have different
    concept scopes, use the LCA (least common ancestor) or split into two rules.
  - Do NOT drop logic that addresses a genuinely distinct failure pattern.
  - DEMOTE priority aggressively: P1 should be reserved for hard overrides.
    Most rules should be P2. When merging, demote P1 → P2 unless genuinely needed.
  - Remove duplicate or near-duplicate rules entirely.
  - Renumber IDs sequentially from R001.
  - Aim to reduce count by 20-30%.

Return ONLY a JSON array of the merged rules:
[
  {
    "id": "R###",
    "rule": "Merged rule text.",
    "priority": 2,
    "applies_to": {
      "ancestor_concept_ids": [12345],
      "sections": ["section name"]
    }
  }
]\
"""


def _build_hierarchy_context(rules: list[dict]) -> str:
    """Build hierarchy context for the merge prompt showing LCA relationships."""
    idx = _get_subsumption_idx()

    # Collect all ancestor_concept_ids used by concept-scoped rules
    all_scope_ids: set[int] = set()
    for r in rules:
        scope = r.get("applies_to", {}).get("ancestor_concept_ids", [])
        all_scope_ids.update(scope)

    if not all_scope_ids:
        return ""

    lines = [
        "\nHIERARCHY CONTEXT — SNOMED relationships between concept-scoped rules:",
    ]

    for cid in sorted(all_scope_ids):
        name = idx.get_name(cid)
        parents = idx.get_parents(cid)
        parent_strs = [f"{p} ({idx.get_name(p)})" for p in sorted(parents)][:3]
        lines.append(f"  {cid}: {name}  parents: {', '.join(parent_strs) or 'none'}")

    # Show potential merge targets (LCA of groups)
    if len(all_scope_ids) >= 2:
        lca_results = idx.find_lca_with_depth(list(all_scope_ids), min_depth=2)
        if lca_results:
            lines.append("\n  Potential merge targets (common ancestors):")
            for anc_id, anc_name, depth in lca_results[:8]:
                lines.append(f"    {anc_id}: {anc_name} (depth={depth})")

    return "\n".join(lines)


def merge_rules_via_llm(
    rules: list[dict],
    rules_file: Path | None = None,
    model: str = "claude-sonnet-4-6",
) -> list[dict]:
    """Merge redundant rules with hierarchy-based abstraction."""
    if len(rules) < 2:
        print("  Nothing to merge (fewer than 2 rules).")
        return rules

    if rules_file is not None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = rules_file.with_stem(f"{rules_file.stem}_backup_{ts}")
        backup.write_text(json.dumps({"rules": rules}, indent=2))
        print(f"  Backup saved → {backup}")

    hierarchy_context = _build_hierarchy_context(rules)
    rules_text = json.dumps(rules, indent=2)
    target_n = max(int(len(rules) * 0.75), len(rules) - 8)
    user_prompt = (
        f"IMPORTANT: Your entire response must be ONLY a JSON array — no analysis, "
        f"no explanation, no preamble. Start your response with [ and end with ].\n\n"
        f"Merge and compress these {len(rules)} rules into ~{target_n} rules "
        f"(20-25% reduction). Do NOT reduce by more than 35%.\n"
        f"{hierarchy_context}\n\n"
        f"CURRENT RULES:\n{rules_text}\n\n"
        "OUTPUT: A JSON array of the merged rules. Nothing else."
    )

    def _extract_json_array(text: str) -> re.Match | None:
        code_block = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
        if code_block:
            return code_block
        for m in reversed(list(re.finditer(r"\[[\s\S]*?\](?=\s*$|\s*```)", text))):
            try:
                parsed = json.loads(m.group())
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    return m
            except json.JSONDecodeError:
                continue
        return re.search(r"\[[\s\S]*\]", text)

    model_label = model.split("/")[-1] if "/" in model else model
    print(f"  Calling {model_label} for merge ...")
    t0 = time.time()
    try:
        raw = _run_sonnet(RULE_MERGE_SYSTEM, user_prompt, model=model)
    except Exception as e:
        print(f"  ERROR calling {model_label} for rule merging: {e}")
        return rules
    print(f"  Sonnet merge call took {time.time() - t0:.1f}s")

    match = _extract_json_array(raw)
    if not match:
        print(f"  WARNING: Could not parse merge output:\n{raw[:200]}")
        return rules

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
    merge_model: str = "claude-sonnet-4-6",
) -> list[dict]:
    import random

    merged = merge_rules_via_llm(rules, rules_file, model=merge_model)
    if merged is rules:
        return rules

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
                f"  Merge REJECTED on holdout — {100 * h_delta:.1f}% drop. "
                f"Keeping original {len(rules)} rules."
            )
            return rules

    print(f"  Merge ACCEPTED ({len(rules)} → {len(merged)} rules).")
    return merged


# ---------------------------------------------------------------------------
# Main loop
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
    parser.add_argument("--context-before", type=int, default=100)
    parser.add_argument("--context-after", type=int, default=50)
    parser.add_argument(
        "--holdout-frac", type=float, default=0.1,
        help="Fraction of (span, section) pairs to hold out (default: 0.1)",
    )
    parser.add_argument("--holdout-seed", type=int, default=42)
    parser.add_argument("--audit-rules", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--auto-advance", action="store_true")
    parser.add_argument("--max-new-rules", type=int, default=5)
    parser.add_argument("--max-rules", type=int, default=25,
                        help="Auto-merge when rule count exceeds this")
    parser.add_argument("--holdout-reject-threshold", type=float, default=-0.03)
    parser.add_argument("--batch-target", type=float, default=0.65)
    parser.add_argument("--merge", action="store_true",
                        help="Run a merge pass and exit")
    parser.add_argument("--merge-sample", type=int, default=30)
    parser.add_argument("--merge-threshold", type=float, default=-0.02)
    parser.add_argument(
        "--index-server", type=str, default="http://127.0.0.1:8421",
        help="URL of the SNOMED index server",
    )
    parser.add_argument("--sonnet-n", type=int, default=1,
                        help="Number of parallel LLM calls per rule-gen step")
    parser.add_argument(
        "--first-stage-model", default="claude-sonnet-4-6",
        help="Claude model for the first N batches",
    )
    parser.add_argument("--first-stage-batches", type=int, default=3)
    parser.add_argument(
        "--remaining-model", default="claude-sonnet-4-6",
        help="Claude model after the first stage",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume a previous run from its directory",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
    )
    args = parser.parse_args()

    rt._index_server_url = args.index_server

    # ------------------------------------------------------------------
    # Pre-flight: verify vLLM server
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
        sys.exit(1)

    # Pre-flight: verify index server
    try:
        req = urllib.request.Request(f"{args.index_server}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5):
            print(f"Index server OK at {args.index_server}")
    except (urllib.error.URLError, OSError) as e:
        print(f"ERROR: Cannot reach SNOMED index server at {args.index_server}")
        print(f"  {e}")
        sys.exit(1)

    # Pre-flight: load subsumption index
    print("Loading SNOMED subsumption index ...")
    sidx = _get_subsumption_idx()
    print(f"  {len(sidx.ancestors):,} concepts with ancestry data")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data from {args.split_dir} ...")
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
    # Extract non-abbreviation items
    # ------------------------------------------------------------------
    print("\nExtracting non-abbreviation annotations ...")
    all_items = extract_non_abbreviation_items(
        ann_df, notes_df, args.context_before, args.context_after,
    )
    print(f"  {len(all_items):,} non-abbreviation instances extracted")

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for item in all_items:
        pair_counts[(item["span"], item["section_header"])] += 1
    all_pairs_sorted = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    all_pair_keys = [k for k, _ in all_pairs_sorted]

    n_pairs = len(all_pair_keys)
    print(f"  {n_pairs:,} unique (span, section) pairs")

    # Holdout split
    holdout_pairs: list[tuple[str, str]] = []
    holdout_history: list[dict] = []
    if not args.resume and args.holdout_frac > 0:
        train_pair_keys, holdout_pairs = _split_holdout(
            all_pair_keys, args.holdout_frac, args.holdout_seed,
        )
    else:
        train_pair_keys = list(all_pair_keys)

    n_train = len(train_pair_keys)
    n_batches = (n_train + args.batch_size - 1) // args.batch_size
    if holdout_pairs:
        print(f"  Holdout: {len(holdout_pairs)} pairs  |  "
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
            print(f"\nMerging {len(rules)} rules (with validation sample={args.merge_sample}) ...")
            merged = _merge_with_validation(
                rules, all_pair_keys, all_items, concept_names,
                args.vllm_url, args.model, args.max_concurrent, args.reasoning_effort,
                rules_file=args.rules_file,
                sample_size=args.merge_sample,
                threshold=args.merge_threshold,
                holdout_pairs=holdout_pairs or None,
                holdout_threshold=args.holdout_reject_threshold,
                merge_model=args.remaining_model,
            )
            if merged is not rules:
                save_rules(merged, args.rules_file)
            print("Done.")
            return
        run_dir = create_run_dir(args.runs_dir, args)
        batch_idx = 0
        save_state(run_dir, batch_idx, rules,
                   holdout_pairs=holdout_pairs, holdout_history=holdout_history)
        print(f"\nRun directory: {run_dir}")

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    def _get_rule_gen_model(batch_idx: int) -> str:
        if batch_idx < args.first_stage_batches:
            return args.first_stage_model
        return args.remaining_model

    # ------------------------------------------------------------------
    # Interactive batch loop
    # ------------------------------------------------------------------
    last_rule_effectiveness: dict | None = None

    # Next-batch prefetch
    import concurrent.futures as _cf
    _prefetch_executor = _cf.ThreadPoolExecutor(max_workers=1)
    _prefetch_future: _cf.Future | None = None
    _prefetch_key: tuple[int, str] | None = None

    def _rules_fingerprint(rules: list[dict]) -> str:
        return json.dumps(rules, sort_keys=True)

    def _prefetch_next_batch(
        next_batch_idx: int, rules: list[dict],
    ) -> tuple[list[list[str]], list[list[dict]]]:
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
        print(f"  [Prefetch] Started search+retrieve for batch {next_idx + 1}")

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
            print(f"  [Prefetch] Using cached results for batch {batch_idx + 1}")
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
            f"instances {batch_start + 1}-{batch_start + len(batch_pairs)})  "
            f"Rules: {len(rules)}  |  {stage_label}  |  {run_dir.name}"
        )
        print(f"{'=' * 70}")

        reps = build_reps_for_batch(all_items, batch_pairs, rules)

        # Auto-merge if too many rules
        if args.max_rules > 0 and len(rules) > args.max_rules:
            print(f"\n  Rule count ({len(rules)}) exceeds --max-rules {args.max_rules}; merging ...")
            merged = _merge_with_validation(
                rules, all_pair_keys, all_items, concept_names,
                args.vllm_url, args.model, args.max_concurrent, args.reasoning_effort,
                rules_file=args.rules_file,
                sample_size=args.merge_sample,
                threshold=args.merge_threshold,
                holdout_pairs=holdout_pairs or None,
                holdout_threshold=args.holdout_reject_threshold,
                merge_model=_get_rule_gen_model(batch_idx),
            )
            if merged is not rules:
                rules = merged
                save_rules(rules, args.rules_file)
                save_state(run_dir, batch_idx, rules)
            reps = build_reps_for_batch(all_items, batch_pairs, rules)

        # Baseline run
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

        # Rule improvement rounds
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
                scope = r.get("applies_to", {})
                scope_parts = []
                if scope.get("ancestor_concept_ids"):
                    scope_parts.append(f"scope={scope['ancestor_concept_ids']}")
                if scope.get("sections"):
                    scope_parts.append(f"sections={scope['sections']}")
                cond = f" [{', '.join(scope_parts)}]" if scope_parts else ""
                print(f"    {r['id']}{cond}: {r['rule']}")

            # Wait for prefetch if needed
            if _prefetch_future is not None and not _prefetch_future.done():
                print("  [Prefetch] Waiting for background prefetch ...")
                try:
                    _prefetch_future.result(timeout=600)
                    print("  [Prefetch] Background prefetch complete")
                except Exception as e:
                    print(f"  [Prefetch] Background prefetch failed: {e}")

            accepted_this_round: list[dict] = []

            def _apply_rule(base_rules: list[dict], rule: dict) -> list[dict]:
                replaces_id = rule.get("replaces")
                if replaces_id:
                    clean_rule = {k: v for k, v in rule.items() if k != "replaces"}
                    result = [clean_rule if r["id"] == replaces_id else r for r in base_rules]
                    if all(r["id"] != clean_rule["id"] for r in result):
                        result.append(clean_rule)
                    return result
                return base_rules + [rule]

            # Baseline retrieval lookup
            baseline_retrieval: dict[tuple[str, str], bool] = {}
            for rep, cands in zip(reps, candidates_list):
                key = (rep["span"], rep["section_header"])
                gold_ids = set(rep["gold_concept_ids_all"])
                hit = any(c["concept_id"] in gold_ids for c in cands)
                baseline_retrieval[key] = hit

            # Build all rule-variant reps and run in one mega-batch
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

            # Phase 1: All LLM calls in one batch (search only)
            all_search_terms, _, _, _ = run_batch(
                all_test_reps, args.vllm_url, args.model,
                args.max_concurrent, args.reasoning_effort,
                search_terms_only=True,
            )

            # Phase 2: Deduplicated batch retrieval
            t_retr = time.time()
            unique_items: dict[tuple[str, ...], str] = {}
            idx_to_dedup: dict[int, str] = {}
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

            # Score each rule variant
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
                    f"({sign}{100 * delta:.1f}%)  net={combined_net}"
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
                    # Holdout regression check
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
                                f"  Holdout: {100 * h_recall:.1f}% (prev: "
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
                                holdout_threshold=args.holdout_reject_threshold,
                                merge_model=_get_rule_gen_model(batch_idx),
                            )
                            if merged is not rules:
                                rules = merged
                                best_rules = merged
                                save_rules(rules, args.rules_file)
                                save_state(run_dir, batch_idx, rules)

            if not current_failures or best_recall >= 0.95:
                print("  Stopping rounds early (no failures or high recall).")
                break

        # Advance
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
                    f"\n  Advance to batch {batch_idx + 2}? [y = yes / r = retry / q = quit]  "
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
