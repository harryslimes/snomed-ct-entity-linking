#!/usr/bin/env python3
"""
Evolutionary prompt optimization for SNOMED CT span discovery.

Uses a genetic algorithm to evolve the LLM discovery prompt by:
  1. Representing prompts as genomes (enabled rules + phrasing variants)
  2. Evaluating fitness = gold-missed span recall - false positive penalty
  3. Selecting top performers, mutating/crossbreeding, repeating

Evaluation runs discovery on a subset of notes for speed (~10s per eval),
then validates the best prompt on the full 60-note holdout.

Usage:
    python scripts/evolve_discovery_prompt.py \
        --vllm-url http://localhost:8000 \
        --eval-notes 15 --population 6 --generations 8
"""

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.llm_rerank_test import _call_vllm_batch, _make_vllm_client

# ---------------------------------------------------------------------------
# Gene pool: rule phrasings and system prompt frames
# ---------------------------------------------------------------------------

# Each rule has a category and multiple phrasing variants.
# A genome picks one phrasing per rule (or disables the rule).
RULE_BANK = {
    "abbreviation": [
        "Find medical abbreviations NOT already in [brackets]. For each, output the abbreviation and its expansion separated by '='. Example: EVD = External Ventricular Drain",
        "Identify clinical abbreviations (like WNL, INR, CXR, ABG, tPA, CVVH) that are not bracketed. Output as: ABBREV = Full Name",
        "Look for unbracketed medical shorthand/acronyms. Write each as ABBREV = expansion so we can look up the SNOMED term from the expansion.",
    ],
    "qualifier": [
        "Find standalone result qualifiers: words like 'normal', 'negative', 'mild', 'moderate', 'severe', 'elevated', 'increased', 'unchanged' when they describe a test result or finding severity.",
        "Annotate severity/result words (normal, negative, mild, severe, elevated, small, increased, unchanged) when used to describe clinical findings or test outcomes.",
        "Look for single-word clinical qualifiers not in brackets: negative, normal, mild, moderate, severe, elevated, increased, small, unchanged. These describe finding severity or test results.",
    ],
    "laterality": [
        "Find 'right' or 'left' when used as laterality for a body part or finding (e.g. right leg, left pleural). Annotate only the laterality word.",
        "Annotate laterality words 'right'/'left' when they modify a body structure or clinical finding. Do not annotate 'right' or 'left' used in other senses.",
    ],
    "action_verb": [
        "Find medication/device action verbs: 'started' (new med), 'held' (med stopped), 'removed' (device taken out), 'placed' (device inserted).",
        "Annotate action words describing treatment changes: started, held, placed, removed — when referring to medications or medical devices.",
    ],
    "procedure_name": [
        "Find procedure/test names not bracketed: Physical Exam, Urine Culture, Gram Stain, Blood Culture, Review of Systems, and similar.",
        "Annotate clinical procedure and test names that are not yet bracketed, such as culture names, imaging studies, and exam types.",
    ],
    "clinical_phrase": [
        "Find multi-word clinical phrases not bracketed: 'shortness of breath', 'Ambulatory - Independent', 'follow up', etc.",
        "Annotate unbracketed clinical status phrases like 'shortness of breath', 'Ambulatory - Independent', 'follow up', 'suicidality'.",
    ],
    "status_qualifier": [
        "Find clinical course words: 'resolving', 'stable', 'worsening', 'improving' — describing how a condition is changing.",
        "Annotate words describing clinical trajectory: resolving, improving, worsening, stable, unchanged.",
    ],
}

RULE_NAMES = list(RULE_BANK.keys())

# System prompt frame variants
FRAME_VARIANTS = [
    # 0: Minimal
    """\
You are a clinical NER annotator. The note below has existing annotations in [brackets]. \
Find clinical entities that are NOT already bracketed.

{rules}

Output one entity per line using EXACT text from the note. \
For abbreviations use: ABBREV = Expansion. \
If nothing is missing, output NONE.""",

    # 1: More structured
    """\
You are annotating a hospital discharge note for SNOMED CT entity linking. \
Text in [brackets] is already annotated. Your task: find MISSING clinical entities.

Rules for what to annotate:
{rules}

Do NOT annotate: medication/drug names, lab numbers, vital sign values, \
dates, section headers, or anything already in [brackets].

Output format: one entity per line, exact text from the note. \
For abbreviations: ABBREV = Full Expansion. \
If nothing to add: NONE""",

    # 2: Concise + examples
    """\
Find unannotated clinical entities in a discharge note. [Brackets] = already done.

{rules}

Skip: drug names, lab numbers, vitals, dates, headers, bracketed text.

Format: one per line, exact text. Abbreviations: ABBREV = Expansion. Nothing found: NONE""",

    # 3: Task-focused
    """\
A dictionary-based system already annotated clinical entities in this discharge note \
(shown in [brackets]). It missed some. Find what it missed.

{rules}

Do NOT re-annotate bracketed text. Do NOT annotate medications, lab numbers, or dates.

One entity per line, exact text from the note. \
Abbreviations: write as ABBREV = Full Name. \
Nothing missing: NONE""",
]


class Genome:
    """A prompt configuration = frame + rule enables + phrasing indices."""

    def __init__(self, frame_idx=0, enabled=None, phrasings=None):
        self.frame_idx = frame_idx
        # Which rules are enabled (dict: rule_name -> bool)
        self.enabled = enabled or {r: True for r in RULE_NAMES}
        # Which phrasing variant per rule (dict: rule_name -> int)
        self.phrasings = phrasings or {r: 0 for r in RULE_NAMES}
        self.fitness = None
        self.metrics = None

    def build_prompt(self) -> str:
        """Assemble the system prompt from this genome."""
        rule_lines = []
        for i, name in enumerate(RULE_NAMES):
            if not self.enabled[name]:
                continue
            pidx = self.phrasings[name] % len(RULE_BANK[name])
            rule_lines.append(f"- {RULE_BANK[name][pidx]}")

        rules_text = "\n".join(rule_lines) if rule_lines else "- Find any clinical entities not already bracketed."
        frame = FRAME_VARIANTS[self.frame_idx % len(FRAME_VARIANTS)]
        return frame.format(rules=rules_text)

    def mutate(self, rate=0.3):
        """Random mutations."""
        g = copy.deepcopy(self)
        # Toggle a rule
        if random.random() < rate:
            rule = random.choice(RULE_NAMES)
            g.enabled[rule] = not g.enabled[rule]
        # Change a phrasing
        if random.random() < rate:
            rule = random.choice(RULE_NAMES)
            n_variants = len(RULE_BANK[rule])
            if n_variants > 1:
                g.phrasings[rule] = (g.phrasings[rule] + random.randint(1, n_variants - 1)) % n_variants
        # Change frame
        if random.random() < rate * 0.5:
            g.frame_idx = (g.frame_idx + random.randint(1, len(FRAME_VARIANTS) - 1)) % len(FRAME_VARIANTS)
        g.fitness = None
        g.metrics = None
        return g

    @staticmethod
    def crossover(a, b):
        """Crossover: take frame from a, mix rules from both."""
        g = Genome(
            frame_idx=a.frame_idx if random.random() < 0.5 else b.frame_idx,
            enabled={r: (a.enabled[r] if random.random() < 0.5 else b.enabled[r]) for r in RULE_NAMES},
            phrasings={r: (a.phrasings[r] if random.random() < 0.5 else b.phrasings[r]) for r in RULE_NAMES},
        )
        return g

    def signature(self) -> str:
        """Short string identifying this genome."""
        en = "".join("1" if self.enabled[r] else "0" for r in RULE_NAMES)
        ph = "".join(str(self.phrasings[r]) for r in RULE_NAMES)
        return f"f{self.frame_idx}_e{en}_p{ph}"

    def __repr__(self):
        return f"Genome({self.signature()}, fitness={self.fitness})"


def load_eval_data(notes_csv, annotations_csv, kiri_pred_csv, train_size, n_eval_notes):
    """Load data needed for fitness evaluation."""
    notes_df = pd.read_csv(notes_csv)
    notes_df["note_id"] = notes_df["note_id"].astype(str)
    note_texts = dict(zip(notes_df["note_id"], notes_df["text"]))

    kiri_pred = pd.read_csv(kiri_pred_csv, dtype={"concept_id": int})
    kiri_pred["note_id"] = kiri_pred["note_id"].astype(str)

    gold = pd.read_csv(annotations_csv, dtype={"concept_id": int})
    gold["note_id"] = gold["note_id"].astype(str)
    gold["start"] = gold["start"].astype(int)
    gold["end"] = gold["end"].astype(int)

    # Reproduce the split
    ids = list(notes_df["note_id"])
    np.random.seed(12345)
    np.random.shuffle(ids)
    test_ids = ids[train_size:]
    if n_eval_notes and n_eval_notes < len(test_ids):
        test_ids = test_ids[:n_eval_notes]

    # Build KIRI intervals and gold-missed spans per note
    existing_per_note = {}
    for nid in test_ids:
        preds = kiri_pred[kiri_pred["note_id"] == nid]
        existing_per_note[nid] = list(zip(preds["start"].astype(int), preds["end"].astype(int)))

    test_gold = gold[gold["note_id"].isin(test_ids)]
    gold_missed_per_note = {}  # note_id -> list of (start, end, span_text_lower)
    for nid in test_ids:
        missed = []
        note_gold = test_gold[test_gold["note_id"] == nid]
        for _, grow in note_gold.iterrows():
            gs, ge = int(grow["start"]), int(grow["end"])
            intervals = existing_per_note.get(nid, [])
            covered = any(ps <= gs and pe >= ge for ps, pe in intervals)
            if not covered:
                text = note_texts.get(nid, "")
                span_text = text[gs:ge].lower().strip() if gs < len(text) else ""
                missed.append((gs, ge, span_text))
        gold_missed_per_note[nid] = missed

    # Build bracket-annotated notes
    annotated_notes = {}
    for nid in test_ids:
        text = note_texts.get(nid, "")
        preds = kiri_pred[kiri_pred["note_id"] == nid]
        spans = preds[["start", "end"]].sort_values("start", ascending=False).values
        for start, end in spans:
            start, end = int(start), int(end)
            if 0 <= start < end <= len(text):
                text = text[:start] + "[" + text[start:end] + "]" + text[end:]
        annotated_notes[nid] = text

    total_missed = sum(len(v) for v in gold_missed_per_note.values())
    print(f"  Eval data: {len(test_ids)} notes, {total_missed} gold-missed spans")

    return test_ids, note_texts, annotated_notes, existing_per_note, gold_missed_per_note


def parse_discovery_output(raw: str) -> list[tuple[str, str | None]]:
    """Parse LLM output into list of (span_text, expansion_or_None).

    Handles:
      - "EVD = External Ventricular Drain" → ("EVD", "External Ventricular Drain")
      - "shortness of breath" → ("shortness of breath", None)
      - "NONE" → []
    """
    if not raw or raw.strip().upper() == "NONE":
        return []

    results = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-\*]\s*", "", line)
        line = line.strip().strip('"').strip("'")
        if line.upper() == "NONE" or len(line) < 1:
            continue

        # Check for abbreviation = expansion format
        eq_match = re.match(r"^([A-Za-z0-9/\-\+]+)\s*=\s*(.+)$", line)
        if eq_match:
            abbrev = eq_match.group(1).strip()
            expansion = eq_match.group(2).strip()
            if len(abbrev) <= 10 and len(expansion) >= 3:
                results.append((abbrev, expansion))
                continue

        if 1 <= len(line) <= 100:
            results.append((line, None))

    return results


def evaluate_genome(
    genome, client, model, test_ids, annotated_notes, note_texts,
    existing_per_note, gold_missed_per_note, reasoning_effort="low",
):
    """Evaluate a genome's fitness by running discovery and scoring against gold."""
    system_prompt = genome.build_prompt()

    # Build prompts
    prompts = [annotated_notes[nid] for nid in test_ids]

    # Run LLM
    outputs = _call_vllm_batch(
        client, model, prompts,
        reasoning_effort=reasoning_effort,
        max_tokens=1024, max_concurrent=256,
        label=genome.signature()[:20],
        system_prompt=system_prompt,
        frequency_penalty=0.3,
    )

    # Score: for each discovered span, check if it matches a gold-missed span
    total_hits = 0
    total_discovered = 0
    total_gold_missed = sum(len(v) for v in gold_missed_per_note.values())
    total_false_pos = 0
    hit_details = []

    for nid, raw in zip(test_ids, outputs):
        parsed = parse_discovery_output(raw)
        gold_missed = gold_missed_per_note.get(nid, [])
        text = note_texts.get(nid, "")
        text_lower = text.lower()
        existing = existing_per_note.get(nid, [])

        # Track which gold-missed spans we've matched
        matched_gold = set()

        for span_text, expansion in parsed:
            total_discovered += 1
            span_lower = span_text.lower()

            # Find this span in the note text
            found_match = False
            start = 0
            while True:
                idx = text_lower.find(span_lower, start)
                if idx == -1:
                    break
                end = idx + len(span_text)
                start = end

                # Skip if overlaps existing KIRI prediction
                overlaps = any(ps < end and pe > idx for ps, pe in existing)
                if overlaps:
                    continue

                # Check if this position matches any gold-missed span
                for gi, (gs, ge, gspan) in enumerate(gold_missed):
                    if gi in matched_gold:
                        continue
                    # Overlap check: the discovered span covers the gold span
                    if idx <= gs and end >= ge:
                        matched_gold.add(gi)
                        found_match = True
                        hit_details.append((nid, span_text, expansion, gspan))
                        break
                    # Or the gold span covers the discovered span
                    if gs <= idx and ge >= end:
                        matched_gold.add(gi)
                        found_match = True
                        hit_details.append((nid, span_text, expansion, gspan))
                        break

                if found_match:
                    break

            if not found_match:
                total_false_pos += 1

        total_hits += len(matched_gold)

    recall = total_hits / total_gold_missed if total_gold_missed > 0 else 0
    precision = total_hits / total_discovered if total_discovered > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Fitness: weighted F1 favoring recall (since false positives can be filtered later)
    fitness = 0.3 * precision + 0.7 * recall

    genome.fitness = fitness
    genome.metrics = {
        "hits": total_hits,
        "discovered": total_discovered,
        "gold_missed": total_gold_missed,
        "false_pos": total_false_pos,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fitness": fitness,
    }
    return genome


def main():
    parser = argparse.ArgumentParser(description="Evolve discovery prompts")
    parser.add_argument("--notes-csv", default="1st Place/data/raw/mimic-iv_notes_training_set.csv")
    parser.add_argument("--annotations-csv", default="1st Place/data/interim/train_annotations_cln.csv")
    parser.add_argument("--kiri-pred-csv", default="outputs/super_dictionary_60test/kiri_super_base_pred.csv")
    parser.add_argument("--train-size", type=int, default=212)
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--reasoning-effort", type=str, default="low")
    parser.add_argument("--eval-notes", type=int, default=15, help="Notes to evaluate on (subset for speed)")
    parser.add_argument("--population", type=int, default=6, help="Population size")
    parser.add_argument("--generations", type=int, default=8, help="Number of generations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="outputs/evolved_prompt.json", help="Save best genome")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 65)
    print("  Loading evaluation data")
    print("=" * 65)
    test_ids, note_texts, annotated_notes, existing_per_note, gold_missed_per_note = load_eval_data(
        args.notes_csv, args.annotations_csv, args.kiri_pred_csv,
        args.train_size, args.eval_notes,
    )

    client = _make_vllm_client(args.vllm_url, args.model)

    # =========================================================================
    # Initialize population
    # =========================================================================
    print("\n" + "=" * 65)
    print(f"  Initializing population (size={args.population})")
    print("=" * 65)

    population = []
    # Seed with diverse starting points
    for fi in range(min(len(FRAME_VARIANTS), args.population)):
        g = Genome(frame_idx=fi)
        population.append(g)
    # Fill rest with random variants
    while len(population) < args.population:
        g = Genome(
            frame_idx=random.randint(0, len(FRAME_VARIANTS) - 1),
            enabled={r: random.random() > 0.2 for r in RULE_NAMES},
            phrasings={r: random.randint(0, len(RULE_BANK[r]) - 1) for r in RULE_NAMES},
        )
        population.append(g)

    best_ever = None

    for gen in range(args.generations):
        print(f"\n{'=' * 65}")
        print(f"  Generation {gen + 1}/{args.generations}")
        print(f"{'=' * 65}")

        # Evaluate unevaluated genomes
        for i, g in enumerate(population):
            if g.fitness is not None:
                continue
            print(f"\n  Evaluating {i + 1}/{len(population)}: {g.signature()}")
            evaluate_genome(
                g, client, args.model, test_ids, annotated_notes,
                note_texts, existing_per_note, gold_missed_per_note,
                reasoning_effort=args.reasoning_effort,
            )
            m = g.metrics
            print(f"    hits={m['hits']}/{m['gold_missed']} "
                  f"discovered={m['discovered']} fp={m['false_pos']} "
                  f"R={m['recall']:.3f} P={m['precision']:.3f} "
                  f"F1={m['f1']:.3f} fit={m['fitness']:.4f}")

        # Sort by fitness
        population.sort(key=lambda g: g.fitness or 0, reverse=True)

        # Print generation summary
        print(f"\n  Generation {gen + 1} rankings:")
        for i, g in enumerate(population):
            m = g.metrics
            marker = " <-- BEST" if i == 0 else ""
            print(f"    {i + 1}. fit={m['fitness']:.4f} "
                  f"R={m['recall']:.3f} P={m['precision']:.3f} "
                  f"hits={m['hits']} disc={m['discovered']} "
                  f"sig={g.signature()}{marker}")

        # Track best ever
        if best_ever is None or population[0].fitness > best_ever.fitness:
            best_ever = copy.deepcopy(population[0])
            print(f"\n  New best ever: fit={best_ever.fitness:.4f}")

        if gen == args.generations - 1:
            break

        # Selection + reproduction
        # Keep top half, generate children from them
        n_keep = max(2, len(population) // 2)
        survivors = population[:n_keep]

        children = []
        while len(children) < args.population - n_keep:
            if random.random() < 0.5 and len(survivors) >= 2:
                # Crossover
                a, b = random.sample(survivors, 2)
                child = Genome.crossover(a, b).mutate(rate=0.3)
            else:
                # Mutate a survivor
                parent = random.choice(survivors)
                child = parent.mutate(rate=0.4)
            children.append(child)

        population = survivors + children

    # =========================================================================
    # Output best
    # =========================================================================
    print(f"\n{'=' * 65}")
    print(f"  Best genome: {best_ever.signature()}")
    print(f"{'=' * 65}")
    print(f"  Fitness: {best_ever.fitness:.4f}")
    m = best_ever.metrics
    print(f"  Recall: {m['recall']:.3f} ({m['hits']}/{m['gold_missed']})")
    print(f"  Precision: {m['precision']:.3f}")
    print(f"  F1: {m['f1']:.3f}")
    print(f"\n  System prompt:\n{'─' * 50}")
    print(best_ever.build_prompt())
    print(f"{'─' * 50}")

    # Save
    result = {
        "signature": best_ever.signature(),
        "fitness": best_ever.fitness,
        "metrics": best_ever.metrics,
        "frame_idx": best_ever.frame_idx,
        "enabled": best_ever.enabled,
        "phrasings": best_ever.phrasings,
        "system_prompt": best_ever.build_prompt(),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved to {args.out}")


if __name__ == "__main__":
    main()
