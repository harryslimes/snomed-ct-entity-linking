# Rule Generation and Testing Guide

This document describes the LLM-driven rule generation and testing pipeline for SNOMED CT entity linking. The system uses Claude Opus to analyze annotated clinical notes and produce structured rules that guide a two-pass LLM agent (search + select) through SNOMED concept disambiguation.

## Overview

The pipeline has three main phases:

1. **Sampling** — Select a representative subset of training notes that maximizes concept coverage.
2. **Rule Generation** — Use Claude Opus to analyze annotated notes and produce structured rules.
3. **Rule Testing** — Evaluate rules on held-out notes using a two-pass LLM agent with SNOMED retrieval.

Rules are versioned (currently v4.0) and stored as timestamped JSON files under `scripts/rules/`.

## Prerequisites

- Python virtual environment with dependencies installed (`.venv/`)
- SNOMED retrieval indexes built (`snomed_index/` — FAISS + BM25)
- SNOMED subsumption index built (`snomed_index/subsumption.pkl`)
- Training data split (`data/old-challenge-split/`)
- Terminology file (`3rd Place/assets/dataflattened_terminology.csv`)
- For rule generation: Claude API access via the Claude Agent SDK
- For rule testing with vLLM: a running vLLM server with `openai/gpt-oss-20b`

### Building the Subsumption Index

The subsumption index is required for v4.0 rules. It maps SNOMED concepts to their ancestors in the hierarchy, enabling rules to declare which concept subtrees they apply to.

```bash
python scripts/build_subsumption_index.py
```

This reads Athena CONCEPT and CONCEPT_RELATIONSHIP CSVs from `data/athena/` and produces `snomed_index/subsumption.pkl`.

## Phase 1: Annotation Sampling

Before generating rules, select a diverse subset of training notes that covers the most unique SNOMED concepts.

```bash
python scripts/sample_annotations.py \
    --target-coverage 0.8 \
    --per-concept-budget 5
```

This runs a greedy set-cover algorithm:
1. Repeatedly picks the note that adds the most uncovered concepts until the target concept coverage is reached.
2. Within selected notes, diversifies annotations per concept using farthest-first traversal on (normalized span, section).

Output: `scripts/sampled_annotations.json` containing:
- `selected_note_ids`: ordered list of notes (most coverage first)
- `annotations`: the sampled annotations with note_id, start, end, concept_id, section, etc.
- Coverage statistics

## Phase 2: Rule Generation

Rule generation uses Claude Opus via the Claude Agent SDK to analyze an annotated clinical note and produce structured rules.

### Initial Generation (First Note)

Generate rules from scratch using the first note in the greedy sample order:

```bash
python scripts/rule_generation.py \
    --sample scripts/sampled_annotations.json \
    --note-index 0
```

### Iterative Extension (Subsequent Notes)

Extend existing rules with patterns from a new note:

```bash
python scripts/rule_generation.py \
    --sample scripts/sampled_annotations.json \
    --note-index 1 \
    --existing-rules scripts/rules/<previous_timestamp>/rules.json
```

In extend mode, the LLM receives the existing rules and is instructed to:
- Preserve existing rules unless they need correction
- Add new rules for uncovered patterns
- Refine existing rules if the new note reveals they are too narrow/broad
- Add new mappings for abbreviations seen in the new note
- Keep rule IDs stable (new rules continue numbering from the highest existing ID)

### Legacy Mode

Generate rules from the first note in the CSV without sampling:

```bash
python scripts/rule_generation.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sample` | None | Path to `sampled_annotations.json` |
| `--note-index` | 0 | Index into the greedy note order |
| `--existing-rules` | None | Path to existing `rules.json` to extend |
| `--model` | `claude-opus-4-20250514` | Model ID for generation |

### Output

Rules are saved to `scripts/rules/<timestamp>/rules.json` along with a `benchmark_generation.json` file containing timing and cost data.

## Rules Format (v4.0)

A rules file contains:

### Mappings

Deterministic, context-free 1:1 translations. Abbreviation expansions, acronym expansions, and direct synonyms where the span text always maps to the same search term regardless of context. At inference time, mapped spans bypass the search LLM entirely.

```json
{
  "mappings": {
    "TB": "tuberculosis",
    "PFO": "patent foramen ovale",
    "CTAB": "clear to auscultation bilaterally"
  }
}
```

### G-Rules (Global Rules)

Universal rules that apply broadly across all annotations. Each can optionally be scoped to specific pipeline stages.

```json
{
  "g_rules": [
    {
      "id": "G1",
      "rule": "Always include the exact span text as one of the search terms...",
      "stages": ["stage_2_search", "stage_3_select"]
    }
  ]
}
```

### Structured Rules (R-Rules)

Specific rules for particular clinical patterns. Each rule declares:

- **concept_type**: semantic category (e.g., `Anatomical_Structure`, `Negated_Finding`, `Medication_Allergy`)
- **applies_to**: which annotations the rule covers, using SNOMED ancestor concept IDs, optional section restrictions, and optional span patterns
- **stage_2_search**: how to generate search terms for this pattern
- **stage_3_select**: how to disambiguate among candidate concepts

```json
{
  "structured_rules": [
    {
      "rule_id": "R1",
      "concept_type": "Anatomical_Structure",
      "applies_to": {
        "ancestor_concept_ids": [91723000],
        "sections": null,
        "span_pattern": null
      },
      "stage_2_search": {
        "filtering_logic": null,
        "intent_translation": "For anatomical terms, search using the exact anatomical name..."
      },
      "stage_3_select": {
        "disambiguation_logic": "Select the most specific anatomical concept...",
        "preferred_hierarchy": "body structure",
        "reject_hierarchies": ["morphologic abnormality", "finding"]
      }
    }
  ]
}
```

The `applies_to.ancestor_concept_ids` field is key to v4.0: at test time, the subsumption index checks whether an annotation's gold concept is a descendant of any listed ancestor. This replaces hard-coded annotation-to-rule mappings with generalizable hierarchy-based matching.

### Annotation Rule Map

A validation artifact mapping each annotation ID to the rules that apply. Used to verify that `applies_to` criteria actually cover what was intended.

```json
{
  "annotation_rule_map": {
    "12345": ["G1", "R1", "R3"]
  }
}
```

## Phase 3: Rule Testing

Test rules by running a two-pass LLM agent on a clinical note:

1. **Pass 1 (Search)**: LLM generates search terms based on rules + clinical context
2. **Retrieval**: Hybrid SNOMED retrieval (SapBERT + FAISS dense + BM25 sparse, fused via RRF)
3. **Pass 2 (Select)**: LLM selects the best concept from retrieved candidates

### Testing with vLLM (gpt-oss-20b)

Start the vLLM server:

```bash
vllm serve openai/gpt-oss-20b \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.93 \
    --enable-prefix-caching \
    --enable-chunked-prefill
```

**Note on determinism:** vLLM inference is non-deterministic with concurrent requests
due to floating-point non-associativity from varying batch compositions. The
`VLLM_BATCH_INVARIANT=1` env var exists in vLLM nightly (0.16+) but does not yet
work with quantized MoE models (e.g. Qwen3-30B-A3B-AWQ). See
[docs/vllm_batch_invariance_investigation.md](../../docs/vllm_batch_invariance_investigation.md)
for details. The rule evaluation loop mitigates this by using per-rule cost gating
(counting actual prediction flips) rather than relying on stable accuracy numbers.

Run the test:

```bash
python scripts/rule_testing.py \
    --backend vllm \
    --rules scripts/rules/<timestamp>/rules.json \
    --note <NOTE_ID>
```

The vLLM backend runs all annotations through a pipelined architecture where each annotation independently flows through search -> retrieve -> select. An async batch retriever accumulates SNOMED queries from concurrent annotations for efficient batched SapBERT encoding and FAISS/BM25 search.

### Testing with Claude Sonnet

```bash
python scripts/rule_testing.py \
    --backend sonnet \
    --rules scripts/rules/<timestamp>/rules.json \
    --note <NOTE_ID>
```

The Sonnet backend runs sequentially via the Claude Agent SDK.

### Replaying Saved Results

Re-run scoring and wider retrieval analysis on previously saved results without re-running inference:

```bash
python scripts/rule_testing.py \
    --replay scripts/rules/<timestamp>/test_results_vllm_<timestamp>.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `sonnet` | `sonnet` (Claude Agent SDK) or `vllm` (local gpt-oss-20b) |
| `--rules` | `scripts/rules_output.json` | Path to rules JSON |
| `--note` | First note in CSV | Note ID to test on |
| `--vllm-url` | `http://localhost:8000` | vLLM server URL |
| `--vllm-model` | `openai/gpt-oss-20b` | vLLM model name |
| `--concurrency` | 0 (unlimited) | Max concurrent vLLM requests |
| `--reasoning-effort` | `low` | Reasoning effort for gpt-oss-20b (`low`/`medium`/`high`) |
| `--replay` | None | Path to saved results JSON to replay |

### Output

Results are saved alongside the rules file:
- `test_results_vllm_<timestamp>.json` or `test_results_sonnet_<timestamp>.json` — per-annotation results with predictions, IoU scores, failure categorization
- `benchmark_testing_<timestamp>.json` — timing breakdown (per-annotation pipeline timings, retrieval batch stats)

### Understanding Results

The summary report breaks down failures by pipeline stage:

- **Stage 2 miss (retrieval_miss)**: The search rules failed to produce search terms that retrieved the gold concept in the top-k candidates. Improve `stage_2_search` rules or add mappings.
- **Stage 3 miss (selection_miss)**: The gold concept was in the candidates but the LLM picked the wrong one. Improve `stage_3_select` disambiguation rules.
- **Parse fail**: The LLM response couldn't be parsed as valid JSON.

The report also checks retrieval misses against a wider top-100 pool to distinguish recoverable misses (gold found in top-100) from true retrieval gaps.

## Typical Workflow

```
# 1. Build subsumption index (one-time)
python scripts/build_subsumption_index.py

# 2. Sample annotations
python scripts/sample_annotations.py --target-coverage 0.8

# 3. Generate initial rules from the first sampled note
python scripts/rule_generation.py \
    --sample scripts/sampled_annotations.json \
    --note-index 0

# 4. Test the rules
vllm serve openai/gpt-oss-20b --enable-prefix-caching --enable-chunked-prefill &
python scripts/rule_testing.py \
    --backend vllm \
    --rules scripts/rules/<timestamp>/rules.json

# 5. Extend rules with the next note
python scripts/rule_generation.py \
    --sample scripts/sampled_annotations.json \
    --note-index 1 \
    --existing-rules scripts/rules/<timestamp>/rules.json

# 6. Test the extended rules
python scripts/rule_testing.py \
    --backend vllm \
    --rules scripts/rules/<new_timestamp>/rules.json

# 7. Repeat steps 5-6 for additional notes
```

## Directory Structure

```
scripts/
  rule_generation.py          # Rule generation via Claude Opus
  rule_testing.py             # Rule testing with two-pass LLM agent
  sample_annotations.py       # Greedy diversity sampler
  sampled_annotations.json    # Output of sampling
  snomed_subsumption.py       # Subsumption index utilities
  build_subsumption_index.py  # Build subsumption index from Athena data
  rules_output.json           # Legacy flat rules (pre-v4.0)
  rules/
    <timestamp>/
      rules.json                          # Generated rules
      benchmark_generation.json           # Generation timing/cost
      test_results_vllm_<timestamp>.json  # Test results
      benchmark_testing_<timestamp>.json  # Testing timing
```
