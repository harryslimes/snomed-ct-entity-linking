# LLM Post-Processor Agent: Plan + Prompt-Search Loop

This document describes a safe, reproducible plan for using an LLM **after** KIRI produces annotations to improve the **leaderboard metric** (macro-averaged per-class character IoU).

The core idea is to treat the LLM as a constrained “annotation editor” that produces a small **edit script** over KIRI’s output (mostly **remove** and **span-align**), and to optimize the prompt via an automated loop on a held-out validation set.

## 1) Why this helps this competition metric

The competition metric is effectively “macro IoU over concept IDs”, where each predicted or gold concept ID is a class. This makes **false-positive concept IDs expensive**: any predicted concept ID that is not in gold gets IoU = 0 for that class and hurts the macro average.

So the highest ROI first step is typically:

- **Delete spurious annotations** (esp. from boilerplate sections / templated text).
- **Fix span boundaries** (shift start/end) so true positives overlap gold more.

Only later should we consider adding new concept IDs (it risks adding new FP classes).

## 2) Guardrails (to avoid leakage + reduce “LLM damage”)

### Data split discipline (no leakage)

- Use `(pred, gold)` pairs from **training data** to design prompts and search over prompt variants.
- Pick the “best prompt” on a **held-out validation split** (or k-fold CV).
- The final prompt is then frozen and applied to:
  - any internal test split evaluation, and
  - the hidden leaderboard test set (where gold is not available).

If the prompt is derived from the same examples it’s scored on, it will overfit and look better than it really is.

### Constrained edit space (start conservative)

V1 agent constraints:

- **No new concept IDs**: the agent may only delete or adjust spans for existing predictions.
- **Bounded span shifts**: e.g., max ±30 chars (configurable).
- **Max edits per note**: e.g., up to 10 edits, to prevent prompt failures from nuking a note.
- **Never edit outside the note bounds**; edits must preserve `start < end`.
- **No overlaps** after edits (reuse existing `remove_overlaps` logic).

Later versions can allow:

- merging adjacent spans of the same `concept_id`,
- adding new spans only from a strict allowlist, or
- swapping to a different concept ID only when strongly justified (high risk).

## 3) Agent contract: Edit-script interface

To make evaluation and optimization reliable, the LLM should output a structured “edit script” instead of free-form prose.

Proposed schema (JSON):

- `note_id`: string
- `edits`: list of:
  - `{ "op": "delete", "idx": <int> }`
  - `{ "op": "shift", "idx": <int>, "start": <int>, "end": <int> }`

Where `idx` refers to the index of the prediction row for that note in a stable, deterministic ordering (e.g., sort by `start`, then `end`, then `concept_id`).

Why this is important:

- avoids ambiguous natural-language changes,
- makes the post-processor deterministic and testable,
- allows us to validate constraints before applying edits.

## 4) Prompt design (V1)

Inputs to the prompt per note:

- the note text (optionally truncated around predicted spans to save tokens),
- the list of predicted spans for that note (with indices),
- a short rules block (generic, not example-specific),
- output must be valid JSON edit script.

Rules to include early (generic, high precision):

- Remove annotations that are clearly part of templated boilerplate (headers, signatures, “disclaimer” blocks).
- Remove annotations in clearly irrelevant sections (e.g., “FAMILY HISTORY” when the concept is an acute finding).
- Align spans to the exact phrase boundaries (exclude surrounding punctuation/whitespace; include the full clinical term).
- Prefer deleting uncertain predictions over keeping them (macro-IoU is FP-sensitive).

## 5) Prompt optimization loop (automated search)

This is “prompt search”, treated like an optimization loop:

1. Start from a **base prompt** (handwritten).
2. Generate `N` prompt variants (mutations) with an LLM:
   - rewrite rules,
   - change formatting / ordering,
   - add clarifying constraints,
   - change how examples are presented (few-shot vs zero-shot).
3. For each prompt candidate:
   - apply the LLM post-processor to validation notes,
   - apply edits to predictions (hard constraint checks),
   - compute the leaderboard metric (macro per-class char IoU).
4. Keep the top `K` prompts and iterate for `R` rounds (bandit / evolutionary search).

Scoring should include penalties:

- invalid JSON output,
- violating constraints (trying to add concept IDs, too many edits),
- excessive deletions (optional penalty to avoid deleting everything).

Stop conditions:

- no improvement for `M` rounds,
- budget limit (tokens / time),
- prompt variants converge.

## 6) Evaluation protocol in this repo

Recommended split:

- Train KIRI dictionary on a training subset (e.g., 150 notes) and generate predictions on held-out notes.
- Run LLM post-processing only on held-out predictions.
- Compute:
  - macro-averaged per-class character IoU (leaderboard proxy),
  - note-level concept-set IoU (secondary diagnostic).

Existing relevant tooling:

- `scripts/super_dictionary/compare_kiri.py` (train/predict/score harness)
- `scripts/super_dictionary/runtime_scoring.py` (`class_char_iou`)
- `scripts/super_dictionary/kiri_delta_report.py` (FP/FN analysis to inform prompt rules)

## 7) Implementation tasks (concrete)

1. Add `scripts/llm_postprocess/` with:
   - `schema.py` (edit script validation + constraints)
   - `apply_edits.py` (apply edit scripts to `*_pred.csv`)
   - `run_agent.py` (calls the model API; caches results per note/prompt hash)
   - `optimize_prompt.py` (prompt-search loop + evaluation)
   - `prompts/base.md` (base prompt template)
2. Add config knobs:
   - max edits per note, max shift chars, allow add concept IDs (default off)
3. Add a small unit test suite for edit application + constraints.

## 8) Optional next steps (after V1 works)

- Add a “boilerplate detector” (regex) so many deletions are deterministic (cheap) before the LLM runs.
- Use delta reports to automatically propose rule candidates (e.g., “top 20 new FP strings”).
- Consider learning a lightweight classifier for “keep vs delete” per predicted span, using features from dict provenance + section headers.
- RLVR-style approaches: only after a stable eval harness exists; otherwise it’s easy to overfit and hard to debug.

