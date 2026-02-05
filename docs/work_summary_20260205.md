# Work Summary (2026-02-05): Super-Dictionary + KIRI Rule Wiring

## What changed

- Wired in **optional** “linguistic rule” knobs for KIRI:
  - `KIRI_LINGUISTIC_RULES=1` adds lightweight abbreviation/“fracture” phrasing variants during dictionary augmentation.
  - `KIRI_STOPWORD_TRANSPARENT=1` enables safer stopword-transparent matching (stopwords allowed **between** tokens only).
- Added a reproducible comparison/eval harness:
  - `scripts/super_dictionary/compare_kiri.py` supports `--super-ablations` and `--eval-all`.
- Added a delta-report tool for diagnosing regressions:
  - `scripts/super_dictionary/kiri_delta_report.py`
- Added tests for the stopword-transparent matcher behavior:
  - `tests/test_kiri_rules.py`
- Added a design doc for an **LLM post-processor** and automated prompt-search loop:
  - `docs/llm_postprocess_agent_plan.md`

## Key implementation notes

- Stopword transparency is intentionally conservative to avoid “term collapse” and large FP spikes:
  - If the **mention itself** contains any stopword (from `KIRI_STOPWORDS`), we fall back to strict matching.
  - Gating knobs:
    - `KIRI_STOPWORD_MIN_TOKENS` (default `3`)
    - `KIRI_STOPWORD_ALLOW_2TOKENS` (default `fracture,fx`)
- Prefilter index behavior:
  - Under stopword-transparent mode, `IndexedDict` switches to a unigram prefilter to reduce false negatives.
- Training robustness:
  - Dictionary building/augmentation now skips non-string mentions (guards against `NaN` sources in synonym tables).

## Full training-set evaluation results (272 notes)

Outputs were written to `outputs/super_dictionary/full_train_20260205/`.

Macro-averaged per-class character IoU (leaderboard proxy) and note-level concept-set IoU:

- `kiri_default_fulltrain`: note IoU `0.7572`, macro-char IoU `0.7197`
- `kiri_super_fulltrain_base`: note IoU `0.7420`, macro-char IoU `0.7253` (best macro-char IoU)
- `kiri_super_fulltrain_linguistic`: note IoU `0.7414`, macro-char IoU `0.7236`
- `kiri_super_fulltrain_stopword`: note IoU `0.7465`, macro-char IoU `0.7220`
- `kiri_super_fulltrain_linguistic_stopword`: note IoU `0.7458`, macro-char IoU `0.7198`

Command used:

```bash
KIRI_PARALLEL=1 KIRI_WORKERS=8 \
KIRI_TRAIN_PARALLEL=1 KIRI_TRAIN_WORKERS=8 \
KIRI_SCORE_PARALLEL=1 KIRI_SCORE_WORKERS=8 \
python scripts/super_dictionary/compare_kiri.py \
  --eval-all --super-ablations \
  --run-name-default kiri_default_fulltrain \
  --run-name-super kiri_super_fulltrain \
  --output-dir outputs/super_dictionary/full_train_20260205
```

## If/when we continue: recommended next steps

1) **Generate a delta report** for `kiri_default_fulltrain` vs `kiri_super_fulltrain_base`:
   - Goal: identify the top “new FP” concept IDs by character mass and the most common offending strings/sections.
2) **Targeted FP suppression** (highest ROI for macro-char IoU):
   - Filter out garbage synonyms during export (e.g., very short/high-frequency phrases like “to the”).
   - Add section-aware suppressions for known noisy headers/templates.
   - Add concept-pair “swap guardrails” for recurring confusions (only if they are stable across many notes).
3) **LLM post-processor (V1)**:
   - Implement the constrained “edit-script only” agent described in `docs/llm_postprocess_agent_plan.md`.
   - Start with delete/shift-only (no new concept IDs), validate on a held-out split, then iterate via prompt search.

## Pointers

- Eval harness: `scripts/super_dictionary/compare_kiri.py`
- Runtime metric implementation: `scripts/super_dictionary/runtime_scoring.py`
- LLM agent plan: `docs/llm_postprocess_agent_plan.md`

