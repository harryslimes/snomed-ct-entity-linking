# FP Concept Analysis Handover

## Task

Investigate pure FP concepts in KIRI super dictionary predictions — specifically which ones appear in both training and test sets — and identify strategies to safely remove them for score improvement.

## Background

The SNOMED CT entity linking competition scores predictions using **macro-averaged character-level IoU** across all unique concept IDs. Every concept counts equally. The KIRI super dictionary baseline scores **0.6152** on the old-challenge-split test set.

A score decomposition revealed:

| Category | Count | % of 4,424 | avg IoU | Description |
|---|---|---|---|---|
| Perfect (IoU=1.0) | 1,882 | 42.5% | 1.000 | Correct concept, correct span |
| Partial (0<IoU<1) | 1,371 | 31.0% | 0.613 | Correct concept, imperfect span |
| Zero overlap | 61 | 1.4% | 0.000 | Right concept, wrong location/note |
| **Completely missed** | **754** | **17.0%** | **0.000** | Gold has them, KIRI never predicts |
| **Pure FP concepts** | **356** | **8.0%** | **0.000** | KIRI predicts, gold never has them |

**1,171 concepts with IoU=0** drag the macro average from 0.837 down to 0.615.

### Why This Matters

- Removing **pure FP concepts** is completely safe — there are zero TPs to accidentally kill. Oracle removal of all 356 pure FP concepts yields **+0.054** (0.615 → 0.669).
- Per-prediction FP filtering was tried and **failed** because it also kills TPs for rare concepts, which is devastating under macro-averaging. That approach is a dead end.
- Concept-level removal is fundamentally different: if concept X has zero gold annotations, removing ALL predictions for concept X can only help.

### The Key Question

Can we predict which concepts will be pure FPs on unseen test data? If certain concepts are consistently spurious across both train and test sets, we can build a blocklist.

## Key Files

### Data
- **Train notes**: `data/old-challenge-split/train_notes.csv` (204 notes)
- **Train gold annotations**: `data/old-challenge-split/train_annotations.csv` (52,078 annotations)
- **Test notes**: `data/old-challenge-split/test_notes.csv` (68 notes)
- **Test gold annotations**: `data/old-challenge-split/test_annotations.csv` (23,413 annotations)

### Predictions
- **KIRI test predictions** (trained on train, predicted on test): `outputs/old_challenge_split/kiri_super_pred.csv` — 24,605 rows, columns: `[note_id, start, end, concept_id]`
- **KIRI train predictions** (trained on train, predicted on train — NOT OOF): `outputs/old_challenge_split/kiri_super_train_pred.csv` — 58,046 rows (if exists, check)
- **KIRI OOF predictions** (4-fold cross-val on train set): `outputs/old_challenge_split/kiri_super_oof_pred.csv` — 56,347 rows

### Scoring
- **Scoring function**: `scripts/super_dictionary/runtime_scoring.py`
  - `macro_char_iou(pred_df, gold_df)` — the competition metric
  - `class_char_iou(pred_df, gold_df)` — per-concept IoU breakdown (returns DataFrame with columns: `concept_id, gt_chars, pred_chars, intersection, union, iou`)

### Dictionary
- **Super dictionary synonyms**: `1st Place/data/interim/flattened_terminology_syn_super.csv`
- **KIRI engine** (trains/predicts): `scripts/super_dictionary/engine.py`

## How to Reproduce the Score Decomposition

```python
import pandas as pd
from scripts.super_dictionary.runtime_scoring import class_char_iou, macro_char_iou

pred = pd.read_csv('outputs/old_challenge_split/kiri_super_pred.csv')
gold = pd.read_csv('data/old-challenge-split/test_annotations.csv')

for df in [pred, gold]:
    df['start'] = df['start'].astype(int)
    df['end'] = df['end'].astype(int)
    df['concept_id'] = df['concept_id'].astype(int)

pc = pred[['note_id','start','end','concept_id']]
gc = gold[['note_id','start','end','concept_id']]

# Per-concept IoU
per_concept = class_char_iou(pc, gc)
valid = per_concept[per_concept['union'] > 0]

# Pure FP concepts: predicted but never in gold
fp_only = valid[(valid['gt_chars'] == 0) & (valid['pred_chars'] > 0)]
print(f"Pure FP concepts: {len(fp_only)}")
print(f"Their concept_ids: {sorted(fp_only['concept_id'].tolist())}")
```

## Analysis To Do

### 1. Find Pure FP Concepts Common to Train and Test

For the **training set**, generate the same decomposition using OOF predictions (or train-on-train predictions) vs train gold. Identify which concepts are pure FPs there. Then intersect with test pure FPs.

```python
# On training set (using OOF predictions for realism)
oof_pred = pd.read_csv('outputs/old_challenge_split/kiri_super_oof_pred.csv')
train_gold = pd.read_csv('data/old-challenge-split/train_annotations.csv')

# Same analysis as above to get train pure FP concepts
# Then: common_fp = train_fp_concepts & test_fp_concepts
```

If a concept is a pure FP in BOTH train and test, it's likely a systematic dictionary artifact (e.g., a common English word that happens to be a SNOMED synonym but is never used as a medical concept in clinical notes).

### 2. Characterize the FP Concepts

For each pure FP concept:
- What text does KIRI match? (look up the span text from predictions)
- What's the SNOMED concept name? (look up in the dictionary)
- Is it a short/common word?
- What sections does it appear in?

### 3. Quantify the Safe Removal Gain

If we blocklist the concepts that are pure FPs in both train and test:
- How many test FP concepts would we catch?
- What's the score improvement?
- What's the risk? (A concept that's FP in train might be a TP in test — check this!)

### 4. Broader Analysis

- Of the 356 test pure FP concepts, how many also appear as TPs in the training set? (These are concepts that are valid medical entities but KIRI matches them in wrong contexts on the test set.)
- Are there patterns: certain concept types, short spans, specific sections?

## Important Caveats

- **NEVER filter at the prediction level** — only at the concept level. Prediction-level filtering kills rare TPs and destroys the macro-averaged score (tested and confirmed: -0.05 to -0.10).
- **The train/test split is `data/old-challenge-split/`** — this is a proxy for the real competition test set. The real test set is hidden.
- **OOF predictions** (`kiri_super_oof_pred.csv`) are more realistic than train-on-train predictions for the training set analysis since they avoid data leakage.
- The scoring function needs `PYTHONPATH=.` set when running from the repo root.

## Other Related Opportunities

### Missed Concepts (+0.136 ceiling)
754 concepts in gold that KIRI never predicts. 410 are novel (not in training gold). Finding these requires NER or LLM-based extraction — a separate workstream but the single biggest opportunity.

### Zero-Overlap Concepts
61 concepts where KIRI predicts the right concept but in wrong notes/locations. Worth investigating — might be fixable with note-level filtering.
