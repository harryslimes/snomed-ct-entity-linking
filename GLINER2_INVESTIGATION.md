# GLiNER2 for Clinical SNOMED-CT Entity Extraction

## Overview

GLiNER2 is a 205M parameter span extraction model built on DeBERTa-v3-large. Unlike token classification (BIO tagging) models, it uses a biaffine decoder that scores all possible (start, end) token pairs against entity type representations. The goal is to use it as a fast first-pass span detector for clinical notes, complementing the existing Qwen3-30B vLLM pipeline.

Training and evaluation use the `data/old-challenge-split/` dataset: 204 training notes (split 85/15 into train/val) and 68 test notes, with SNOMED-CT concept annotations.

## Entity Classes

Annotations are mapped from SNOMED concept IDs to 3 semantic classes via the SNOMED subsumption hierarchy (`sctid_to_tag` from RF2 files):

| Semantic Tag | Entity Class | Description |
|---|---|---|
| finding, disorder | medical finding, symptom, or disease | Clinical findings, diagnoses, disorders, symptoms, and signs |
| procedure, regime/therapy | procedure | Medical procedures, surgeries, therapies, and treatments |
| body structure, morphologic abnormality, cell structure | anatomical body part | Body structures, anatomical parts, and morphologic features |

## Training Configuration

- **Base model:** `fastino/gliner2-large-v1` (DeBERTa-v3-large encoder)
- **Fine-tuning:** LoRA (r=16, alpha=32, dropout=0.05, target=encoder)
- **Hardware:** RTX 5090, 32GB VRAM
- **Chunking:** Section-aware overlapping windows (w=200 tokens, o=30 overlap). Section headers prepended as `[Section Name]` prefix to match training/inference.
- **Max sequence length:** 350 subword tokens per chunk (hard cap via processor monkey-patch; GLiNER2 adds ~50 schema tokens on top)

### Bug Fixes Required

1. **Early stopping bug (Issue #70):** GLiNER2's `_evaluate()` updates `self.best_metric` before `_check_early_stopping()` compares against it, so patience never triggers. Fixed via monkey-patch in `train_gliner2.py` that saves the pre-eval best metric.

2. **Gradient checkpointing:** The stock GLiNER2 wheel uses `use_reentrant=True` which is incompatible with LoRA. A custom wheel (`gliner2-1.2.4-py3-none-any.whl`) with `use_reentrant=False` is required. Install with: `pip install /workspaces/snomed-ct-entity-linking/gliner2-1.2.4-py3-none-any.whl --force-reinstall --no-deps`

3. **Processor max length:** GLiNER2's processor pads to the longest sequence in the batch with no truncation. DeBERTa's O(n^2) disentangled attention makes this catastrophic for long sequences. Fixed via monkey-patch that truncates `input_ids` and `mapped_indices` at the processor level.

## Training Run History

All metrics below are **honest** (no abbreviation filtering, no lab table exclusion):

| Run | Config | Exact F1 | Partial F1 | Notes |
|-----|--------|----------|------------|-------|
| v6 | 4 classes, r=16, bs=32, ~9 epochs | 0.5616 | 0.7247 | Reference baseline |
| v7-gc | 4 classes, r=16, bs=128, GC, 20 epochs | 0.5782 | 0.7252 | Gradient checkpointing |
| v8 | 4 classes, r=16, bs=32, GC, 20 epochs | 0.5683 | 0.7191 | |
| v10 | 4 classes, r=8, bs=16, 5 epochs | 0.5017 | 0.7039 | Lower rank underperforms |
| v11 | 4 classes, r=16, bs=32, 15 epochs | 0.5659 | 0.7247 | |
| **v12** | **3 classes, r=16, bs=32, GC, 15 epochs** | **0.6473** | **0.7931** | **Best — no abbreviation class** |

v12 training command:
```bash
python scripts/train_gliner2.py \
    --data-dir data/gliner2-v12 \
    --output-dir models/gliner2-snomed-large-v12 \
    --base-model fastino/gliner2-large-v1 \
    --epochs 15 --batch-size 32 --grad-accum 1 \
    --encoder-lr 1e-5 --task-lr 5e-4 \
    --lora-r 16 --gradient-checkpointing --patience 10
```

GPU utilization: 98%, 13.5GB VRAM with batch=32 and gradient checkpointing.

## Key Finding: Abbreviation Class Removal

### The Problem

Early runs (v6-v11) used 4 entity classes, including "medical abbreviation". This caused two problems:

1. **Overlapping labels:** Abbreviations like `WBC`, `HTN`, `COPD` are also findings or procedures. In training data, these spans were labeled ONLY as "medical abbreviation" — never dual-labeled with their semantic class. This meant 4,128 abbreviation predictions (19% of all predictions) were discarded at eval time.

2. **Inflated metrics:** The original grid search filtered abbreviation annotations from the gold set too, removing ~6,000 annotations and inflating F1. The reported v6 score of 0.6547 was actually 0.5616 when evaluated honestly.

### The Fix

Removed the abbreviation class entirely in v12. All spans are now labeled with their semantic class only. The existing LLM pipeline handles abbreviation detection independently.

**Impact:** v12 achieved Exact F1=0.6473 (+8.6 points over v6's honest 0.5616).

## Key Finding: Structured Lab Table Problem

### Discovery

Error analysis on v12 revealed that procedures had a 50% miss rate. Investigation showed the bulk of failures were lab test abbreviations in structured result tables:

```
___ 06:25AM BLOOD WBC-8.4 RBC-4.06* Hgb-12.8* Hct-39.7*
MCV-98 MCH-31.7 MCHC-32.4 RDW-12.4 Plt ___
```

### Quantification

89% of training notes contain structured lab tables. 36.4% of all procedure annotations fall within these regions — almost entirely short abbreviations like `WBC`, `RBC`, `Hgb`, `Creat`, `Na`.

The model produces only **17 predictions** inside lab table regions out of 2,461 gold spans — essentially 0% recall. GLiNER2's biaffine span extraction architecture is fundamentally unsuited to these dense, repetitive structured patterns (unlike BIO token classification which handles them natively — the 2nd place competition solution trains on lab tables successfully with BIO tagging).

### Segmented Evaluation (v12)

| Segment | Gold | Pred | Exact F1 | Partial F1 |
|---|---|---|---|---|
| All annotations | 23,413 | 17,599 | 0.6098 | 0.7564 |
| **Non-lab only** | **20,952** | **17,582** | **0.6477** | **0.7976** |
| Lab table only | 2,461 | 17 | 0.0073 | 0.0105 |

Non-lab procedure F1 is 0.5733 vs 0.4604 overall, confirming lab tables drag the class down.

### Lab Table Detection (v2 filter)

A regex-based filter detects lab table regions with high precision:

```python
# Requires 2+ alpha chars in test name (excludes spinal levels like L1-2, C4-5)
LAB_VALUE_PATTERN = re.compile(r"[A-Z][A-Za-z][A-Za-z0-9]{0,4}-\d+\.?\d*\*?")
# Region = first match start to last match end + 15 char buffer (not full line)
```

Filter accuracy on training data:
- **False positives (non-procedures incorrectly removed):** 9 findings, 0 body parts out of 52,078 annotations (0.02% error)
- 3 of the 9 are genuinely debatable (`Bacteri-FEW` in urinalysis lines, `FOLATE-11.6` as a lab value, `Temp-38.0` in blood gas lines)

Code: `scripts/test_lab_table_filter.py`

### Recommendation

Remove lab table annotations from training. At inference time, use regex to extract lab test names from structured tables and map to SNOMED codes via a lookup dictionary — near-perfect recall on a pattern that the neural model can't learn.

## Error Analysis: Non-Lab Failures

With lab tables excluded, the remaining failures on test data break down as follows.

### Per-Class Performance (non-lab, v12)

| Class | Gold | Pred | Exact P | Exact R | Exact F1 |
|---|---|---|---|---|---|
| finding | 11,941 | 10,496 | 0.7457 | 0.6555 | 0.6977 |
| procedure | 5,535 | 4,097 | 0.6739 | 0.4988 | 0.5733 |
| body part | 3,476 | 2,989 | 0.6327 | 0.5440 | 0.5850 |

### Top Failure Categories

#### 1. Single-line lab values escaping the filter (~400 annotations)

Lab values on lines with only 1 `NAME-VALUE` pattern aren't caught by the 2-match-per-line rule:
- `Glucose` (124/128 missed, 97%), `Calcium` (81/81, 100%), `PTT` (68/68, 100%), `pH` (28/28, 100%), `TotBili` (20/23, 87%)

These are still structured data, just less dense. Expanding the regex or handling via dictionary post-processing at inference would cover them.

#### 2. Physical exam sub-headers as procedures

Section abbreviations in physical exam notes are annotated as examination procedures:
- `Physical examination` (83/173 missed) — spans like `Neck`, `PHYSICAL EXAM`
- `General examination` (26/96 missed) — `GEN`, `Gen`
- `Cardiovascular exam` (23/90 missed) — `CV`, `CARDIAC`

Annotation rate analysis shows these are annotated >80% of the time — the model should learn them.

#### 3. Drug names as therapy procedures

Drug names annotated as their therapy procedure:
- `prednisone` — 151 occurrences, only 27 annotated (17.9%)
- `coumadin`/`Coumadin` — 122 occurrences, 34 annotated (27.9%)
- `heparin` — 76 occurrences, 45 annotated (59.2%)

These have genuinely low annotation rates — the drug appears in medication lists (never annotated) and in clinical narrative (sometimes annotated as therapy).

#### 4. Ambiguous common words

Words with context-dependent annotation:
- `mild` — 366 occurrences, 167 annotated (45.6%). Annotated 100% in discharge diagnosis, 0% in medication sections.
- `negative` — 298 occurrences, 121 annotated (40.6%)
- `worse` — 53 occurrences, 21 annotated (39.6%)
- `normal` — 658 occurrences, 396 annotated (60.2%)

#### 5. Boundary mismatches (826 annotations)

Partial match (IoU >= 0.5) but not exact — the model finds the entity but with wrong boundaries. Common patterns: `mitral regurgitation` vs `Mild mitral regurgitation`, `blood cultures` vs `BLOOD CULTURE`.

### Annotation Rate Analysis

The commonly-missed spans fall into three tiers:

| Tier | Annotation Rate | Examples | Implication |
|---|---|---|---|
| High (>80%) | 80-100% | `CTAB`, `CV`, `HEENT`, `edema`, `Neuro`, `pain`, `antibiotics` | Model should learn these — failure is threshold/architecture, not imbalance |
| Medium (40-60%) | 40-60% | `mild` (45.6%), `normal` (60.2%), `heparin` (59.2%) | Genuinely ambiguous — annotation depends on section context |
| Low (<40%) | 18-40% | `prednisone` (17.9%), `coumadin` (26.5%), `evaluated` (30.5%), `worse` (39.6%) | Heavy imbalance — model sees these unannotated far more than annotated |

Note: Section headers are already prepended to training chunks (e.g., `[Discharge Medications]`), so the model has context to distinguish. The fact that it still struggles on the medium/low tier suggests the signal isn't strong enough through the section prefix alone.

## Comparison with 2nd Place Solution

The 2nd place competition entry used BERT-based BIO token classification:

- **Trains on everything** including lab tables — no special filtering
- **Removes only 3 sections:** Medications on Admission, Discharge Medications, and one variant
- **Class weights:** O label weighted at 0.142 vs 0.571 for all entity B/I labels (4x upweighting)
- **Chunking:** Standard sliding window over tokenized text (max 512, BiomedBERT), random offset during training for augmentation
- **Section headers:** Optionally prepended to chunks with zero offsets (same approach as ours)

BIO token classification handles dense structured text natively because each token independently gets a label. GLiNER2's biaffine decoder must score all (start, end) pairs, which struggles when entities are packed tightly.

## Next Steps

### Tier 1 — No retraining needed
1. **Threshold sweep** — the current 0.5 threshold may be too conservative. Many missed spans might be predicted at 0.3-0.4 confidence. Lower threshold with higher recall could be better since downstream linking filters false positives.
2. **Expand lab filter** — catch single-value lab lines to stop penalizing the model for ~400 more annotations it will never learn.

### Tier 2 — Retrain with cleaner data
3. **Remove lab table annotations from training** — stop teaching the model a pattern it can't learn. Let the procedure class focus on natural language mentions.
4. **Grid search window/overlap** on the retrained model — old grid search results are stale.

### Tier 3 — Targeted improvements
5. **Oversample positive examples** for low-annotation-rate spans (`prednisone`, `coumadin`, `worse`, etc.)
6. **Boundary post-processing** — snap predictions to word boundaries, expand to include common modifiers. Could recover some of the 826 boundary-mismatch annotations.
7. **Entity description tuning** — make procedure description explicitly mention drug/medication names and physical exam abbreviations.

### Inference Pipeline Design
At inference time, combine:
- **GLiNER2** for natural language span extraction (findings, procedures, body parts in running text)
- **Regex post-processor** for structured lab tables (near-perfect recall via `TESTNAME-VALUE` pattern matching + SNOMED code lookup dictionary)
- **Separate abbreviation detection** via existing LLM pipeline

## File Inventory

| File | Purpose |
|---|---|
| `scripts/prepare_gliner2_data.py` | CSV → GLiNER2 training data with section-aware chunking |
| `scripts/train_gliner2.py` | Training with LoRA, early stopping fix, gradient checkpointing |
| `scripts/eval_gliner2.py` | Test set evaluation with span-level P/R/F1 |
| `scripts/grid_search_window_overlap.py` | Grid search over window/overlap at inference |
| `scripts/analyze_lab_tables.py` | Lab table detection and annotation overlap analysis |
| `scripts/test_lab_table_filter.py` | Lab table filter testing with false positive/negative checks |
| `models/gliner2-snomed-large-v12/best/` | Best model checkpoint (LoRA adapter) |
| `data/gliner2-v12/` | Training data (3,813 train, 673 val, 1,655 test chunks) |
