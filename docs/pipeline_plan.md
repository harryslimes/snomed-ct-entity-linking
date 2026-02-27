# SNOMED CT Entity Linking Pipeline Plan

## Competition Metric

Macro-averaged character-level IoU across all SNOMED concept IDs. Each concept gets equal weight regardless of frequency. This means:

- Rare/unseen concepts matter as much as common ones
- False positive concept IDs are extremely expensive (each contributes IoU=0 to the average AND inflates the denominator)
- Span boundary accuracy matters (character-level overlap, not just detection)

## Current Baseline Performance (old-challenge-split)

| Segment | Concepts | Weight | KIRI IoU | Notes |
|---|---|---|---|---|
| Seen + Full-term | 2,299 | 56.7% | 0.58 | Dictionary works OK |
| Seen + Abbreviation | 476 | 11.7% | 0.53 | Dictionary has gaps |
| Unseen + Full-term | 1,111 | 27.4% | 0.29 | Dictionary misses 67% entirely |
| Unseen + Abbreviation | 166 | 4.1% | 0.04 | Nearly zero detection |
| Pure FP concepts | 544 | drag | 0.00 | All contribute zero, inflate denominator |
| **Overall** | **4,596** | | **0.474** | |

GLiNER2 span detection performance (v12, 3-class, lab tables excluded):
- Seen full-term spans: 74% overlap detection, 64% exact match
- Unseen full-term spans: 74% overlap detection, 47% exact match
- Seen abbreviations: 69% overlap, 63% exact
- Unseen abbreviations: 52% overlap, 31% exact

## Target Estimates

| Scenario | Score | vs KIRI | Key assumption |
|---|---|---|---|
| Conservative | 0.55 | +0.08 | Decent linking, some FP leakage |
| Moderate | 0.61 | +0.14 | Good linking, good FP filtering |
| Optimistic | 0.68 | +0.21 | Strong linking, minimal FPs |

Competition 1st place on private LB: 0.4202 (different, harder test set).

---

## Pipeline Architecture

```
Clinical Note
     │
     ├──► [1] Abbreviation Detector (LLM + Dictionary)
     │         → abbreviation spans with candidate concept IDs
     │
     ├──► [2] GLiNER2 Span Extractor
     │         → medical entity spans (finding/procedure/body)
     │
     ▼
[3] Merge & Deduplicate Spans
     │
     ▼
[4] Concept Linker (Embedding Search + LLM Reranker)
     │    → each span gets a SNOMED concept ID
     │
     ▼
[5] FP Validation (LLM + Rules)
     │    → remove spurious annotations
     │
     ▼
Final Predictions (note_id, start, end, concept_id)
```

---

## Phase 1: Evaluation Harness & Concept Linking on Seen+Full

**Goal**: Build the scoring infrastructure and get concept linking working well on the easiest segment (57% of competition weight).

**Why first**: Concept linking is the real bottleneck. GLiNER2 detects spans but assigns entity types, not SNOMED concept IDs. The gap between span detection ceiling (0.68) and actual KIRI score (0.47) is almost entirely linking quality + FP drag.

### 1.1 Evaluation Harness

- Wrap `runtime_scoring.py` to score any pipeline output CSV
- Per-concept IoU breakdown with frequency bucketing
- Per-segment reporting (seen/unseen × abbrev/full)
- Diff two pipeline runs to see exactly which concepts improved/regressed

### 1.2 SNOMED Embedding Index

- Embed all SNOMED concept descriptions (FSN + synonyms) using a medical embedding model
- Index with FAISS or similar for fast top-k retrieval
- At query time: embed the detected span text → retrieve top-k candidate concepts

### 1.3 Concept Linking Pipeline

For each GLiNER2-detected span:
1. Embed span text + surrounding context window (~50 chars each side)
2. Retrieve top-k SNOMED candidates by cosine similarity
3. If top-1 confidence is high enough, accept directly
4. Otherwise, LLM reranks candidates given the full sentence context

### 1.4 Validation

- Run on training data where we know ground truth concept IDs
- Measure concept linking accuracy: what % of correctly-detected spans get the right concept?
- Target: >80% linking accuracy on seen+full concepts

---

## Phase 2: FP Elimination Rules

**Goal**: Reduce false positive concepts from ~544 to under 100. Each eliminated FP shrinks the denominator and removes a zero from the numerator.

**Why second**: This is the highest-ROI intervention after basic linking. Purely subtractive — we're deleting bad predictions, not adding new ones, so there's no risk of creating new errors.

### 2.1 Training Data Error Analysis

- Run the full Phase 1 pipeline on training data
- Collect every predicted concept that doesn't appear in gold
- Categorise FP types:
  - **Section-based FPs**: medical terms in boilerplate sections (discharge medications, instructions, headers)
  - **Ambiguity FPs**: common English words matched to medical concepts ("right" → laterality, "regular" → heart rhythm)
  - **Partial match FPs**: substring of a longer term matched to wrong concept
  - **Lab table FPs**: annotations in structured lab value tables (already have filter for this)

### 2.2 Rule Generation

- For each FP category, generate compact rules (<200 words each)
- Rules are either deterministic (section blacklists, known false matches) or probabilistic (LLM applies contextual judgement)
- Store rules as JSON with: pattern, action (remove/flag), confidence threshold, examples

### 2.3 LLM Validation Pass

- For each predicted annotation, check against rules
- LLM receives: the span, surrounding context, predicted concept description, and relevant rules
- LLM outputs: keep/remove with confidence
- High-confidence removes applied automatically, borderline cases kept (bias toward recall)

---

## Phase 3: Abbreviation Dictionary & LLM Disambiguation

**Goal**: Build a comprehensive abbreviation → SNOMED mapping and handle disambiguation in context.

**Why third**: Abbreviations are the worst-performing segment (unseen+abbrev = 0.04 IoU). A separate abbreviation pipeline handles them better than GLiNER2 because abbreviations are fundamentally a lookup problem, not a span extraction problem.

### 3.1 Build Abbreviation Dictionary

Sources:
- **SNOMED synonyms**: Many concepts have abbreviation synonyms in their description files (e.g., "DVT" is a synonym for deep vein thrombosis)
- **Training data mining**: Find concepts that have both abbreviated and expanded annotations in training data
- **LLM classification**: Use the abbreviation classifier we already built (`abbreviation_classification_test.json`) to identify which training spans are abbreviations, then map abbreviation → concept ID from the training annotations
- **Manual curation**: Common clinical abbreviations not in SNOMED (e.g., "BKA" → below-knee amputation)

Dictionary structure:
```json
{
  "DVT": [{"concept_id": 128053003, "term": "Deep vein thrombosis", "confidence": 0.95}],
  "Ca": [
    {"concept_id": 312472004, "term": "Calcium measurement", "confidence": 0.6, "context_hint": "lab/chemistry"},
    {"concept_id": 363346000, "term": "Malignant neoplasm", "confidence": 0.6, "context_hint": "diagnosis/oncology"}
  ]
}
```

### 3.2 Abbreviation Detection

- Scan clinical note for tokens matching dictionary keys
- Use section context to filter (e.g., don't match single-letter abbreviations in narrative text)
- Apply section-specific rules (e.g., "Ca" in "Pertinent Results" = calcium, "Ca" in "History of Present Illness" = more likely cancer)

### 3.3 LLM Disambiguation

For abbreviations with multiple candidate concepts:
- Pass LLM the surrounding sentence + candidate concept descriptions
- LLM picks the most likely concept given clinical context
- Include section-specific rules in the prompt

### 3.4 Negative Rules

Rules for when NOT to create an abbreviation annotation:
- Section blacklists (e.g., don't annotate abbreviations in medication lists unless they're diagnosis abbreviations)
- Contextual negation (e.g., "no DVT" — still annotate, the span is the concept)
- Vertebral levels (T7, L2, C4-5) vs abbreviations — use surrounding context to disambiguate

---

## Phase 4: Unseen Concept Linking via Embedding Search

**Goal**: Link the 1,111 unseen full-term concepts that GLiNER2 detects but KIRI's dictionary can't match.

**Why fourth**: This is the single biggest opportunity by weight (27.4% of concepts, KIRI=0.29 IoU). But it requires the embedding index and linking pipeline from Phase 1 to already be working well.

### 4.1 Enhanced Embedding Search

- For spans that don't match any dictionary entry exactly, fall back to embedding similarity
- Use the span text + context window as the query
- Retrieve top-10 SNOMED candidates
- Apply a confidence threshold — if top-1 similarity is below threshold, don't annotate (avoid creating FP concepts)

### 4.2 LLM Reranking for Unseen Concepts

- For borderline cases (multiple candidates with similar scores), use LLM to pick
- The LLM sees: span text, sentence context, section header, and top-5 candidate concept descriptions
- Critical: the LLM must be allowed to say "none of these" to avoid FP concepts

### 4.3 Confidence Calibration

- On training data, measure: at what embedding similarity threshold does linking accuracy drop below 50%?
- Set the threshold conservatively — for unseen concepts, a missed annotation (FN) costs less than a wrong concept (which creates both an FN and an FP)

---

## Phase 5: Rules-Based Final Pass

**Goal**: Learn systematic error patterns from training data and apply corrections.

### 5.1 Error Pattern Mining

Run the complete pipeline (Phases 1-4) on training data. For each concept where pipeline IoU < 1.0, categorise the error:
- **Span boundary errors**: predicted span is too short or too long (e.g., "chest pain" vs "chest")
- **Missed spans**: gold annotation exists but pipeline produced nothing
- **Wrong concept**: span detected but linked to wrong concept ID
- **Spurious annotation**: pipeline predicted something with no gold match

### 5.2 Rule Inference

For each common error pattern, generate a rule:
- Span boundary rules: "For concept X, the span should include/exclude the following words..."
- Contextual rules: "Concept X should only be annotated when preceded/followed by Y..."
- Section rules: "In section Z, don't annotate concepts of type W..."
- Mutual exclusion rules: "If concept A is annotated, don't also annotate concept B at overlapping positions..."

### 5.3 LLM Application

- Final pass over all annotations with the rulebook
- LLM receives: the annotation, context, and relevant rules
- Can adjust span boundaries, remove annotations, or flag for manual review
- Focus on the concepts the pipeline most struggles with (lowest IoU on training data)

---

## Data & Infrastructure Requirements

| Component | Resource | Notes |
|---|---|---|
| SNOMED embedding index | FAISS index + embedding model | Build once, reuse |
| Abbreviation dictionary | JSON file | Built from SNOMED + training data |
| GLiNER2 model | v12 adapter on DeBERTa-v3-large | Already trained |
| LLM (Qwen3-30B) | vLLM on GPU | For disambiguation, reranking, validation |
| Rule store | JSON files | Generated from training error analysis |
| Evaluation | `runtime_scoring.py` | Competition metric |

## Files

| File | Purpose |
|---|---|
| `scripts/analyze_concept_performance.py` | Per-concept performance analysis |
| `scripts/classify_abbreviations_llm.py` | LLM abbreviation classification |
| `abbreviation_classification_test.json` | Abbreviation labels for all spans |
| `super-dictionary/runtime_scoring.py` | Competition metric scorer |
| `scripts/eval_gliner2.py` | GLiNER2 span detection evaluation |

## Key Risks

1. **Concept linking accuracy**: If linking is wrong, we create FP concepts (double penalty). Must be high-precision, even at cost of some recall.
2. **LLM latency/cost**: Multiple LLM calls per note (abbreviation disambiguation, concept reranking, validation). Need efficient batching via vLLM.
3. **Rule overfitting**: Rules learned from training data may not generalise to test concepts. Keep rules general and concept-agnostic where possible.
4. **Abbreviation ambiguity**: Many abbreviations have multiple valid expansions. Context helps but isn't always sufficient. Default to not annotating when uncertain.
