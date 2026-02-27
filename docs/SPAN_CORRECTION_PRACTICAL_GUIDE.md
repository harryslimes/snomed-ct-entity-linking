# Practical Guide: LLM Span Boundary Correction

## Key Findings from Analysis

### Critical Insight: Two Types of Errors

Looking at real examples, we have:
1. **Small boundary errors**: Off by a few characters (e.g., "work-up" at position 8280 vs 8412 = 132 chars off)
2. **Wrong instance matching**: Matching the wrong occurrence of a concept in the note (e.g., "DKA" at position 262 vs "Diabetic Ketoacidosis" at position 7719 = **7,457 chars off!**)

**The majority of "span errors" are actually matching the WRONG INSTANCE of the concept!**

This changes our approach significantly.

---

## Context Requirements (Empirical Analysis)

From 200 gold annotations:
- **Mean context needed**: 42 chars (one sentence)
- **Median**: 46 chars
- **95th percentile**: 65 chars

**Recommendation**: Use **150-200 chars on each side** for safety → ~300-500 char total context

---

## Entity Density (For Batching)

Window size analysis:
- **300 chars**: ~9 entities (good)
- **500 chars**: ~15 entities (optimal!)
- **1000 chars**: ~28 entities (too many)
- **2000 chars**: ~52 entities (way too many)

**95th percentile**: 32 entities in a 500-char window (manageable)

**Recommendation**: Use **500-char windows with 100-char overlap**

---

## Efficiency Gains from Batching

- **24,605 predictions** in test set
- **Per-entity approach**: 24,605 LLM calls
- **Batched approach (500-char windows)**: ~5,000 LLM calls
- **Efficiency gain**: **5x reduction!**

At $0.001 per 1K tokens:
- Per-entity: ~$50 for test set
- Batched: **~$10 for test set**

---

## Recommended Approach: Two-Stage Pipeline

### Stage 1: Instance Matching (High Priority)
**Problem**: Super dictionary often matches the wrong occurrence of a concept in the note

**Example**:
```
Note has "DKA" (position 262) and "Diabetic Ketoacidosis" (position 7719)
Prediction: 262-265 "DKA"
Gold: 7719-7740 "Diabetic Ketoacidosis"
Error: 7,457 chars off! (wrong instance)
```

**Solution**: Use LLM to choose the correct instance from all candidates

**Prompt format**:
```
You are correcting clinical entity annotations. A concept has been predicted in this note, but may be at the wrong location.

Full note (truncated if long):
{note_text}

Predicted location: [{pred_start}, {pred_end}] "{pred_span}"
Concept: {concept_name}

Task: Find ALL mentions of this concept in the note. Choose the most appropriate one.

Output format (one per line):
CANDIDATE: <start> <end> "<exact text>"

After listing all candidates, output:
CORRECT: <start> <end>
```

**Expected impact**: Fixes large-distance errors (7K+ chars off) → probably **+0.05-0.08 IoU**

### Stage 2: Boundary Refinement (Medium Priority)
**Problem**: Once we have the right instance, the span boundaries might be slightly off

**Example**:
```
Predicted: [8280, 8287] "work-up"
Gold:      [8412, 8419] "work-up"
Error: 132 chars off (same word, different occurrence)
```

**Solution**: For matches within ~200 chars, use LLM to refine exact boundaries

**Prompt format**: See below (multi-entity batch)

**Expected impact**: Fixes small boundary errors → probably **+0.03-0.05 IoU**

---

## Optimal Prompt Format: Multi-Entity Batch

Based on analysis, **Format 3 (multi-entity batch)** is best:

### Why?
- **5x more efficient** (fewer LLM calls)
- LLM sees **relationships between nearby entities**
- More **natural context** (full text region)
- **Better for fine-tuning** (more diverse examples)

### Format

```
Correct span boundaries for clinical entities in this note excerpt.

Text:
{500_char_window}

Entities to correct:
E1: [10, 28] "No Known Allergies" - Allergic disposition (finding)
E2: [150, 162] "colon cancer" - Malignant neoplasm of colon (disorder)
E3: [200, 215] "right colectomy" - Colectomy (procedure)

For each entity, output one line with corrected positions (relative to the text above):
<ID>: <start> <end> "<exact text>"

If an entity is already correct:
<ID>: CORRECT

If an entity should be deleted (false positive):
<ID>: DELETE

Output:
```

### Expected output:
```
E1: CORRECT
E2: 150 175 "colon
cancer"
E3: DELETE
```

---

## Implementation Strategy

### Option A: Two-Stage Pipeline (Recommended for Best Results)

**Stage 1**: Instance matching (find correct occurrence)
- For each prediction, extract all candidate mentions in the note
- LLM chooses the best instance
- Updates the coarse position

**Stage 2**: Boundary refinement (fix exact spans)
- Use 500-char windows with overlap
- Batch 10-15 entities per LLM call
- Refine exact character positions

**Pros**:
- Addresses both error types
- Highest potential gain (+0.08-0.13 IoU)
- Can run stages separately for debugging

**Cons**:
- Two passes over the data
- More complex implementation

### Option B: Single-Stage Batch Processing (Recommended for Simplicity)

**Approach**: Process notes in 500-char windows, handling both instance and boundary errors together

**Prompt format**:
```
Correct clinical entity annotations in this note excerpt.

Context (showing ±200 chars around this region):
{extended_context}

Region to correct:
{500_char_window}

Entities predicted in this region:
E1: [10, 28] "No Known Allergies" - Allergic disposition
E2: [150, 162] "colon cancer" - Malignant neoplasm of colon

For each entity:
1. If it's in the WRONG location (wrong instance of concept), find the correct instance in the CONTEXT
2. If the span boundary is wrong, fix it
3. If it's a false positive, DELETE it
4. If it's correct, mark CORRECT

Output one line per entity:
<ID>: <start> <end> "<exact text>" [MOVED if position changed significantly]
<ID>: CORRECT
<ID>: DELETE

Output:
```

**Pros**:
- Simpler implementation (one pass)
- Still gets 5x efficiency from batching
- Handles both error types

**Cons**:
- Slightly lower potential gain than two-stage

---

## Fine-Tuning Data Generation

### Training Data Format

For each 500-char window with entities:

**Input**:
```json
{
  "window_text": "The patient presented to clinic with a diagnosis of right colon cancer...",
  "window_start": 200,
  "entities": [
    {
      "id": "E1",
      "pred_start": 67,
      "pred_end": 85,
      "pred_text": "colon cancer",
      "concept_id": 363406005,
      "concept_name": "Malignant neoplasm of colon"
    }
  ]
}
```

**Output (Gold)**:
```json
{
  "corrections": [
    {
      "id": "E1",
      "action": "CORRECT",
      "gold_start": 67,
      "gold_end": 85,
      "gold_text": "colon cancer"
    }
  ]
}
```

Or for corrections:
```json
{
  "corrections": [
    {
      "id": "E1",
      "action": "FIX",
      "gold_start": 70,
      "gold_end": 88,
      "gold_text": "colon \ncancer"
    }
  ]
}
```

### Generation Process

```python
def generate_training_data(train_notes, train_gold, train_pred):
    """Generate training examples for span correction."""
    examples = []

    for note_id in train_notes.index:
        note_text = train_notes.loc[note_id]['text']
        note_gold = train_gold[train_gold['note_id'] == note_id]
        note_pred = train_pred[train_pred['note_id'] == note_id]

        # Process note in 500-char windows with 100-char overlap
        window_size = 500
        stride = 400

        for window_start in range(0, len(note_text), stride):
            window_end = min(window_start + window_size, len(note_text))
            window_text = note_text[window_start:window_end]

            # Get predictions in this window
            window_preds = note_pred[
                (note_pred['start'] >= window_start) &
                (note_pred['start'] < window_end)
            ]

            if len(window_preds) == 0:
                continue

            # Build entity list with corrections
            entities = []
            corrections = []

            for idx, pred in window_preds.iterrows():
                pred_start = int(pred['start']) - window_start
                pred_end = int(pred['end']) - window_start
                concept_id = int(pred['concept_id'])

                # Find matching gold annotation
                matching_gold = note_gold[
                    (note_gold['concept_id'] == concept_id) &
                    # Within reasonable distance (not a different instance)
                    (abs(note_gold['start'] - pred['start']) < 200)
                ]

                if len(matching_gold) == 0:
                    # False positive - should be deleted
                    corrections.append({
                        'id': f'E{len(entities)+1}',
                        'action': 'DELETE'
                    })
                elif len(matching_gold) == 1:
                    gold = matching_gold.iloc[0]
                    gold_start = int(gold['start']) - window_start
                    gold_end = int(gold['end']) - window_start

                    if pred_start == gold_start and pred_end == gold_end:
                        # Perfect match
                        corrections.append({
                            'id': f'E{len(entities)+1}',
                            'action': 'CORRECT'
                        })
                    else:
                        # Needs boundary fix
                        corrections.append({
                            'id': f'E{len(entities)+1}',
                            'action': 'FIX',
                            'gold_start': gold_start,
                            'gold_end': gold_end,
                            'gold_text': note_text[gold['start']:gold['end']]
                        })

                entities.append({
                    'id': f'E{len(entities)+1}',
                    'pred_start': pred_start,
                    'pred_end': pred_end,
                    'pred_text': note_text[pred['start']:pred['end']],
                    'concept_id': concept_id,
                    'concept_name': concept_names.get(concept_id, 'Unknown')
                })

            examples.append({
                'window_text': window_text,
                'window_start': window_start,
                'entities': entities,
                'corrections': corrections
            })

    return examples
```

---

## Expected Results

### Conservative Estimates

| Stage | Gain | Cumulative Score |
|-------|------|------------------|
| Baseline | - | 0.6152 |
| Instance matching (50% fixed) | +0.05 | 0.6652 |
| Boundary refinement (50% fixed) | +0.03 | 0.6952 |
| **Total** | **+0.08** | **0.6952** |

### Optimistic Estimates

| Stage | Gain | Cumulative Score |
|-------|------|------------------|
| Baseline | - | 0.6152 |
| Instance matching (75% fixed) | +0.08 | 0.6952 |
| Boundary refinement (75% fixed) | +0.05 | 0.7452 |
| **Total** | **+0.13** | **0.7452** |

---

## Implementation Checklist

- [ ] Generate training data from 204 training notes
  - [ ] Run super dictionary on train notes
  - [ ] Create 500-char windows
  - [ ] Match predictions to gold annotations
  - [ ] Generate ~5,000-10,000 training examples

- [ ] Fine-tune model (Llama-3-8B with LoRA)
  - [ ] Format data for instruction tuning
  - [ ] Train for 3-5 epochs
  - [ ] Validate on hold-out set (20% of train)
  - [ ] Monitor: exact match accuracy, IoU improvement

- [ ] Test on old-challenge-split
  - [ ] Apply model to super dict predictions
  - [ ] Calculate new char-level IoU
  - [ ] Analyze error types (instance vs boundary)
  - [ ] Iterate if needed

- [ ] Combine with FP deletion (Phase 2)
  - [ ] Train binary classifier
  - [ ] Apply to span-corrected predictions
  - [ ] Target: **0.75+ mean IoU**

---

## Compute Requirements

### For 5K examples (500-char windows, ~10 entities each)

**Fine-tuning**:
- GPU: 1x A100 (40GB) or 2x RTX 4090
- Training time: 6-12 hours
- Storage: ~50GB
- Cost (cloud): ~$50-75

**Inference** (test set):
- ~5K LLM calls at 500 tokens each = 2.5M tokens
- Using vLLM with batching: ~30-60 minutes
- Cost (OpenAI API): ~$2.50
- Cost (local vLLM): Free!

---

## Key Success Factors

1. **Address instance matching first** - this is the bigger problem (7K+ char errors)
2. **Use batching** - 5x efficiency gain, better context
3. **Start with strong baseline** - Llama-3-8B or better
4. **Validate carefully** - watch for overfitting to training concepts
5. **Measure both metrics**:
   - Exact match accuracy (did we output correct spans?)
   - Char-level IoU (actual competition metric)

---

## Next Steps

1. **Run training data generation script** (create the script from pseudo-code above)
2. **Analyze training data quality** (check distribution of corrections)
3. **Fine-tune small model first** (Llama-3-8B, 3 epochs, see if it learns)
4. **Test on validation set** (hold out 20% of train notes)
5. **Apply to test set** (measure actual IoU gain)
6. **Iterate** based on results

**Expected timeline**: 1-2 weeks to Phase 1 completion (span correction only)

**Expected result**: 0.6152 → 0.70-0.75 mean IoU (**26% improvement!**)
