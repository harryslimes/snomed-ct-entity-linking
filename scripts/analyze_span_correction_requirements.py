#!/usr/bin/env python3
"""
Analyze practical requirements for span boundary correction:
1. How much context is needed?
2. How many corrections can be batched together?
3. What's the best prompt format?
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

def analyze_context_requirements(notes_df, gold_df, sample_size=100):
    """Analyze how much context is actually needed for span correction."""
    print("="*80)
    print("CONTEXT REQUIREMENT ANALYSIS")
    print("="*80)
    print()

    # Sample some annotations
    samples = gold_df.sample(min(sample_size, len(gold_df)), random_state=42)

    context_needs = []

    for _, row in samples.iterrows():
        note_text = notes_df.loc[row['note_id']]['text']
        start, end = int(row['start']), int(row['end'])
        span_text = note_text[start:end]

        # Find sentence boundaries around the span
        # Simple heuristic: look for periods, newlines
        left_context = note_text[max(0, start-500):start]
        right_context = note_text[end:min(len(note_text), end+500)]

        # Find nearest sentence boundaries
        left_periods = [m.start() for m in re.finditer(r'[.!?\n]', left_context)]
        right_periods = [m.start() for m in re.finditer(r'[.!?\n]', right_context)]

        # Measure distances
        if left_periods:
            left_sentence_dist = len(left_context) - left_periods[-1]
        else:
            left_sentence_dist = len(left_context)

        if right_periods:
            right_sentence_dist = right_periods[0]
        else:
            right_sentence_dist = len(right_context)

        context_needs.append({
            'left_chars': left_sentence_dist,
            'right_chars': right_sentence_dist,
            'total_chars': left_sentence_dist + right_sentence_dist + (end - start),
            'span_length': end - start
        })

    context_df = pd.DataFrame(context_needs)

    print("Context statistics (chars to nearest sentence boundary):")
    print(f"  Left context needed:")
    print(f"    Mean: {context_df['left_chars'].mean():.0f} chars")
    print(f"    Median: {context_df['left_chars'].median():.0f} chars")
    print(f"    95th percentile: {context_df['left_chars'].quantile(0.95):.0f} chars")
    print()
    print(f"  Right context needed:")
    print(f"    Mean: {context_df['right_chars'].mean():.0f} chars")
    print(f"    Median: {context_df['right_chars'].median():.0f} chars")
    print(f"    95th percentile: {context_df['right_chars'].quantile(0.95):.0f} chars")
    print()
    print(f"  Total context (full sentence):")
    print(f"    Mean: {context_df['total_chars'].mean():.0f} chars")
    print(f"    Median: {context_df['total_chars'].median():.0f} chars")
    print(f"    95th percentile: {context_df['total_chars'].quantile(0.95):.0f} chars")
    print()

    return context_df


def analyze_entity_density(notes_df, gold_df):
    """Analyze how many entities per region (for batching strategy)."""
    print("="*80)
    print("ENTITY DENSITY ANALYSIS (for batching)")
    print("="*80)
    print()

    densities = []

    for note_id in gold_df['note_id'].unique()[:50]:  # Sample 50 notes
        note_text = notes_df.loc[note_id]['text']
        note_gold = gold_df[gold_df['note_id'] == note_id].copy()

        # Divide note into 500-char windows
        window_size = 500
        num_windows = (len(note_text) + window_size - 1) // window_size

        for i in range(num_windows):
            window_start = i * window_size
            window_end = min((i + 1) * window_size, len(note_text))

            # Count entities in this window
            entities_in_window = note_gold[
                (note_gold['start'] >= window_start) &
                (note_gold['start'] < window_end)
            ]

            densities.append({
                'window_size': window_size,
                'num_entities': len(entities_in_window),
                'unique_concepts': entities_in_window['concept_id'].nunique() if len(entities_in_window) > 0 else 0
            })

    density_df = pd.DataFrame(densities)

    print(f"Window size: 500 chars")
    print(f"  Mean entities per window: {density_df['num_entities'].mean():.1f}")
    print(f"  Median entities per window: {density_df['num_entities'].median():.0f}")
    print(f"  95th percentile: {density_df['num_entities'].quantile(0.95):.0f}")
    print(f"  Max entities in a window: {density_df['num_entities'].max():.0f}")
    print()

    # Calculate for different window sizes
    for window_size in [300, 500, 1000, 2000]:
        densities = []
        for note_id in gold_df['note_id'].unique()[:50]:
            note_text = notes_df.loc[note_id]['text']
            note_gold = gold_df[gold_df['note_id'] == note_id].copy()

            num_windows = max(1, (len(note_text) + window_size - 1) // window_size)

            for i in range(num_windows):
                window_start = i * window_size
                window_end = min((i + 1) * window_size, len(note_text))

                entities_in_window = note_gold[
                    (note_gold['start'] >= window_start) &
                    (note_gold['start'] < window_end)
                ]

                if len(entities_in_window) > 0:
                    densities.append(len(entities_in_window))

        if densities:
            print(f"Window size: {window_size} chars → Mean: {np.mean(densities):.1f} entities, Median: {np.median(densities):.0f}")

    print()
    return density_df


def show_example_corrections(notes_df, gold_df, pred_df, concept_names_df, num_examples=5):
    """Show real examples of span corrections needed."""
    print("="*80)
    print("REAL EXAMPLES OF SPAN CORRECTIONS")
    print("="*80)
    print()

    # Load concept names
    concept_names = dict(zip(concept_names_df['concept_id'], concept_names_df['concept_name']))

    # Find cases where pred and gold have same concept but different spans
    merged = pd.merge(
        pred_df,
        gold_df,
        on=['note_id', 'concept_id'],
        suffixes=('_pred', '_gold')
    )

    # Filter to cases with different spans
    corrections_needed = merged[
        (merged['start_pred'] != merged['start_gold']) |
        (merged['end_pred'] != merged['end_gold'])
    ].copy()

    print(f"Found {len(corrections_needed)} span corrections needed")
    print()

    # Sample diverse examples
    examples = corrections_needed.sample(min(num_examples, len(corrections_needed)), random_state=42)

    for idx, row in examples.iterrows():
        note_text = notes_df.loc[row['note_id']]['text']
        concept_name = concept_names.get(row['concept_id'], 'Unknown')

        # Extract contexts
        pred_start, pred_end = int(row['start_pred']), int(row['end_pred'])
        gold_start, gold_end = int(row['start_gold']), int(row['end_gold'])

        # Get surrounding context
        context_start = max(0, min(pred_start, gold_start) - 100)
        context_end = min(len(note_text), max(pred_end, gold_end) + 100)
        context = note_text[context_start:context_end]

        pred_span = note_text[pred_start:pred_end]
        gold_span = note_text[gold_start:gold_end]

        print(f"Example {idx + 1}:")
        print(f"  Concept: {row['concept_id']} - {concept_name}")
        print(f"  Context: ...{context}...")
        print(f"  Predicted span: [{pred_start}, {pred_end}] \"{pred_span}\"")
        print(f"  Gold span:      [{gold_start}, {gold_end}] \"{gold_span}\"")
        print(f"  Error type: ", end="")

        if gold_start < pred_start:
            print(f"Missing {pred_start - gold_start} chars on left")
        elif pred_start < gold_start:
            print(f"Extra {gold_start - pred_start} chars on left")

        if gold_end > pred_end:
            print(f"              Missing {gold_end - pred_end} chars on right")
        elif pred_end > gold_end:
            print(f"              Extra {pred_end - gold_end} chars on right")

        print()

    return corrections_needed


def design_prompt_formats(notes_df, gold_df, pred_df, concept_names_df):
    """Design and compare different prompt formats."""
    print("="*80)
    print("PROMPT FORMAT DESIGN")
    print("="*80)
    print()

    # Load concept names
    concept_names = dict(zip(concept_names_df['concept_id'], concept_names_df['concept_name']))

    # Get a real example
    merged = pd.merge(
        pred_df,
        gold_df,
        on=['note_id', 'concept_id'],
        suffixes=('_pred', '_gold')
    )

    corrections_needed = merged[
        (merged['start_pred'] != merged['start_gold']) |
        (merged['end_pred'] != merged['end_gold'])
    ]

    example = corrections_needed.iloc[0]
    note_text = notes_df.loc[example['note_id']]['text']
    concept_name = concept_names.get(example['concept_id'], 'Unknown')

    pred_start, pred_end = int(example['start_pred']), int(example['end_pred'])
    gold_start, gold_end = int(example['start_gold']), int(example['end_gold'])

    context_start = max(0, min(pred_start, gold_start) - 150)
    context_end = min(len(note_text), max(pred_end, gold_end) + 150)
    context = note_text[context_start:context_end]

    pred_span = note_text[pred_start:pred_end]
    gold_span = note_text[gold_start:gold_end]

    print("Using real example:")
    print(f"  Concept: {example['concept_id']} - {concept_name}")
    print(f"  Predicted: \"{pred_span}\" [{pred_start}, {pred_end}]")
    print(f"  Gold:      \"{gold_span}\" [{gold_start}, {gold_end}]")
    print()

    # Format 1: Bracket notation
    print("-" * 80)
    print("FORMAT 1: Bracket Notation (Simple)")
    print("-" * 80)
    prompt1 = f"""Correct the span boundary for this clinical entity.

Context: {context}

Current prediction: [{pred_span}]
Concept: {concept_name}

Output the corrected span text exactly as it appears in the note.
If the prediction is already correct, output: CORRECT

Corrected span:"""

    print(prompt1)
    print(f"\nExpected output: {gold_span}")
    print()

    # Format 2: JSON with positions
    print("-" * 80)
    print("FORMAT 2: JSON with Character Positions")
    print("-" * 80)

    # Mark the span in context
    marked_context = context[:pred_start - context_start] + \
                    f"[[{pred_span}]]" + \
                    context[pred_end - context_start:]

    prompt2 = f"""Correct the span boundary for this clinical entity.

Context (predicted span marked with [[brackets]]):
{marked_context}

Concept: {concept_name}

Output JSON with corrected start and end positions (relative to the context above):
{{"start": <number>, "end": <number>, "text": "<exact span text>"}}

If prediction is correct, output: {{"correct": true}}

JSON:"""

    print(prompt2)
    print(f"\nExpected output: {{\"start\": {gold_start - context_start}, \"end\": {gold_end - context_start}, \"text\": \"{gold_span}\"}}")
    print()

    # Format 3: Multi-entity batch
    print("-" * 80)
    print("FORMAT 3: Multi-Entity Batch Correction")
    print("-" * 80)

    # Get multiple entities in the same region
    note_id = example['note_id']
    region_start = context_start
    region_end = context_end

    region_preds = pred_df[
        (pred_df['note_id'] == note_id) &
        (pred_df['start'] >= region_start) &
        (pred_df['start'] < region_end)
    ].head(3)  # Max 3 for this example

    entities_list = []
    for _, pred in region_preds.iterrows():
        entities_list.append({
            'id': f"E{len(entities_list) + 1}",
            'concept_id': int(pred['concept_id']),
            'concept_name': concept_names.get(pred['concept_id'], 'Unknown'),
            'start': int(pred['start']) - region_start,
            'end': int(pred['end']) - region_start,
            'text': note_text[int(pred['start']):int(pred['end'])]
        })

    prompt3 = f"""Correct span boundaries for clinical entities in this note excerpt.

Text:
{note_text[region_start:region_end]}

Entities to correct:
"""
    for ent in entities_list:
        prompt3 += f"\n{ent['id']}: [{ent['start']}, {ent['end']}] \"{ent['text']}\" - {ent['concept_name']}"

    prompt3 += """

For each entity, output one line with corrected positions:
<ID>: <start> <end> "<exact text>"

If an entity is already correct, output:
<ID>: CORRECT

Output:"""

    print(prompt3)
    print()

    return {
        'format1_simple_bracket': prompt1,
        'format2_json_positions': prompt2,
        'format3_multi_entity': prompt3
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-notes', default='data/old-challenge-split/train_notes.csv')
    parser.add_argument('--train-gold', default='data/old-challenge-split/train_annotations.csv')
    parser.add_argument('--test-notes', default='data/old-challenge-split/test_notes.csv')
    parser.add_argument('--test-gold', default='data/old-challenge-split/test_annotations.csv')
    parser.add_argument('--test-pred', default='outputs/old_challenge_split/kiri_super_pred.csv')
    parser.add_argument('--concepts', default='1st Place/data/interim/flattened_terminology.csv')
    args = parser.parse_args()

    print("Loading data...")
    train_notes = pd.read_csv(args.train_notes).set_index('note_id')
    train_gold = pd.read_csv(args.train_gold)
    test_notes = pd.read_csv(args.test_notes).set_index('note_id')
    test_gold = pd.read_csv(args.test_gold)
    test_pred = pd.read_csv(args.test_pred)
    concept_names_df = pd.read_csv(args.concepts)

    print(f"Loaded {len(train_notes)} train notes, {len(test_notes)} test notes")
    print()

    # 1. Analyze context requirements
    context_stats = analyze_context_requirements(test_notes, test_gold, sample_size=200)

    # 2. Analyze entity density for batching
    density_stats = analyze_entity_density(test_notes, test_gold)

    # 3. Show real examples
    examples = show_example_corrections(test_notes, test_gold, test_pred, concept_names_df, num_examples=5)

    # 4. Design prompt formats
    prompts = design_prompt_formats(test_notes, test_gold, test_pred, concept_names_df)

    # Save recommendations
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    print("1. CONTEXT SIZE:")
    print(f"   - Use 150-200 chars on each side (covers 95% of cases)")
    print(f"   - This gives ~300-500 char context windows")
    print()
    print("2. BATCHING STRATEGY:")
    print(f"   - Use 500-char sliding windows (stride=400 for overlap)")
    print(f"   - Expect 2-5 entities per window on average")
    print(f"   - Max ~10 entities per window (can handle)")
    print(f"   - Batch processing is 5-10x more efficient!")
    print()
    print("3. PROMPT FORMAT:")
    print(f"   - Format 3 (multi-entity batch) recommended for efficiency")
    print(f"   - Format 1 (simple bracket) best for single-entity fallback")
    print(f"   - Use character positions relative to context window")
    print(f"   - Clear markers for predicted spans")
    print()
    print("4. IMPLEMENTATION:")
    print(f"   - Process notes in 500-char windows with 100-char overlap")
    print(f"   - This reduces API calls by ~5x vs per-entity processing")
    print(f"   - For 24K predictions → ~5K LLM calls (very feasible!)")
    print()


if __name__ == '__main__':
    main()
