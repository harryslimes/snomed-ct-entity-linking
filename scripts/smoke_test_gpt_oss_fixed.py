#!/usr/bin/env python3
"""
Fixed smoke test for GPT-OSS-20B using working prompt format.
"""
import argparse
import re
from typing import Dict, List

import pandas as pd
import requests
from tqdm import tqdm


def call_vllm(vllm_url: str, model: str, prompt: str, max_tokens: int = 50, stop=None) -> str:
    """Call vLLM server."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if stop:
        payload["stop"] = stop

    try:
        response = requests.post(f"{vllm_url}/v1/completions", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""


def create_window_examples(
    notes_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    concept_names: Dict[int, str],
    num_examples: int = 50,
) -> List[Dict]:
    """Create single-entity examples for testing."""
    examples = []

    # Sample diverse notes
    sampled_notes = pred_df['note_id'].unique()[:20]

    for note_id in sampled_notes:
        if len(examples) >= num_examples:
            break

        note_text = notes_df.loc[note_id]['text']
        note_pred = pred_df[pred_df['note_id'] == note_id]
        note_gold = gold_df[gold_df['note_id'] == note_id]

        for _, pred in note_pred.head(10).iterrows():
            if len(examples) >= num_examples:
                break

            pred_start = int(pred['start'])
            pred_end = int(pred['end'])
            concept_id = int(pred['concept_id'])

            # Get context window
            window_start = max(0, pred_start - 200)
            window_end = min(len(note_text), pred_end + 200)
            context = note_text[window_start:window_end]

            # Find matching gold (within 200 chars)
            matching_gold = note_gold[
                (note_gold['concept_id'] == concept_id) &
                (abs(note_gold['start'] - pred_start) < 200)
            ]

            # Determine gold action
            if len(matching_gold) == 0:
                gold_action = "DELETE"
            elif len(matching_gold) == 1:
                gold = matching_gold.iloc[0]
                gold_start = int(gold['start'])
                gold_end = int(gold['end'])
                if pred_start == gold_start and pred_end == gold_end:
                    gold_action = "CORRECT"
                else:
                    gold_action = "FIX"
            else:
                continue  # Multiple matches, skip

            examples.append({
                'note_id': note_id,
                'context': context,
                'pred_text': note_text[pred_start:pred_end],
                'pred_start': pred_start,
                'pred_end': pred_end,
                'concept_id': concept_id,
                'concept_name': concept_names.get(concept_id, 'Unknown'),
                'gold_action': gold_action,
            })

    return examples


def format_prompt(example: Dict) -> str:
    """Format prompt using working format."""
    prompt = f"""Is this clinical entity annotation correct?

Text: {example['context']}

Entity: "{example['pred_text']}"
Concept: {example['concept_name']}

Answer (YES if correct, NO if wrong):"""
    return prompt


def parse_response(response: str) -> str:
    """Parse model response."""
    response = response.strip().upper()

    # YES/NO format
    if "YES" in response or response == "Y":
        return "YES"
    if "NO" in response or response == "N":
        return "NO"

    return "UNKNOWN"


def run_smoke_test(vllm_url: str, model: str, examples: List[Dict]):
    """Run smoke test."""
    print("="*80)
    print(f"SMOKE TEST: {model}")
    print("="*80)
    print()

    results = {
        'total': len(examples),
        'correct': 0,
        'parse_failures': 0,
    }

    for example in tqdm(examples, desc="Testing"):
        prompt = format_prompt(example)

        # Call model
        response = call_vllm(vllm_url, model, prompt, max_tokens=5, stop=None)

        if not response:
            results['parse_failures'] += 1
            continue

        # Parse response
        pred_action = parse_response(response)

        if pred_action == "UNKNOWN":
            results['parse_failures'] += 1
            continue

        # Check if correct
        # YES means CORRECT, NO means DELETE or FIX
        gold_correct = (example['gold_action'] == "CORRECT")
        pred_correct = (pred_action == "YES")

        if gold_correct == pred_correct:
            results['correct'] += 1

    # Print results
    print()
    print(f"Results:")
    print(f"  Total examples: {results['total']}")
    print(f"  Correct: {results['correct']} ({100*results['correct']/results['total']:.1f}%)")
    print(f"  Parse failures: {results['parse_failures']} ({100*results['parse_failures']/results['total']:.1f}%)")
    print()

    accuracy = 100 * results['correct'] / results['total']
    parse_rate = 100 * results['parse_failures'] / results['total']

    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    if accuracy > 60 and parse_rate < 10:
        print("✅ EXCELLENT: Ready for fine-tuning!")
    elif accuracy > 40:
        print("✅ GOOD: Promising, proceed with fine-tuning")
    else:
        print("⚠️  NEEDS WORK: Consider different model or more prompt engineering")
    print()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm-url', default='http://localhost:8000')
    parser.add_argument('--model', default='openai/gpt-oss-20b')
    parser.add_argument('--test-notes', default='data/old-challenge-split/test_notes.csv')
    parser.add_argument('--test-gold', default='data/old-challenge-split/test_annotations.csv')
    parser.add_argument('--test-pred', default='outputs/old_challenge_split/kiri_super_pred.csv')
    parser.add_argument('--concepts', default='1st Place/data/interim/flattened_terminology.csv')
    parser.add_argument('--num-examples', type=int, default=50)
    args = parser.parse_args()

    print("Loading data...")
    test_notes = pd.read_csv(args.test_notes).set_index('note_id')
    test_gold = pd.read_csv(args.test_gold)
    test_pred = pd.read_csv(args.test_pred)
    concept_names_df = pd.read_csv(args.concepts)
    concept_names = dict(zip(concept_names_df['concept_id'], concept_names_df['concept_name']))

    print(f"Creating {args.num_examples} test examples...")
    examples = create_window_examples(
        test_notes, test_pred, test_gold, concept_names,
        num_examples=args.num_examples
    )

    print(f"Created {len(examples)} examples")
    print()

    # Run test
    results = run_smoke_test(args.vllm_url, args.model, examples)


if __name__ == '__main__':
    main()
