#!/usr/bin/env python3
"""
Smoke test: Compare zero-shot span correction between models.
Tests Llama-3-8B vs GPT-OSS-20B on same examples.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re

import pandas as pd
import requests
from tqdm import tqdm

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.super_dictionary.runtime_scoring import class_char_iou


def call_vllm(vllm_url: str, model: str, prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
    """Call vLLM server."""
    try:
        response = requests.post(
            f"{vllm_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error calling vLLM: {e}")
        return ""


def create_window_examples(
    notes_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    concept_names: Dict[int, str],
    num_examples: int = 50,
    window_size: int = 500
) -> List[Dict]:
    """Create window-based examples for testing."""
    examples = []

    # Sample diverse notes
    sampled_notes = pred_df['note_id'].unique()[:20]  # From first 20 notes

    for note_id in sampled_notes:
        if len(examples) >= num_examples:
            break

        note_text = notes_df.loc[note_id]['text']
        note_pred = pred_df[pred_df['note_id'] == note_id]
        note_gold = gold_df[gold_df['note_id'] == note_id]

        # Find windows with corrections needed
        for window_start in range(0, len(note_text), 400):  # Stride of 400
            if len(examples) >= num_examples:
                break

            window_end = min(window_start + window_size, len(note_text))
            window_text = note_text[window_start:window_end]

            # Get predictions in this window
            window_preds = note_pred[
                (note_pred['start'] >= window_start) &
                (note_pred['start'] < window_end)
            ].head(10)  # Max 10 entities per window

            if len(window_preds) == 0:
                continue

            # Build entities and corrections
            entities = []
            gold_corrections = []
            has_corrections = False

            for idx, pred in window_preds.iterrows():
                pred_start = int(pred['start'])
                pred_end = int(pred['end'])
                concept_id = int(pred['concept_id'])

                # Find matching gold (within 200 chars)
                matching_gold = note_gold[
                    (note_gold['concept_id'] == concept_id) &
                    (abs(note_gold['start'] - pred_start) < 200)
                ]

                entity_id = f"E{len(entities) + 1}"

                entities.append({
                    'id': entity_id,
                    'start': pred_start - window_start,
                    'end': pred_end - window_start,
                    'text': note_text[pred_start:pred_end],
                    'concept_id': concept_id,
                    'concept_name': concept_names.get(concept_id, 'Unknown')
                })

                if len(matching_gold) == 0:
                    # False positive
                    gold_corrections.append({
                        'id': entity_id,
                        'action': 'DELETE'
                    })
                    has_corrections = True
                elif len(matching_gold) == 1:
                    gold = matching_gold.iloc[0]
                    gold_start = int(gold['start'])
                    gold_end = int(gold['end'])

                    if pred_start == gold_start and pred_end == gold_end:
                        gold_corrections.append({
                            'id': entity_id,
                            'action': 'CORRECT'
                        })
                    else:
                        gold_corrections.append({
                            'id': entity_id,
                            'action': 'FIX',
                            'start': gold_start - window_start,
                            'end': gold_end - window_start,
                            'text': note_text[gold_start:gold_end]
                        })
                        has_corrections = True

            # Only include windows with at least one correction needed
            if has_corrections and len(entities) > 0:
                examples.append({
                    'note_id': note_id,
                    'window_start': window_start,
                    'window_text': window_text,
                    'entities': entities,
                    'gold_corrections': gold_corrections
                })

    return examples


def format_prompt(example: Dict) -> str:
    """Format prompt for span correction."""
    prompt = """Correct span boundaries for clinical entities in this note excerpt.

Text:
"""
    prompt += example['window_text']
    prompt += "\n\nEntities to correct:\n"

    for entity in example['entities']:
        prompt += f"{entity['id']}: [{entity['start']}, {entity['end']}] \"{entity['text']}\" - {entity['concept_name']}\n"

    prompt += """
For each entity, output ONE line:
<ID>: <start> <end> "<exact text>"  (if needs fixing)
<ID>: CORRECT  (if already right)
<ID>: DELETE   (if false positive)

Output:
"""
    return prompt


def parse_response(response: str, entities: List[Dict]) -> List[Dict]:
    """Parse model response into corrections."""
    corrections = []

    for line in response.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Try to parse: E1: CORRECT
        match_correct = re.match(r'(E\d+):\s*CORRECT', line, re.IGNORECASE)
        if match_correct:
            corrections.append({
                'id': match_correct.group(1),
                'action': 'CORRECT'
            })
            continue

        # Try to parse: E1: DELETE
        match_delete = re.match(r'(E\d+):\s*DELETE', line, re.IGNORECASE)
        if match_delete:
            corrections.append({
                'id': match_delete.group(1),
                'action': 'DELETE'
            })
            continue

        # Try to parse: E1: 10 25 "some text"
        match_fix = re.match(r'(E\d+):\s*(\d+)\s+(\d+)\s+"([^"]*)"', line)
        if match_fix:
            corrections.append({
                'id': match_fix.group(1),
                'action': 'FIX',
                'start': int(match_fix.group(2)),
                'end': int(match_fix.group(3)),
                'text': match_fix.group(4)
            })
            continue

    return corrections


def evaluate_corrections(pred_corrections: List[Dict], gold_corrections: List[Dict]) -> Dict:
    """Evaluate predicted corrections against gold."""
    results = {
        'total': len(gold_corrections),
        'correct': 0,
        'action_correct': 0,
        'span_correct': 0,
    }

    # Build lookup
    gold_dict = {c['id']: c for c in gold_corrections}
    pred_dict = {c['id']: c for c in pred_corrections}

    for entity_id in gold_dict:
        gold = gold_dict[entity_id]
        pred = pred_dict.get(entity_id, {'action': 'UNKNOWN'})

        # Check action
        if gold['action'] == pred['action']:
            results['action_correct'] += 1

            # Check span if FIX
            if gold['action'] == 'FIX' and pred['action'] == 'FIX':
                if (gold['start'] == pred.get('start') and
                    gold['end'] == pred.get('end')):
                    results['span_correct'] += 1
                    results['correct'] += 1
            else:
                results['correct'] += 1

    return results


def run_smoke_test(
    vllm_url: str,
    models: List[str],
    examples: List[Dict],
    verbose: bool = False
):
    """Run smoke test on multiple models."""
    print("="*80)
    print("SMOKE TEST: Zero-Shot Span Correction")
    print("="*80)
    print()

    print(f"Testing {len(models)} models on {len(examples)} examples")
    print()

    results = {}

    for model in models:
        print(f"\nTesting model: {model}")
        print("-"*80)

        model_results = {
            'total_entities': 0,
            'total_correct': 0,
            'total_action_correct': 0,
            'total_span_correct': 0,
            'parse_failures': 0,
        }

        for example in tqdm(examples, desc=f"Testing {model}"):
            prompt = format_prompt(example)

            # Call model
            response = call_vllm(vllm_url, model, prompt, max_tokens=500, temperature=0.0)

            if not response:
                model_results['parse_failures'] += 1
                continue

            # Parse response
            pred_corrections = parse_response(response, example['entities'])

            if len(pred_corrections) == 0:
                model_results['parse_failures'] += 1
                if verbose:
                    print(f"\nFailed to parse response:")
                    print(f"Prompt: {prompt[:200]}...")
                    print(f"Response: {response}")
                continue

            # Evaluate
            eval_results = evaluate_corrections(pred_corrections, example['gold_corrections'])

            model_results['total_entities'] += eval_results['total']
            model_results['total_correct'] += eval_results['correct']
            model_results['total_action_correct'] += eval_results['action_correct']
            model_results['total_span_correct'] += eval_results['span_correct']

        results[model] = model_results

        # Print summary
        print(f"\nResults for {model}:")
        print(f"  Total entities: {model_results['total_entities']}")
        print(f"  Fully correct: {model_results['total_correct']} ({100*model_results['total_correct']/max(1, model_results['total_entities']):.1f}%)")
        print(f"  Action correct: {model_results['total_action_correct']} ({100*model_results['total_action_correct']/max(1, model_results['total_entities']):.1f}%)")
        print(f"  Span correct (for FIX): {model_results['total_span_correct']}")
        print(f"  Parse failures: {model_results['parse_failures']}/{len(examples)}")

    # Compare models
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print()

    print(f"{'Model':<30} {'Fully Correct':<15} {'Action Correct':<15} {'Parse Failures'}")
    print("-"*80)

    for model in models:
        r = results[model]
        fully_correct = 100*r['total_correct']/max(1, r['total_entities'])
        action_correct = 100*r['total_action_correct']/max(1, r['total_entities'])
        parse_fail = r['parse_failures']

        print(f"{model:<30} {fully_correct:>6.1f}%         {action_correct:>6.1f}%         {parse_fail}")

    print()
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    print("Fully Correct: Entity action AND span boundaries are perfect")
    print("Action Correct: Chose right action (CORRECT/FIX/DELETE) but maybe wrong span")
    print()
    print("Good performance: >60% action correct, <10% parse failures")
    print("Needs fine-tuning: 40-60% action correct")
    print("Model unsuitable: <40% action correct or >20% parse failures")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Smoke test span correction models")
    parser.add_argument('--vllm-url', default='http://localhost:8000', help='vLLM server URL')
    parser.add_argument('--models', nargs='+', default=['openai/gpt-oss-20b'], help='Models to test')
    parser.add_argument('--test-notes', default='data/old-challenge-split/test_notes.csv')
    parser.add_argument('--test-gold', default='data/old-challenge-split/test_annotations.csv')
    parser.add_argument('--test-pred', default='outputs/old_challenge_split/kiri_super_pred.csv')
    parser.add_argument('--concepts', default='1st Place/data/interim/flattened_terminology.csv')
    parser.add_argument('--num-examples', type=int, default=50, help='Number of examples to test')
    parser.add_argument('--verbose', action='store_true', help='Show parse failures')
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

    print(f"Created {len(examples)} examples with corrections needed")
    print()

    # Run smoke test
    results = run_smoke_test(args.vllm_url, args.models, examples, verbose=args.verbose)


if __name__ == '__main__':
    main()
