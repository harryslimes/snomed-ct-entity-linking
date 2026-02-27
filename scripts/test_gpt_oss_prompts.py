#!/usr/bin/env python3
"""
Test different prompt formats for GPT-OSS-20B to find what works.
"""
import requests
import json
from pathlib import Path

import pandas as pd

def call_vllm(vllm_url: str, model: str, prompt: str, max_tokens: int = 500, temperature: float = 0.0, stop=None) -> str:
    """Call vLLM server."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        payload["stop"] = stop

    try:
        response = requests.post(
            f"{vllm_url}/v1/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"].strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""


# Load one real example
test_notes = pd.read_csv('data/old-challenge-split/test_notes.csv').set_index('note_id')
test_pred = pd.read_csv('outputs/old_challenge_split/kiri_super_pred.csv')
test_gold = pd.read_csv('data/old-challenge-split/test_annotations.csv')
concept_names_df = pd.read_csv('1st Place/data/interim/flattened_terminology.csv')
concept_names = dict(zip(concept_names_df['concept_id'], concept_names_df['concept_name']))

# Get a real example
note_id = test_pred['note_id'].iloc[0]
note_text = test_notes.loc[note_id]['text']
note_pred = test_pred[test_pred['note_id'] == note_id].head(5)
note_gold = test_gold[test_gold['note_id'] == note_id]

# Create window example
window_start = 200
window_end = 700
window_text = note_text[window_start:window_end]

entities = []
for idx, pred in note_pred.iterrows():
    pred_start = int(pred['start'])
    pred_end = int(pred['end'])
    if pred_start >= window_start and pred_start < window_end:
        entities.append({
            'id': f"E{len(entities)+1}",
            'start': pred_start - window_start,
            'end': pred_end - window_start,
            'text': note_text[pred_start:pred_end],
            'concept_id': int(pred['concept_id']),
            'concept_name': concept_names.get(int(pred['concept_id']), 'Unknown')
        })

if len(entities) < 2:
    # Try different window
    window_start = 500
    window_end = 1000
    window_text = note_text[window_start:window_end]
    entities = []
    for idx, pred in test_pred[test_pred['note_id'] == note_id].head(10).iterrows():
        pred_start = int(pred['start'])
        pred_end = int(pred['end'])
        if pred_start >= window_start and pred_start < window_end:
            entities.append({
                'id': f"E{len(entities)+1}",
                'start': pred_start - window_start,
                'end': pred_end - window_start,
                'text': note_text[pred_start:pred_end],
                'concept_id': int(pred['concept_id']),
                'concept_name': concept_names.get(int(pred['concept_id']), 'Unknown')
            })

print("="*80)
print("TESTING DIFFERENT PROMPTS FOR GPT-OSS-20B")
print("="*80)
print()
print(f"Window text: {window_text[:200]}...")
print()
print("Entities:")
for e in entities[:3]:
    print(f"  {e['id']}: [{e['start']}, {e['end']}] \"{e['text']}\" - {e['concept_name']}")
print()

vllm_url = "http://localhost:8000"

# Restart with GPT-OSS-20B
import subprocess
import time
subprocess.run(["pkill", "-9", "-f", "vllm"], check=False)
time.sleep(3)

print("Starting GPT-OSS-20B server...")
subprocess.Popen([
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "openai/gpt-oss-20b",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--max-model-len", "8192"
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Waiting for server...")
time.sleep(45)

print("Testing different prompt formats...\n")

# ============================================================================
# PROMPT 1: Original (that failed)
# ============================================================================
print("="*80)
print("PROMPT 1: Original Format (Structured)")
print("="*80)

prompt1 = f"""Correct span boundaries for clinical entities in this note excerpt.

Text:
{window_text}

Entities to correct:
"""
for e in entities[:3]:
    prompt1 += f"{e['id']}: [{e['start']}, {e['end']}] \"{e['text']}\" - {e['concept_name']}\n"

prompt1 += """
For each entity, output ONE line:
<ID>: <start> <end> "<exact text>"  (if needs fixing)
<ID>: CORRECT  (if already right)
<ID>: DELETE   (if false positive)

Output:
"""

response1 = call_vllm(vllm_url, "openai/gpt-oss-20b", prompt1, max_tokens=300, temperature=0.0)
print(f"Response: {response1[:500]}")
print()

# ============================================================================
# PROMPT 2: Direct, no formatting
# ============================================================================
print("="*80)
print("PROMPT 2: Direct Question Format")
print("="*80)

prompt2 = f"""Text: {window_text}

Entity 1: "{entities[0]['text']}" at position [{entities[0]['start']}, {entities[0]['end']}]
Concept: {entities[0]['concept_name']}

Is this span correct? Answer only: CORRECT, DELETE, or WRONG

Answer:"""

response2 = call_vllm(vllm_url, "openai/gpt-oss-20b", prompt2, max_tokens=50, temperature=0.0, stop=["\n"])
print(f"Response: {response2}")
print()

# ============================================================================
# PROMPT 3: Few-shot examples
# ============================================================================
print("="*80)
print("PROMPT 3: Few-Shot Examples")
print("="*80)

prompt3 = """Task: Verify clinical entity span boundaries.

Example 1:
Text: "Patient has diabetes mellitus type 2"
Entity: [13, 30] "diabetes mellitus"
Concept: Diabetes mellitus
Output: CORRECT

Example 2:
Text: "History of MI and CHF"
Entity: [15, 18] "CHF"
Concept: Congestive heart failure
Output: CORRECT

Example 3:
Text: "No signs of infection present"
Entity: [12, 21] "infection"
Concept: Infectious disease
Output: DELETE (no infection present)

Now you:
Text: """ + window_text + f"""
Entity: [{entities[0]['start']}, {entities[0]['end']}] "{entities[0]['text']}"
Concept: {entities[0]['concept_name']}
Output:"""

response3 = call_vllm(vllm_url, "openai/gpt-oss-20b", prompt3, max_tokens=50, temperature=0.0, stop=["\n\n"])
print(f"Response: {response3}")
print()

# ============================================================================
# PROMPT 4: JSON output
# ============================================================================
print("="*80)
print("PROMPT 4: JSON Output Format")
print("="*80)

prompt4 = f"""Verify clinical entity spans. Output JSON only.

Text: {window_text[:300]}...

Entities:
{json.dumps([{
    'id': e['id'],
    'span': [e['start'], e['end']],
    'text': e['text'],
    'concept': e['concept_name']
} for e in entities[:3]], indent=2)}

Output JSON with corrections:
{{"corrections": [{{"id": "E1", "action": "CORRECT or DELETE or FIX"}}, ...]}}

JSON:
"""

response4 = call_vllm(vllm_url, "openai/gpt-oss-20b", prompt4, max_tokens=200, temperature=0.0)
print(f"Response: {response4[:500]}")
print()

# ============================================================================
# PROMPT 5: System message style
# ============================================================================
print("="*80)
print("PROMPT 5: System Message Style (Chat Format)")
print("="*80)

prompt5 = f"""<|system|>
You are a medical entity annotation verifier. You output only the verification result, no explanations.
<|endofsystem|>

<|user|>
Text: {window_text}

Verify entity: "{entities[0]['text']}" at [{entities[0]['start']}, {entities[0]['end']}]
Concept: {entities[0]['concept_name']}

Output only: CORRECT, DELETE, or FIX <start> <end> "<text>"
<|endofuser|>

<|assistant|>
"""

response5 = call_vllm(vllm_url, "openai/gpt-oss-20b", prompt5, max_tokens=50, temperature=0.0, stop=["<|"])
print(f"Response: {response5}")
print()

# ============================================================================
# PROMPT 6: Instruction following with explicit constraints
# ============================================================================
print("="*80)
print("PROMPT 6: Explicit Constraints (No Explanation)")
print("="*80)

prompt6 = f"""INSTRUCTIONS: Output ONLY the answer. Do NOT explain. Do NOT think out loud.

Task: Verify if this entity span is correct.

Text: {window_text}

Entity: "{entities[0]['text']}" at position [{entities[0]['start']}, {entities[0]['end']}]
Concept: {entities[0]['concept_name']}

Your answer (one word only):"""

response6 = call_vllm(vllm_url, "openai/gpt-oss-20b", prompt6, max_tokens=10, temperature=0.0, stop=["\n", ".", "Explanation"])
print(f"Response: {response6}")
print()

# ============================================================================
# PROMPT 7: Completion style (not instruction)
# ============================================================================
print("="*80)
print("PROMPT 7: Completion Style (Not Instruction)")
print("="*80)

prompt7 = f"""Medical entity annotation:

Text: {window_text}

Entity verification results:
- Entity: "{entities[0]['text']}" [{entities[0]['start']}, {entities[0]['end']}] - {entities[0]['concept_name']}
  Status:"""

response7 = call_vllm(vllm_url, "openai/gpt-oss-20b", prompt7, max_tokens=20, temperature=0.0, stop=["\n-", "\n\n"])
print(f"Response: {response7}")
print()

print("="*80)
print("SUMMARY")
print("="*80)
print()
print("Prompt 1 (Original):         ", "✓ Parseable" if any(x in response1 for x in ["CORRECT", "DELETE", "E1:", "E2:"]) else "✗ Not parseable")
print("Prompt 2 (Direct):           ", "✓ Parseable" if any(x in response2 for x in ["CORRECT", "DELETE", "WRONG"]) else "✗ Not parseable")
print("Prompt 3 (Few-shot):         ", "✓ Parseable" if any(x in response3 for x in ["CORRECT", "DELETE", "FIX"]) else "✗ Not parseable")
print("Prompt 4 (JSON):             ", "✓ Parseable" if "corrections" in response4 or "id" in response4 else "✗ Not parseable")
print("Prompt 5 (System message):   ", "✓ Parseable" if any(x in response5 for x in ["CORRECT", "DELETE", "FIX"]) else "✗ Not parseable")
print("Prompt 6 (No explanation):   ", "✓ Parseable" if any(x in response6 for x in ["CORRECT", "DELETE", "WRONG"]) else "✗ Not parseable")
print("Prompt 7 (Completion):       ", "✓ Parseable" if any(x in response7 for x in ["CORRECT", "DELETE", "correct", "incorrect"]) else "✗ Not parseable")
print()
print("Best performing prompt will be used for full smoke test.")
