#!/usr/bin/env python3
import requests

def call_vllm(prompt, max_tokens=300, stop=None):
    payload = {
        "model": "openai/gpt-oss-20b",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if stop:
        payload["stop"] = stop

    response = requests.post("http://localhost:8000/v1/completions", json=payload, timeout=120)
    return response.json()["choices"][0]["text"].strip()

# Test case
text = """Patient has history of diabetes mellitus type 2 with neuropathy. Also has hypertension."""

entity_text = "diabetes mellitus"
entity_pos = [24, 41]
concept = "Diabetes mellitus (disorder)"

print("="*80)
print("Testing different prompts with GPT-OSS-20B")
print("="*80)
print()

# Prompt 1: Few-shot
print("1. FEW-SHOT PROMPT")
print("-"*80)
prompt1 = """Task: Verify if clinical entity spans are correct.

Example 1:
Text: "Patient diagnosed with pneumonia"
Entity: [23, 32] "pneumonia" - Pneumonia (disorder)
Answer: CORRECT

Example 2:
Text: "No history of cancer found"
Entity: [14, 20] "cancer" - Malignant neoplasm (disorder)
Answer: DELETE

Example 3:
Text: "Right hip fracture present"
Entity: [6, 18] "hip fracture" - Fracture of hip (disorder)
Answer: CORRECT

Now you:
Text: """ + text + f"""
Entity: {entity_pos} "{entity_text}" - {concept}
Answer:"""

response1 = call_vllm(prompt1, max_tokens=20, stop=["\n\n", "Example"])
print(f"Response: '{response1}'")
print()

# Prompt 2: Direct with explicit instruction
print("2. DIRECT WITH STOP REASONING")
print("-"*80)
prompt2 = f"""Verify clinical entity span. Answer ONLY with: CORRECT, DELETE, or WRONG
DO NOT explain. DO NOT think step-by-step.

Text: {text}
Entity: {entity_pos} "{entity_text}" - {concept}

Answer (one word):"""

response2 = call_vllm(prompt2, max_tokens=10, stop=["\n", ".", "Ex"])
print(f"Response: '{response2}'")
print()

# Prompt 3: Completion style
print("3. COMPLETION STYLE")
print("-"*80)
prompt3 = f"""Clinical entity annotation verification:

Text: {text}

Entity: "{entity_text}" [{entity_pos[0]}, {entity_pos[1]}] - {concept}
Verification: """

response3 = call_vllm(prompt3, max_tokens=15, stop=["\n\n", "Entity"])
print(f"Response: '{response3}'")
print()

# Prompt 4: JSON format
print("4. JSON FORMAT")
print("-"*80)
prompt4 = f"""Verify clinical entity span. Output JSON only.

Text: {text}
Entity: {entity_pos} "{entity_text}" - {concept}

Output JSON:
{{"status": "CORRECT or DELETE or WRONG"}}

JSON:
"""

response4 = call_vllm(prompt4, max_tokens=30, stop=["}\n"])
print(f"Response: '{response4}'")
print()

# Prompt 5: Very simple yes/no
print("5. SIMPLE YES/NO")
print("-"*80)
prompt5 = f"""Text: {text}

Is this entity correct?
Entity: "{entity_text}" at position {entity_pos}
Concept: {concept}

Answer (YES or NO):"""

response5 = call_vllm(prompt5, max_tokens=5, stop=["\n"])
print(f"Response: '{response5}'")
print()

print("="*80)
print("SUMMARY")
print("="*80)
responses = [response1, response2, response3, response4, response5]
for i, r in enumerate(responses, 1):
    parseable = any(x in r.upper() for x in ["CORRECT", "DELETE", "WRONG", "YES", "NO", "status"])
    has_explanation = len(r.split()) > 5 or "We need" in r or "Let's" in r
    print(f"Prompt {i}: {'✓ Parseable' if parseable else '✗ Not parseable'} | {'⚠ Has explanation' if has_explanation else '✓ Clean'}")
