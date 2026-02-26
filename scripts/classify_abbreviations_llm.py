#!/usr/bin/env python3
"""Classify medical annotation spans as abbreviation vs non-abbreviation using vLLM.

Extracts unique span texts from training annotations, sends batched classification
requests to a vLLM server, and outputs a JSON file mapping spans to labels.

Usage:
    # Start vLLM first:
    # vllm serve models/Qwen3.5-35B-A3B-AWQ-4bit --max-model-len 4096 --gpu-memory-utilization 0.95

    python scripts/classify_abbreviations_llm.py \
        --data-dir data/old-challenge-split \
        --output abbreviation_classification_llm.json \
        --concurrency 64
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import aiohttp
import pandas as pd

SYSTEM_PROMPT = """\
You are a clinical NLP expert classifying medical spans extracted from clinical notes.

For each span, classify it as ABBREV or FULL.

ABBREV = an abbreviation, acronym, or initialism where the CORE term is shortened and \
requires expansion to understand. The key question: is the MAIN medical concept \
abbreviated? If so, ABBREV.

FULL = a full word, standard medical term, drug name, anatomical term, or descriptive \
phrase — even if written in ALL CAPS, even if misspelled, even if informal.

## CRITICAL RULES (apply these FIRST):

1. IGNORE CAPITALIZATION. A word in ALL CAPS is still FULL if it's a recognizable word. \
"CORONARY" → FULL, "RASH" → FULL, "YEAST" → FULL, "ANEMIA" → FULL.

2. FULL PHRASES ARE FULL even if they describe something that has a common abbreviation. \
"deep vein thrombosis" → FULL (it's spelled out!). "DVT" → ABBREV (it's abbreviated). \
"Congestive Heart Failure" → FULL. "CHF" → ABBREV. \
"Non-ST Elevation Myocardial Infarction" → FULL. "NSTEMI" → ABBREV. \
"Paroxysmal atrial fibrillation" → FULL. "PAF" → ABBREV. \
"Community Acquired Pneumonia" → FULL. "CAP" → ABBREV. \
"Transient Ischemic Attack" → FULL. "TIA" → ABBREV.

3. MISSPELLED WORDS ARE FULL. If the span is a misspelling of a real word, it's FULL. \
"antibiotocs" → FULL (misspelling of antibiotics). "celluliis" → FULL. \
"steriods" → FULL. "palpaitations" → FULL. "meninigitis" → FULL.

4. EPONYMS AND BRAND NAMES ARE FULL. "Foley" → FULL, "Guaiac" → FULL, "Mohs" → FULL, \
"Plavix" → FULL, "Lasix" → FULL.

5. STANDARD CLINICAL VOCABULARY IS FULL even if informal/shortened in common usage: \
"meds" → FULL, "labs" → FULL, "vitals" → FULL, "eval" → FULL, "pressors" → FULL, \
"flu" → FULL, "chemo" → FULL, "sarcoid" → FULL, "tachy" → FULL, "nebs" → FULL.

6. SINGLE CHARACTERS AND PURE NUMBERS ARE FULL (not abbreviations): \
"A" → FULL, "5" → FULL, "02" → FULL.

7. GARBAGE/TRUNCATED SPANS ARE FULL. If the span is a sentence fragment, partial word, \
or contains HTML like "<br>", classify as FULL.

8. MIXED PHRASES: if a phrase contains BOTH an abbreviation AND full words, classify \
based on whether the CORE medical concept is abbreviated: \
"COPD exacerbation" → ABBREV (COPD is the core concept, abbreviated). \
"elevated WBC" → ABBREV (WBC is the core lab value, abbreviated). \
"cardiac cath" → ABBREV (catheterization is abbreviated to cath). \
"stress test" → FULL (no abbreviation present). \
"sliding scale insulin" → FULL (no abbreviation present). \
"flow cytometry" → FULL (no abbreviation present). \
"reticulocyte count" → FULL (no abbreviation present).

## Examples:

ABBREV:
- "WBC", "RBC", "HTN", "CAD", "Hct", "Na", "UreaN", "AnGap", "cTropnT"
- "CXR", "HEENT", "RRR", "BP", "DVT", "NSTEMI", "PERRL", "BNP", "HFpEF"
- "CT A/P", "PLT COUNT", "COPD exacerbation", "elevated LFTs", "etoh abuse"
- "cardiac cath", "Hep C", "Type 2 DM"

FULL:
- "chest pain", "cholecystectomy", "levofloxacin", "hypertension", "abdomen"
- "deep vein thrombosis", "Congestive Heart Failure", "stress test", "flow cytometry"
- "sarcoid", "euvolemic", "presyncope", "Foley", "Guaiac", "Mohs"
- "CORONARY", "RASH", "YEAST", "Pancytopenia", "Orthostatics"
- "meds", "labs", "vitals", "eval", "pressors", "flu", "chemo", "nebs"
- "within normal limits", "status post", "sliding scale insulin", "reticulocyte count"
- "antibiotocs", "celluliis", "steriods" (misspellings are FULL)
- "Foley catheter", "plain films", "finger sticks", "color Doppler"
"""

USER_PROMPT_TEMPLATE = """\
Classify each span below as ABBREV or FULL. Reply with ONLY a JSON object mapping \
each span to its label. No explanation.

Spans:
{spans_list}
"""


def build_batches(spans: list[str], batch_size: int = 25) -> list[list[str]]:
    """Split spans into batches."""
    return [spans[i:i + batch_size] for i in range(0, len(spans), batch_size)]


async def classify_batch(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    batch: list[str],
    batch_idx: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict[str, str]:
    """Send a batch of spans to vLLM for classification."""
    spans_formatted = "\n".join(f'- "{s}"' for s in batch)
    user_msg = USER_PROMPT_TEMPLATE.format(spans_list=spans_formatted)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }

    for attempt in range(max_retries):
        async with semaphore:
            try:
                async with session.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"  Batch {batch_idx}: HTTP {resp.status}: {text[:200]}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return {}

                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]

                    # Parse JSON from response (handle markdown code blocks)
                    content = content.strip()
                    if content.startswith("```"):
                        content = re.sub(r"^```\w*\n?", "", content)
                        content = re.sub(r"\n?```$", "", content)
                        content = content.strip()

                    result = json.loads(content)

                    # Normalize values
                    normalized = {}
                    for span, label in result.items():
                        label = label.strip().upper()
                        if label in ("ABBREV", "ABBREVIATION", "ABB"):
                            normalized[span] = "ABBREV"
                        elif label in ("FULL", "NOT_ABBREV", "NOT ABBREV", "NOTABBREV"):
                            normalized[span] = "FULL"
                        else:
                            normalized[span] = label
                    return normalized

            except json.JSONDecodeError as e:
                print(f"  Batch {batch_idx}: JSON parse error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                # Fall back: try to classify individually
                return {}
            except Exception as e:
                print(f"  Batch {batch_idx}: Error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {}

    return {}


async def run_classification(
    spans: list[str],
    base_url: str,
    model: str,
    batch_size: int,
    concurrency: int,
) -> dict[str, str]:
    """Classify all spans with concurrent batched requests."""
    batches = build_batches(spans, batch_size)
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, str] = {}
    failed_spans: list[str] = []

    print(f"  {len(spans)} spans in {len(batches)} batches "
          f"(batch_size={batch_size}, concurrency={concurrency})")

    async with aiohttp.ClientSession() as session:
        tasks = [
            classify_batch(session, base_url, model, batch, i, semaphore)
            for i, batch in enumerate(batches)
        ]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            batch_result = await coro
            results.update(batch_result)
            completed += 1
            if completed % 20 == 0 or completed == len(batches):
                print(f"  Completed {completed}/{len(batches)} batches "
                      f"({len(results)}/{len(spans)} classified)")

    # Find spans that weren't classified (batch failures or missing from response)
    for span in spans:
        if span not in results:
            failed_spans.append(span)

    # Retry failed spans individually
    if failed_spans:
        print(f"\n  Retrying {len(failed_spans)} failed spans individually...")
        individual_batches = build_batches(failed_spans, batch_size=5)
        async with aiohttp.ClientSession() as session:
            retry_tasks = [
                classify_batch(session, base_url, model, batch, i, semaphore)
                for i, batch in enumerate(individual_batches)
            ]
            for coro in asyncio.as_completed(retry_tasks):
                batch_result = await coro
                results.update(batch_result)

    still_missing = [s for s in spans if s not in results]
    if still_missing:
        print(f"  WARNING: {len(still_missing)} spans still unclassified after retries")
        for s in still_missing[:20]:
            print(f"    - \"{s}\"")

    return results


# --- Deterministic overrides for known classification errors ---
# These spans are unambiguously one category regardless of what the LLM says.
FORCE_FULL = {
    # Full medical terms the LLM sometimes calls ABBREV
    "stress test", "sinus tachycardia", "sinus bradycardia",
    "type 2 diabetes", "type II diabetes", "type I diabetes",
    "type 1 diabetes", "Graves Disease", "Graves' disease", "Grave's Disease",
    "Sulfonamides", "calculi", "positive nitrite", "positive nitrites",
    "non-small cell carcinoma", "non-small cell lung cancer",
    "Left anterior descending", "left anterior descending",
    "pressors", "presyncope", "presyncopal",
    "Plavix", "plavix", "Guaiac", "guaiac",
    "anion gap",
}
FORCE_ABBREV = {
    # Abbreviations the LLM sometimes calls FULL (case-insensitive matching applied below)
    "BRBPR", "HENT",
}

# Spans whose ABBREV status should NOT propagate to case variants.
# These are standard words that happen to share a lowercase form with abbreviations.
# e.g. "LOW" might be ABBREV in one context but "low" is a plain English word.
NO_CASE_PROPAGATE = {
    "low", "old", "head", "card", "tab", "core", "catch", "wash",
    "lymph", "lymphs", "mmm", "stress test",
}


def postprocess_classifications(results: dict[str, str]) -> dict[str, str]:
    """Apply deterministic fixes after LLM classification.

    1. Apply forced overrides for known errors.
    2. Enforce case consistency: if any case variant of a span is ABBREV,
       all variants become ABBREV (unless in NO_CASE_PROPAGATE).
    """
    out = dict(results)
    n_forced_full = 0
    n_forced_abbrev = 0
    n_case_fixed = 0

    # Step 1: forced overrides (exact match)
    for span in out:
        if span in FORCE_FULL or span.lower() in {s.lower() for s in FORCE_FULL}:
            if out[span] != "FULL":
                out[span] = "FULL"
                n_forced_full += 1
        if span.upper() in FORCE_ABBREV or span in FORCE_ABBREV:
            if out[span] != "ABBREV":
                out[span] = "ABBREV"
                n_forced_abbrev += 1

    # Step 2: case-consistency propagation
    from collections import defaultdict
    by_lower = defaultdict(dict)
    for span, label in out.items():
        by_lower[span.lower()][span] = label

    for lower, variants in by_lower.items():
        if lower in NO_CASE_PROPAGATE:
            continue
        has_abbrev = any(l == "ABBREV" for l in variants.values())
        if has_abbrev:
            for span, label in variants.items():
                if label != "ABBREV":
                    out[span] = "ABBREV"
                    n_case_fixed += 1

    print(f"\n  Post-processing:")
    print(f"    Forced FULL:  {n_forced_full}")
    print(f"    Forced ABBREV: {n_forced_abbrev}")
    print(f"    Case-consistency fixes: {n_case_fixed}")

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Classify annotation spans as abbreviation/full using vLLM")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/old-challenge-split"))
    parser.add_argument("--output", type=Path,
                        default=Path("abbreviation_classification_llm.json"))
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model", type=str, default="Qwen3.5-35B-A3B-AWQ-4bit",
                        help="Model name as served by vLLM")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--sample", type=int, default=0,
                        help="If >0, only classify a random sample of N spans (for sanity checking)")
    args = parser.parse_args()

    # Extract unique spans from TRAIN annotations only
    print("Loading train annotations...")
    all_spans = set()
    notes = pd.read_csv(args.data_dir / "train_notes.csv")
    ann = pd.read_csv(args.data_dir / "train_annotations.csv",
                      dtype={"concept_id": int})
    note_texts = dict(zip(notes["note_id"], notes["text"]))
    for _, row in ann.iterrows():
        nid = row["note_id"]
        if nid not in note_texts:
            continue
        text = note_texts[nid]
        span = re.sub(r"\s+", " ", text[int(row["start"]):int(row["end"])]).strip()
        if span:
            all_spans.add(span)

    spans_list = sorted(all_spans)
    print(f"  Unique train spans: {len(spans_list)}")

    # Optional: sample for sanity checking
    if args.sample > 0:
        import random
        random.seed(42)
        spans_list = sorted(random.sample(spans_list, min(args.sample, len(spans_list))))
        print(f"  Sampled {len(spans_list)} spans for sanity check")

    # Check vLLM is running
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"{args.base_url}/v1/models", timeout=5)
        models_data = json.loads(resp.read())
        available = [m["id"] for m in models_data["data"]]
        print(f"  vLLM models available: {available}")
        if args.model not in available and available:
            print(f"  WARNING: --model={args.model} not found, using {available[0]}")
            args.model = available[0]
    except Exception as e:
        print(f"  ERROR: Cannot connect to vLLM at {args.base_url}: {e}")
        print("  Start vLLM first, e.g.:")
        print("    vllm serve models/Qwen3.5-35B-A3B-AWQ-4bit "
              "--max-model-len 4096 --gpu-memory-utilization 0.95")
        return

    # Run classification
    print(f"\nClassifying spans via {args.model}...")
    t0 = time.time()
    results = asyncio.run(run_classification(
        spans_list, args.base_url, args.model,
        args.batch_size, args.concurrency,
    ))
    elapsed = time.time() - t0

    # --- Post-processing: fix known errors and enforce case consistency ---
    results = postprocess_classifications(results)

    # Summarize
    abbrev_count = sum(1 for v in results.values() if v == "ABBREV")
    full_count = sum(1 for v in results.values() if v == "FULL")
    other_count = len(results) - abbrev_count - full_count
    unclassified = len(spans_list) - len(results)

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  ABBREV: {abbrev_count}")
    print(f"  FULL: {full_count}")
    print(f"  Other labels: {other_count}")
    print(f"  Unclassified: {unclassified}")

    # If sample mode, print all results for manual review
    if args.sample > 0:
        print(f"\n{'='*60}")
        print("SANITY CHECK — all classifications:")
        print(f"{'='*60}")
        print(f"\n  ABBREV ({abbrev_count}):")
        for span, label in sorted(results.items()):
            if label == "ABBREV":
                print(f"    {span}")
        print(f"\n  FULL ({full_count}):")
        for span, label in sorted(results.items()):
            if label == "FULL":
                print(f"    {span}")
        if other_count > 0:
            print(f"\n  OTHER ({other_count}):")
            for span, label in sorted(results.items()):
                if label not in ("ABBREV", "FULL"):
                    print(f"    {span} → {label}")

    # Save output
    output = {
        "abbreviations": sorted(k for k, v in results.items() if v == "ABBREV"),
        "not_abbreviations": sorted(k for k, v in results.items() if v == "FULL"),
        "unclassified": sorted(s for s in spans_list if s not in results),
        "raw_classifications": results,
        "stats": {
            "total_spans": len(spans_list),
            "abbrev": abbrev_count,
            "full": full_count,
            "other": other_count,
            "unclassified": unclassified,
            "elapsed_seconds": round(elapsed, 1),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
