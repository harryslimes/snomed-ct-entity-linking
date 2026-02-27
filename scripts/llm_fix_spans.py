#!/usr/bin/env python3
"""
Test if LLM can improve span boundaries for concepts with low IoU.
Focuses on the top concepts with worst span boundaries.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.super_dictionary.runtime_scoring import class_char_iou


def call_llm(vllm_url: str, model: str, prompt: str, max_tokens: int = 500) -> str:
    """Call vLLM server."""
    response = requests.post(
        f"{vllm_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"].strip()


SPAN_FIX_PROMPT = """You are a clinical entity annotator. You will see a note excerpt with predicted entity spans marked by [brackets].

Your task: For the concept "{concept_name}" (ID: {concept_id}), verify each predicted span is correct.

Note excerpt:
{context}

Current predictions for this concept:
{predictions}

Instructions:
1. Check if each span is the correct clinical term
2. Fix any incorrect boundaries (extend/trim to match the full clinical phrase)
3. Merge adjacent/overlapping spans if they refer to the same instance
4. Delete any spans that are clearly wrong

Output format (one per line):
KEEP: <start>-<end>
FIX: <old_start>-<old_end> -> <new_start>-<new_end>
DELETE: <start>-<end>

If all spans are correct, output: ALL_CORRECT
"""


def extract_concept_context(note_text: str, spans: list[tuple[int, int]], context_chars: int = 200) -> str:
    """Extract context around spans."""
    if not spans:
        return note_text[:1000]  # First 1000 chars if no spans

    # Get min/max positions
    all_starts = [s for s, e in spans]
    all_ends = [e for s, e in spans]
    min_pos = max(0, min(all_starts) - context_chars)
    max_pos = min(len(note_text), max(all_ends) + context_chars)

    context = note_text[min_pos:max_pos]

    # Mark predicted spans with brackets
    offset = min_pos
    marked = []
    last_end = 0

    for start, end in sorted(spans):
        rel_start = start - offset
        rel_end = end - offset
        if rel_start >= 0 and rel_end <= len(context):
            marked.append(context[last_end:rel_start])
            marked.append(f"[{context[rel_start:rel_end]}]")
            last_end = rel_end

    marked.append(context[last_end:])
    return "".join(marked)


def parse_llm_response(response: str, note_text: str) -> dict:
    """Parse LLM response to extract edits."""
    edits = {"keep": [], "fix": [], "delete": []}

    if "ALL_CORRECT" in response:
        return edits

    for line in response.split("\n"):
        line = line.strip()

        # KEEP: start-end
        if line.startswith("KEEP:"):
            match = re.search(r"(\d+)-(\d+)", line)
            if match:
                edits["keep"].append((int(match.group(1)), int(match.group(2))))

        # FIX: old_start-old_end -> new_start-new_end
        elif line.startswith("FIX:"):
            match = re.search(r"(\d+)-(\d+)\s*->\s*(\d+)-(\d+)", line)
            if match:
                old = (int(match.group(1)), int(match.group(2)))
                new = (int(match.group(3)), int(match.group(4)))
                edits["fix"].append((old, new))

        # DELETE: start-end
        elif line.startswith("DELETE:"):
            match = re.search(r"(\d+)-(\d+)", line)
            if match:
                edits["delete"].append((int(match.group(1)), int(match.group(2))))

    return edits


def apply_edits(pred_df: pd.DataFrame, concept_id: int, note_id: str, edits: dict) -> pd.DataFrame:
    """Apply edits to predictions."""
    # Get all rows for this concept+note
    mask = (pred_df["concept_id"] == concept_id) & (pred_df["note_id"] == note_id)
    concept_rows = pred_df[mask].copy()

    # Start with keeping all, then apply edits
    kept_rows = []

    for _, row in concept_rows.iterrows():
        span = (int(row["start"]), int(row["end"]))

        # Check if deleted
        if span in edits["delete"]:
            continue  # Skip this row

        # Check if fixed
        fixed = False
        for old, new in edits["fix"]:
            if span == old:
                row["start"] = new[0]
                row["end"] = new[1]
                fixed = True
                break

        kept_rows.append(row)

    # Remove old rows and add updated ones
    pred_df = pred_df[~mask]
    if kept_rows:
        pred_df = pd.concat([pred_df, pd.DataFrame(kept_rows)], ignore_index=True)

    return pred_df


def main():
    parser = argparse.ArgumentParser(description="Test LLM span boundary fixing")
    parser.add_argument("--pred-csv", required=True, help="Super dictionary predictions")
    parser.add_argument("--notes-csv", required=True, help="Notes CSV")
    parser.add_argument("--gold-csv", required=True, help="Gold annotations")
    parser.add_argument("--vllm-url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="Model name")
    parser.add_argument("--top-n", type=int, default=10, help="Fix top N worst concepts")
    parser.add_argument("--out-csv", required=True, help="Output predictions")
    args = parser.parse_args()

    print("="*80)
    print("LLM Span Boundary Fixing Test")
    print("="*80)

    # Load data
    pred_df = pd.read_csv(args.pred_csv)
    notes_df = pd.read_csv(args.notes_csv)
    gold_df = pd.read_csv(args.gold_csv)

    # Load concept names
    concept_names_df = pd.read_csv("1st Place/data/interim/flattened_terminology.csv")
    concept_names = dict(zip(concept_names_df["concept_id"], concept_names_df["concept_name"]))

    # Create note text lookup
    note_texts = dict(zip(notes_df["note_id"], notes_df["text"]))

    # Calculate current IoU
    print("\nCalculating baseline IoU...")
    baseline_class_iou = class_char_iou(
        pred_df[["note_id", "start", "end", "concept_id"]],
        gold_df[["note_id", "start", "end", "concept_id"]]
    )

    baseline_score = baseline_class_iou[baseline_class_iou["union"] > 0]["iou"].mean()
    print(f"Baseline score: {baseline_score:.4f}")

    # Find worst concepts (low IoU, high gold chars)
    true_positives = baseline_class_iou[baseline_class_iou["gt_chars"] > 0].copy()
    true_positives["chars_lost"] = true_positives["gt_chars"] * (1.0 - true_positives["iou"])
    worst_concepts = true_positives.nlargest(args.top_n, "chars_lost")

    print(f"\nFixing top {args.top_n} concepts with worst span boundaries:")
    for _, row in worst_concepts.iterrows():
        cid = int(row["concept_id"])
        name = concept_names.get(cid, "Unknown")
        print(f"  {cid:>10}  IoU={row['iou']:.3f}  gt={int(row['gt_chars']):>4} chars  lost={int(row['chars_lost']):>4}")
        print(f"              {name[:60]}")

    # Fix each concept
    modified_pred_df = pred_df.copy()

    for idx, row in worst_concepts.iterrows():
        concept_id = int(row["concept_id"])
        concept_name = concept_names.get(concept_id, "Unknown")

        print(f"\nProcessing concept {concept_id} ({concept_name})...")

        # Get all predictions for this concept
        concept_preds = pred_df[pred_df["concept_id"] == concept_id]

        # Group by note
        notes_fixed = 0
        for note_id, note_preds in concept_preds.groupby("note_id"):
            if note_id not in note_texts:
                continue

            note_text = note_texts[note_id]
            spans = list(zip(note_preds["start"].astype(int), note_preds["end"].astype(int)))

            # Extract context
            context = extract_concept_context(note_text, spans, context_chars=200)

            # Format predictions
            pred_list = "\n".join([f"  {s}-{e}: \"{note_text[s:e]}\"" for s, e in spans[:10]])
            if len(spans) > 10:
                pred_list += f"\n  ... +{len(spans)-10} more"

            # Build prompt
            prompt = SPAN_FIX_PROMPT.format(
                concept_name=concept_name,
                concept_id=concept_id,
                context=context,
                predictions=pred_list
            )

            # Call LLM
            try:
                response = call_llm(args.vllm_url, args.model, prompt)

                # Parse edits
                edits = parse_llm_response(response, note_text)

                if edits["keep"] or edits["fix"] or edits["delete"]:
                    print(f"  Note {note_id}: {len(edits['keep'])} keep, {len(edits['fix'])} fix, {len(edits['delete'])} delete")

                    # Apply edits
                    modified_pred_df = apply_edits(modified_pred_df, concept_id, note_id, edits)
                    notes_fixed += 1

            except Exception as e:
                print(f"  Error processing note {note_id}: {e}")
                continue

        print(f"  Fixed {notes_fixed} notes for this concept")

    # Calculate new IoU
    print("\nCalculating new IoU after LLM fixes...")
    new_class_iou = class_char_iou(
        modified_pred_df[["note_id", "start", "end", "concept_id"]],
        gold_df[["note_id", "start", "end", "concept_id"]]
    )

    new_score = new_class_iou[new_class_iou["union"] > 0]["iou"].mean()
    print(f"New score: {new_score:.4f}")
    print(f"Improvement: {new_score - baseline_score:+.4f}")

    # Save
    modified_pred_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved to: {args.out_csv}")


if __name__ == "__main__":
    main()
