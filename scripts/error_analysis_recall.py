#!/usr/bin/env python3
"""Analyze missed gold spans (false negatives) from GLiNER2 evaluation.

Dumps missed spans with context, categorizes by pattern, and identifies
what the model is systematically failing to extract.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from gliner2 import GLiNER2

# Same entity types as eval script
ENTITY_TYPES = [
    "medical finding, symptom, or disease",
    "procedure",
    "anatomical body part",
]
ALL_ENTITY_TYPES = ENTITY_TYPES + ["medical abbreviation"]

# Reuse functions from eval and data prep scripts
import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_gliner2 import predict_note, chunk_note_by_section
from prepare_gliner2_data import load_sctid_to_tag, TAG_TO_CLASS


def load_abbrev_set(path: Path) -> set[str]:
    with open(path) as f:
        data = json.load(f)
    return set(data.get("abbreviations", []))


def get_context(text: str, start: int, end: int, window: int = 60) -> str:
    """Get surrounding context for a span."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    before = text[ctx_start:start].replace("\n", "\\n")
    span = text[start:end]
    after = text[end:ctx_end].replace("\n", "\\n")
    return f"...{before}<<{span}>>{after}..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/old-challenge-split"))
    parser.add_argument("--base-model", type=str, default="fastino/gliner2-large-v1")
    parser.add_argument("--abbrev-classification", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--project-root", type=Path, default=Path("/workspaces/snomed-ct-entity-linking"))
    parser.add_argument("--max-notes", type=int, default=0, help="Limit notes for faster analysis")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_dir}... (device=cuda)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (args.model_dir / "adapter_config.json").exists():
        print(f"Loading as adapter on {args.base_model}...")
        model = GLiNER2.from_pretrained(
            args.base_model,
            adapter_path=str(args.model_dir),
            device=device,
        )
    else:
        model = GLiNER2.from_pretrained(str(args.model_dir), device=device)

    # Load SNOMED tags
    sctid_to_tag = load_sctid_to_tag(args.project_root)
    print(f"Loaded {len(sctid_to_tag)} concept → semantic tag mappings")

    # Load test data
    notes_df = pd.read_csv(args.data_dir / "test_notes.csv")
    ann_df = pd.read_csv(args.data_dir / "test_annotations.csv", dtype={"concept_id": int})
    note_texts = dict(zip(notes_df["note_id"], notes_df["text"]))

    # Load abbreviation set
    abbrev_set = set()
    if args.abbrev_classification and args.abbrev_classification.exists():
        abbrev_set = load_abbrev_set(args.abbrev_classification)
        print(f"Loaded {len(abbrev_set)} abbreviation patterns")

    # Map concept_id to class
    def concept_to_class(concept_id: int) -> str | None:
        tag = sctid_to_tag.get(concept_id)
        if not tag:
            return None
        return TAG_TO_CLASS.get(tag)

    # Run inference and collect misses
    test_notes = []
    for _, row in notes_df.iterrows():
        nid = row["note_id"]
        text = note_texts.get(nid, "")
        note_ann = ann_df[ann_df["note_id"] == nid]
        if len(text) > 0:
            test_notes.append((nid, text, note_ann))

    if args.max_notes > 0:
        test_notes = test_notes[:args.max_notes]

    all_misses = []
    all_gold = []
    all_pred = []
    n_abbrev_skipped = 0

    print(f"\nAnalyzing {len(test_notes)} test notes at threshold={args.threshold}...")

    for i, (nid, text, note_ann) in enumerate(test_notes):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(test_notes)} notes")

        # Gold spans (non-abbreviation)
        gold_spans = []
        for _, row in note_ann.iterrows():
            cls = concept_to_class(int(row["concept_id"]))
            if cls is None:
                cls = "medical finding, symptom, or disease"
            span_text = re.sub(r"\s+", " ", text[int(row["start"]):int(row["end"])]).strip()
            if abbrev_set and span_text in abbrev_set:
                n_abbrev_skipped += 1
                continue
            gold_spans.append({
                "start": int(row["start"]),
                "end": int(row["end"]),
                "cls": cls,
                "span_text": span_text,
                "concept_id": int(row["concept_id"]),
                "note_id": nid,
            })

        # Predictions
        preds = predict_note(model, text, threshold=args.threshold)
        pred_spans = []
        for pred in preds:
            if pred["entity_type"] == "medical abbreviation":
                continue
            pred_spans.append({
                "start": pred["start"],
                "end": pred["end"],
                "cls": pred["entity_type"],
                "span_text": pred.get("span", text[pred["start"]:pred["end"]]),
                "confidence": pred.get("confidence", 1.0),
            })

        # Match: find unmatched gold spans (exact match)
        pred_set = set((p["start"], p["end"], p["cls"]) for p in pred_spans)
        # Also check partial matches (IoU > 0.5)
        for g in gold_spans:
            g_tuple = (g["start"], g["end"], g["cls"])
            if g_tuple in pred_set:
                g["matched"] = "exact"
            else:
                # Check partial match
                best_iou = 0
                best_pred = None
                for p in pred_spans:
                    if p["cls"] != g["cls"]:
                        continue
                    overlap = max(0, min(g["end"], p["end"]) - max(g["start"], p["start"]))
                    union = max(g["end"], p["end"]) - min(g["start"], p["start"])
                    iou = overlap / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = p
                if best_iou >= 0.5:
                    g["matched"] = "partial"
                    g["best_iou"] = best_iou
                    g["pred_span"] = best_pred["span_text"] if best_pred else ""
                elif best_iou > 0:
                    g["matched"] = "overlap_low"
                    g["best_iou"] = best_iou
                    g["pred_span"] = best_pred["span_text"] if best_pred else ""
                else:
                    # Check if any pred overlaps regardless of type
                    any_overlap = False
                    for p in pred_spans:
                        overlap = max(0, min(g["end"], p["end"]) - max(g["start"], p["start"]))
                        if overlap > 0:
                            any_overlap = True
                            g["wrong_type_pred"] = p["cls"]
                            g["wrong_type_span"] = p["span_text"]
                            break
                    g["matched"] = "wrong_type" if any_overlap else "missed"

            g["context"] = get_context(text, g["start"], g["end"])
            all_gold.append(g)

        all_pred.extend(pred_spans)

    # Analyze misses
    print(f"\nAbbrev gold spans skipped: {n_abbrev_skipped}")
    print(f"Total non-abbrev gold spans: {len(all_gold)}")

    match_counts = Counter(g["matched"] for g in all_gold)
    print(f"\nMatch breakdown:")
    for match_type, count in match_counts.most_common():
        pct = 100 * count / len(all_gold)
        print(f"  {match_type:20s}: {count:6d} ({pct:.1f}%)")

    # Focus on completely missed spans
    missed = [g for g in all_gold if g["matched"] == "missed"]
    wrong_type = [g for g in all_gold if g["matched"] == "wrong_type"]

    print(f"\n{'='*70}")
    print(f"COMPLETELY MISSED SPANS: {len(missed)}")
    print(f"{'='*70}")

    # Categorize by class
    missed_by_class = defaultdict(list)
    for g in missed:
        missed_by_class[g["cls"]].append(g)

    for cls in ENTITY_TYPES:
        items = missed_by_class[cls]
        print(f"\n--- {cls}: {len(items)} missed ---")

        # Group by span text to find patterns
        span_counts = Counter(g["span_text"] for g in items)
        print(f"  Unique span texts: {len(span_counts)}")
        print(f"  Top 30 most-missed spans:")
        for span, count in span_counts.most_common(30):
            print(f"    {count:3d}x  \"{span}\"")

    # Analyze span length distribution of misses vs hits
    hit_lens = [g["end"] - g["start"] for g in all_gold if g["matched"] in ("exact", "partial")]
    miss_lens = [g["end"] - g["start"] for g in missed]

    import statistics
    if hit_lens and miss_lens:
        print(f"\n{'='*70}")
        print(f"SPAN LENGTH ANALYSIS")
        print(f"{'='*70}")
        print(f"  Hits:   median={statistics.median(hit_lens):.0f}, mean={statistics.mean(hit_lens):.1f}, "
              f"p10={sorted(hit_lens)[len(hit_lens)//10]}, p90={sorted(hit_lens)[9*len(hit_lens)//10]}")
        print(f"  Misses: median={statistics.median(miss_lens):.0f}, mean={statistics.mean(miss_lens):.1f}, "
              f"p10={sorted(miss_lens)[len(miss_lens)//10]}, p90={sorted(miss_lens)[9*len(miss_lens)//10]}")

    # Wrong type analysis
    if wrong_type:
        print(f"\n{'='*70}")
        print(f"WRONG TYPE PREDICTIONS: {len(wrong_type)}")
        print(f"{'='*70}")
        type_confusions = Counter(
            (g["cls"], g.get("wrong_type_pred", "?")) for g in wrong_type
        )
        for (gold_t, pred_t), count in type_confusions.most_common(20):
            print(f"  {count:4d}x  gold={gold_t:45s} pred={pred_t}")

    # Sample missed spans with context
    print(f"\n{'='*70}")
    print(f"SAMPLE MISSED SPANS WITH CONTEXT")
    print(f"{'='*70}")
    import random
    random.seed(42)
    for cls in ENTITY_TYPES:
        items = missed_by_class[cls]
        if not items:
            continue
        sample = random.sample(items, min(20, len(items)))
        print(f"\n--- {cls} (showing {len(sample)} of {len(items)}) ---")
        for g in sample:
            print(f"  [{g['concept_id']}] \"{g['span_text']}\"")
            print(f"    {g['context']}")


if __name__ == "__main__":
    main()
