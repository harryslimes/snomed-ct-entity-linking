#!/usr/bin/env python3
"""Evaluate GLiNER2 on the test set from old-challenge-split.

Runs inference on test notes (chunked), merges predictions across overlapping
chunks, and computes span-level P/R/F1 metrics.

Usage:
    python scripts/eval_gliner2.py \
        --model-dir models/gliner2-snomed/best \
        --data-dir data/old-challenge-split
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from gliner2 import GLiNER2


# ---------------------------------------------------------------------------
# Entity types and descriptions (must match training)
# ---------------------------------------------------------------------------

ENTITY_TYPES = [
    "medical finding, symptom, or disease",
    "procedure",
    "anatomical body part",
]

ALL_ENTITY_TYPES = ENTITY_TYPES

ENTITY_DESCRIPTIONS = {
    "medical finding, symptom, or disease": "Clinical findings, symptoms, disorders, diseases, diagnoses, and abnormal observations",
    "procedure": "Medical procedures, surgeries, therapies, laboratory tests, diagnostic tests, and clinical assessments",
    "anatomical body part": "Body structures, organs, anatomical regions, and morphologic abnormalities",
}


# ---------------------------------------------------------------------------
# Section-aware chunking (reuse from prepare_gliner2_data.py)
# ---------------------------------------------------------------------------

# Import the section-aware chunking functions from prep script
sys.path.insert(0, str(Path(__file__).parent))
from prepare_gliner2_data import (
    chunk_note_by_section,
    HEADER_DISPLAY,
)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_note(
    model: GLiNER2,
    text: str,
    window_tokens: int = 400,
    overlap_tokens: int = 50,
    threshold: float = 0.5,
) -> list[dict]:
    """Run GLiNER2 on a full clinical note with section-aware chunking and dedup.

    Each chunk gets its section header prepended (same as training).
    Returns list of dicts: {start, end, span, entity_type, confidence}
    """
    chunks = chunk_note_by_section(text, window_tokens, overlap_tokens)
    all_preds = []

    for section_header, chunk_start, chunk_end in chunks:
        raw_chunk_text = text[chunk_start:chunk_end].strip()
        if not raw_chunk_text:
            continue

        # Prepend section header (same as training)
        display_header = HEADER_DISPLAY.get(section_header, section_header.title())
        if display_header:
            chunk_text = f"[{display_header}] {raw_chunk_text}"
        else:
            chunk_text = raw_chunk_text

        result = model.extract_entities(
            chunk_text,
            ALL_ENTITY_TYPES,
            threshold=threshold,
            include_confidence=True,
            include_spans=True,
        )

        # The header prefix shifts character offsets. Calculate the offset.
        header_prefix_len = len(chunk_text) - len(raw_chunk_text)

        # result format: {"entities": {"entity_type": [{"text": ..., "confidence": ...,
        #                  "start": ..., "end": ...}, ...]}}
        if isinstance(result, dict):
            # GLiNER2 wraps output in {"entities": {...}}
            entity_dict = result.get("entities", result)
            for etype, entities in entity_dict.items():
                if not isinstance(entities, list):
                    continue
                for ent in entities:
                    if isinstance(ent, dict):
                        conf = ent.get("confidence", 1.0)
                        if conf < threshold:
                            continue
                        # Convert chunk-relative offsets to note-level,
                        # accounting for the prepended header
                        ent_start_in_chunk = ent.get("start", 0)
                        ent_end_in_chunk = ent.get("end", 0)

                        # Skip entities that fall within the header prefix
                        if ent_end_in_chunk <= header_prefix_len:
                            continue

                        ent_start = ent_start_in_chunk - header_prefix_len + chunk_start
                        ent_end = ent_end_in_chunk - header_prefix_len + chunk_start
                        # Clamp to chunk bounds
                        ent_start = max(ent_start, chunk_start)
                        ent_end = min(ent_end, chunk_end)

                        all_preds.append({
                            "start": ent_start,
                            "end": ent_end,
                            "span": ent.get("text", text[ent_start:ent_end]),
                            "entity_type": etype,
                            "confidence": conf,
                        })

    # Deduplicate overlapping predictions (from chunk overlap)
    all_preds = deduplicate_predictions(all_preds)
    return all_preds


def deduplicate_predictions(preds: list[dict]) -> list[dict]:
    """Remove duplicate predictions from overlapping chunks.

    For predictions with IoU > 0.5 and same entity type, keep the one with
    higher confidence.
    """
    if not preds:
        return preds

    # Sort by confidence descending
    preds = sorted(preds, key=lambda x: -x["confidence"])
    kept = []

    for pred in preds:
        is_dup = False
        for existing in kept:
            if existing["entity_type"] != pred["entity_type"]:
                continue
            # Compute IoU
            overlap_start = max(pred["start"], existing["start"])
            overlap_end = min(pred["end"], existing["end"])
            if overlap_start >= overlap_end:
                continue
            overlap = overlap_end - overlap_start
            union = (max(pred["end"], existing["end"])
                     - min(pred["start"], existing["start"]))
            iou = overlap / union if union > 0 else 0
            if iou > 0.5:
                is_dup = True
                break
        if not is_dup:
            kept.append(pred)

    return kept


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_span_metrics(
    gold_spans: list[tuple[int, int, str]],
    pred_spans: list[tuple[int, int, str]],
    iou_threshold: float = 0.0,
) -> dict:
    """Compute span-level P/R/F1.

    Each span is (start, end, entity_type).
    iou_threshold=0.0 means exact match, >0 means partial match.
    """
    if iou_threshold == 0.0:
        # Exact match
        gold_set = set(gold_spans)
        pred_set = set(pred_spans)
        tp = len(gold_set & pred_set)
    else:
        # Partial match via IoU
        matched_gold = set()
        matched_pred = set()
        for i, (gs, ge, gt) in enumerate(gold_spans):
            for j, (ps, pe, pt) in enumerate(pred_spans):
                if gt != pt:
                    continue
                overlap_start = max(gs, ps)
                overlap_end = min(ge, pe)
                if overlap_start >= overlap_end:
                    continue
                overlap = overlap_end - overlap_start
                union = max(ge, pe) - min(gs, ps)
                iou = overlap / union if union > 0 else 0
                if iou >= iou_threshold and i not in matched_gold and j not in matched_pred:
                    matched_gold.add(i)
                    matched_pred.add(j)
        tp = len(matched_gold)

    precision = tp / len(pred_spans) if pred_spans else 0
    recall = tp / len(gold_spans) if gold_spans else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "tp": tp,
        "fp": len(pred_spans) - tp,
        "fn": len(gold_spans) - tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate GLiNER2 on test set")
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to trained GLiNER2 model or adapter directory",
    )
    parser.add_argument(
        "--base-model", type=str, default="fastino/gliner2-large-v1",
        help="Base model (used if model-dir is an adapter)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/old-challenge-split"),
        help="Directory containing test_notes.csv and test_annotations.csv",
    )
    parser.add_argument(
        "--project-root", type=Path,
        default=Path("/workspaces/snomed-ct-entity-linking"),
        help="Main project root (for SNOMED tag lookup)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--window-tokens", type=int, default=200)
    parser.add_argument("--overlap-tokens", type=int, default=30)
    parser.add_argument("--filter-lab-tables", action="store_true",
                        help="Exclude gold annotations inside structured lab tables")
    args = parser.parse_args()

    # Load model
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model_dir}... (device={device})")
    try:
        model = GLiNER2.from_pretrained(args.model_dir).to(device)
    except Exception:
        # Try loading as adapter on base model
        print(f"Loading as adapter on {args.base_model}...")
        model = GLiNER2.from_pretrained(args.base_model).to(device)
        model.load_adapter(args.model_dir)

    # Load SNOMED tag mapping for gold class labels
    from prepare_gliner2_data import (
        load_sctid_to_tag, map_concept_to_fine_class,
        find_lab_regions, merge_adjacent_regions, annotation_in_lab_region,
    )
    sctid_to_tag = load_sctid_to_tag(args.project_root)

    # Load test data
    print("Loading test data...")
    test_notes = pd.read_csv(args.data_dir / "test_notes.csv")
    test_ann = pd.read_csv(args.data_dir / "test_annotations.csv",
                           dtype={"concept_id": int})
    test_ann["start"] = test_ann["start"].astype(int)
    test_ann["end"] = test_ann["end"].astype(int)

    # Map gold annotations to 3-class
    test_ann["cls"] = test_ann["concept_id"].apply(
        lambda cid: map_concept_to_fine_class(int(cid), sctid_to_tag)
    )

    note_texts = dict(zip(test_notes["note_id"], test_notes["text"]))

    # Lab table filtering on gold annotations
    if args.filter_lab_tables:
        n_before = len(test_ann)
        keep_mask = pd.Series(True, index=test_ann.index)
        for note_id, text in note_texts.items():
            regions = merge_adjacent_regions(find_lab_regions(text))
            if not regions:
                continue
            note_mask = test_ann["note_id"] == note_id
            for idx in test_ann[note_mask].index:
                row = test_ann.loc[idx]
                if annotation_in_lab_region(row["start"], row["end"], regions):
                    keep_mask[idx] = False
        test_ann = test_ann[keep_mask]
        print(f"  Lab table filter: {n_before - len(test_ann):,} gold annotations removed")

    # Run predictions
    print(f"\nRunning inference on {len(test_notes)} test notes...")
    all_gold_spans = []
    all_pred_spans = []
    per_class_gold = defaultdict(list)
    per_class_pred = defaultdict(list)
    for idx, (note_id, text) in enumerate(note_texts.items()):
        note_ann = test_ann[test_ann["note_id"] == note_id]

        # Gold spans
        for _, row in note_ann.iterrows():
            span_tuple = (int(row["start"]), int(row["end"]), row["cls"])
            all_gold_spans.append(span_tuple)
            per_class_gold[row["cls"]].append(span_tuple)

        # Predictions
        preds = predict_note(
            model, text,
            window_tokens=args.window_tokens,
            overlap_tokens=args.overlap_tokens,
            threshold=args.threshold,
        )

        for pred in preds:
            span_tuple = (pred["start"], pred["end"], pred["entity_type"])
            all_pred_spans.append(span_tuple)
            per_class_pred[pred["entity_type"]].append(span_tuple)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(note_texts)} notes")

    # Compute metrics
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total gold spans: {len(all_gold_spans):,}")
    print(f"Total pred spans: {len(all_pred_spans):,}")

    # Exact match
    print(f"\n--- Exact Match ---")
    exact = compute_span_metrics(all_gold_spans, all_pred_spans, iou_threshold=0.0)
    print(f"  Precision: {exact['precision']:.4f}")
    print(f"  Recall:    {exact['recall']:.4f}")
    print(f"  F1:        {exact['f1']:.4f}")

    # Partial match (IoU >= 0.5)
    print(f"\n--- Partial Match (IoU >= 0.5) ---")
    partial = compute_span_metrics(all_gold_spans, all_pred_spans, iou_threshold=0.5)
    print(f"  Precision: {partial['precision']:.4f}")
    print(f"  Recall:    {partial['recall']:.4f}")
    print(f"  F1:        {partial['f1']:.4f}")

    # Per-class metrics
    print(f"\n--- Per-Class (Exact Match) ---")
    for cls in ENTITY_TYPES:
        m = compute_span_metrics(
            per_class_gold.get(cls, []),
            per_class_pred.get(cls, []),
            iou_threshold=0.0,
        )
        n_gold = len(per_class_gold.get(cls, []))
        n_pred = len(per_class_pred.get(cls, []))
        print(f"  {cls:12s}: P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f}  (gold={n_gold:,} pred={n_pred:,})")


if __name__ == "__main__":
    main()
