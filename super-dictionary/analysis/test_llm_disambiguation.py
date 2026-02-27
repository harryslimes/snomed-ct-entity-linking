#!/usr/bin/env python3
"""LLM-based concept disambiguation using local MedGemma model.

For each prediction where the training dictionary has multiple candidate
concepts for the same mention, ask MedGemma to pick the best concept
given the clinical context.
"""
from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from train_dictionary import (
    COMMON_HEADERS, BLACKLIST_THRESH, INTERNAL_BLACKLIST,
    IndexedDict, annotate_with_dict, remove_overlaps,
    CASE_SENSITIVE_DICT, train,
    build_dict_from_annotations,
)
from runtime_scoring import macro_char_iou, class_char_iou

MODEL_PATH = REPO_ROOT / "models" / "medgemma-1.5-4b-it-Q8_0.gguf"
CONTEXT_WINDOW = 200  # chars on each side of mention


def load_concept_names():
    """Load SNOMED concept names from flattened terminology."""
    ft = pd.read_csv(REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv")
    return dict(zip(ft["concept_id"].astype(int), ft["concept_name"]))


def predict_with_dict_keep_metadata(d, uc_d, notes):
    """Standard prediction keeping section/dict_entry metadata."""
    uc_d_full = dict(uc_d)
    uc_d_full.update(CASE_SENSITIVE_DICT)

    headers_lc = [h.lower() for h in COMMON_HEADERS]
    headers_orig = list(COMMON_HEADERS)
    d_indexed = IndexedDict(d, prefilter="bigram")
    uc_indexed = IndexedDict(uc_d_full, prefilter="bigram")

    all_preds = []
    for _, note_row in notes.iterrows():
        note_id = str(note_row["note_id"])
        text = note_row["text"]
        text_lc = text.lower()
        ann_lc = annotate_with_dict(text_lc, d_indexed, headers_lc, note_id)
        ann_uc = annotate_with_dict(text, uc_indexed, headers_orig, note_id)
        combined = pd.concat([ann_lc, ann_uc], ignore_index=True)
        if not combined.empty:
            combined = remove_overlaps(combined)
            all_preds.append(combined)

    if not all_preds:
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id",
                                      "section", "dict_entry"])
    pred = pd.concat(all_preds, ignore_index=True)
    for c in ["start", "end", "concept_id"]:
        pred[c] = pred[c].astype(int)
    return pred


def build_mention_concept_priors(d_combined):
    """Aggregate (mention -> {concept_id: count}) across sections."""
    mention_concepts = defaultdict(Counter)
    for (section, mention), counter in d_combined.items():
        mention_concepts[mention] += counter
    return dict(mention_concepts)


def get_context(note_text, start, end, window=CONTEXT_WINDOW):
    """Extract context window around a mention."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(note_text), end + window)
    before = note_text[ctx_start:start]
    mention = note_text[start:end]
    after = note_text[end:ctx_end]
    return before, mention, after


def build_disambiguation_prompt(mention, before, after, candidates, concept_names):
    """Build a prompt for MedGemma to disambiguate concepts."""
    cand_lines = []
    for i, (cid, count) in enumerate(candidates):
        name = concept_names.get(cid, f"Unknown ({cid})")
        cand_lines.append(f"{i}: {name}")

    prompt = f"""Select the SNOMED CT concept that best matches the highlighted term in this clinical note excerpt.

Context: ...{before}**{mention}**{after}...

Term: {mention}
Candidates:
{chr(10).join(cand_lines)}

Reply with ONLY the number of the best matching candidate."""
    return prompt


def parse_llm_response(response, n_candidates):
    """Parse LLM response to extract chosen candidate index."""
    text = response.strip()
    # Try to find a number
    for token in text.split():
        token = token.strip(".:,;")
        try:
            idx = int(token)
            if 0 <= idx < n_candidates:
                return idx
        except ValueError:
            continue
    # Fallback: look for first digit
    for ch in text:
        if ch.isdigit():
            idx = int(ch)
            if 0 <= idx < n_candidates:
                return idx
    return 0  # Default to first candidate


def main():
    t0 = time.perf_counter()
    split_dir = REPO_ROOT / "data" / "old-challenge-split"
    super_dict_path = REPO_ROOT / "data" / "interim" / "super_dictionary_full.tsv"
    flat_term_path = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"

    # Load data
    print("Loading data...", flush=True)
    train_notes_df = pd.read_csv(split_dir / "train_notes.csv")
    train_annotations_df = pd.read_csv(split_dir / "train_annotations.csv")
    test_notes = pd.read_csv(split_dir / "test_notes.csv")
    test_gold = pd.read_csv(split_dir / "test_annotations.csv")
    for col in ["start", "end", "concept_id"]:
        test_gold[col] = test_gold[col].astype(int)

    concept_names = load_concept_names()
    print(f"  Loaded {len(concept_names):,} concept names")

    # Build test note text lookup
    test_text_by_id = {}
    for _, row in test_notes.iterrows():
        test_text_by_id[str(row["note_id"])] = row["text"]

    # Build training state
    print("Building training state...", flush=True)
    texts = train_notes_df.set_index("note_id")["text"]
    texts_lc = texts.str.lower()
    headers = [h.lower() for h in COMMON_HEADERS]

    annotations = train_annotations_df.copy()
    if "span" in annotations.columns:
        annotations["source orig"] = annotations["span"]
        annotations["source"] = annotations["span"].str.lower()

    words_counter = Counter()
    for text in texts_lc:
        words_counter.update(text.split())
    blacklist_set = {w for w in words_counter if words_counter[w] > BLACKLIST_THRESH}
    blacklist_set.update(INTERNAL_BLACKLIST)

    d_combined: dict[tuple, Counter] = {}
    for nid in texts_lc.index:
        note_anns = annotations[annotations["note_id"] == nid]
        if note_anns.empty:
            continue
        t = build_dict_from_annotations(texts_lc[nid], note_anns, headers, blacklist_set)
        for k in t:
            d_combined.setdefault(k, Counter()).update(t[k])

    mention_freq = build_mention_concept_priors(d_combined)
    print(f"  {len(mention_freq)} unique mentions with concept priors")

    # Build baseline
    print("\nBuilding baseline...", flush=True)
    d_baseline, uc_baseline = train(
        train_notes_df, train_annotations_df,
        super_dict_path=super_dict_path,
        flat_terminology_path=flat_term_path,
    )

    pred_baseline = predict_with_dict_keep_metadata(d_baseline, uc_baseline, test_notes)
    pred_eval = pred_baseline[["note_id", "start", "end", "concept_id"]].copy()
    iou_baseline = macro_char_iou(pred_eval, test_gold[["note_id", "start", "end", "concept_id"]])
    print(f"  Baseline IoU: {iou_baseline:.4f}  ({len(pred_baseline):,} predictions)")

    # Identify ambiguous predictions
    print("\nIdentifying ambiguous predictions...", flush=True)
    ambiguous = []
    for idx in pred_baseline.index:
        s, e = int(pred_baseline.loc[idx, "start"]), int(pred_baseline.loc[idx, "end"])
        cid = int(pred_baseline.loc[idx, "concept_id"])
        nid = str(pred_baseline.loc[idx, "note_id"])

        note_text = test_text_by_id.get(nid, "")
        mention = note_text[s:e].lower()
        mention_norm = " ".join(mention.split())

        alternatives = mention_freq.get(mention_norm, {})
        if len(alternatives) >= 2:
            ambiguous.append({
                "pred_idx": idx,
                "note_id": nid,
                "start": s,
                "end": e,
                "current_cid": cid,
                "mention": mention_norm,
                "alternatives": alternatives,
            })

    print(f"  Total predictions: {len(pred_baseline):,}")
    print(f"  Ambiguous (multiple training concepts): {len(ambiguous)}")

    # ===================================================================
    # Load MedGemma
    # ===================================================================
    print(f"\nLoading MedGemma from {MODEL_PATH}...", flush=True)
    from llama_cpp import Llama

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,  # Full GPU offload
        n_ctx=2048,
        verbose=False,
    )
    print("  Model loaded!")

    # ===================================================================
    # Run LLM disambiguation
    # ===================================================================
    print(f"\nRunning LLM disambiguation on {len(ambiguous)} ambiguous predictions...", flush=True)

    pred_llm = pred_baseline.copy()
    swaps = 0
    correct_swaps = 0
    incorrect_swaps = 0
    kept_correct = 0
    kept_wrong = 0

    gold_by_note = {}
    for nid, g in test_gold.groupby("note_id"):
        gold_by_note[str(nid)] = g

    t_llm = time.perf_counter()
    for i, item in enumerate(ambiguous):
        nid = item["note_id"]
        s, e = item["start"], item["end"]
        current_cid = item["current_cid"]
        mention = item["mention"]

        note_text = test_text_by_id.get(nid, "")
        before, mention_text, after = get_context(note_text, s, e)

        # Sort candidates by training frequency (most common first)
        candidates = sorted(item["alternatives"].items(), key=lambda x: -x[1])
        # Limit to top 6 candidates
        candidates = candidates[:6]

        prompt = build_disambiguation_prompt(
            mention_text, before, after, candidates, concept_names
        )

        try:
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0.0,
            )
            answer = response["choices"][0]["message"]["content"]
            chosen_idx = parse_llm_response(answer, len(candidates))
        except Exception as exc:
            chosen_idx = 0  # Default to most common

        chosen_cid = candidates[chosen_idx][0]

        # Check against gold
        gold = gold_by_note.get(nid)
        gold_cids = set()
        if gold is not None:
            overlaps = gold[(gold["start"] < e) & (gold["end"] > s)]
            gold_cids = set(overlaps["concept_id"].astype(int))

        if chosen_cid != current_cid:
            pred_llm.loc[item["pred_idx"], "concept_id"] = chosen_cid
            swaps += 1
            if chosen_cid in gold_cids:
                correct_swaps += 1
            elif current_cid in gold_cids:
                incorrect_swaps += 1
        else:
            if current_cid in gold_cids:
                kept_correct += 1
            elif gold_cids:
                kept_wrong += 1

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_llm
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(ambiguous)} ({rate:.1f}/s) "
                  f"swaps={swaps} correct={correct_swaps} incorrect={incorrect_swaps}",
                  flush=True)

    elapsed_llm = time.perf_counter() - t_llm
    print(f"\n  LLM disambiguation done in {elapsed_llm:.1f}s "
          f"({len(ambiguous)/elapsed_llm:.1f} queries/s)")

    # Evaluate
    pred_llm_eval = pred_llm[["note_id", "start", "end", "concept_id"]].copy()
    for c in ["start", "end", "concept_id"]:
        pred_llm_eval[c] = pred_llm_eval[c].astype(int)
    iou_llm = macro_char_iou(pred_llm_eval, test_gold[["note_id", "start", "end", "concept_id"]])

    print(f"\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"  Baseline IoU:       {iou_baseline:.4f}")
    print(f"  LLM disambig IoU:   {iou_llm:.4f} (delta: {iou_llm - iou_baseline:+.4f})")
    print(f"  Queries: {len(ambiguous)}")
    print(f"  Swaps:   {swaps}")
    print(f"    Correct swaps (wrong→right):  {correct_swaps}")
    print(f"    Incorrect swaps (right→wrong): {incorrect_swaps}")
    print(f"    Kept correct: {kept_correct}")
    print(f"    Kept wrong:   {kept_wrong}")
    if swaps > 0:
        print(f"    Swap accuracy: {correct_swaps}/{swaps} = {correct_swaps/swaps:.1%}")

    # ===================================================================
    # Per-concept IoU comparison
    # ===================================================================
    baseline_class = class_char_iou(pred_eval, test_gold[["note_id", "start", "end", "concept_id"]])
    llm_class = class_char_iou(pred_llm_eval, test_gold[["note_id", "start", "end", "concept_id"]])

    merged = baseline_class.merge(llm_class, on="concept_id", suffixes=("_base", "_llm"))
    merged["iou_delta"] = merged["iou_llm"] - merged["iou_base"]

    improved = merged[merged["iou_delta"] > 0.001].sort_values("iou_delta", ascending=False)
    degraded = merged[merged["iou_delta"] < -0.001].sort_values("iou_delta")

    print(f"\n  Concepts improved: {len(improved)}")
    if not improved.empty:
        for _, row in improved.head(10).iterrows():
            name = concept_names.get(int(row["concept_id"]), "?")[:50]
            print(f"    {int(row['concept_id']):>12} {row['iou_base']:.4f}→{row['iou_llm']:.4f} "
                  f"({row['iou_delta']:+.4f}) {name}")

    print(f"\n  Concepts degraded: {len(degraded)}")
    if not degraded.empty:
        for _, row in degraded.head(10).iterrows():
            name = concept_names.get(int(row["concept_id"]), "?")[:50]
            print(f"    {int(row['concept_id']):>12} {row['iou_base']:.4f}→{row['iou_llm']:.4f} "
                  f"({row['iou_delta']:+.4f}) {name}")

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
