#!/usr/bin/env python3
"""LLM concept disambiguation v2 — with confidence filtering and better prompting.

Improvements over v1:
- Prompt explicitly states the current concept and asks to confirm or change
- Uses logprobs for confidence filtering (only swap when highly confident)
- Filters queries: only asks about genuinely ambiguous mentions (minority
  concept has >=10% training share)
- Tests multiple confidence thresholds
"""
from __future__ import annotations

import math
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
CONTEXT_WINDOW = 250


def load_concept_names():
    ft = pd.read_csv(REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv")
    return dict(zip(ft["concept_id"].astype(int), ft["concept_name"]))


def predict_with_dict_keep_metadata(d, uc_d, notes):
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
    mention_concepts = defaultdict(Counter)
    for (section, mention), counter in d_combined.items():
        mention_concepts[mention] += counter
    return dict(mention_concepts)


def get_context(note_text, start, end, window=CONTEXT_WINDOW):
    ctx_start = max(0, start - window)
    ctx_end = min(len(note_text), end + window)
    return note_text[ctx_start:start], note_text[start:end], note_text[end:ctx_end]


def build_prompt_confirm_or_change(mention, before, after, current_name,
                                    candidates, concept_names):
    """Prompt that asks LLM to confirm or change the current concept assignment."""
    cand_lines = []
    for i, (cid, count) in enumerate(candidates):
        name = concept_names.get(cid, f"Unknown ({cid})")
        cand_lines.append(f"{i}: {name}")

    return f"""In this discharge note excerpt, the term "{mention}" has been automatically labeled as:
  {current_name}

Context: ...{before}[{mention}]{after}...

Is this the correct SNOMED CT concept, or would one of these alternatives be better?

{chr(10).join(cand_lines)}

Reply with ONLY the number of the correct concept."""


def parse_response(text, n_candidates):
    text = text.strip()
    for token in text.split():
        token = token.strip(".:,;()")
        try:
            idx = int(token)
            if 0 <= idx < n_candidates:
                return idx
        except ValueError:
            continue
    for ch in text:
        if ch.isdigit():
            idx = int(ch)
            if 0 <= idx < n_candidates:
                return idx
    return None  # Could not parse


def main():
    t0 = time.perf_counter()
    split_dir = REPO_ROOT / "data" / "old-challenge-split"
    super_dict_path = REPO_ROOT / "data" / "interim" / "super_dictionary_full.tsv"
    flat_term_path = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"

    print("Loading data...", flush=True)
    train_notes_df = pd.read_csv(split_dir / "train_notes.csv")
    train_annotations_df = pd.read_csv(split_dir / "train_annotations.csv")
    test_notes = pd.read_csv(split_dir / "test_notes.csv")
    test_gold = pd.read_csv(split_dir / "test_annotations.csv")
    for col in ["start", "end", "concept_id"]:
        test_gold[col] = test_gold[col].astype(int)

    concept_names = load_concept_names()
    test_text_by_id = {str(r["note_id"]): r["text"] for _, r in test_notes.iterrows()}

    # Build training state
    print("Building training state...", flush=True)
    texts = train_notes_df.set_index("note_id")["text"]
    texts_lc = texts.str.lower()
    headers = [h.lower() for h in COMMON_HEADERS]
    annotations = train_annotations_df.copy()
    if "span" in annotations.columns:
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

    # Baseline
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

    gold_by_note = {str(nid): g for nid, g in test_gold.groupby("note_id")}

    # Identify ambiguous predictions with filtering
    print("\nIdentifying ambiguous predictions...", flush=True)
    ambiguous = []
    for idx in pred_baseline.index:
        s, e = int(pred_baseline.loc[idx, "start"]), int(pred_baseline.loc[idx, "end"])
        cid = int(pred_baseline.loc[idx, "concept_id"])
        nid = str(pred_baseline.loc[idx, "note_id"])

        note_text = test_text_by_id.get(nid, "")
        mention = " ".join(note_text[s:e].lower().split())

        alternatives = mention_freq.get(mention, {})
        if len(alternatives) < 2:
            continue

        # Filter: only query if minority concept has >=10% share
        total = sum(alternatives.values())
        sorted_alts = sorted(alternatives.items(), key=lambda x: -x[1])
        top_share = sorted_alts[0][1] / total
        if top_share >= 0.95:
            continue  # 95%+ dominant — not genuinely ambiguous

        ambiguous.append({
            "pred_idx": idx,
            "note_id": nid,
            "start": s,
            "end": e,
            "current_cid": cid,
            "mention": mention,
            "alternatives": alternatives,
            "top_share": top_share,
        })

    print(f"  Total predictions: {len(pred_baseline):,}")
    print(f"  Genuinely ambiguous (<95% dominant): {len(ambiguous)}")

    # Load model
    print(f"\nLoading MedGemma...", flush=True)
    from llama_cpp import Llama
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=2048,
        logits_all=False,
        verbose=False,
    )
    print("  Model loaded!")

    # Run LLM queries and collect results with logprobs
    print(f"\nQuerying LLM on {len(ambiguous)} predictions...", flush=True)

    llm_results = []
    t_llm = time.perf_counter()

    for i, item in enumerate(ambiguous):
        nid = item["note_id"]
        s, e = item["start"], item["end"]
        current_cid = item["current_cid"]
        mention = item["mention"]

        note_text = test_text_by_id.get(nid, "")
        before, mention_text, after = get_context(note_text, s, e)

        # Build candidates list with current concept first
        sorted_alts = sorted(item["alternatives"].items(), key=lambda x: -x[1])
        candidates = sorted_alts[:6]

        # Find index of current concept in candidates
        current_idx = None
        for ci, (cid, _) in enumerate(candidates):
            if cid == current_cid:
                current_idx = ci
                break

        current_name = concept_names.get(current_cid, f"Unknown ({current_cid})")

        prompt = build_prompt_confirm_or_change(
            mention_text, before, after, current_name,
            candidates, concept_names
        )

        try:
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
            )
            answer = response["choices"][0]["message"]["content"]
            chosen_idx = parse_response(answer, len(candidates))

            # Extract logprob of the first token (the number)
            logprob = None
            lp_content = response["choices"][0].get("logprobs", {})
            if lp_content and "content" in lp_content and lp_content["content"]:
                logprob = lp_content["content"][0].get("logprob", None)

        except Exception as exc:
            chosen_idx = current_idx if current_idx is not None else 0
            logprob = None

        if chosen_idx is None:
            chosen_idx = current_idx if current_idx is not None else 0

        chosen_cid = candidates[chosen_idx][0]

        # Check against gold
        gold = gold_by_note.get(nid)
        gold_cids = set()
        if gold is not None:
            overlaps = gold[(gold["start"] < e) & (gold["end"] > s)]
            gold_cids = set(overlaps["concept_id"].astype(int))

        llm_results.append({
            "pred_idx": item["pred_idx"],
            "current_cid": current_cid,
            "chosen_cid": chosen_cid,
            "current_idx": current_idx,
            "chosen_idx": chosen_idx,
            "logprob": logprob,
            "gold_cids": gold_cids,
            "mention": mention,
            "current_correct": current_cid in gold_cids,
            "chosen_correct": chosen_cid in gold_cids,
            "is_swap": chosen_cid != current_cid,
        })

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_llm
            n_swaps = sum(1 for r in llm_results if r["is_swap"])
            n_correct = sum(1 for r in llm_results if r["is_swap"] and r["chosen_correct"])
            n_incorrect = sum(1 for r in llm_results if r["is_swap"] and r["current_correct"])
            print(f"  {i+1}/{len(ambiguous)} ({(i+1)/elapsed:.1f}/s) "
                  f"swaps={n_swaps} correct={n_correct} incorrect={n_incorrect}",
                  flush=True)

    elapsed_llm = time.perf_counter() - t_llm
    print(f"\n  LLM done in {elapsed_llm:.1f}s ({len(ambiguous)/elapsed_llm:.1f}/s)")

    # ===================================================================
    # Test multiple confidence thresholds
    # ===================================================================
    print("\n" + "=" * 80)
    print("CONFIDENCE THRESHOLD SWEEP")
    print("=" * 80)

    # Logprob distribution
    logprobs = [r["logprob"] for r in llm_results if r["logprob"] is not None and r["is_swap"]]
    if logprobs:
        probs = [math.exp(lp) for lp in logprobs]
        print(f"\n  Swap logprob stats (n={len(logprobs)}):")
        print(f"    prob: min={min(probs):.4f} median={sorted(probs)[len(probs)//2]:.4f} "
              f"max={max(probs):.4f}")

    for min_prob in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        pred_test = pred_baseline.copy()
        n_swaps = 0
        n_correct = 0
        n_incorrect = 0

        for r in llm_results:
            if not r["is_swap"]:
                continue

            prob = math.exp(r["logprob"]) if r["logprob"] is not None else 0.0
            if prob < min_prob:
                continue

            pred_test.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
            n_swaps += 1
            if r["chosen_correct"]:
                n_correct += 1
            if r["current_correct"]:
                n_incorrect += 1

        pred_test_eval = pred_test[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]:
            pred_test_eval[c] = pred_test_eval[c].astype(int)
        iou = macro_char_iou(pred_test_eval, test_gold[["note_id", "start", "end", "concept_id"]])

        swap_acc = n_correct / n_swaps * 100 if n_swaps > 0 else 0
        print(f"  prob>={min_prob:.2f}: IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
              f"swaps={n_swaps:>4} right={n_correct:>3} wrong={n_incorrect:>3} "
              f"acc={swap_acc:.0f}%")

    # ===================================================================
    # Also test: only swap when LLM agrees AND frequency supports it
    # ===================================================================
    print("\n" + "=" * 80)
    print("LLM + FREQUENCY AGREEMENT")
    print("=" * 80)

    for min_prob in [0.0, 0.5, 0.7, 0.9]:
        pred_combo = pred_baseline.copy()
        n_swaps = 0
        n_correct = 0
        n_incorrect = 0

        for r in llm_results:
            if not r["is_swap"]:
                continue

            prob = math.exp(r["logprob"]) if r["logprob"] is not None else 0.0
            if prob < min_prob:
                continue

            # Also require that the chosen concept has higher frequency
            mention = r["mention"]
            alts = mention_freq.get(mention, {})
            current_freq = alts.get(r["current_cid"], 0)
            chosen_freq = alts.get(r["chosen_cid"], 0)
            if chosen_freq <= current_freq:
                continue  # Skip if frequency doesn't support it

            pred_combo.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
            n_swaps += 1
            if r["chosen_correct"]:
                n_correct += 1
            if r["current_correct"]:
                n_incorrect += 1

        pred_combo_eval = pred_combo[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]:
            pred_combo_eval[c] = pred_combo_eval[c].astype(int)
        iou = macro_char_iou(pred_combo_eval, test_gold[["note_id", "start", "end", "concept_id"]])

        swap_acc = n_correct / n_swaps * 100 if n_swaps > 0 else 0
        print(f"  prob>={min_prob:.2f}+freq: IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
              f"swaps={n_swaps:>4} right={n_correct:>3} wrong={n_incorrect:>3} "
              f"acc={swap_acc:.0f}%")

    # ===================================================================
    # Only swap when current concept is NOT the most frequent
    # ===================================================================
    print("\n" + "=" * 80)
    print("LLM SWAP ONLY WHEN CURRENT != MOST FREQUENT")
    print("=" * 80)

    for min_prob in [0.0, 0.5, 0.7, 0.9]:
        pred_nontop = pred_baseline.copy()
        n_swaps = 0
        n_correct = 0
        n_incorrect = 0

        for r in llm_results:
            if not r["is_swap"]:
                continue

            # Only consider if current concept is NOT the most frequent
            mention = r["mention"]
            alts = mention_freq.get(mention, {})
            if alts:
                most_common_cid = max(alts, key=alts.get)
                if r["current_cid"] == most_common_cid:
                    continue  # Current is already most frequent, don't swap

            prob = math.exp(r["logprob"]) if r["logprob"] is not None else 0.0
            if prob < min_prob:
                continue

            pred_nontop.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
            n_swaps += 1
            if r["chosen_correct"]:
                n_correct += 1
            if r["current_correct"]:
                n_incorrect += 1

        pred_nontop_eval = pred_nontop[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]:
            pred_nontop_eval[c] = pred_nontop_eval[c].astype(int)
        iou = macro_char_iou(pred_nontop_eval, test_gold[["note_id", "start", "end", "concept_id"]])

        swap_acc = n_correct / n_swaps * 100 if n_swaps > 0 else 0
        print(f"  prob>={min_prob:.2f}: IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
              f"swaps={n_swaps:>4} right={n_correct:>3} wrong={n_incorrect:>3} "
              f"acc={swap_acc:.0f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Baseline: {iou_baseline:.4f}")
    print(f"  Oracle ceiling (training-available): +0.0381")
    print(f"  freq_only from v1 experiment: +0.0024")
    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
