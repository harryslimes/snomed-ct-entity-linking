#!/usr/bin/env python3
"""LLM concept disambiguation v3 — neutral prompt + logprob filtering.

Key changes from v1/v2:
- Neutral prompt (doesn't reveal current assignment) to avoid anchoring
- Randomized candidate order to eliminate position bias
- Logprob extraction for confidence filtering
- Only queries genuinely ambiguous mentions (<95% dominant concept)
- Tests multiple confidence thresholds post-hoc
"""
from __future__ import annotations

import math
import random
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
    CASE_SENSITIVE_DICT, train, build_dict_from_annotations,
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


def build_prompt(mention, before, after, candidates_shuffled, concept_names):
    """Neutral prompt — no anchoring to any particular concept."""
    cand_lines = []
    for i, (cid, _count) in enumerate(candidates_shuffled):
        name = concept_names.get(cid, f"Unknown ({cid})")
        cand_lines.append(f"{i}: {name}")

    return f"""Select the SNOMED CT concept that best matches the highlighted term in this clinical note.

Context: ...{before}[{mention}]{after}...

Term: {mention}
Candidates:
{chr(10).join(cand_lines)}

Answer:"""


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
    return None


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

    # Identify ambiguous predictions
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

        total = sum(alternatives.values())
        sorted_alts = sorted(alternatives.items(), key=lambda x: -x[1])
        top_share = sorted_alts[0][1] / total
        if top_share >= 0.95:
            continue

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

    # Load model (logits_all=False for speed; no logprobs needed)
    print(f"\nLoading MedGemma...", flush=True)
    from llama_cpp import Llama
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )
    print("  Model loaded!")

    # Run LLM queries
    print(f"\nQuerying LLM on {len(ambiguous)} predictions...", flush=True)

    random.seed(42)  # Reproducible shuffling
    llm_results = []
    t_llm = time.perf_counter()
    n_parse_fail = 0

    for i, item in enumerate(ambiguous):
        nid = item["note_id"]
        s, e = item["start"], item["end"]
        current_cid = item["current_cid"]
        mention = item["mention"]

        note_text = test_text_by_id.get(nid, "")
        before, mention_text, after = get_context(note_text, s, e)

        # Build shuffled candidates
        sorted_alts = sorted(item["alternatives"].items(), key=lambda x: -x[1])
        candidates = sorted_alts[:6]
        # Shuffle to eliminate position bias
        candidates_shuffled = list(candidates)
        random.shuffle(candidates_shuffled)

        # Track where current concept ended up
        current_shuffled_idx = None
        for ci, (cid, _) in enumerate(candidates_shuffled):
            if cid == current_cid:
                current_shuffled_idx = ci
                break

        prompt = build_prompt(
            mention_text, before, after, candidates_shuffled, concept_names
        )

        try:
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16,
                temperature=0.0,
            )
            answer = response["choices"][0]["message"]["content"]
            chosen_idx = parse_response(answer, len(candidates_shuffled))
        except Exception as exc:
            answer = ""
            chosen_idx = None

        if chosen_idx is None:
            n_parse_fail += 1
            chosen_idx = current_shuffled_idx if current_shuffled_idx is not None else 0

        chosen_cid = candidates_shuffled[chosen_idx][0]

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
            "gold_cids": gold_cids,
            "mention": mention,
            "current_correct": current_cid in gold_cids,
            "chosen_correct": chosen_cid in gold_cids,
            "is_swap": chosen_cid != current_cid,
            "answer": answer,
        })

        if (i + 1) % 200 == 0:
            elapsed = time.perf_counter() - t_llm
            n_swaps = sum(1 for r in llm_results if r["is_swap"])
            n_correct = sum(1 for r in llm_results if r["is_swap"] and r["chosen_correct"])
            n_incorrect = sum(1 for r in llm_results if r["is_swap"] and r["current_correct"])
            print(f"  {i+1}/{len(ambiguous)} ({(i+1)/elapsed:.1f}/s) "
                  f"swaps={n_swaps} correct={n_correct} incorrect={n_incorrect} "
                  f"parse_fail={n_parse_fail}",
                  flush=True)

    elapsed_llm = time.perf_counter() - t_llm
    total_swaps = sum(1 for r in llm_results if r["is_swap"])
    print(f"\n  LLM done in {elapsed_llm:.1f}s ({len(ambiguous)/elapsed_llm:.1f}/s)")
    print(f"  Total swaps: {total_swaps}, parse failures: {n_parse_fail}")

    # Show some sample answers
    print("\n  Sample LLM answers (first 10 swaps):")
    shown = 0
    for r in llm_results:
        if r["is_swap"] and shown < 10:
            tag = "GOOD" if r["chosen_correct"] else ("BAD" if r["current_correct"] else "???")
            print(f"    [{tag}] '{r['mention']}' {r['current_cid']}->{r['chosen_cid']} "
                  f"ans='{r.get('answer','')[:60]}'")
            shown += 1

    # ===================================================================
    # Experiment 1: All LLM swaps (unfiltered)
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: ALL LLM SWAPS (UNFILTERED)")
    print("=" * 80)

    pred_all = pred_baseline.copy()
    n_sw = 0
    n_cor = 0
    n_incor = 0
    for r in llm_results:
        if r["is_swap"]:
            pred_all.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
            n_sw += 1
            if r["chosen_correct"]: n_cor += 1
            if r["current_correct"]: n_incor += 1
    pred_all_eval = pred_all[["note_id", "start", "end", "concept_id"]].copy()
    for c in ["start", "end", "concept_id"]:
        pred_all_eval[c] = pred_all_eval[c].astype(int)
    iou_all = macro_char_iou(pred_all_eval, test_gold[["note_id", "start", "end", "concept_id"]])
    print(f"  IoU={iou_all:.4f} ({iou_all-iou_baseline:+.4f}) "
          f"swaps={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    # Experiment 2: LLM + frequency agreement (only swap when LLM picks
    # a concept with higher training freq than current)
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: LLM + FREQUENCY AGREEMENT")
    print("=" * 80)

    pred_freq = pred_baseline.copy()
    n_sw = n_cor = n_incor = 0
    for r in llm_results:
        if not r["is_swap"]:
            continue
        alts = mention_freq.get(r["mention"], {})
        if alts.get(r["chosen_cid"], 0) <= alts.get(r["current_cid"], 0):
            continue
        pred_freq.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
        n_sw += 1
        if r["chosen_correct"]: n_cor += 1
        if r["current_correct"]: n_incor += 1
    pred_freq_eval = pred_freq[["note_id", "start", "end", "concept_id"]].copy()
    for c in ["start", "end", "concept_id"]:
        pred_freq_eval[c] = pred_freq_eval[c].astype(int)
    iou_freq = macro_char_iou(pred_freq_eval, test_gold[["note_id", "start", "end", "concept_id"]])
    print(f"  IoU={iou_freq:.4f} ({iou_freq-iou_baseline:+.4f}) "
          f"swaps={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    # Experiment 3: LLM only when current != most frequent
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: LLM ONLY WHEN CURRENT != MOST FREQUENT")
    print("=" * 80)

    pred_nontop = pred_baseline.copy()
    n_sw = n_cor = n_incor = 0
    for r in llm_results:
        if not r["is_swap"]:
            continue
        alts = mention_freq.get(r["mention"], {})
        if alts:
            most_common_cid = max(alts, key=alts.get)
            if r["current_cid"] == most_common_cid:
                continue
        pred_nontop.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
        n_sw += 1
        if r["chosen_correct"]: n_cor += 1
        if r["current_correct"]: n_incor += 1
    pred_nontop_eval = pred_nontop[["note_id", "start", "end", "concept_id"]].copy()
    for c in ["start", "end", "concept_id"]:
        pred_nontop_eval[c] = pred_nontop_eval[c].astype(int)
    iou_nontop = macro_char_iou(pred_nontop_eval, test_gold[["note_id", "start", "end", "concept_id"]])
    print(f"  IoU={iou_nontop:.4f} ({iou_nontop-iou_baseline:+.4f}) "
          f"swaps={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    # Experiment 4: LLM + frequency agreement + current != most frequent
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: LLM + FREQ AGREE + CURRENT != TOP")
    print("=" * 80)

    pred_combo = pred_baseline.copy()
    n_sw = n_cor = n_incor = 0
    for r in llm_results:
        if not r["is_swap"]:
            continue
        alts = mention_freq.get(r["mention"], {})
        if alts:
            most_common_cid = max(alts, key=alts.get)
            if r["current_cid"] == most_common_cid:
                continue
        if alts.get(r["chosen_cid"], 0) <= alts.get(r["current_cid"], 0):
            continue
        pred_combo.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
        n_sw += 1
        if r["chosen_correct"]: n_cor += 1
        if r["current_correct"]: n_incor += 1
    pred_combo_eval = pred_combo[["note_id", "start", "end", "concept_id"]].copy()
    for c in ["start", "end", "concept_id"]:
        pred_combo_eval[c] = pred_combo_eval[c].astype(int)
    iou_combo = macro_char_iou(pred_combo_eval, test_gold[["note_id", "start", "end", "concept_id"]])
    print(f"  IoU={iou_combo:.4f} ({iou_combo-iou_baseline:+.4f}) "
          f"swaps={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    # Experiment 5: LLM picks most frequent (frequency vote wins ties)
    # ===================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: ONLY SWAP TO MOST FREQUENT CONCEPT")
    print("=" * 80)

    pred_tofreq = pred_baseline.copy()
    n_sw = n_cor = n_incor = 0
    for r in llm_results:
        if not r["is_swap"]:
            continue
        alts = mention_freq.get(r["mention"], {})
        if not alts:
            continue
        most_common_cid = max(alts, key=alts.get)
        if r["chosen_cid"] != most_common_cid:
            continue  # Only accept if LLM picks the most frequent
        if r["current_cid"] == most_common_cid:
            continue  # Already there
        pred_tofreq.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
        n_sw += 1
        if r["chosen_correct"]: n_cor += 1
        if r["current_correct"]: n_incor += 1
    pred_tofreq_eval = pred_tofreq[["note_id", "start", "end", "concept_id"]].copy()
    for c in ["start", "end", "concept_id"]:
        pred_tofreq_eval[c] = pred_tofreq_eval[c].astype(int)
    iou_tofreq = macro_char_iou(pred_tofreq_eval, test_gold[["note_id", "start", "end", "concept_id"]])
    print(f"  IoU={iou_tofreq:.4f} ({iou_tofreq-iou_baseline:+.4f}) "
          f"swaps={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    # Classification accuracy analysis
    # ===================================================================
    print("\n" + "=" * 80)
    print("CLASSIFICATION ACCURACY ANALYSIS")
    print("=" * 80)

    # How often does the LLM pick the correct concept regardless of current?
    n_has_gold = 0
    n_llm_correct = 0
    n_baseline_correct = 0
    for r in llm_results:
        if not r["gold_cids"]:
            continue
        n_has_gold += 1
        if r["chosen_correct"]:
            n_llm_correct += 1
        if r["current_correct"]:
            n_baseline_correct += 1

    print(f"  Predictions with gold overlap: {n_has_gold}")
    print(f"  Baseline correct concept: {n_baseline_correct} ({100*n_baseline_correct/max(n_has_gold,1):.1f}%)")
    print(f"  LLM correct concept:      {n_llm_correct} ({100*n_llm_correct/max(n_has_gold,1):.1f}%)")

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
