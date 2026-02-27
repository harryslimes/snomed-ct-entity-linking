#!/usr/bin/env python3
"""LLM concept disambiguation v5 — OpenBioLLM-8B full run with better parsing.

Findings from v4:
- OpenBioLLM-8B logprob confidence works: 90% acc at logprob>=-0.1, 73% at >=-0.5
- But 87% parse failures due to verbose output
- MedGemma logprobs don't discriminate at all

This version:
- Forces shorter output via stronger prompt instruction
- Better regex-based parsing for verbose responses
- Runs on ALL 3924 genuinely ambiguous predictions
- Uses logprob confidence for selective swaps
"""
from __future__ import annotations

import re
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

OPENBIO_MODEL = "bartowski/OpenBioLLM-Llama3-8B-AWQ"
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
    """Concise prompt that strongly constrains output format."""
    cand_lines = []
    for i, (cid, _count) in enumerate(candidates_shuffled):
        name = concept_names.get(cid, f"Unknown ({cid})")
        cand_lines.append(f"{i}: {name}")

    return f"""Clinical note excerpt: ...{before}[{mention}]{after}...

Which SNOMED CT concept best matches "{mention}" in this context?
{chr(10).join(cand_lines)}

Reply with ONLY the number (e.g. "0" or "1"). Do not explain."""


def parse_response(text, n_candidates):
    """Robust parsing that handles verbose model outputs."""
    if not text:
        return None
    text = text.strip()

    # Try exact single digit/number first
    m = re.match(r'^(\d+)\s*$', text)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n_candidates:
            return idx

    # "The answer is X" pattern
    m = re.search(r'(?:answer|best|match)\s+(?:is\s+)?(?:\*\*)?(\d+)', text, re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n_candidates:
            return idx

    # Look for "X:" pattern (like "0: Surgical procedure")
    m = re.search(r'\b(\d+)\s*:', text)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n_candidates:
            return idx

    # Look for any standalone digit
    for token in text.split():
        token = re.sub(r'[^0-9]', '', token)
        if token:
            try:
                idx = int(token)
                if 0 <= idx < n_candidates:
                    return idx
            except ValueError:
                continue

    # Last resort: first digit in text
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

    # Identify ALL ambiguous predictions (<95% dominant)
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

    # Build prompts
    random.seed(42)
    queries = []
    for item in ambiguous:
        nid = item["note_id"]
        s, e = item["start"], item["end"]
        note_text = test_text_by_id.get(nid, "")
        before, mention_text, after = get_context(note_text, s, e)

        sorted_alts = sorted(item["alternatives"].items(), key=lambda x: -x[1])
        candidates = sorted_alts[:6]
        candidates_shuffled = list(candidates)
        random.shuffle(candidates_shuffled)

        current_shuffled_idx = None
        for ci, (cid, _) in enumerate(candidates_shuffled):
            if cid == item["current_cid"]:
                current_shuffled_idx = ci
                break

        prompt = build_prompt(
            mention_text, before, after, candidates_shuffled, concept_names
        )
        queries.append({
            "item": item,
            "prompt": prompt,
            "candidates_shuffled": candidates_shuffled,
            "current_shuffled_idx": current_shuffled_idx,
        })

    # ===================================================================
    # Run OpenBioLLM via vLLM
    # ===================================================================
    print(f"\nLoading OpenBioLLM-Llama3-8B-AWQ via vLLM...", flush=True)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=OPENBIO_MODEL,
        quantization="awq",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        max_num_seqs=256,
        seed=42,
    )
    print("  Model loaded!")

    sampling_params = SamplingParams(
        max_tokens=8,  # Shorter — we only need a single digit
        temperature=0.0,
        logprobs=5,
    )

    # Format prompts with Llama 3 template
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for q in queries:
        try:
            messages = [{"role": "user", "content": q["prompt"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except (ValueError, AttributeError):
            text = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{q['prompt']}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        formatted_prompts.append(text)

    print(f"\n  Running vLLM batch inference on {len(formatted_prompts)} prompts...", flush=True)
    t_llm = time.perf_counter()
    outputs = llm.generate(formatted_prompts, sampling_params)
    elapsed_llm = time.perf_counter() - t_llm
    print(f"  vLLM done in {elapsed_llm:.1f}s ({len(formatted_prompts)/elapsed_llm:.1f} queries/s)")

    # Process results
    llm_results = []
    n_parse_fail = 0
    for q, output in zip(queries, outputs):
        item = q["item"]
        candidates_shuffled = q["candidates_shuffled"]
        current_shuffled_idx = q["current_shuffled_idx"]

        answer = output.outputs[0].text.strip()
        chosen_idx = parse_response(answer, len(candidates_shuffled))

        # Extract logprob for the first generated token
        logprob = None
        if output.outputs[0].logprobs and len(output.outputs[0].logprobs) > 0:
            first_token_logprobs = output.outputs[0].logprobs[0]
            if first_token_logprobs:
                top_lp = max(first_token_logprobs.values(), key=lambda x: x.logprob)
                logprob = top_lp.logprob

        parse_fail = chosen_idx is None
        if parse_fail:
            n_parse_fail += 1
            chosen_idx = current_shuffled_idx if current_shuffled_idx is not None else 0

        chosen_cid = candidates_shuffled[chosen_idx][0]

        # Check against gold
        gold = gold_by_note.get(item["note_id"])
        gold_cids = set()
        if gold is not None:
            overlaps = gold[(gold["start"] < item["end"]) & (gold["end"] > item["start"])]
            gold_cids = set(overlaps["concept_id"].astype(int))

        llm_results.append({
            "pred_idx": item["pred_idx"],
            "current_cid": item["current_cid"],
            "chosen_cid": chosen_cid,
            "gold_cids": gold_cids,
            "mention": item["mention"],
            "current_correct": item["current_cid"] in gold_cids,
            "chosen_correct": chosen_cid in gold_cids,
            "is_swap": chosen_cid != item["current_cid"],
            "answer": answer,
            "logprob": logprob,
            "parse_fail": parse_fail,
        })

    print(f"  Parse failures: {n_parse_fail}/{len(queries)} ({100*n_parse_fail/len(queries):.1f}%)")

    # Show parse failure samples
    print("\n  Sample parse failures:")
    shown = 0
    for r in llm_results:
        if r["parse_fail"] and shown < 15:
            print(f"    '{r['answer'][:80]}'")
            shown += 1

    # ===================================================================
    # RESULTS
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("EXPERIMENT 1: ALL LLM SWAPS (UNFILTERED)")
    print("=" * 80)
    pred = pred_baseline.copy()
    n_sw = n_cor = n_incor = 0
    for r in llm_results:
        if r["is_swap"]:
            pred.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
            n_sw += 1
            if r["chosen_correct"]: n_cor += 1
            if r["current_correct"]: n_incor += 1
    ev = pred[["note_id", "start", "end", "concept_id"]].copy()
    for c in ["start", "end", "concept_id"]: ev[c] = ev[c].astype(int)
    iou = macro_char_iou(ev, test_gold[["note_id", "start", "end", "concept_id"]])
    print(f"  IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
          f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    print(f"\n{'=' * 80}")
    print("EXPERIMENT 2: LOGPROB CONFIDENCE THRESHOLDS")
    print("=" * 80)
    for thresh in [-0.05, -0.1, -0.2, -0.3, -0.5, -0.7, -1.0, -1.5, -2.0]:
        pred = pred_baseline.copy()
        n_sw = n_cor = n_incor = 0
        for r in llm_results:
            if not r["is_swap"]:
                continue
            lp = r.get("logprob")
            if lp is None or lp < thresh:
                continue
            pred.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
            n_sw += 1
            if r["chosen_correct"]: n_cor += 1
            if r["current_correct"]: n_incor += 1
        ev = pred[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]: ev[c] = ev[c].astype(int)
        iou = macro_char_iou(ev, test_gold[["note_id", "start", "end", "concept_id"]])
        print(f"  logprob>={thresh:+.2f}: IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
              f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    print(f"\n{'=' * 80}")
    print("EXPERIMENT 3: LOGPROB + FREQUENCY AGREEMENT")
    print("=" * 80)
    for thresh in [-0.1, -0.3, -0.5, -0.7, -1.0]:
        pred = pred_baseline.copy()
        n_sw = n_cor = n_incor = 0
        for r in llm_results:
            if not r["is_swap"]:
                continue
            lp = r.get("logprob")
            if lp is None or lp < thresh:
                continue
            alts = mention_freq.get(r["mention"], {})
            if alts.get(r["chosen_cid"], 0) <= alts.get(r["current_cid"], 0):
                continue
            pred.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
            n_sw += 1
            if r["chosen_correct"]: n_cor += 1
            if r["current_correct"]: n_incor += 1
        ev = pred[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]: ev[c] = ev[c].astype(int)
        iou = macro_char_iou(ev, test_gold[["note_id", "start", "end", "concept_id"]])
        print(f"  lp>={thresh:+.1f}+freq: IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
              f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    print(f"\n{'=' * 80}")
    print("EXPERIMENT 4: LOGPROB + SWAP TO MOST FREQUENT")
    print("=" * 80)
    for thresh in [-0.1, -0.3, -0.5, -0.7, -1.0]:
        pred = pred_baseline.copy()
        n_sw = n_cor = n_incor = 0
        for r in llm_results:
            if not r["is_swap"]:
                continue
            lp = r.get("logprob")
            if lp is None or lp < thresh:
                continue
            alts = mention_freq.get(r["mention"], {})
            if not alts:
                continue
            most_common_cid = max(alts, key=alts.get)
            if r["chosen_cid"] != most_common_cid:
                continue
            if r["current_cid"] == most_common_cid:
                continue
            pred.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
            n_sw += 1
            if r["chosen_correct"]: n_cor += 1
            if r["current_correct"]: n_incor += 1
        ev = pred[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]: ev[c] = ev[c].astype(int)
        iou = macro_char_iou(ev, test_gold[["note_id", "start", "end", "concept_id"]])
        print(f"  lp>={thresh:+.1f}+top: IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
              f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    print(f"\n{'=' * 80}")
    print("EXPERIMENT 5: PURE FREQUENCY SWAP (no LLM, for comparison)")
    print("=" * 80)
    pred = pred_baseline.copy()
    n_sw = n_cor = n_incor = 0
    for item in ambiguous:
        alts = mention_freq.get(item["mention"], {})
        if not alts:
            continue
        most_common_cid = max(alts, key=alts.get)
        if most_common_cid == item["current_cid"]:
            continue
        pred.loc[item["pred_idx"], "concept_id"] = most_common_cid

        gold = gold_by_note.get(item["note_id"])
        gold_cids = set()
        if gold is not None:
            overlaps = gold[(gold["start"] < item["end"]) & (gold["end"] > item["start"])]
            gold_cids = set(overlaps["concept_id"].astype(int))
        n_sw += 1
        if most_common_cid in gold_cids: n_cor += 1
        if item["current_cid"] in gold_cids: n_incor += 1
    ev = pred[["note_id", "start", "end", "concept_id"]].copy()
    for c in ["start", "end", "concept_id"]: ev[c] = ev[c].astype(int)
    iou = macro_char_iou(ev, test_gold[["note_id", "start", "end", "concept_id"]])
    print(f"  IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
          f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    print(f"\n{'=' * 80}")
    print("CLASSIFICATION ACCURACY")
    print("=" * 80)
    n_has_gold = n_llm_correct = n_baseline_correct = 0
    for r in llm_results:
        if not r["gold_cids"]:
            continue
        n_has_gold += 1
        if r["chosen_correct"]: n_llm_correct += 1
        if r["current_correct"]: n_baseline_correct += 1
    print(f"  Predictions with gold: {n_has_gold}")
    print(f"  Baseline correct: {n_baseline_correct} ({100*n_baseline_correct/max(n_has_gold,1):.1f}%)")
    print(f"  OpenBioLLM correct: {n_llm_correct} ({100*n_llm_correct/max(n_has_gold,1):.1f}%)")

    # Accuracy by confidence bucket
    print(f"\n  Accuracy by logprob bucket:")
    buckets = [(-0.1, 0.0), (-0.3, -0.1), (-0.5, -0.3), (-0.7, -0.5), (-1.0, -0.7), (-2.0, -1.0)]
    for lo, hi in buckets:
        bucket = [r for r in llm_results
                  if r["logprob"] is not None and lo <= r["logprob"] < hi and r["gold_cids"]]
        if bucket:
            n_ok = sum(1 for r in bucket if r["chosen_correct"])
            n_base = sum(1 for r in bucket if r["current_correct"])
            print(f"    [{lo:.1f}, {hi:.1f}): n={len(bucket)} "
                  f"LLM_acc={100*n_ok/len(bucket):.1f}% "
                  f"base_acc={100*n_base/len(bucket):.1f}%")

    # Logprob distribution
    lps = [r["logprob"] for r in llm_results if r["logprob"] is not None]
    if lps:
        arr = np.array(lps)
        print(f"\n  Logprob distribution: median={np.median(arr):.2f} "
              f"mean={np.mean(arr):.2f} min={np.min(arr):.2f} max={np.max(arr):.2f}")
        for pct in [10, 25, 50, 75, 90]:
            print(f"    p{pct}: {np.percentile(arr, pct):.3f}")

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
