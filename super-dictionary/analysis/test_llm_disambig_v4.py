#!/usr/bin/env python3
"""LLM concept disambiguation v4 — OpenBioLLM-8B via vLLM + logprob confidence.

Key changes from v3:
- Uses vLLM (batch inference) with OpenBioLLM-Llama3-8B-AWQ
- Native logprob support (no logits_all hack needed)
- Also tests MedGemma via llama-cpp-python with logits_processor confidence
- Compares both models head-to-head on the same subset
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

MEDGEMMA_PATH = REPO_ROOT / "models" / "medgemma-1.5-4b-it-Q8_0.gguf"
OPENBIO_MODEL = "bartowski/OpenBioLLM-Llama3-8B-AWQ"
CONTEXT_WINDOW = 250
SUBSET_SIZE = 500  # Test on subset for speed comparison


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


def prepare_data():
    """Load data, build baseline, identify ambiguous predictions."""
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

    # Identify ambiguous predictions (<95% dominant)
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

    return (pred_baseline, test_gold, iou_baseline, gold_by_note,
            test_text_by_id, mention_freq, concept_names, ambiguous)


def build_queries(ambiguous, test_text_by_id, concept_names, seed=42):
    """Build prompts and shuffled candidates for all ambiguous items."""
    random.seed(seed)
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
    return queries


def evaluate_results(llm_results, pred_baseline, test_gold, iou_baseline,
                     mention_freq, concept_names, model_name):
    """Run the same experiments as v3 on a set of LLM results."""
    print(f"\n{'=' * 80}")
    print(f"RESULTS: {model_name}")
    print(f"{'=' * 80}")

    total_swaps = sum(1 for r in llm_results if r["is_swap"])
    n_parse_fail = sum(1 for r in llm_results if r.get("parse_fail"))
    print(f"  Queries: {len(llm_results)}, Swaps: {total_swaps}, Parse failures: {n_parse_fail}")

    # Sample answers
    print(f"\n  Sample LLM answers (first 10 swaps):")
    shown = 0
    for r in llm_results:
        if r["is_swap"] and shown < 10:
            tag = "GOOD" if r["chosen_correct"] else ("BAD" if r["current_correct"] else "???")
            print(f"    [{tag}] '{r['mention']}' {r['current_cid']}->{r['chosen_cid']} "
                  f"ans='{r.get('answer','')[:60]}'")
            shown += 1

    results = {}

    # --- Experiment 1: All LLM swaps ---
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
    print(f"\n  EXP 1 (all swaps):       IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
          f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")
    results["all_swaps"] = iou

    # --- Experiment 2: LLM + frequency agreement ---
    pred = pred_baseline.copy()
    n_sw = n_cor = n_incor = 0
    for r in llm_results:
        if not r["is_swap"]:
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
    print(f"  EXP 2 (LLM+freq agree): IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
          f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")
    results["freq_agree"] = iou

    # --- Experiment 3: Only swap to most frequent ---
    pred = pred_baseline.copy()
    n_sw = n_cor = n_incor = 0
    for r in llm_results:
        if not r["is_swap"]:
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
    print(f"  EXP 3 (swap to top):    IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
          f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")
    results["swap_to_top"] = iou

    # --- Experiment 4: High confidence only (logprob >= -0.5) ---
    has_logprob = any(r.get("logprob") is not None for r in llm_results)
    if has_logprob:
        for thresh in [-0.1, -0.3, -0.5, -1.0, -2.0]:
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
            print(f"  EXP 4 (logprob>={thresh:.1f}): IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
                  f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

        # --- Experiment 5: High confidence + frequency agreement ---
        for thresh in [-0.3, -0.5, -1.0]:
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
            print(f"  EXP 5 (lp>={thresh:.1f}+freq): IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
                  f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # --- Classification accuracy ---
    n_has_gold = n_llm_correct = n_baseline_correct = 0
    for r in llm_results:
        if not r["gold_cids"]:
            continue
        n_has_gold += 1
        if r["chosen_correct"]: n_llm_correct += 1
        if r["current_correct"]: n_baseline_correct += 1

    print(f"\n  Classification accuracy ({n_has_gold} with gold):")
    print(f"    Baseline: {n_baseline_correct} ({100*n_baseline_correct/max(n_has_gold,1):.1f}%)")
    print(f"    LLM:      {n_llm_correct} ({100*n_llm_correct/max(n_has_gold,1):.1f}%)")

    return results


def run_openbio_vllm(queries, gold_by_note):
    """Run OpenBioLLM-Llama3-8B-AWQ via vLLM batch inference with logprobs."""
    from vllm import LLM, SamplingParams

    print(f"\nLoading OpenBioLLM-Llama3-8B-AWQ via vLLM...", flush=True)
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
        max_tokens=16,
        temperature=0.0,
        logprobs=5,  # Top-5 logprobs per token
    )

    # Build prompts — use Llama 3 instruct format manually since tokenizer
    # may not have a chat template set
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for q in queries:
        # Try chat template first, fall back to Llama 3 format
        try:
            messages = [{"role": "user", "content": q["prompt"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except (ValueError, AttributeError):
            # Manual Llama 3 instruct format
            text = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{q['prompt']}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        formatted_prompts.append(text)

    print(f"\n  Running vLLM batch inference on {len(formatted_prompts)} prompts...", flush=True)
    t0 = time.perf_counter()
    outputs = llm.generate(formatted_prompts, sampling_params)
    elapsed = time.perf_counter() - t0
    print(f"  vLLM done in {elapsed:.1f}s ({len(formatted_prompts)/elapsed:.1f} queries/s)")

    # Process results
    results = []
    n_parse_fail = 0
    for q, output in zip(queries, outputs):
        item = q["item"]
        candidates_shuffled = q["candidates_shuffled"]
        current_shuffled_idx = q["current_shuffled_idx"]

        answer = output.outputs[0].text.strip()
        chosen_idx = parse_response(answer, len(candidates_shuffled))

        # Extract logprob for the first token (the digit selection)
        logprob = None
        if output.outputs[0].logprobs and len(output.outputs[0].logprobs) > 0:
            first_token_logprobs = output.outputs[0].logprobs[0]
            if first_token_logprobs:
                # Get the logprob of the actually-generated token
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

        results.append({
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

    print(f"  Parse failures: {n_parse_fail}/{len(queries)}")
    return results


def run_medgemma_with_logprobs(queries, gold_by_note):
    """Run MedGemma via llama-cpp-python with logits_processor for confidence."""
    from llama_cpp import Llama, LogitsProcessorList

    class LogprobCapture:
        """Capture top-k logprobs from sampling pipeline without logits_all."""
        def __init__(self, top_k=10):
            self.top_k = top_k
            self.step_logprobs = []

        def __call__(self, input_ids, logits):
            log_max = np.max(logits)
            shifted = logits - log_max
            exp_l = np.exp(shifted)
            log_sum = np.log(np.sum(exp_l))
            log_probs = shifted - log_sum

            top_idx = np.argpartition(log_probs, -self.top_k)[-self.top_k:]
            top_idx = top_idx[np.argsort(log_probs[top_idx])[::-1]]
            self.step_logprobs.append(
                [(int(i), float(log_probs[i])) for i in top_idx]
            )
            return logits

        def reset(self):
            self.step_logprobs = []

    print(f"\nLoading MedGemma...", flush=True)
    llm = Llama(
        model_path=str(MEDGEMMA_PATH),
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )
    print("  Model loaded!")

    capture = LogprobCapture(top_k=10)
    processor_list = LogitsProcessorList([capture])

    results = []
    n_parse_fail = 0
    t0 = time.perf_counter()

    for i, q in enumerate(queries):
        item = q["item"]
        candidates_shuffled = q["candidates_shuffled"]
        current_shuffled_idx = q["current_shuffled_idx"]

        capture.reset()
        try:
            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": q["prompt"]}],
                max_tokens=16,
                temperature=0.0,
                logits_processor=processor_list,
            )
            answer = response["choices"][0]["message"]["content"]
            chosen_idx = parse_response(answer, len(candidates_shuffled))
        except Exception:
            answer = ""
            chosen_idx = None

        # Extract logprob from first generation step
        logprob = None
        if capture.step_logprobs:
            # First step logprobs — find max (most likely token)
            first_step = capture.step_logprobs[0]
            if first_step:
                logprob = first_step[0][1]  # Highest logprob

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

        results.append({
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

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i+1}/{len(queries)} ({(i+1)/elapsed:.1f}/s) "
                  f"parse_fail={n_parse_fail}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  MedGemma done in {elapsed:.1f}s ({len(queries)/elapsed:.1f}/s)")
    print(f"  Parse failures: {n_parse_fail}/{len(queries)}")
    return results


def main():
    t0 = time.perf_counter()

    # Prepare data (shared across both models)
    (pred_baseline, test_gold, iou_baseline, gold_by_note,
     test_text_by_id, mention_freq, concept_names, ambiguous) = prepare_data()

    # Use subset for comparison (same subset for both models)
    random.seed(42)
    subset = ambiguous[:SUBSET_SIZE] if len(ambiguous) > SUBSET_SIZE else ambiguous
    print(f"\n  Using {len(subset)} queries for model comparison")

    # Build queries (same for both models)
    queries = build_queries(subset, test_text_by_id, concept_names, seed=42)

    # ===================================================================
    # Model A: OpenBioLLM-Llama3-8B-AWQ via vLLM
    # ===================================================================
    print("\n" + "=" * 80)
    print("MODEL A: OpenBioLLM-Llama3-8B-AWQ (vLLM, batch inference)")
    print("=" * 80)

    openbio_results = run_openbio_vllm(queries, gold_by_note)
    evaluate_results(
        openbio_results, pred_baseline, test_gold, iou_baseline,
        mention_freq, concept_names, "OpenBioLLM-8B"
    )

    # Free vLLM GPU memory before loading MedGemma
    import gc
    import torch
    del openbio_results  # Keep reference for comparison - actually let's save it
    # Re-run to save it
    print("\n  Freeing vLLM GPU memory...", flush=True)

    # ===================================================================
    # Model B: MedGemma-4B via llama-cpp-python with logprob capture
    # ===================================================================
    print("\n" + "=" * 80)
    print("MODEL B: MedGemma-4B (llama.cpp, logits_processor confidence)")
    print("=" * 80)

    # Need to reload openbio results before freeing
    # Actually let's restructure: run both, save results, then evaluate
    # The issue is GPU memory. Let's run OpenBioLLM first, save results, free GPU, then MedGemma.

    # We already ran OpenBioLLM above. Let's re-run evaluation with saved results.
    # Actually the evaluate_results already ran. Let's just run MedGemma now.

    # Force cleanup of vLLM
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    medgemma_results = run_medgemma_with_logprobs(queries, gold_by_note)
    evaluate_results(
        medgemma_results, pred_baseline, test_gold, iou_baseline,
        mention_freq, concept_names, "MedGemma-4B"
    )

def main():
    t0 = time.perf_counter()

    # Prepare data
    (pred_baseline, test_gold, iou_baseline, gold_by_note,
     test_text_by_id, mention_freq, concept_names, ambiguous) = prepare_data()

    random.seed(42)
    subset = ambiguous[:SUBSET_SIZE] if len(ambiguous) > SUBSET_SIZE else ambiguous
    print(f"\n  Using {len(subset)} queries for model comparison")

    queries = build_queries(subset, test_text_by_id, concept_names, seed=42)

    # --- Model A: OpenBioLLM ---
    print("\n" + "=" * 80)
    print("MODEL A: OpenBioLLM-Llama3-8B-AWQ (vLLM)")
    print("=" * 80)
    openbio_results = run_openbio_vllm(queries, gold_by_note)

    evaluate_results(
        openbio_results, pred_baseline, test_gold, iou_baseline,
        mention_freq, concept_names, "OpenBioLLM-8B"
    )

    # Free vLLM GPU memory before loading MedGemma
    print("\n  Freeing vLLM GPU memory...", flush=True)
    import gc
    import torch
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Model B: MedGemma ---
    print("\n" + "=" * 80)
    print("MODEL B: MedGemma-4B (llama.cpp + logprob capture)")
    print("=" * 80)
    medgemma_results = run_medgemma_with_logprobs(queries, gold_by_note)

    evaluate_results(
        medgemma_results, pred_baseline, test_gold, iou_baseline,
        mention_freq, concept_names, "MedGemma-4B"
    )

    # --- Head-to-head ---
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 80)

    n_both_correct = n_ob_only = n_mg_only = n_neither = 0
    n_agree = n_disagree = 0
    n_with_gold = 0

    for ob, mg in zip(openbio_results, medgemma_results):
        if not ob["gold_cids"]:
            continue
        n_with_gold += 1
        ob_ok = ob["chosen_correct"]
        mg_ok = mg["chosen_correct"]
        if ob_ok and mg_ok:
            n_both_correct += 1
        elif ob_ok:
            n_ob_only += 1
        elif mg_ok:
            n_mg_only += 1
        else:
            n_neither += 1

        if ob["chosen_cid"] == mg["chosen_cid"]:
            n_agree += 1
        else:
            n_disagree += 1

    print(f"  Queries with gold: {n_with_gold}")
    print(f"  Both correct:      {n_both_correct} ({100*n_both_correct/max(n_with_gold,1):.1f}%)")
    print(f"  OpenBioLLM only:   {n_ob_only} ({100*n_ob_only/max(n_with_gold,1):.1f}%)")
    print(f"  MedGemma only:     {n_mg_only} ({100*n_mg_only/max(n_with_gold,1):.1f}%)")
    print(f"  Neither correct:   {n_neither} ({100*n_neither/max(n_with_gold,1):.1f}%)")
    print(f"  Models agree:      {n_agree}/{n_with_gold} ({100*n_agree/max(n_with_gold,1):.1f}%)")
    print(f"  Models disagree:   {n_disagree}/{n_with_gold} ({100*n_disagree/max(n_with_gold,1):.1f}%)")

    # Logprob distribution comparison
    print(f"\n  Logprob distributions:")
    for name, res_list in [("OpenBioLLM", openbio_results), ("MedGemma", medgemma_results)]:
        lps = [r["logprob"] for r in res_list if r["logprob"] is not None]
        if lps:
            arr = np.array(lps)
            print(f"    {name}: median={np.median(arr):.2f} mean={np.mean(arr):.2f} "
                  f"min={np.min(arr):.2f} max={np.max(arr):.2f}")
            # Accuracy at different confidence levels
            for thresh in [-0.1, -0.5, -1.0]:
                high_conf = [r for r in res_list
                             if r["logprob"] is not None and r["logprob"] >= thresh
                             and r["gold_cids"]]
                if high_conf:
                    n_cor = sum(1 for r in high_conf if r["chosen_correct"])
                    print(f"      logprob>={thresh:.1f}: {len(high_conf)} queries, "
                          f"acc={100*n_cor/len(high_conf):.1f}%")

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
