#!/usr/bin/env python3
"""Test gpt-oss-20b on concept disambiguation — 1000 examples, low reasoning effort."""
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
from engine import segment_sections, get_section_for_pos
from runtime_scoring import macro_char_iou

CONTEXT_WINDOW = 250
N_EXAMPLES = 1000


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


def build_prompt(mention, before, after, section_name, candidates_shuffled, concept_names):
    cand_lines = []
    for i, (cid, _count) in enumerate(candidates_shuffled):
        name = concept_names.get(cid, f"Unknown ({cid})")
        cand_lines.append(f"{i}: {name}")

    section_line = f"Section: {section_name}\n" if section_name else ""

    return f"""{section_line}Clinical note excerpt: ...{before}[{mention}]{after}...

Which SNOMED CT concept best matches "{mention}" in this context?
{chr(10).join(cand_lines)}

Reply with ONLY the number (e.g. "0" or "1"). Do not explain."""


def parse_response(text, n_candidates):
    """Parse gpt-oss output: extract answer after 'assistantfinal' marker."""
    if not text:
        return None

    # gpt-oss uses "analysisXXXassistantfinalY" format
    if "assistantfinal" in text:
        final = text.split("assistantfinal")[-1].strip()
        m = re.match(r'^(\d+)', final)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < n_candidates:
                return idx

    # Fallback: try standard parsing
    text = text.strip()
    m = re.match(r'^(\d+)\s*$', text)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n_candidates:
            return idx

    # Look for any digit
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

    # Section segmentation for test notes
    test_sections_by_id = {}
    for nid, text in test_text_by_id.items():
        test_sections_by_id[nid] = segment_sections(text)

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

        sections = test_sections_by_id.get(nid, [])
        sec = get_section_for_pos(s, sections)
        section_name = sec.header if sec else None

        ambiguous.append({
            "pred_idx": idx,
            "note_id": nid,
            "start": s,
            "end": e,
            "current_cid": cid,
            "mention": mention,
            "alternatives": alternatives,
            "top_share": top_share,
            "section": section_name,
        })

    print(f"  Total predictions: {len(pred_baseline):,}")
    print(f"  Genuinely ambiguous: {len(ambiguous)}")

    # Take first N_EXAMPLES
    subset = ambiguous[:N_EXAMPLES]
    print(f"  Using first {len(subset)} for this test")

    # Build prompts
    random.seed(42)
    queries = []
    for item in subset:
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
            mention_text, before, after, item["section"],
            candidates_shuffled, concept_names,
        )
        queries.append({
            "item": item,
            "prompt": prompt,
            "candidates_shuffled": candidates_shuffled,
            "current_shuffled_idx": current_shuffled_idx,
        })

    # Load model
    print(f"\nLoading gpt-oss-20b via vLLM...", flush=True)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="openai/gpt-oss-20b",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        max_num_seqs=16,
        seed=42,
    )
    print("  Model loaded!")

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.0,
    )

    # Format prompts with chat template + low reasoning effort
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for q in queries:
        messages = [
            {"role": "system", "content": "Reasoning effort: low"},
            {"role": "user", "content": q["prompt"]},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except (ValueError, AttributeError):
            text = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"Reasoning effort: low<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{q['prompt']}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")
        formatted_prompts.append(text)

    prompt_lengths = [len(p) for p in formatted_prompts]
    print(f"  Prompt lengths: median={np.median(prompt_lengths):.0f} "
          f"max={max(prompt_lengths)} mean={np.mean(prompt_lengths):.0f}")

    print(f"\n  Running vLLM batch inference on {len(formatted_prompts)} prompts "
          f"(max_num_seqs=16, reasoning=low)...", flush=True)
    t_llm = time.perf_counter()
    outputs = llm.generate(formatted_prompts, sampling_params)
    elapsed_llm = time.perf_counter() - t_llm
    print(f"  vLLM done in {elapsed_llm:.1f}s ({len(formatted_prompts)/elapsed_llm:.1f} queries/s)")

    # Count total output tokens
    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  Total output tokens: {total_output_tokens:,} "
          f"(avg {total_output_tokens/len(outputs):.0f}/query)")

    # Process results
    llm_results = []
    n_parse_fail = 0
    for q, output in zip(queries, outputs):
        item = q["item"]
        candidates_shuffled = q["candidates_shuffled"]
        current_shuffled_idx = q["current_shuffled_idx"]

        answer = output.outputs[0].text
        chosen_idx = parse_response(answer, len(candidates_shuffled))

        parse_fail = chosen_idx is None
        if parse_fail:
            n_parse_fail += 1
            chosen_idx = current_shuffled_idx if current_shuffled_idx is not None else 0

        chosen_cid = candidates_shuffled[chosen_idx][0]

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
            "section": item["section"],
            "current_correct": item["current_cid"] in gold_cids,
            "chosen_correct": chosen_cid in gold_cids,
            "is_swap": chosen_cid != item["current_cid"],
            "answer": answer,
            "parse_fail": parse_fail,
            "n_tokens": len(output.outputs[0].token_ids),
        })

    print(f"  Parse failures: {n_parse_fail}/{len(queries)} ({100*n_parse_fail/len(queries):.1f}%)")

    if n_parse_fail > 0:
        print("\n  Sample parse failures:")
        shown = 0
        for r in llm_results:
            if r["parse_fail"] and shown < 10:
                print(f"    '{r['answer'][:120]}'")
                shown += 1

    # ===================================================================
    # RESULTS
    # ===================================================================
    def run_experiment(name, filter_fn):
        pred = pred_baseline.copy()
        n_sw = n_cor = n_incor = 0
        for r in llm_results:
            if not r["is_swap"]:
                continue
            if not filter_fn(r):
                continue
            pred.loc[r["pred_idx"], "concept_id"] = r["chosen_cid"]
            n_sw += 1
            if r["chosen_correct"]: n_cor += 1
            if r["current_correct"]: n_incor += 1
        ev = pred[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]: ev[c] = ev[c].astype(int)
        iou = macro_char_iou(ev, test_gold[["note_id", "start", "end", "concept_id"]])
        acc = 100 * n_cor / max(n_sw, 1)
        print(f"  {name:40s} IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
              f"sw={n_sw} right={n_cor} wrong={n_incor} acc={acc:.0f}%")
        return iou

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENTS (gpt-oss-20b, low effort, {len(subset)} queries)")
    print("=" * 80)

    run_experiment("All LLM swaps", lambda r: True)

    # Frequency agreement
    run_experiment(
        "LLM + freq agree",
        lambda r: mention_freq.get(r["mention"], {}).get(r["chosen_cid"], 0) >
                  mention_freq.get(r["mention"], {}).get(r["current_cid"], 0)
    )

    # Swap to most frequent
    def swap_to_top(r):
        alts = mention_freq.get(r["mention"], {})
        if not alts:
            return False
        most_common = max(alts, key=alts.get)
        return r["chosen_cid"] == most_common and r["current_cid"] != most_common
    run_experiment("Swap to top freq", swap_to_top)

    # Pure frequency swap on same subset
    print(f"\n  --- Pure frequency swap (no LLM, same {len(subset)} queries) ---")
    pred = pred_baseline.copy()
    n_sw = n_cor = n_incor = 0
    for item in subset:
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
    print(f"  {'Pure freq swap':40s} IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
          f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    # CLASSIFICATION ACCURACY
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
    print(f"  gpt-oss-20b correct: {n_llm_correct} ({100*n_llm_correct/max(n_has_gold,1):.1f}%)")

    # Per-section accuracy
    print(f"\n  Accuracy by section:")
    sec_groups = defaultdict(list)
    for r in llm_results:
        if r["gold_cids"]:
            sec_groups[r["section"] or "unknown"].append(r)
    for sec in sorted(sec_groups, key=lambda s: -len(sec_groups[s])):
        group = sec_groups[sec]
        n_llm = sum(1 for r in group if r["chosen_correct"])
        n_base = sum(1 for r in group if r["current_correct"])
        print(f"    {sec:35s} n={len(group):4d}  "
              f"base={100*n_base/len(group):.1f}%  "
              f"gpt-oss={100*n_llm/len(group):.1f}%")

    # Token distribution
    token_counts = [r["n_tokens"] for r in llm_results]
    arr = np.array(token_counts)
    print(f"\n  Output tokens: median={np.median(arr):.0f} "
          f"mean={np.mean(arr):.0f} min={np.min(arr)} max={np.max(arr)}")

    # Show sample correct swaps
    print(f"\n  Sample CORRECT swaps (LLM right, baseline wrong):")
    shown = 0
    for r in llm_results:
        if r["is_swap"] and r["chosen_correct"] and not r["current_correct"] and shown < 15:
            name = concept_names.get(r["chosen_cid"], str(r["chosen_cid"]))
            print(f"    [{r['section']}] '{r['mention']}' -> {name}")
            shown += 1

    print(f"\n  Sample INCORRECT swaps (LLM wrong, baseline was right):")
    shown = 0
    for r in llm_results:
        if r["is_swap"] and not r["chosen_correct"] and r["current_correct"] and shown < 15:
            name_wrong = concept_names.get(r["chosen_cid"], str(r["chosen_cid"]))
            name_right = concept_names.get(r["current_cid"], str(r["current_cid"]))
            print(f"    [{r['section']}] '{r['mention']}' chose={name_wrong} was={name_right}")
            shown += 1

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
