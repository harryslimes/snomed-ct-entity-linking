#!/usr/bin/env python3
"""gpt-oss-20b confidence-gated disambiguation.

Instead of asking the LLM to pick from scratch, we tell it:
  "Frequency analysis suggests concept X. Here are the alternatives.
   Do you agree, or override? Rate confidence 1-10."

Then sweep confidence thresholds to find where LLM overrides beat frequency.
"""
from __future__ import annotations

import json
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
N_EXAMPLES = None  # None = all ambiguous predictions


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


def build_confidence_prompt(
    mention, before, after, section_name,
    freq_choice_idx, candidates, concept_names,
):
    """Prompt that presents the frequency suggestion and asks for agreement or override."""
    cand_lines = []
    for i, (cid, count) in enumerate(candidates):
        name = concept_names.get(cid, f"Unknown ({cid})")
        marker = " <-- frequency suggestion" if i == freq_choice_idx else ""
        cand_lines.append(f"{i}: {name} (seen {count}x in training){marker}")

    freq_name = concept_names.get(candidates[freq_choice_idx][0], "Unknown")

    section_line = f"Section: {section_name}\n" if section_name else ""

    return f"""{section_line}Clinical note excerpt: ...{before}[{mention}]{after}...

The mention "{mention}" is ambiguous. Based on training frequency, the best match is:
  {freq_choice_idx}: {freq_name}

All candidates (with training counts):
{chr(10).join(cand_lines)}

IMPORTANT: These annotations follow SNOMED CT conventions where annotators strongly prefer GENERIC concepts over body-site-specific ones. For example:
- "pain" is usually Pain (finding), NOT Abdominal pain, even when the abdomen is mentioned
- "cyst" is usually Cyst (morphologic abnormality), NOT Cyst of kidney
- "ct" is usually Computed tomography (procedure), NOT CT of chest/abdomen
- "consolidation" is usually Consolidation (morphologic abnormality), NOT Lung consolidation
- "intubation" vs "insertion of endotracheal tube" — prefer whichever the frequency suggests
The frequency suggestion already captures these annotation conventions. Only override it when the context makes the frequency choice clearly WRONG (e.g. wrong organ system, wrong clinical meaning entirely).

Do you agree with the frequency suggestion, or should it be a different concept?

Reply in this exact format:
ANSWER: <number>
CONFIDENCE: <1-10>

Where 1 = very uncertain, 10 = absolutely certain. If you agree with the frequency suggestion, repeat its number."""


def parse_confidence_response(text, n_candidates):
    """Parse gpt-oss response to extract answer number and confidence."""
    if not text:
        return None, None

    # Handle gpt-oss reasoning format: look after "assistantfinal"
    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1].strip()

    answer = None
    confidence = None

    # Look for ANSWER: N
    m = re.search(r'ANSWER\s*:\s*(\d+)', text, re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n_candidates:
            answer = idx

    # Look for CONFIDENCE: N
    m = re.search(r'CONFIDENCE\s*:\s*(\d+)', text, re.IGNORECASE)
    if m:
        conf = int(m.group(1))
        if 1 <= conf <= 10:
            confidence = conf

    # Fallback: if no structured match, try to find any digit for answer
    if answer is None:
        for ch in text:
            if ch.isdigit():
                idx = int(ch)
                if 0 <= idx < n_candidates:
                    answer = idx
                    break

    return answer, confidence


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

    print(f"  Genuinely ambiguous: {len(ambiguous)}")

    if N_EXAMPLES is not None:
        subset = ambiguous[:N_EXAMPLES]
        print(f"  Using first {len(subset)} of {len(ambiguous)} for this test")
    else:
        subset = ambiguous
        print(f"  Using ALL {len(subset)} ambiguous predictions")

    # Build prompts — frequency suggestion is the most-common concept
    random.seed(42)
    queries = []
    for item in subset:
        nid = item["note_id"]
        s, e = item["start"], item["end"]
        note_text = test_text_by_id.get(nid, "")
        before, mention_text, after = get_context(note_text, s, e)

        sorted_alts = sorted(item["alternatives"].items(), key=lambda x: -x[1])
        candidates = sorted_alts[:6]

        # The frequency suggestion is candidate 0 (most frequent)
        freq_cid = candidates[0][0]

        # Shuffle for position bias, but track where freq ended up
        candidates_shuffled = list(candidates)
        random.shuffle(candidates_shuffled)

        freq_shuffled_idx = None
        current_shuffled_idx = None
        for ci, (cid, _) in enumerate(candidates_shuffled):
            if cid == freq_cid:
                freq_shuffled_idx = ci
            if cid == item["current_cid"]:
                current_shuffled_idx = ci

        prompt = build_confidence_prompt(
            mention_text, before, after, item["section"],
            freq_shuffled_idx, candidates_shuffled, concept_names,
        )
        queries.append({
            "item": item,
            "prompt": prompt,
            "candidates_shuffled": candidates_shuffled,
            "freq_shuffled_idx": freq_shuffled_idx,
            "freq_cid": freq_cid,
            "current_shuffled_idx": current_shuffled_idx,
        })

    # Show sample prompt
    print("\n  === SAMPLE PROMPT ===")
    print(queries[0]["prompt"][:1500])
    print("  === END SAMPLE ===\n")

    # Load model
    print(f"Loading gpt-oss-20b via vLLM...", flush=True)
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

    print(f"\n  Running vLLM batch inference on {len(formatted_prompts)} prompts...", flush=True)
    t_llm = time.perf_counter()
    outputs = llm.generate(formatted_prompts, sampling_params)
    elapsed_llm = time.perf_counter() - t_llm
    print(f"  vLLM done in {elapsed_llm:.1f}s ({len(formatted_prompts)/elapsed_llm:.1f} queries/s)")

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  Total output tokens: {total_output_tokens:,} "
          f"(avg {total_output_tokens/len(outputs):.0f}/query)")

    # Process results
    llm_results = []
    n_parse_fail = 0
    n_no_confidence = 0
    for q, output in zip(queries, outputs):
        item = q["item"]
        candidates_shuffled = q["candidates_shuffled"]
        freq_shuffled_idx = q["freq_shuffled_idx"]

        raw_answer = output.outputs[0].text
        chosen_idx, confidence = parse_confidence_response(raw_answer, len(candidates_shuffled))

        parse_fail = chosen_idx is None
        if parse_fail:
            n_parse_fail += 1
            chosen_idx = freq_shuffled_idx  # fall back to freq
        if confidence is None:
            n_no_confidence += 1
            confidence = 5  # neutral default

        chosen_cid = candidates_shuffled[chosen_idx][0]
        freq_cid = q["freq_cid"]
        agrees_with_freq = (chosen_cid == freq_cid)

        gold = gold_by_note.get(item["note_id"])
        gold_cids = set()
        if gold is not None:
            overlaps = gold[(gold["start"] < item["end"]) & (gold["end"] > item["start"])]
            gold_cids = set(overlaps["concept_id"].astype(int))

        llm_results.append({
            "pred_idx": item["pred_idx"],
            "current_cid": item["current_cid"],
            "freq_cid": freq_cid,
            "chosen_cid": chosen_cid,
            "gold_cids": gold_cids,
            "mention": item["mention"],
            "section": item["section"],
            "freq_correct": freq_cid in gold_cids,
            "chosen_correct": chosen_cid in gold_cids,
            "current_correct": item["current_cid"] in gold_cids,
            "agrees_with_freq": agrees_with_freq,
            "confidence": confidence,
            "raw_answer": raw_answer,
            "parse_fail": parse_fail,
            "n_tokens": len(output.outputs[0].token_ids),
        })

    print(f"  Parse failures: {n_parse_fail}/{len(queries)} ({100*n_parse_fail/len(queries):.1f}%)")
    print(f"  Missing confidence: {n_no_confidence}/{len(queries)} ({100*n_no_confidence/len(queries):.1f}%)")

    # Show sample outputs
    print("\n  Sample raw outputs:")
    for r in llm_results[:5]:
        final = r["raw_answer"].split("assistantfinal")[-1].strip() if "assistantfinal" in r["raw_answer"] else r["raw_answer"][:100]
        print(f"    conf={r['confidence']} agrees={r['agrees_with_freq']} | {final[:80]}")

    # ===================================================================
    # AGREEMENT ANALYSIS
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("AGREEMENT ANALYSIS")
    print("=" * 80)

    n_agree = sum(1 for r in llm_results if r["agrees_with_freq"])
    n_disagree = sum(1 for r in llm_results if not r["agrees_with_freq"])
    print(f"  Agrees with frequency: {n_agree} ({100*n_agree/len(llm_results):.1f}%)")
    print(f"  Disagrees (overrides): {n_disagree} ({100*n_disagree/len(llm_results):.1f}%)")

    # Among overrides, how many are correct?
    overrides = [r for r in llm_results if not r["agrees_with_freq"] and r["gold_cids"]]
    if overrides:
        n_override_correct = sum(1 for r in overrides if r["chosen_correct"])
        n_freq_was_correct = sum(1 for r in overrides if r["freq_correct"])
        print(f"\n  Among overrides with gold (n={len(overrides)}):")
        print(f"    LLM override correct: {n_override_correct} ({100*n_override_correct/len(overrides):.1f}%)")
        print(f"    Freq was correct:     {n_freq_was_correct} ({100*n_freq_was_correct/len(overrides):.1f}%)")

    # ===================================================================
    # CONFIDENCE DISTRIBUTION
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 80)

    for conf in range(1, 11):
        at_conf = [r for r in llm_results if r["confidence"] == conf]
        if not at_conf:
            continue
        n_agree_c = sum(1 for r in at_conf if r["agrees_with_freq"])
        n_disagree_c = len(at_conf) - n_agree_c
        with_gold = [r for r in at_conf if r["gold_cids"]]
        n_correct = sum(1 for r in with_gold if r["chosen_correct"])
        n_freq_cor = sum(1 for r in with_gold if r["freq_correct"])
        print(f"  Conf={conf:2d}: n={len(at_conf):4d}  agree={n_agree_c:4d}  "
              f"disagree={n_disagree_c:3d}  "
              f"LLM_acc={100*n_correct/max(len(with_gold),1):.1f}%  "
              f"freq_acc={100*n_freq_cor/max(len(with_gold),1):.1f}%")

    # ===================================================================
    # CONFIDENCE-GATED OVERRIDE: sweep thresholds
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("CONFIDENCE-GATED OVERRIDE: IoU by threshold")
    print("=" * 80)
    print("  (Take LLM answer when it disagrees with freq AND confidence >= threshold)")

    for min_conf in range(1, 11):
        pred = pred_baseline.copy()
        n_sw = n_cor = n_incor = 0
        for r in llm_results:
            # Only apply LLM override when it disagrees with freq AND meets confidence
            if r["agrees_with_freq"]:
                # LLM agrees — apply freq swap if freq != current
                if r["freq_cid"] != r["current_cid"]:
                    pred.loc[r["pred_idx"], "concept_id"] = r["freq_cid"]
                continue

            # LLM disagrees — only override if confident enough
            if r["confidence"] >= min_conf:
                new_cid = r["chosen_cid"]
            else:
                new_cid = r["freq_cid"]  # fall back to freq

            if new_cid == r["current_cid"]:
                continue
            pred.loc[r["pred_idx"], "concept_id"] = new_cid
            n_sw += 1
            if new_cid in r["gold_cids"]: n_cor += 1
            if r["current_cid"] in r["gold_cids"]: n_incor += 1

        ev = pred[["note_id", "start", "end", "concept_id"]].copy()
        for c in ["start", "end", "concept_id"]: ev[c] = ev[c].astype(int)
        iou = macro_char_iou(ev, test_gold[["note_id", "start", "end", "concept_id"]])
        n_overrides = sum(1 for r in llm_results
                         if not r["agrees_with_freq"] and r["confidence"] >= min_conf)
        print(f"  conf>={min_conf:2d}: IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
              f"overrides={n_overrides:3d} sw={n_sw} right={n_cor} wrong={n_incor}")

    # Also test: pure freq swap (ignore LLM entirely) on same subset
    print(f"\n  --- Reference: pure freq swap on same {len(subset)} queries ---")
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
    print(f"  Pure freq swap:  IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
          f"sw={n_sw} right={n_cor} wrong={n_incor}")

    # ===================================================================
    # OVERRIDE-ONLY ANALYSIS: just look at the disagreements
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("OVERRIDE-ONLY: accuracy of LLM overrides by confidence")
    print("=" * 80)

    for min_conf in range(1, 11):
        overrides = [r for r in llm_results
                     if not r["agrees_with_freq"] and r["confidence"] >= min_conf and r["gold_cids"]]
        if not overrides:
            print(f"  conf>={min_conf:2d}: no overrides")
            continue
        n_llm_right = sum(1 for r in overrides if r["chosen_correct"])
        n_freq_right = sum(1 for r in overrides if r["freq_correct"])
        net = n_llm_right - n_freq_right
        print(f"  conf>={min_conf:2d}: n={len(overrides):3d}  "
              f"LLM_correct={n_llm_right:3d} ({100*n_llm_right/len(overrides):.0f}%)  "
              f"freq_correct={n_freq_right:3d} ({100*n_freq_right/len(overrides):.0f}%)  "
              f"net={net:+d}")

    # Show sample high-confidence overrides
    print(f"\n  Sample high-confidence CORRECT overrides (conf>=8):")
    shown = 0
    for r in llm_results:
        if (not r["agrees_with_freq"] and r["confidence"] >= 8 and
                r["chosen_correct"] and not r["freq_correct"] and shown < 15):
            name = concept_names.get(r["chosen_cid"], str(r["chosen_cid"]))
            freq_name = concept_names.get(r["freq_cid"], str(r["freq_cid"]))
            print(f"    conf={r['confidence']} [{r['section']}] '{r['mention']}' "
                  f"freq={freq_name[:35]} -> LLM={name[:35]}")
            shown += 1

    print(f"\n  Sample high-confidence WRONG overrides (conf>=8):")
    shown = 0
    for r in llm_results:
        if (not r["agrees_with_freq"] and r["confidence"] >= 8 and
                not r["chosen_correct"] and r["freq_correct"] and shown < 15):
            name = concept_names.get(r["chosen_cid"], str(r["chosen_cid"]))
            freq_name = concept_names.get(r["freq_cid"], str(r["freq_cid"]))
            print(f"    conf={r['confidence']} [{r['section']}] '{r['mention']}' "
                  f"freq={freq_name[:35]} -> LLM={name[:35]}")
            shown += 1

    # Token stats
    token_counts = [r["n_tokens"] for r in llm_results]
    arr = np.array(token_counts)
    print(f"\n  Output tokens: median={np.median(arr):.0f} "
          f"mean={np.mean(arr):.0f} min={np.min(arr)} max={np.max(arr)}")

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
