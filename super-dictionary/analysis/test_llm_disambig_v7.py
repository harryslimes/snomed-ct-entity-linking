#!/usr/bin/env python3
"""LLM concept disambiguation v7 — section-header-aware prompts.

Key insight from v6: SNOMED hierarchy/definitions overloaded the prompt (41.9% acc
vs 56.2% bare). Section headers are different: compact, directly disambiguating.

Examples where section headers help:
  - "tender to palpation" in Physical Exam over right hip → generic Tenderness
  - "bright red blood" in History of Present Illness (bowel) → Hematochezia
  - "ischemic changes" in Pertinent Results (EKG) → ECG finding, not Ischemia
  - "pink conjunctiva" in Physical Exam → Conjunctiva normal, not Hyperemia

This version:
- Adds section header to each prompt (compact, high-signal)
- Adds section-aware training examples showing concept usage by section
- Drops SNOMED hierarchy/definitions (they hurt in v6)
- Compares section-enriched vs bare prompts on same model run
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
from engine import segment_sections, get_section_for_pos
from runtime_scoring import macro_char_iou

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


def build_section_concept_priors(d_combined):
    """Build (mention, section) -> Counter of concept IDs from training dict.

    This captures section-specific disambiguation signal:
    e.g., "drainage" in "physical exam" → 307488001 (Drainage (finding))
          "drainage" in "history of present illness" → 122462000 (Drainage (procedure))
    """
    section_mention_concepts: dict[tuple[str, str], Counter] = {}
    for (section, mention), counter in d_combined.items():
        key = (mention, section)
        section_mention_concepts[key] = counter
    return section_mention_concepts


def build_training_examples_with_sections(
    train_notes_df: pd.DataFrame,
    train_annotations_df: pd.DataFrame,
    candidate_cids: set[int],
    max_per_concept: int = 3,
    context_chars: int = 60,
) -> dict[int, list[dict]]:
    """Extract training examples with their section headers.

    Returns {concept_id: [{"section": str, "context": str}, ...]}
    """
    print(f"  Building section-aware training examples for {len(candidate_cids):,} concepts...",
          flush=True)
    texts_by_id = {}
    sections_by_id = {}
    for _, row in train_notes_df.iterrows():
        nid = row["note_id"]
        text = row["text"]
        texts_by_id[nid] = text
        sections_by_id[nid] = segment_sections(text)

    relevant = train_annotations_df[train_annotations_df["concept_id"].isin(candidate_cids)]

    examples: dict[int, list[dict]] = defaultdict(list)
    for _, row in relevant.iterrows():
        cid = int(row["concept_id"])
        if len(examples[cid]) >= max_per_concept:
            continue
        nid = row["note_id"]
        note_text = texts_by_id.get(nid, "")
        if not note_text:
            continue
        s, e = int(row["start"]), int(row["end"])
        mention = note_text[s:e]

        # Get section header
        sections = sections_by_id.get(nid, [])
        sec = get_section_for_pos(s, sections)
        section_name = sec.header if sec else "unknown"

        # Build context snippet
        before = note_text[max(0, s - context_chars):s].split(" ", 1)[-1] if s > 0 else ""
        after = note_text[e:min(len(note_text), e + context_chars)].rsplit(" ", 1)[0] if e < len(note_text) else ""
        ctx = f"...{before}[{mention}]{after}..."
        ctx = re.sub(r"\s+", " ", ctx)

        examples[cid].append({"section": section_name, "context": ctx})

    n_with = sum(1 for v in examples.values() if v)
    print(f"    Concepts with examples: {n_with:,}")
    return dict(examples)


def get_context(note_text, start, end, window=CONTEXT_WINDOW):
    ctx_start = max(0, start - window)
    ctx_end = min(len(note_text), end + window)
    return note_text[ctx_start:start], note_text[start:end], note_text[end:ctx_end]


def build_prompt_with_section(
    mention, before, after, section_name, candidates_shuffled,
    concept_names, training_examples,
):
    """Build prompt with section header and section-aware training examples."""
    cand_lines = []
    for i, (cid, _count) in enumerate(candidates_shuffled):
        name = concept_names.get(cid, f"Unknown ({cid})")
        line = f"{i}: {name}"
        # Add compact training examples with their sections
        exs = training_examples.get(cid, [])
        if exs:
            ex_strs = [f"[{ex['section']}] {ex['context']}" for ex in exs[:2]]
            line += "\n   Seen in: " + " | ".join(ex_strs)
        cand_lines.append(line)

    section_line = f"Section: {section_name}\n" if section_name else ""

    return f"""{section_line}Clinical note excerpt: ...{before}[{mention}]{after}...

Which SNOMED CT concept best matches "{mention}" in this context?
{chr(10).join(cand_lines)}

Reply with ONLY the number (e.g. "0" or "1"). Do not explain."""


def build_prompt_bare(mention, before, after, candidates_shuffled, concept_names):
    """Same as v5 bare prompt for comparison."""
    cand_lines = []
    for i, (cid, _count) in enumerate(candidates_shuffled):
        name = concept_names.get(cid, f"Unknown ({cid})")
        cand_lines.append(f"{i}: {name}")

    return f"""Clinical note excerpt: ...{before}[{mention}]{after}...

Which SNOMED CT concept best matches "{mention}" in this context?
{chr(10).join(cand_lines)}

Reply with ONLY the number (e.g. "0" or "1"). Do not explain."""


def parse_response(text, n_candidates):
    if not text:
        return None
    text = text.strip()

    m = re.match(r'^(\d+)\s*$', text)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n_candidates:
            return idx

    m = re.search(r'(?:answer|best|match)\s+(?:is\s+)?(?:\*\*)?(\d+)', text, re.IGNORECASE)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n_candidates:
            return idx

    m = re.search(r'\b(\d+)\s*:', text)
    if m:
        idx = int(m.group(1))
        if 0 <= idx < n_candidates:
            return idx

    for token in text.split():
        token = re.sub(r'[^0-9]', '', token)
        if token:
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

    # Pre-compute section segmentations for test notes
    print("Segmenting sections for test notes...", flush=True)
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
    section_freq = build_section_concept_priors(d_combined)

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

        # Get section header for this prediction
        # segment_sections returns header without colon (e.g. "physical exam")
        # but training dict uses "physical exam:" — we store both forms
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

    n_with_section = sum(1 for a in ambiguous if a["section"])
    print(f"  Total predictions: {len(pred_baseline):,}")
    print(f"  Genuinely ambiguous (<95% dominant): {len(ambiguous)}")
    print(f"  With section header: {n_with_section}/{len(ambiguous)} "
          f"({100*n_with_section/max(len(ambiguous),1):.0f}%)")

    # Section distribution
    sec_counter = Counter(a["section"] or "unknown" for a in ambiguous)
    print(f"\n  Section distribution:")
    for sec, cnt in sec_counter.most_common(10):
        print(f"    {sec}: {cnt}")

    # ===================================================================
    # Build training examples
    # ===================================================================
    print("\nLoading section-aware training examples...", flush=True)
    all_candidate_cids = set()
    for item in ambiguous:
        for cid in item["alternatives"]:
            all_candidate_cids.add(cid)
    print(f"  Unique candidate concepts: {len(all_candidate_cids):,}")

    training_examples = build_training_examples_with_sections(
        train_notes_df, train_annotations_df, all_candidate_cids,
    )

    # ===================================================================
    # Section-aware frequency oracle (how much can section info help?)
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("SECTION-AWARE FREQUENCY ORACLE")
    print("=" * 80)

    # For each ambiguous prediction, check if section-specific frequency
    # gives a different (better) answer than global frequency
    n_section_diff = 0
    n_section_correct = 0
    n_global_correct = 0
    n_with_section_freq = 0
    for item in ambiguous:
        sec = item["section"]
        mention = item["mention"]

        # Global most frequent
        alts = mention_freq.get(mention, {})
        if not alts:
            continue
        global_top = max(alts, key=alts.get)

        # Section-specific most frequent
        # Training dict keys use section with colon, e.g. "physical exam:"
        sec_with_colon = sec + ":" if sec else None
        sec_key = (mention, sec_with_colon) if sec_with_colon else None
        sec_alts = section_freq.get(sec_key, Counter()) if sec_key else Counter()

        gold = gold_by_note.get(item["note_id"])
        gold_cids = set()
        if gold is not None:
            overlaps = gold[(gold["start"] < item["end"]) & (gold["end"] > item["start"])]
            gold_cids = set(overlaps["concept_id"].astype(int))

        if not gold_cids:
            continue

        if sec_alts:
            n_with_section_freq += 1
            sec_top = max(sec_alts, key=sec_alts.get)
            if sec_top != global_top:
                n_section_diff += 1
            if sec_top in gold_cids:
                n_section_correct += 1
        if global_top in gold_cids:
            n_global_correct += 1

    print(f"  Queries with section-specific freq data: {n_with_section_freq}")
    print(f"  Section freq differs from global: {n_section_diff}")
    print(f"  Global freq correct: {n_global_correct}")
    print(f"  Section freq correct: {n_section_correct}")

    # ===================================================================
    # Build prompts — BOTH section-enriched AND bare for A/B comparison
    # ===================================================================
    print("\nBuilding prompts...", flush=True)
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

        prompt_section = build_prompt_with_section(
            mention_text, before, after, item["section"],
            candidates_shuffled, concept_names, training_examples,
        )
        prompt_bare = build_prompt_bare(
            mention_text, before, after, candidates_shuffled, concept_names,
        )

        queries.append({
            "item": item,
            "prompt_section": prompt_section,
            "prompt_bare": prompt_bare,
            "candidates_shuffled": candidates_shuffled,
            "current_shuffled_idx": current_shuffled_idx,
        })

    # Show sample section-enriched prompt
    print("\n  === SAMPLE SECTION PROMPT ===")
    for q in queries[:30]:
        if q["item"]["section"]:
            print(q["prompt_section"][:2000])
            print("  === END SAMPLE ===\n")
            break

    # ===================================================================
    # Run OpenBioLLM via vLLM — both prompt variants in one batch
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
        max_tokens=8,
        temperature=0.0,
        logprobs=5,
    )

    tokenizer = llm.get_tokenizer()

    def format_prompt(raw_prompt):
        try:
            messages = [{"role": "user", "content": raw_prompt}]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except (ValueError, AttributeError):
            return (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{raw_prompt}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")

    # Batch: first all section prompts, then all bare prompts
    n_queries = len(queries)
    all_prompts = []
    for q in queries:
        all_prompts.append(format_prompt(q["prompt_section"]))
    for q in queries:
        all_prompts.append(format_prompt(q["prompt_bare"]))

    prompt_lengths_section = [len(all_prompts[i]) for i in range(n_queries)]
    prompt_lengths_bare = [len(all_prompts[n_queries + i]) for i in range(n_queries)]
    print(f"  Section prompt lengths: median={np.median(prompt_lengths_section):.0f} "
          f"max={max(prompt_lengths_section)} mean={np.mean(prompt_lengths_section):.0f}")
    print(f"  Bare prompt lengths: median={np.median(prompt_lengths_bare):.0f} "
          f"max={max(prompt_lengths_bare)} mean={np.mean(prompt_lengths_bare):.0f}")

    print(f"\n  Running vLLM batch inference on {len(all_prompts)} prompts "
          f"({n_queries} section + {n_queries} bare)...", flush=True)
    t_llm = time.perf_counter()
    outputs = llm.generate(all_prompts, sampling_params)
    elapsed_llm = time.perf_counter() - t_llm
    print(f"  vLLM done in {elapsed_llm:.1f}s ({len(all_prompts)/elapsed_llm:.1f} queries/s)")

    # Process results for both variants
    def process_outputs(output_list, queries, variant_name):
        results = []
        n_parse_fail = 0
        for q, output in zip(queries, output_list):
            item = q["item"]
            candidates_shuffled = q["candidates_shuffled"]
            current_shuffled_idx = q["current_shuffled_idx"]

            answer = output.outputs[0].text.strip()
            chosen_idx = parse_response(answer, len(candidates_shuffled))

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
                "section": item["section"],
                "current_correct": item["current_cid"] in gold_cids,
                "chosen_correct": chosen_cid in gold_cids,
                "is_swap": chosen_cid != item["current_cid"],
                "answer": answer,
                "logprob": logprob,
                "parse_fail": parse_fail,
            })

        print(f"  [{variant_name}] Parse failures: {n_parse_fail}/{len(queries)} "
              f"({100*n_parse_fail/len(queries):.1f}%)")
        return results

    section_results = process_outputs(outputs[:n_queries], queries, "section")
    bare_results = process_outputs(outputs[n_queries:], queries, "bare")

    # ===================================================================
    # RESULTS
    # ===================================================================

    def run_experiment(name, results, filter_fn):
        pred = pred_baseline.copy()
        n_sw = n_cor = n_incor = 0
        for r in results:
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
        print(f"  {name:45s} IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
              f"sw={n_sw} right={n_cor} wrong={n_incor} acc={acc:.0f}%")
        return iou

    print(f"\n{'=' * 80}")
    print("A/B COMPARISON: SECTION vs BARE PROMPTS")
    print("=" * 80)

    run_experiment("SECTION: All LLM swaps", section_results, lambda r: True)
    run_experiment("BARE: All LLM swaps", bare_results, lambda r: True)

    for thresh in [-0.3, -0.5, -0.7, -1.0]:
        run_experiment(
            f"SECTION: Logprob >= {thresh:.1f}",
            section_results,
            lambda r, t=thresh: r.get("logprob") is not None and r["logprob"] >= t
        )
        run_experiment(
            f"BARE: Logprob >= {thresh:.1f}",
            bare_results,
            lambda r, t=thresh: r.get("logprob") is not None and r["logprob"] >= t
        )

    # Frequency agreement
    run_experiment(
        "SECTION: LLM + freq agree",
        section_results,
        lambda r: mention_freq.get(r["mention"], {}).get(r["chosen_cid"], 0) >
                  mention_freq.get(r["mention"], {}).get(r["current_cid"], 0)
    )
    run_experiment(
        "BARE: LLM + freq agree",
        bare_results,
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

    run_experiment("SECTION: Swap to top freq", section_results, swap_to_top)
    run_experiment("BARE: Swap to top freq", bare_results, swap_to_top)

    # Pure frequency swap (no LLM baseline)
    print(f"\n  --- Pure frequency swap (no LLM) ---")
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
    print(f"  {'Pure freq swap':45s} IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
          f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # Section-aware frequency swap
    print(f"\n  --- Section-aware frequency swap (no LLM) ---")
    pred = pred_baseline.copy()
    n_sw = n_cor = n_incor = n_used_section = 0
    for item in ambiguous:
        sec = item["section"]
        mention = item["mention"]

        # Try section-specific frequency first, fallback to global
        # Training dict keys use section with colon, e.g. "physical exam:"
        sec_with_colon = sec + ":" if sec else None
        sec_key = (mention, sec_with_colon) if sec_with_colon else None
        sec_alts = section_freq.get(sec_key, Counter()) if sec_key else Counter()

        if sec_alts and len(sec_alts) >= 1:
            best_cid = max(sec_alts, key=sec_alts.get)
            n_used_section += 1
        else:
            alts = mention_freq.get(mention, {})
            if not alts:
                continue
            best_cid = max(alts, key=alts.get)

        if best_cid == item["current_cid"]:
            continue
        pred.loc[item["pred_idx"], "concept_id"] = best_cid
        gold = gold_by_note.get(item["note_id"])
        gold_cids = set()
        if gold is not None:
            overlaps = gold[(gold["start"] < item["end"]) & (gold["end"] > item["start"])]
            gold_cids = set(overlaps["concept_id"].astype(int))
        n_sw += 1
        if best_cid in gold_cids: n_cor += 1
        if item["current_cid"] in gold_cids: n_incor += 1
    ev = pred[["note_id", "start", "end", "concept_id"]].copy()
    for c in ["start", "end", "concept_id"]: ev[c] = ev[c].astype(int)
    iou = macro_char_iou(ev, test_gold[["note_id", "start", "end", "concept_id"]])
    print(f"  {'Section freq swap':45s} IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
          f"sw={n_sw} right={n_cor} wrong={n_incor} "
          f"used_section={n_used_section} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    # CLASSIFICATION ACCURACY
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("CLASSIFICATION ACCURACY")
    print("=" * 80)

    for variant_name, results in [("Section", section_results), ("Bare", bare_results)]:
        for subset_name, filter_fn in [
            ("All queries", lambda r: True),
            ("With section", lambda r: r["section"] is not None),
            ("Without section", lambda r: r["section"] is None),
        ]:
            subset = [r for r in results if filter_fn(r) and r["gold_cids"]]
            if not subset:
                continue
            n_llm = sum(1 for r in subset if r["chosen_correct"])
            n_base = sum(1 for r in subset if r["current_correct"])
            print(f"  [{variant_name}] {subset_name} (n={len(subset)}): "
                  f"baseline={100*n_base/len(subset):.1f}% "
                  f"LLM={100*n_llm/len(subset):.1f}%")

    # Head-to-head: section vs bare
    print(f"\n  Head-to-head: Section vs Bare prompt")
    both_correct = both_wrong = section_only = bare_only = 0
    for sr, br in zip(section_results, bare_results):
        if not sr["gold_cids"]:
            continue
        sc = sr["chosen_correct"]
        bc = br["chosen_correct"]
        if sc and bc: both_correct += 1
        elif sc and not bc: section_only += 1
        elif not sc and bc: bare_only += 1
        else: both_wrong += 1
    total = both_correct + both_wrong + section_only + bare_only
    print(f"    Both correct: {both_correct} ({100*both_correct/total:.1f}%)")
    print(f"    Section only: {section_only} ({100*section_only/total:.1f}%)")
    print(f"    Bare only:    {bare_only} ({100*bare_only/total:.1f}%)")
    print(f"    Both wrong:   {both_wrong} ({100*both_wrong/total:.1f}%)")

    # Per-section accuracy
    print(f"\n  Accuracy by section (section-enriched prompt):")
    sec_groups = defaultdict(list)
    for r in section_results:
        if r["gold_cids"]:
            sec_groups[r["section"] or "unknown"].append(r)
    for sec in sorted(sec_groups, key=lambda s: -len(sec_groups[s])):
        group = sec_groups[sec]
        n_llm = sum(1 for r in group if r["chosen_correct"])
        n_base = sum(1 for r in group if r["current_correct"])
        print(f"    {sec:35s} n={len(group):4d}  "
              f"base={100*n_base/len(group):.1f}%  "
              f"LLM={100*n_llm/len(group):.1f}%")

    # Show sample section wins
    print(f"\n  Sample wins from section prompt (correct where bare was wrong):")
    shown = 0
    for sr, br in zip(section_results, bare_results):
        if sr["chosen_correct"] and not br["chosen_correct"] and shown < 15:
            name = concept_names.get(sr["chosen_cid"], str(sr["chosen_cid"]))
            print(f"    [{sr['section']}] '{sr['mention']}' -> {name}")
            shown += 1

    print(f"\n  Sample losses from section prompt (wrong where bare was correct):")
    shown = 0
    for sr, br in zip(section_results, bare_results):
        if not sr["chosen_correct"] and br["chosen_correct"] and shown < 15:
            name_s = concept_names.get(sr["chosen_cid"], str(sr["chosen_cid"]))
            name_b = concept_names.get(br["chosen_cid"], str(br["chosen_cid"]))
            print(f"    [{sr['section']}] '{sr['mention']}' section->{name_s} bare->{name_b}")
            shown += 1

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
