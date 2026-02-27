#!/usr/bin/env python3
"""LLM concept disambiguation v6 — SNOMED-enriched prompts.

Enrichments over v5:
- IS-A hierarchy relationships between candidates ("X is a more specific type of Y")
- Text definitions from SNOMED RF2 where available
- Training context examples showing how each concept was actually annotated
- Semantic tag from concept names
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
from runtime_scoring import macro_char_iou

OPENBIO_MODEL = "bartowski/OpenBioLLM-Llama3-8B-AWQ"
CONTEXT_WINDOW = 250
RF2_DIR = REPO_ROOT / "data" / "SnomedCT_InternationalRF2_PRODUCTION_20260101T120000Z" / "Full" / "Terminology"
IS_A_TYPE = 116680003


# =====================================================================
# SNOMED knowledge loading
# =====================================================================

def load_concept_names():
    ft = pd.read_csv(REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv")
    return dict(zip(ft["concept_id"].astype(int), ft["concept_name"]))


def load_parent_map(candidate_cids: set[int]) -> dict[int, set[int]]:
    """Load IS-A relationships. Only keep entries relevant to candidate concepts."""
    rel_path = RF2_DIR / "sct2_Relationship_Full_INT_20260101.txt"
    print(f"  Loading IS-A hierarchy from {rel_path.name}...", flush=True)
    t0 = time.perf_counter()

    rels = pd.read_csv(
        rel_path, sep="\t",
        dtype={"sourceId": int, "destinationId": int, "typeId": int, "active": int},
        usecols=["active", "sourceId", "destinationId", "typeId"],
    )
    is_a = rels[(rels["active"] == 1) & (rels["typeId"] == IS_A_TYPE)]
    print(f"    Active IS-A relationships: {len(is_a):,}")

    # Build full parent map (we need transitive closure for candidates)
    parent_map: dict[int, set[int]] = {}
    for src, dst in zip(is_a["sourceId"].values, is_a["destinationId"].values):
        src, dst = int(src), int(dst)
        parent_map.setdefault(src, set()).add(dst)

    print(f"    Parent map: {len(parent_map):,} concepts ({time.perf_counter()-t0:.1f}s)")
    return parent_map


def find_hierarchy_relations(candidates_cids: list[int], parent_map: dict[int, set[int]]) -> list[tuple[int, int]]:
    """Find direct parent-child pairs among a list of candidate concept IDs.

    Returns list of (child_cid, parent_cid) tuples.
    """
    cid_set = set(candidates_cids)
    relations = []
    for cid in candidates_cids:
        parents = parent_map.get(cid, set())
        for p in parents:
            if p in cid_set:
                relations.append((cid, p))
    return relations


def load_text_definitions(candidate_cids: set[int]) -> dict[int, str]:
    """Load SNOMED text definitions for candidate concepts."""
    def_path = RF2_DIR / "sct2_TextDefinition_Full-en_INT_20260101.txt"
    print(f"  Loading text definitions from {def_path.name}...", flush=True)

    defs = pd.read_csv(
        def_path, sep="\t",
        usecols=["active", "conceptId", "term"],
        dtype={"active": int, "conceptId": int},
    )
    active = defs[(defs["active"] == 1) & (defs["conceptId"].isin(candidate_cids))]
    active = active.drop_duplicates("conceptId")

    result = dict(zip(active["conceptId"].astype(int), active["term"]))
    print(f"    Definitions found: {len(result):,} / {len(candidate_cids):,} candidates")
    return result


def build_training_contexts(
    train_notes_df: pd.DataFrame,
    train_annotations_df: pd.DataFrame,
    candidate_cids: set[int],
    max_examples: int = 3,
    context_chars: int = 80,
) -> dict[int, list[str]]:
    """Extract example contexts from training data for each candidate concept.

    Returns {concept_id: [context_string, ...]}
    """
    print(f"  Building training contexts for {len(candidate_cids):,} concepts...", flush=True)
    texts_by_id = {row["note_id"]: row["text"] for _, row in train_notes_df.iterrows()}

    # Filter to relevant annotations
    relevant = train_annotations_df[train_annotations_df["concept_id"].isin(candidate_cids)]

    contexts: dict[int, list[str]] = defaultdict(list)
    for _, row in relevant.iterrows():
        cid = int(row["concept_id"])
        if len(contexts[cid]) >= max_examples:
            continue
        note_text = texts_by_id.get(row["note_id"], "")
        s, e = int(row["start"]), int(row["end"])
        mention = note_text[s:e]
        before = note_text[max(0, s - context_chars):s].split(" ", 1)[-1] if s > 0 else ""
        after = note_text[e:min(len(note_text), e + context_chars)].rsplit(" ", 1)[0] if e < len(note_text) else ""
        ctx = f"...{before}[{mention}]{after}..."
        # Collapse whitespace
        ctx = re.sub(r"\s+", " ", ctx)
        contexts[cid].append(ctx)

    n_with = sum(1 for v in contexts.values() if v)
    print(f"    Concepts with training examples: {n_with:,}")
    return dict(contexts)


# =====================================================================
# Prediction and disambiguation infrastructure
# =====================================================================

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


def build_enriched_prompt(
    mention, before, after, candidates_shuffled, concept_names,
    parent_map, text_defs, training_contexts,
):
    """Build prompt with SNOMED hierarchy, definitions, and training examples."""
    cids = [cid for cid, _ in candidates_shuffled]

    # Build candidate lines with definitions
    cand_lines = []
    for i, (cid, _count) in enumerate(candidates_shuffled):
        name = concept_names.get(cid, f"Unknown ({cid})")
        line = f"{i}: {name}"
        defn = text_defs.get(cid)
        if defn:
            # Truncate long definitions
            if len(defn) > 150:
                defn = defn[:147] + "..."
            line += f"\n   Definition: {defn}"
        # Add training examples
        examples = training_contexts.get(cid, [])
        if examples:
            line += f"\n   Training examples: " + " | ".join(examples[:2])
        cand_lines.append(line)

    # Find hierarchy relationships between candidates
    hier_lines = []
    relations = find_hierarchy_relations(cids, parent_map)
    for child, parent in relations:
        ci = cids.index(child)
        pi = cids.index(parent)
        child_name = concept_names.get(child, str(child))
        parent_name = concept_names.get(parent, str(parent))
        hier_lines.append(f"  - {ci} ({child_name}) is a more specific type of {pi} ({parent_name})")

    hier_section = ""
    if hier_lines:
        hier_section = "\nHierarchy:\n" + "\n".join(hier_lines) + "\n"

    return f"""Clinical note excerpt: ...{before}[{mention}]{after}...

Which SNOMED CT concept best matches "{mention}" in this context?
{chr(10).join(cand_lines)}
{hier_section}
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

    # ===================================================================
    # Load SNOMED knowledge
    # ===================================================================
    print("\nLoading SNOMED knowledge...", flush=True)

    # Collect all candidate concept IDs
    all_candidate_cids = set()
    for item in ambiguous:
        for cid in item["alternatives"]:
            all_candidate_cids.add(cid)
    print(f"  Unique candidate concepts: {len(all_candidate_cids):,}")

    parent_map = load_parent_map(all_candidate_cids)
    text_defs = load_text_definitions(all_candidate_cids)
    training_contexts = build_training_contexts(
        train_notes_df, train_annotations_df, all_candidate_cids
    )

    # Count how many queries will benefit from hierarchy info
    n_with_hier = 0
    for item in ambiguous:
        cids = [c for c, _ in sorted(item["alternatives"].items(), key=lambda x: -x[1])[:6]]
        if find_hierarchy_relations(cids, parent_map):
            n_with_hier += 1
    print(f"  Queries with hierarchy info between candidates: {n_with_hier}/{len(ambiguous)}")

    # ===================================================================
    # Build prompts
    # ===================================================================
    print("\nBuilding enriched prompts...", flush=True)
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

        prompt = build_enriched_prompt(
            mention_text, before, after, candidates_shuffled, concept_names,
            parent_map, text_defs, training_contexts,
        )
        queries.append({
            "item": item,
            "prompt": prompt,
            "candidates_shuffled": candidates_shuffled,
            "current_shuffled_idx": current_shuffled_idx,
        })

    # Show sample prompt
    print("\n  === SAMPLE ENRICHED PROMPT ===")
    for q in queries[:20]:
        cids = [c for c, _ in q["candidates_shuffled"]]
        rels = find_hierarchy_relations(cids, parent_map)
        has_def = any(c in text_defs for c in cids)
        has_ctx = any(c in training_contexts for c in cids)
        if rels and (has_def or has_ctx):
            print(q["prompt"][:1500])
            print("  === END SAMPLE ===\n")
            break

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
        max_tokens=8,
        temperature=0.0,
        logprobs=5,
    )

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

    # Check prompt lengths
    prompt_lengths = [len(p) for p in formatted_prompts]
    print(f"  Prompt lengths: median={np.median(prompt_lengths):.0f} "
          f"max={max(prompt_lengths)} mean={np.mean(prompt_lengths):.0f}")

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

        # Check if this query had hierarchy info
        cids = [c for c, _ in candidates_shuffled]
        has_hier = bool(find_hierarchy_relations(cids, parent_map))

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
            "has_hier": has_hier,
        })

    print(f"  Parse failures: {n_parse_fail}/{len(queries)} ({100*n_parse_fail/len(queries):.1f}%)")

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
    print("EXPERIMENTS")
    print("=" * 80)

    run_experiment("All LLM swaps (unfiltered)", lambda r: True)

    # Logprob thresholds
    for thresh in [-0.3, -0.5, -0.7, -1.0]:
        run_experiment(
            f"Logprob >= {thresh:.1f}",
            lambda r, t=thresh: r.get("logprob") is not None and r["logprob"] >= t
        )

    # Frequency agreement
    run_experiment(
        "LLM + frequency agreement",
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
    run_experiment("Swap to most frequent only", swap_to_top)

    # Logprob + freq agreement
    for thresh in [-0.5, -0.7, -1.0]:
        run_experiment(
            f"LP >= {thresh:.1f} + freq agree",
            lambda r, t=thresh: (
                r.get("logprob") is not None and r["logprob"] >= t and
                mention_freq.get(r["mention"], {}).get(r["chosen_cid"], 0) >
                mention_freq.get(r["mention"], {}).get(r["current_cid"], 0)
            )
        )

    # Logprob + swap to top
    for thresh in [-0.5, -0.7, -1.0]:
        run_experiment(
            f"LP >= {thresh:.1f} + swap to top",
            lambda r, t=thresh: (
                r.get("logprob") is not None and r["logprob"] >= t and swap_to_top(r)
            )
        )

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
    print(f"  {'Pure freq swap':40s} IoU={iou:.4f} ({iou-iou_baseline:+.4f}) "
          f"sw={n_sw} right={n_cor} wrong={n_incor} acc={100*n_cor/max(n_sw,1):.0f}%")

    # ===================================================================
    # Classification accuracy — overall and by enrichment type
    # ===================================================================
    print(f"\n{'=' * 80}")
    print("CLASSIFICATION ACCURACY")
    print("=" * 80)

    for subset_name, filter_fn in [
        ("All queries", lambda r: True),
        ("With hierarchy info", lambda r: r["has_hier"]),
        ("Without hierarchy info", lambda r: not r["has_hier"]),
    ]:
        subset = [r for r in llm_results if filter_fn(r) and r["gold_cids"]]
        if not subset:
            continue
        n_llm = sum(1 for r in subset if r["chosen_correct"])
        n_base = sum(1 for r in subset if r["current_correct"])
        print(f"  {subset_name} (n={len(subset)}): "
              f"baseline={100*n_base/len(subset):.1f}% "
              f"LLM={100*n_llm/len(subset):.1f}%")

    # Accuracy by logprob bucket
    print(f"\n  Accuracy by logprob bucket:")
    buckets = [(-0.3, -0.1), (-0.5, -0.3), (-0.7, -0.5), (-1.0, -0.7), (-2.0, -1.0)]
    for lo, hi in buckets:
        bucket = [r for r in llm_results
                  if r["logprob"] is not None and lo <= r["logprob"] < hi and r["gold_cids"]]
        if bucket:
            n_ok = sum(1 for r in bucket if r["chosen_correct"])
            n_base = sum(1 for r in bucket if r["current_correct"])
            print(f"    [{lo:.1f}, {hi:.1f}): n={len(bucket)} "
                  f"LLM={100*n_ok/len(bucket):.1f}% "
                  f"base={100*n_base/len(bucket):.1f}%")

    # Show sample correct swaps
    print(f"\n  Sample CORRECT swaps:")
    shown = 0
    for r in llm_results:
        if r["is_swap"] and r["chosen_correct"] and not r["current_correct"] and shown < 10:
            print(f"    '{r['mention']}' {r['current_cid']}->{r['chosen_cid']} "
                  f"hier={r['has_hier']} lp={r.get('logprob','?'):.2f}")
            shown += 1

    print(f"\n  Sample INCORRECT swaps:")
    shown = 0
    for r in llm_results:
        if r["is_swap"] and not r["chosen_correct"] and r["current_correct"] and shown < 10:
            print(f"    '{r['mention']}' {r['current_cid']}->{r['chosen_cid']} "
                  f"gold={r['gold_cids']} hier={r['has_hier']} lp={r.get('logprob','?'):.2f}")
            shown += 1

    print(f"\nTotal time: {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
