#!/usr/bin/env python3
"""Diagnose why dictionary entries fail to match gold annotations.

Focuses on the 824 "in_dict_but_not_matched" FNs from the error analysis.
For each one, determines WHY the regex didn't fire or why the match was
filtered out.
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "super-dictionary"))

from train_dictionary import (
    COMMON_HEADERS, IndexedDict, annotate_with_dict, remove_overlaps,
    CASE_SENSITIVE_DICT, get_pattern, train,
    get_sections, get_header_by_pos, is_in_header,
)


def diagnose_single_fn(
    gold_text: str, gold_start: int, gold_end: int, gold_cid: int,
    note_text: str, note_text_lc: str, d: dict, uc_d: dict,
    headers_lc: list[str], headers_orig: list[str],
) -> dict:
    """Diagnose why a gold annotation wasn't matched."""
    gold_mention = gold_text.lower().strip()
    gold_mention_norm = " ".join(gold_mention.split())

    result = {
        "gold_text": gold_text,
        "gold_mention_norm": gold_mention_norm,
        "gold_start": gold_start,
        "gold_end": gold_end,
        "gold_cid": gold_cid,
        "reasons": [],
    }

    # Step 1: Is the mention in the dictionary?
    matching_keys = [(k, d[k]) for k in d if k[1] == gold_mention_norm]
    matching_uc = [(k, uc_d[k]) for k in uc_d if k[1] == gold_text.strip()]

    if not matching_keys and not matching_uc:
        # Try fuzzy: maybe whitespace difference
        close_keys = []
        for k in d:
            if isinstance(k[1], str) and k[1].replace(" ", "") == gold_mention_norm.replace(" ", ""):
                close_keys.append((k, d[k]))
        if close_keys:
            result["reasons"].append("whitespace_mismatch_in_dict")
            result["close_keys"] = close_keys[:3]
        else:
            result["reasons"].append("not_in_dict")
        return result

    result["matching_keys"] = matching_keys[:5]
    result["in_dict"] = True

    # Step 2: Does the regex pattern match at the gold location?
    p = get_pattern(gold_mention_norm)
    if p is None:
        result["reasons"].append("pattern_compile_failed")
        return result

    # Check if pattern matches anywhere in the note
    all_matches = list(p.finditer(note_text_lc))
    result["n_pattern_matches"] = len(all_matches)

    # Check if pattern matches at the gold location specifically
    gold_region = note_text_lc[max(0, gold_start - 5):gold_end + 5]
    region_matches = list(p.finditer(gold_region))

    if not all_matches:
        # Pattern doesn't match anywhere - why?
        # Check if it's a newline issue
        gold_span_raw = note_text_lc[gold_start:gold_end]
        has_newline = "\n" in gold_span_raw
        has_tab = "\t" in gold_span_raw
        has_multi_space = "  " in gold_span_raw

        result["gold_span_raw"] = repr(gold_span_raw)
        result["pattern"] = p.pattern

        if has_newline:
            # Try matching with newline replaced by space
            fixed = gold_span_raw.replace("\n", " ")
            fixed = " ".join(fixed.split())
            if p.match(fixed):
                result["reasons"].append("newline_in_span")
            else:
                result["reasons"].append("pattern_no_match_even_fixed")
        elif has_tab:
            result["reasons"].append("tab_in_span")
        elif has_multi_space:
            result["reasons"].append("multi_space_in_span")
        else:
            # Something else - check char by char
            result["reasons"].append("pattern_no_match_unknown")
            # Check if the pattern has special chars that might be wrong
            if any(c in gold_mention_norm for c in ".^$?|\\"):
                result["reasons"].append("special_chars_in_mention")
        return result

    # Pattern matches somewhere - check if it matches at the gold position
    match_at_gold = None
    for m in all_matches:
        if m.start() == gold_start or (m.start() >= gold_start - 2 and m.end() <= gold_end + 2):
            match_at_gold = m
            break

    if match_at_gold is None:
        # Pattern matches elsewhere but not at gold position
        # Check the filters in annotate_with_dict
        closest = min(all_matches, key=lambda m: abs(m.start() - gold_start))
        result["closest_match"] = (closest.start(), closest.end())
        result["reasons"].append("pattern_matches_elsewhere_not_at_gold")
        return result

    # Step 3: Pattern matches at gold position - check annotate_with_dict filters
    i, j = match_at_gold.start(), match_at_gold.end()

    # Filter: skip first 100 chars
    if i < 100:
        result["reasons"].append("filtered_preamble_lt_100")
        return result

    # Filter: word boundary check
    if i > 0 and note_text_lc[i - 1].isalnum():
        result["reasons"].append("filtered_left_boundary")
        result["left_context"] = repr(note_text_lc[max(0, i-5):i+5])
        return result

    if j < len(note_text_lc) and note_text_lc[j].isalnum():
        result["reasons"].append("filtered_right_boundary")
        result["right_context"] = repr(note_text_lc[max(0, j-5):j+5])
        return result

    # Filter: in header
    if is_in_header(note_text_lc, i):
        result["reasons"].append("filtered_in_header")
        return result

    # Filter: section check
    h_positions, pos_header = get_sections(note_text_lc, headers_lc)
    h = get_header_by_pos(i, h_positions, pos_header, headers_lc)
    if h is None:
        result["reasons"].append("filtered_no_header")
        return result

    hl = h.lower()
    if "medication" in hl or "service" in hl or "date of birth" in hl:
        result["reasons"].append("filtered_excluded_section")
        result["section"] = h
        return result

    # Check if any matching key has a compatible section
    section_match = False
    for (sec, _mention), cid in matching_keys:
        if sec == "any" or h == sec or h in sec:
            section_match = True
            # But is the concept right?
            if cid == gold_cid:
                result["reasons"].append("SHOULD_HAVE_MATCHED")
                result["section"] = h
                result["matching_section"] = sec
                return result

    if not section_match:
        result["reasons"].append("filtered_section_mismatch")
        result["note_section"] = h
        result["dict_sections"] = [k[0] for k, _ in matching_keys[:5]]
        return result

    # Section matches but concept_id doesn't
    result["reasons"].append("wrong_concept_in_dict")
    result["dict_concepts"] = [cid for _, cid in matching_keys[:5]]
    return result


def main():
    t0 = time.perf_counter()
    split_dir = REPO_ROOT / "data" / "old-challenge-split"
    super_dict_path = REPO_ROOT / "data" / "interim" / "super_dictionary_full.tsv"
    flat_term_path = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"

    print("Loading data...", flush=True)
    train_notes = pd.read_csv(split_dir / "train_notes.csv")
    train_annotations = pd.read_csv(split_dir / "train_annotations.csv")
    test_notes = pd.read_csv(split_dir / "test_notes.csv")
    test_gold = pd.read_csv(split_dir / "test_annotations.csv")
    for col in ["start", "end", "concept_id"]:
        test_gold[col] = test_gold[col].astype(int)

    print("Training dictionary...", flush=True)
    d, uc_d = train(
        train_notes, train_annotations,
        super_dict_path=super_dict_path,
        flat_terminology_path=flat_term_path,
    )
    print(f"  Dict: {len(d):,}, UC: {len(uc_d):,}")

    # Generate predictions
    print("Predicting...", flush=True)
    headers_lc = [h.lower() for h in COMMON_HEADERS]
    headers_orig = list(COMMON_HEADERS)

    uc_d_full = dict(uc_d)
    uc_d_full.update(CASE_SENSITIVE_DICT)
    d_indexed = IndexedDict(d, prefilter="bigram")
    uc_indexed = IndexedDict(uc_d_full, prefilter="bigram")

    texts = test_notes.set_index("note_id")["text"]
    texts_lc = texts.str.lower()

    all_preds = []
    for _, note_row in test_notes.iterrows():
        note_id = str(note_row["note_id"])
        text = note_row["text"]
        text_lc = text.lower()
        ann_lc = annotate_with_dict(text_lc, d_indexed, headers_lc, note_id)
        ann_uc = annotate_with_dict(text, uc_indexed, headers_orig, note_id)
        combined = pd.concat([ann_lc, ann_uc], ignore_index=True)
        if not combined.empty:
            combined = remove_overlaps(combined)
            all_preds.append(combined)

    pred = pd.concat(all_preds, ignore_index=True)
    for c in ["start", "end", "concept_id"]:
        pred[c] = pred[c].astype(int)
    print(f"  {len(pred):,} predictions")

    # Find FNs that are "in dict but not matched"
    print("\nFinding 'in_dict_but_not_matched' FNs...", flush=True)
    fn_diagnoses = []
    n_total = 0
    n_in_dict_not_matched = 0

    for _, gold_row in test_gold.iterrows():
        nid = str(gold_row["note_id"])
        gs, ge, gcid = int(gold_row["start"]), int(gold_row["end"]), int(gold_row["concept_id"])

        note_preds = pred[pred["note_id"].astype(str) == nid]
        # Check if this gold annotation is covered
        if not note_preds.empty:
            overlaps = note_preds[
                (note_preds["start"] < ge) &
                (note_preds["end"] > gs) &
                (note_preds["concept_id"] == gcid)
            ]
            if not overlaps.empty:
                continue  # It's a TP

        n_total += 1
        # Get the gold text
        text = texts.get(nid, texts.get(int(nid) if nid.isdigit() else nid, ""))
        if isinstance(text, pd.Series):
            text = text.iloc[0] if len(text) > 0 else ""
        text_lc_note = text.lower()
        gold_text = text[gs:ge]
        gold_mention_norm = " ".join(gold_text.lower().strip().split())

        # Check if it's in the dictionary
        in_d = any(k[1] == gold_mention_norm for k in d)
        in_uc = any(k[1] == gold_text.strip() for k in uc_d)

        if not in_d and not in_uc:
            continue  # not_in_dict FN, not our focus

        n_in_dict_not_matched += 1
        diag = diagnose_single_fn(
            gold_text, gs, ge, gcid,
            text, text_lc_note, d, uc_d_full,
            headers_lc, headers_orig,
        )
        fn_diagnoses.append(diag)

    print(f"  Total FNs: {n_total}")
    print(f"  In-dict-but-not-matched: {n_in_dict_not_matched}")

    # Aggregate diagnoses
    print("\n" + "=" * 80)
    print("DIAGNOSIS RESULTS")
    print("=" * 80)

    reason_counter = Counter()
    for diag in fn_diagnoses:
        for r in diag["reasons"]:
            reason_counter[r] += 1

    print(f"\n  Reason breakdown ({len(fn_diagnoses)} FNs):")
    for reason, count in reason_counter.most_common():
        chars = sum(
            d["gold_end"] - d["gold_start"]
            for d in fn_diagnoses if reason in d["reasons"]
        )
        print(f"    {reason:<40} {count:>5} ({chars:>6} chars)")

    # Show samples for each reason
    for reason in reason_counter:
        samples = [d for d in fn_diagnoses if reason in d["reasons"]][:5]
        print(f"\n  --- Samples: {reason} ---")
        for s in samples:
            print(f"    gold='{s['gold_text'][:60]}' norm='{s['gold_mention_norm'][:40]}'")
            print(f"      cid={s['gold_cid']} pos={s['gold_start']}-{s['gold_end']}")
            if "gold_span_raw" in s:
                print(f"      raw={s['gold_span_raw'][:80]}")
            if "pattern" in s:
                print(f"      pattern={s['pattern'][:80]}")
            if "left_context" in s:
                print(f"      left_ctx={s['left_context']}")
            if "right_context" in s:
                print(f"      right_ctx={s['right_context']}")
            if "section" in s:
                print(f"      section={s.get('section', '?')}")
            if "note_section" in s:
                print(f"      note_section={s['note_section']}")
            if "dict_sections" in s:
                print(f"      dict_sections={s['dict_sections']}")
            if "dict_concepts" in s:
                print(f"      dict_concepts={s['dict_concepts']}")
            if "closest_match" in s:
                print(f"      closest_match={s['closest_match']}")

    # Quantify the "overlap removal ate it" case
    print("\n" + "=" * 80)
    print("OVERLAP REMOVAL ANALYSIS")
    print("=" * 80)
    print("Checking if matches exist pre-overlap-removal but get eaten...")

    # Re-run prediction with keep_overlaps=True to see what's lost
    n_recovered = 0
    recovered_chars = 0
    overlap_victims = []
    for _, note_row in test_notes.iterrows():
        note_id = str(note_row["note_id"])
        text = note_row["text"]
        text_lc = text.lower()

        # Get all matches (before overlap removal)
        ann_all = annotate_with_dict(text_lc, d_indexed, headers_lc, note_id, keep_overlaps=True)
        ann_uc_all = annotate_with_dict(text, uc_indexed, headers_orig, note_id, keep_overlaps=True)
        all_matches = pd.concat([ann_all, ann_uc_all], ignore_index=True)

        if all_matches.empty:
            continue

        # Get post-overlap matches
        deduped = remove_overlaps(all_matches.copy())

        # Find gold annotations for this note
        note_gold = test_gold[test_gold["note_id"].astype(str) == note_id]

        for _, gold_row in note_gold.iterrows():
            gs, ge, gcid = int(gold_row["start"]), int(gold_row["end"]), int(gold_row["concept_id"])

            # Is it in the deduped predictions?
            in_deduped = not deduped[
                (deduped["start"].astype(int) < ge) &
                (deduped["end"].astype(int) > gs) &
                (deduped["concept_id"].astype(int) == gcid)
            ].empty

            if in_deduped:
                continue

            # Was it in the pre-overlap set?
            in_all = not all_matches[
                (all_matches["start"].astype(int) < ge) &
                (all_matches["end"].astype(int) > gs) &
                (all_matches["concept_id"].astype(int) == gcid)
            ].empty

            if in_all:
                n_recovered += 1
                recovered_chars += ge - gs
                overlap_victims.append({
                    "note_id": note_id,
                    "gold_text": text[gs:ge][:60],
                    "gold_cid": gcid,
                    "start": gs, "end": ge,
                })

    print(f"\n  Matches eaten by overlap removal: {n_recovered} ({recovered_chars:,} chars)")
    if overlap_victims:
        print(f"\n  Samples of overlap victims:")
        for v in overlap_victims[:20]:
            print(f"    '{v['gold_text']}'  cid={v['gold_cid']}  pos={v['start']}-{v['end']}")

    # Check what the \s+ pattern does with newlines
    print("\n" + "=" * 80)
    print("REGEX PATTERN BEHAVIOR TESTS")
    print("=" * 80)

    test_cases = [
        ("aortic valve", "aortic valve"),
        ("aortic valve", "aortic\nvalve"),
        ("aortic valve", "aortic  valve"),
        ("aortic valve", "aortic\n valve"),
        ("blood pressure", "blood\npressure"),
        ("rib fx", "rib\nfx"),
        ("no acute", "no\nacute"),
        ("st elevation", "st\nelevation"),
    ]

    print(f"\n  Pattern matching behavior:")
    for mention, test_text in test_cases:
        p = get_pattern(mention)
        if p is None:
            print(f"    '{mention}' vs '{repr(test_text)}': PATTERN FAILED")
            continue
        m = p.search(test_text)
        matched = "MATCH" if m else "NO MATCH"
        print(f"    '{mention}' pattern='{p.pattern}' vs {repr(test_text)}: {matched}")

    print(f"\nTotal time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
