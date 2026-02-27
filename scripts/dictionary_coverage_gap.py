#!/usr/bin/env python3
"""
Measure how many gold training annotations are NOT covered by the super dictionary.

Compares:
  - Gold training annotations from data/train_annotations.csv
  - Super dictionary from data/interim/super_dictionary_full.tsv
  - (Also) 1st Place flattened terminology from 1st Place/data/interim/flattened_terminology_syn_super.csv

Reports coverage at span-text level (case-insensitive) and span+concept level.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/workspaces/snomed-ct-entity-linking")

# ── 1. Load gold training annotations ───────────────────────────────────────

ann_path = ROOT / "data" / "train_annotations.csv"
print(f"Loading gold annotations from {ann_path} ...")

gold_annotations = []  # list of (span_text, concept_id)
type_counts = Counter()

with open(ann_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        atype = row["annotation_type"]
        type_counts[atype] += 1
        # We only evaluate "train" annotations (the training gold set)
        if atype == "train":
            span = row["span"].strip()
            cid = row["concept_id"].strip()
            gold_annotations.append((span, cid))

print(f"  Total rows in file: {sum(type_counts.values())}")
for k, v in type_counts.items():
    print(f"    annotation_type={k!r}: {v}")
print(f"  Gold TRAIN annotations to evaluate: {len(gold_annotations)}")

# ── 2. Load the super dictionary (full) ─────────────────────────────────────

dict_path = ROOT / "data" / "interim" / "super_dictionary_full.tsv"
print(f"\nLoading super dictionary from {dict_path} ...")

# Build: term_lower -> set of concept_ids
dict_term_to_cids: dict[str, set[str]] = defaultdict(set)
dict_rows = 0

with open(dict_path) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        term = row["term"].strip().lower()
        cid = row["snomed_concept_id"].strip()
        dict_term_to_cids[term].add(cid)
        dict_rows += 1

print(f"  Dictionary rows: {dict_rows}")
print(f"  Unique terms (lowercased): {len(dict_term_to_cids)}")

# count unique concept_ids
all_cids = set()
for cids in dict_term_to_cids.values():
    all_cids.update(cids)
print(f"  Unique concept IDs: {len(all_cids)}")

# ── 3. Also load 1st-place flattened super terminology for comparison ────────

flat_path = ROOT / "1st Place" / "data" / "interim" / "flattened_terminology_syn_super.csv"
flat_term_to_cids: dict[str, set[str]] = defaultdict(set)
flat_rows = 0

if flat_path.exists():
    print(f"\nLoading 1st-place flattened terminology from {flat_path} ...")
    with open(flat_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row["concept_name"].strip().lower()
            cid = row["concept_id"].strip()
            flat_term_to_cids[term].add(cid)
            flat_rows += 1
    print(f"  Rows: {flat_rows}")
    print(f"  Unique terms (lowercased): {len(flat_term_to_cids)}")

# ── 4. Evaluate coverage ────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("COVERAGE ANALYSIS: Super Dictionary vs Gold Train Annotations")
print("=" * 80)

total = len(gold_annotations)
found_any = 0          # span found in dict (any concept)
found_correct = 0      # span found AND maps to correct concept
found_wrong_only = 0   # span found but only maps to WRONG concepts
not_found = 0          # span not in dict at all

missing_spans = Counter()       # span_text -> count (not found at all)
wrong_concept_spans = Counter() # span_text -> count (found but wrong concept)
concept_id_for_missing = {}     # span_text -> concept_id (for display)

for span, cid in gold_annotations:
    span_lower = span.lower()
    if span_lower in dict_term_to_cids:
        found_any += 1
        if cid in dict_term_to_cids[span_lower]:
            found_correct += 1
        else:
            found_wrong_only += 1
            wrong_concept_spans[span] += 1
            concept_id_for_missing[span] = cid
    else:
        not_found += 1
        missing_spans[span] += 1
        concept_id_for_missing[span] = cid

pct = lambda n: f"{n:>7,}  ({100*n/total:5.1f}%)"

print(f"\nTotal gold train annotations:               {total:>7,}")
print(f"")
print(f"Span found in dictionary (any concept):     {pct(found_any)}")
print(f"  ... and maps to CORRECT concept:          {pct(found_correct)}")
print(f"  ... but only WRONG concept(s):            {pct(found_wrong_only)}")
print(f"Span NOT found in dictionary at all:        {pct(not_found)}")
print(f"")
print(f"Effective recall (correct concept):          {100*found_correct/total:.1f}%")
print(f"Coverage gap (not found at all):             {100*not_found/total:.1f}%")
print(f"Concept mismatch gap (found, wrong concept): {100*found_wrong_only/total:.1f}%")
print(f"Total gap (not found + wrong concept):       {100*(not_found+found_wrong_only)/total:.1f}%")

# ── 5. Unique span analysis ─────────────────────────────────────────────────

print("\n" + "-" * 80)
print("UNIQUE SPAN ANALYSIS")
print("-" * 80)

unique_gold_pairs = set(gold_annotations)
unique_spans = set(s.lower() for s, _ in gold_annotations)

unique_found = sum(1 for s in unique_spans if s in dict_term_to_cids)
unique_not_found = len(unique_spans) - unique_found

unique_pair_correct = 0
for span, cid in unique_gold_pairs:
    sl = span.lower()
    if sl in dict_term_to_cids and cid in dict_term_to_cids[sl]:
        unique_pair_correct += 1

print(f"Unique span texts (case-insensitive): {len(unique_spans):,}")
print(f"  Found in dictionary:                {unique_found:,}  ({100*unique_found/len(unique_spans):.1f}%)")
print(f"  NOT found in dictionary:            {unique_not_found:,}  ({100*unique_not_found/len(unique_spans):.1f}%)")
print(f"")
print(f"Unique (span, concept_id) pairs:      {len(unique_gold_pairs):,}")
print(f"  Correctly covered:                  {unique_pair_correct:,}  ({100*unique_pair_correct/len(unique_gold_pairs):.1f}%)")

# ── 6. Top missing spans ────────────────────────────────────────────────────

print("\n" + "-" * 80)
print("TOP 30 MOST FREQUENT MISSING SPANS (not in dictionary at all)")
print("-" * 80)
print(f"{'Rank':>4}  {'Count':>6}  {'% of total':>10}  {'Concept ID':>15}  Span Text")
print(f"{'----':>4}  {'-----':>6}  {'----------':>10}  {'-'*15:>15}  {'-'*40}")

for rank, (span, count) in enumerate(missing_spans.most_common(30), 1):
    cid = concept_id_for_missing.get(span, "?")
    print(f"{rank:>4}  {count:>6}  {100*count/total:>9.2f}%  {cid:>15}  {span}")

# ── 7. Top wrong-concept spans ──────────────────────────────────────────────

print("\n" + "-" * 80)
print("TOP 20 MOST FREQUENT WRONG-CONCEPT SPANS (found but wrong concept)")
print("-" * 80)
print(f"{'Rank':>4}  {'Count':>6}  {'Gold CID':>15}  {'Dict CIDs (sample)':>40}  Span Text")
print(f"{'----':>4}  {'-----':>6}  {'-'*15:>15}  {'-'*40:>40}  {'-'*40}")

for rank, (span, count) in enumerate(wrong_concept_spans.most_common(20), 1):
    gold_cid = concept_id_for_missing.get(span, "?")
    dict_cids = dict_term_to_cids.get(span.lower(), set())
    sample_cids = ", ".join(sorted(dict_cids)[:3])
    if len(dict_cids) > 3:
        sample_cids += f" (+{len(dict_cids)-3} more)"
    print(f"{rank:>4}  {count:>6}  {gold_cid:>15}  {sample_cids:>40}  {span}")

# ── 8. Comparison with 1st-place flattened terminology ───────────────────────

if flat_term_to_cids:
    print("\n" + "=" * 80)
    print("COMPARISON: 1st-Place Flattened Super Terminology")
    print("=" * 80)

    flat_found_any = 0
    flat_found_correct = 0
    flat_not_found = 0

    for span, cid in gold_annotations:
        sl = span.lower()
        if sl in flat_term_to_cids:
            flat_found_any += 1
            if cid in flat_term_to_cids[sl]:
                flat_found_correct += 1
        else:
            flat_not_found += 1

    print(f"\nTotal gold train annotations:               {total:>7,}")
    print(f"Span found (any concept):                   {pct(flat_found_any)}")
    print(f"  ... correct concept:                      {pct(flat_found_correct)}")
    print(f"Span NOT found at all:                      {pct(flat_not_found)}")
    print(f"Effective recall (correct concept):          {100*flat_found_correct/total:.1f}%")

# ── 9. Bucket analysis by span length ───────────────────────────────────────

print("\n" + "-" * 80)
print("COVERAGE BY SPAN LENGTH (word count)")
print("-" * 80)

from collections import defaultdict as dd

len_buckets = dd(lambda: {"total": 0, "found_correct": 0, "found_wrong": 0, "missing": 0})

for span, cid in gold_annotations:
    wc = len(span.split())
    bucket = "1" if wc == 1 else "2" if wc == 2 else "3" if wc == 3 else "4-5" if wc <= 5 else "6+"
    len_buckets[bucket]["total"] += 1
    sl = span.lower()
    if sl in dict_term_to_cids:
        if cid in dict_term_to_cids[sl]:
            len_buckets[bucket]["found_correct"] += 1
        else:
            len_buckets[bucket]["found_wrong"] += 1
    else:
        len_buckets[bucket]["missing"] += 1

print(f"{'Bucket':>8}  {'Total':>7}  {'Correct':>10}  {'Wrong CID':>10}  {'Missing':>10}")
for bucket in ["1", "2", "3", "4-5", "6+"]:
    b = len_buckets[bucket]
    t = b["total"]
    if t == 0:
        continue
    print(f"{bucket:>8}  {t:>7,}  {b['found_correct']:>6,} ({100*b['found_correct']/t:4.1f}%)  {b['found_wrong']:>6,} ({100*b['found_wrong']/t:4.1f}%)  {b['missing']:>6,} ({100*b['missing']/t:4.1f}%)")

print("\nDone.")
