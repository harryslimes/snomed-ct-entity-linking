"""
Analyze 2nd Place SNOMED CT Entity Linking solution performance
broken down by entity class (find/proc/body) and concept frequency.
"""
import json
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_2ND = os.path.join(BASE, "2nd Place")

# ── 1. Load sctid_syn files to build concept_id -> class mapping ──
print("=" * 80)
print("STEP 1: Building concept_id -> class mapping from sctid_syn files")
print("=" * 80)

concept_to_class = {}
class_counts = {}
for cls_name in ["find", "proc", "body"]:
    fpath = os.path.join(DATA_2ND, "data", "preprocess_data", f"{cls_name}_sctid_syn.json")
    with open(fpath) as f:
        d = json.load(f)
    count = 0
    for sctid in d.keys():
        concept_to_class[sctid] = cls_name
        count += 1
    class_counts[cls_name] = count
    print(f"  {cls_name}: {count:,} unique concept IDs")

print(f"  Total mapped concept IDs: {len(concept_to_class):,}")

# ── 2. Load full training annotations and add class labels ──
print("\n" + "=" * 80)
print("STEP 2: Loading full training annotations + class labels")
print("=" * 80)

full_ann = pd.read_csv(os.path.join(DATA_2ND, "data", "competition_data", "cutmed_fixed_train_annotations.csv"))
print(f"  Full training annotations: {len(full_ann):,} rows")
print(f"  Columns: {list(full_ann.columns)}")

# Add class labels
full_ann["concept_id_str"] = full_ann["concept_id"].astype(str)
full_ann["cls"] = full_ann["concept_id_str"].map(concept_to_class)

cls_dist = full_ann["cls"].value_counts(dropna=False)
print(f"\n  Class distribution in full training data:")
for c, n in cls_dist.items():
    print(f"    {c}: {n:,} ({100*n/len(full_ann):.1f}%)")

# ── 3. Compute concept frequency from TRAINING split ──
print("\n" + "=" * 80)
print("STEP 3: Computing concept frequency from train_ann_split_0")
print("=" * 80)

train_split = pd.read_csv(os.path.join(DATA_2ND, "data", "preprocess_data", "splits", "train_ann_split_0.csv"))
print(f"  Training split annotations: {len(train_split):,} rows")

train_split["concept_id_str"] = train_split["concept_id"].astype(str)
concept_freq = train_split["concept_id_str"].value_counts()
print(f"  Unique concepts in training split: {len(concept_freq):,}")
print(f"  Frequency stats:")
print(f"    Mean: {concept_freq.mean():.1f}")
print(f"    Median: {concept_freq.median():.1f}")
print(f"    Min: {concept_freq.min()}")
print(f"    Max: {concept_freq.max()}")

# Define frequency buckets
def freq_bucket(count):
    if count == 0:
        return "unseen (0)"
    elif count == 1:
        return "rare (1)"
    elif count <= 5:
        return "low (2-5)"
    elif count <= 20:
        return "medium (6-20)"
    elif count <= 100:
        return "high (21-100)"
    else:
        return "very_high (>100)"

BUCKET_ORDER = ["unseen (0)", "rare (1)", "low (2-5)", "medium (6-20)", "high (21-100)", "very_high (>100)"]

# ── 4. Load validation annotations and submission ──
print("\n" + "=" * 80)
print("STEP 4: Loading validation annotations and submission")
print("=" * 80)

val_ann = pd.read_csv(os.path.join(DATA_2ND, "data", "preprocess_data", "splits", "val_ann_split_0.csv"))
print(f"  Validation annotations: {len(val_ann):,} rows")

submission = pd.read_csv(os.path.join(DATA_2ND, "submission.csv"))
print(f"  Submission predictions: {len(submission):,} rows")
print(f"  Submission columns: {list(submission.columns)}")

# Normalize types
val_ann["concept_id_str"] = val_ann["concept_id"].astype(str)
val_ann["start"] = val_ann["start"].astype(float)
val_ann["end"] = val_ann["end"].astype(float)
val_ann["note_id"] = val_ann["note_id"].astype(str)

submission["concept_id_str"] = submission["concept_id"].astype(str)
submission["start"] = submission["start"].astype(float)
submission["end"] = submission["end"].astype(float)
submission["note_id"] = submission["note_id"].astype(str)

# Add class and frequency info to val annotations
val_ann["cls"] = val_ann["concept_id_str"].map(concept_to_class)

# Map frequency from training split
val_ann["train_freq"] = val_ann["concept_id_str"].map(concept_freq).fillna(0).astype(int)
val_ann["freq_bucket"] = val_ann["train_freq"].apply(freq_bucket)

print(f"\n  Validation class distribution:")
vc = val_ann["cls"].value_counts(dropna=False)
for c, n in vc.items():
    print(f"    {c}: {n:,}")

print(f"\n  Validation frequency bucket distribution:")
fb = val_ann["freq_bucket"].value_counts()
for b in BUCKET_ORDER:
    if b in fb.index:
        print(f"    {b}: {fb[b]:,}")

# ── 5. Match predictions to validation annotations ──
print("\n" + "=" * 80)
print("STEP 5: Matching predictions to validation annotations")
print("=" * 80)

# Build a lookup: for each note_id, store list of (start, end, concept_id) from submission
from collections import defaultdict

pred_by_note = defaultdict(list)
for _, row in submission.iterrows():
    pred_by_note[row["note_id"]].append((row["start"], row["end"], row["concept_id_str"]))

print(f"  Notes with predictions: {len(pred_by_note):,}")
val_note_ids = set(val_ann["note_id"].unique())
print(f"  Notes in validation: {len(val_note_ids):,}")
common_notes = val_note_ids.intersection(set(pred_by_note.keys()))
print(f"  Common notes: {len(common_notes):,}")

# For each validation annotation, check if there's a matching prediction
# Match = same note_id, overlapping span, same concept_id
def check_match(val_row, preds):
    """Check if any prediction matches this validation annotation."""
    v_start, v_end = val_row["start"], val_row["end"]
    v_concept = val_row["concept_id_str"]

    for p_start, p_end, p_concept in preds:
        # Check concept match
        if p_concept != v_concept:
            continue
        # Check span overlap
        if p_start < v_end and p_end > v_start:
            return True
    return False

matches = []
for idx, row in val_ann.iterrows():
    note_id = row["note_id"]
    preds = pred_by_note.get(note_id, [])
    matched = check_match(row, preds)
    matches.append(matched)

val_ann["matched"] = matches
total_matched = sum(matches)
print(f"\n  Total validation annotations: {len(val_ann):,}")
print(f"  Matched: {total_matched:,} ({100*total_matched/len(val_ann):.1f}%)")
print(f"  Unmatched: {len(val_ann)-total_matched:,} ({100*(len(val_ann)-total_matched)/len(val_ann):.1f}%)")

# ── 6. Breakdown by entity class ──
print("\n" + "=" * 80)
print("RESULTS: Match Rate by Entity Class")
print("=" * 80)

print(f"\n{'Class':<12} {'Total':>8} {'Matched':>8} {'Match Rate':>12}")
print("-" * 44)
for cls in ["find", "proc", "body", None]:
    if cls is None:
        mask = val_ann["cls"].isna()
        label = "unknown"
    else:
        mask = val_ann["cls"] == cls
        label = cls
    subset = val_ann[mask]
    if len(subset) == 0:
        continue
    n_matched = subset["matched"].sum()
    rate = 100 * n_matched / len(subset)
    print(f"{label:<12} {len(subset):>8,} {n_matched:>8,} {rate:>11.1f}%")

# Total row
print("-" * 44)
print(f"{'TOTAL':<12} {len(val_ann):>8,} {total_matched:>8,} {100*total_matched/len(val_ann):>11.1f}%")

# ── 7. Breakdown by frequency bucket ──
print("\n" + "=" * 80)
print("RESULTS: Match Rate by Frequency Bucket")
print("=" * 80)

print(f"\n{'Freq Bucket':<18} {'Total':>8} {'Matched':>8} {'Match Rate':>12}")
print("-" * 50)
for bucket in BUCKET_ORDER:
    mask = val_ann["freq_bucket"] == bucket
    subset = val_ann[mask]
    if len(subset) == 0:
        continue
    n_matched = subset["matched"].sum()
    rate = 100 * n_matched / len(subset)
    print(f"{bucket:<18} {len(subset):>8,} {n_matched:>8,} {rate:>11.1f}%")

print("-" * 50)
print(f"{'TOTAL':<18} {len(val_ann):>8,} {total_matched:>8,} {100*total_matched/len(val_ann):>11.1f}%")

# ── 8. Breakdown by frequency bucket AND entity class (combined) ──
print("\n" + "=" * 80)
print("RESULTS: Match Rate by Frequency Bucket x Entity Class")
print("=" * 80)

classes = ["find", "proc", "body"]

# Header
header = f"{'Freq Bucket':<18}"
for cls in classes:
    header += f" | {cls:>18}"
print(f"\n{header}")
print("-" * (18 + 3 * 21))

for bucket in BUCKET_ORDER:
    row_str = f"{bucket:<18}"
    for cls in classes:
        mask = (val_ann["freq_bucket"] == bucket) & (val_ann["cls"] == cls)
        subset = val_ann[mask]
        if len(subset) == 0:
            row_str += f" | {'--':>18}"
        else:
            n_matched = subset["matched"].sum()
            rate = 100 * n_matched / len(subset)
            cell = f"{n_matched}/{len(subset)} ({rate:.1f}%)"
            row_str += f" | {cell:>18}"
    print(row_str)

# Total row per class
print("-" * (18 + 3 * 21))
row_str = f"{'TOTAL':<18}"
for cls in classes:
    mask = val_ann["cls"] == cls
    subset = val_ann[mask]
    if len(subset) == 0:
        row_str += f" | {'--':>18}"
    else:
        n_matched = subset["matched"].sum()
        rate = 100 * n_matched / len(subset)
        cell = f"{n_matched}/{len(subset)} ({rate:.1f}%)"
        row_str += f" | {cell:>18}"
print(row_str)

# ── 9. Additional: top missed concepts ──
print("\n" + "=" * 80)
print("ANALYSIS: Top 15 Most Frequently Missed Concepts (in validation)")
print("=" * 80)

missed = val_ann[~val_ann["matched"]]
missed_concept_counts = missed.groupby(["concept_id_str", "cls"]).agg(
    miss_count=("matched", "size"),
    span_examples=("span", lambda x: list(x.head(3)))
).reset_index()
missed_concept_counts = missed_concept_counts.sort_values("miss_count", ascending=False).head(15)

print(f"\n{'Concept ID':<15} {'Class':<6} {'Misses':>7} {'Train Freq':>11}  Sample Spans")
print("-" * 90)
for _, row in missed_concept_counts.iterrows():
    tf = concept_freq.get(row["concept_id_str"], 0)
    spans = "; ".join(str(s)[:25] for s in row["span_examples"])
    print(f"{row['concept_id_str']:<15} {str(row['cls']):<6} {row['miss_count']:>7} {tf:>11}  {spans}")

# ── 10. Summary statistics ──
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

# Check how many val concepts are unseen in training
unseen_mask = val_ann["train_freq"] == 0
print(f"\n  Validation annotations with unseen concepts: {unseen_mask.sum():,} / {len(val_ann):,} ({100*unseen_mask.sum()/len(val_ann):.1f}%)")
print(f"  Match rate for unseen concepts: {100*val_ann[unseen_mask]['matched'].mean():.1f}%")
seen_mask = val_ann["train_freq"] > 0
print(f"  Match rate for seen concepts: {100*val_ann[seen_mask]['matched'].mean():.1f}%")

print(f"\n  Overall match rate: {100*val_ann['matched'].mean():.1f}%")
print(f"\n  Class match rates:")
for cls in ["find", "proc", "body"]:
    mask = val_ann["cls"] == cls
    if mask.sum() > 0:
        print(f"    {cls}: {100*val_ann[mask]['matched'].mean():.1f}%")

print()
