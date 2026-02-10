#!/usr/bin/env python3
"""Estimate token count for inline-annotated clinical notes with SNOMED labels."""

import csv

# Read all notes
notes = {}
with open('data/train_notes.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        notes[row['note_id']] = row['text']

print(f"Total notes: {len(notes)}")
total_chars_raw = sum(len(t) for t in notes.values())
print(f"Total raw note characters: {total_chars_raw:,}")
print(f"Avg chars per note: {total_chars_raw / len(notes):,.0f}")

# Read cleaned annotations (with concept text / SNOMED labels)
annotations = {}
ann_count = 0
type_counts = {}
with open('1st Place/data/interim/train_annotations_cln.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        nid = row['note_id']
        if nid not in annotations:
            annotations[nid] = []
        annotations[nid].append({
            'start': int(row['start']),
            'end': int(row['end']),
            'span': row['source'],
            'concept_text': row['concept text'],
            'ann_type': row['annotation_type']
        })
        ann_count += 1
        atype = row['annotation_type']
        type_counts[atype] = type_counts.get(atype, 0) + 1

print(f"\nTotal annotations: {ann_count}")
print(f"Annotation types: {type_counts}")
print(f"Notes with annotations: {len(annotations)}")
print(f"Avg annotations per annotated note: {ann_count / len(annotations):.1f}")


def inline_annotate(note_text, anns):
    """Insert SNOMED labels inline after each annotated span."""
    sorted_anns = sorted(anns, key=lambda a: a['start'], reverse=True)
    result = note_text
    for ann in sorted_anns:
        start = ann['start']
        end = ann['end']
        label = ann['concept_text']
        result = result[:end] + f" [SNOMED: {label}]" + result[end:]
    return result


# Calculate for different annotation subsets
def calc_stats(ann_filter_fn, label):
    filtered = {}
    count = 0
    for nid, anns in annotations.items():
        subset = [a for a in anns if ann_filter_fn(a)]
        if subset:
            filtered[nid] = subset
            count += len(subset)

    total_chars = 0
    for nid, note_text in notes.items():
        if nid in filtered:
            annotated = inline_annotate(note_text, filtered[nid])
        else:
            annotated = note_text
        total_chars += len(annotated)

    overhead = total_chars - total_chars_raw
    print(f"\n--- {label} ---")
    print(f"  Annotations used: {count:,}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Overhead from labels: {overhead:,} chars")
    print(f"  Token estimate (3.5 c/t): {total_chars / 3.5:,.0f}")
    print(f"  Token estimate (4.0 c/t): {total_chars / 4.0:,.0f}")
    return total_chars


# All annotations
calc_stats(lambda a: True, "All annotations (train+test+proposed)")

# Train only
calc_stats(lambda a: a['ann_type'] == 'train', "Train annotations only")

# Train + proposed_ACCEPTED
calc_stats(lambda a: a['ann_type'] in ('train', 'proposed_ACCEPTED'),
           "Train + proposed_ACCEPTED")

# Test only (the golden evaluation set)
calc_stats(lambda a: a['ann_type'] == 'test', "Test annotations only")

# Show a sample annotated note
sample_nid = '10043750-DS-6'
if sample_nid in notes and sample_nid in annotations:
    sample = inline_annotate(notes[sample_nid], annotations[sample_nid])
    print(f"\n=== SAMPLE ANNOTATED NOTE ({sample_nid}) first 3000 chars ===")
    print(sample[:3000])
    print(f"\n=== Sample stats ===")
    print(f"  Original chars: {len(notes[sample_nid]):,}")
    print(f"  Annotated chars: {len(sample):,}")
    print(f"  Annotations: {len(annotations[sample_nid])}")

# Per-note size distribution
print(f"\n=== Per-note annotated size distribution ===")
sizes = []
for nid, note_text in notes.items():
    if nid in annotations:
        annotated = inline_annotate(note_text, annotations[nid])
    else:
        annotated = note_text
    sizes.append(len(annotated))
sizes.sort()
print(f"  Min: {sizes[0]:,} chars")
print(f"  Median: {sizes[len(sizes)//2]:,} chars")
print(f"  Mean: {sum(sizes)/len(sizes):,.0f} chars")
print(f"  Max: {sizes[-1]:,} chars")
print(f"  P90: {sizes[int(len(sizes)*0.9)]:,} chars")
print(f"  P95: {sizes[int(len(sizes)*0.95)]:,} chars")

# How many notes fit in different context windows?
for window_tokens in [128000, 200000, 1000000]:
    window_chars = window_tokens * 3.5  # conservative
    cumulative = 0
    fit_count = 0
    for s in sorted(sizes):
        if cumulative + s <= window_chars:
            cumulative += s
            fit_count += 1
        else:
            break
    print(f"  Notes fitting in {window_tokens:,} token window: {fit_count}/{len(sizes)}")
