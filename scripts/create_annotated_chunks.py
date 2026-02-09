#!/usr/bin/env python3
"""Create chunked text files of inline-annotated clinical notes with SNOMED labels."""

import csv
import os
import math

NOTES_PER_CHUNK = 15
OUTPUT_DIR = "outputs/annotated_chunks"

# Read all notes
notes = {}
note_ids_ordered = []
with open("data/train_notes.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        nid = row["note_id"]
        notes[nid] = row["text"]
        note_ids_ordered.append(nid)

# Read cleaned annotations with SNOMED labels
annotations = {}
with open("1st Place/data/interim/train_annotations_cln.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        nid = row["note_id"]
        if nid not in annotations:
            annotations[nid] = []
        annotations[nid].append({
            "start": int(row["start"]),
            "end": int(row["end"]),
            "span": row["source"],
            "concept_text": row["concept text"],
            "ann_type": row["annotation_type"],
        })


def inline_annotate(note_text, anns):
    """Replace each annotated span with [span text | SNOMED concept label]."""
    sorted_anns = sorted(anns, key=lambda a: a["start"], reverse=True)
    result = note_text
    for ann in sorted_anns:
        start = ann["start"]
        end = ann["end"]
        label = ann["concept_text"]
        result = result[:start] + f"[{note_text[start:end]} | {label}]" + result[end:]
    return result


num_chunks = math.ceil(len(note_ids_ordered) / NOTES_PER_CHUNK)

for chunk_idx in range(num_chunks):
    start = chunk_idx * NOTES_PER_CHUNK
    end = min(start + NOTES_PER_CHUNK, len(note_ids_ordered))
    chunk_note_ids = note_ids_ordered[start:end]

    lines = []
    for i, nid in enumerate(chunk_note_ids):
        note_text = notes[nid]
        if nid in annotations:
            annotated = inline_annotate(note_text, annotations[nid])
        else:
            annotated = note_text

        ann_count = len(annotations.get(nid, []))
        lines.append(f"{'='*80}")
        lines.append(f"NOTE {start + i + 1}/{len(note_ids_ordered)} | ID: {nid} | Annotations: {ann_count}")
        lines.append(f"{'='*80}")
        lines.append(annotated)
        lines.append("")

    chunk_text = "\n".join(lines)

    # Create directory for this chunk
    chunk_dir = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx + 1:02d}")
    os.makedirs(chunk_dir, exist_ok=True)

    filename = f"notes_{start + 1}_to_{end}.txt"
    filepath = os.path.join(chunk_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(chunk_text)

    print(f"chunk_{chunk_idx + 1:02d}/{filename}: {len(chunk_note_ids)} notes, {len(chunk_text):,} chars")

print(f"\nDone. {num_chunks} chunks written to {OUTPUT_DIR}/")
