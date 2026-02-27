#!/usr/bin/env python3
"""Generate inline-annotated version of the first note."""
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TERMINOLOGY_CSV = REPO_ROOT / "3rd Place" / "assets" / "dataflattened_terminology.csv"

# Load data
notes_df = pd.read_csv(REPO_ROOT / "data" / "old-challenge-split" / "train_notes.csv")
ann_df = pd.read_csv(REPO_ROOT / "data" / "old-challenge-split" / "train_annotations.csv")

# Get first note
first_note_id = notes_df.iloc[0]["note_id"]
note_text = notes_df.iloc[0]["text"]
note_anns = ann_df[ann_df["note_id"] == first_note_id].copy()
note_anns = note_anns.sort_values("start", ascending=False)

# Load concept info
ft = pd.read_csv(TERMINOLOGY_CSV)
concept_info = {
    int(row.concept_id): (row.concept_name, row.hierarchy)
    for row in ft.itertuples()
}

# Build inline-annotated version
annotated = note_text
for _, row in note_anns.iterrows():
    s, e = int(row["start"]), int(row["end"])
    cid = int(row["concept_id"])
    ann_id = int(row["annotation_id"])
    span_text = note_text[s:e]

    cname, hierarchy = concept_info.get(cid, ("Unknown", "unknown"))
    tag = f"[{span_text} | {cname}]{{id={ann_id}}}"
    annotated = annotated[:s] + tag + annotated[e:]

# Save to file
with open("inline_annotated_note.txt", "w") as f:
    f.write(annotated)

print(f"Generated inline-annotated note with {len(note_anns)} annotations")
print("Saved to inline_annotated_note.txt")