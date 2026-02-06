You are improving a noisy span-annotation output for a clinical note.

Your goal is to improve **macro-averaged per-class character IoU**. This metric is very sensitive to **false-positive concept IDs**.

You are given:
- `note_id`
- `note_len`
- a list of predicted spans (with stable indices)

You must return an **edit script** in JSON that only:
- deletes an existing predicted span.

Notes about indices:
- The `idx` values are stable indices into the model's full prediction list for the note.
- `idx` values may be non-contiguous (some spans may be omitted from the prompt).

Hard rules:
- You MUST NOT add new concept IDs.
- You MUST NOT create new spans.
- You MUST NOT output more than {{MAX_EDITS}} edits.
- Prefer returning **no edits** over deleting something that might be correct.

Delete guidance (high-confidence spurious patterns):
- Template/section labels and structural tokens: `Labs`, `vs`, `vitals`, `ADMISSION LABS`, `Pertinent Results`.
- Lab/vital abbreviations that are usually not annotated as entities: `Plt/PLT`, `WBC`, `RBC`, `HGB`, `HCT`, `MCV`, `MCH`, `MCHC`, `RDW`, `UREA N`, single-letter labs like `K`, and vital abbreviations like `BP`.
- Single letters / roman numerals / unit fragments: `T`, `L`, `II` when they appear as formatting/measurement artifacts.
- Generic exam adjectives when used as boilerplate (e.g., `Clear` in \"lungs clear\"), not as a diagnosis.

Do NOT delete when the mention is plausibly a real clinical entity:
- Allergies (e.g., `No Known Allergies`) in an Allergies section.
- Concrete diagnoses, symptoms, medications, procedures, or devices in the HPI / Assessment & Plan / Problem List.
- Mentions that are part of a more specific phrase in context (e.g., `pain` inside \"chest pain\").

Edits JSON schema:
{
  "note_id": "<string>",
  "edits": [
    {"op": "delete", "idx": <int>}
  ]
}

Return ONLY the JSON. No commentary.

note_id: {{NOTE_ID}}
note_len: {{NOTE_LEN}}

predictions (stable indices):
{{PRED_ROWS_JSON}}
