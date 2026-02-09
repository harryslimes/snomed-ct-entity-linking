# Super-Dictionary (Athena-only bootstrap)

This is a stopgap to build a SNOMED synonym table **without** UMLS `MRCONSO.RRF`.

It uses OHDSI Athena OMOP files to:

- map **CHV** concepts → **SNOMED** concepts via `CONCEPT_RELATIONSHIP.csv` (`relationship_id = "Maps to"`)
- emit CHV `concept_name` (and **optionally** CHV `CONCEPT_SYNONYM.csv`) as terms for the mapped SNOMED concept IDs

## Requirements

Your Athena download must include at least:

- `CONCEPT.csv`
- `CONCEPT_RELATIONSHIP.csv`

If you also have:

- `CONCEPT_SYNONYM.csv`

…you’ll get more CHV terms per SNOMED concept.

## Run

```bash
python scripts/super_dictionary/build_from_athena.py \
  --athena-dir data/athena \
  --out data/interim/super_dictionary_chv_athena.tsv
```

## Athena-only SNOMED synonyms (Revised Task 1)

If you want to build the “Super-Dictionary” strictly from Athena:

```bash
python scripts/super_dictionary/build_snomed_from_athena.py \
  --athena-dir data/athena \
  --out data/interim/super_dictionary_snomed_athena.tsv \
  --include-concept-name
```

This script:

- filters `CONCEPT.csv` to `vocabulary_id = SNOMED` and `invalid_reason IS NULL`
- joins to `CONCEPT_SYNONYM.csv` on `concept_id`
- keeps **English** synonyms by default (`language_concept_id = 4180186`)

Output columns:

- `snomed_concept_id` (SNOMED concept code)
- `omop_concept_id` (Athena concept_id)
- `term`
- `source_kind` (`concept_name` or `concept_synonym`)
- `language_concept_id`

If `CONCEPT_SYNONYM.csv` isn’t in your Athena bundle, re-download with that table included.

## Full Super-Dictionary (Athena + SNOMED RF2 + training spans)

```bash
python scripts/super_dictionary/build_super_dictionary.py \
  --athena-dir data/athena \
  --snomed-dir data/SnomedCT_InternationalRF2_PRODUCTION_20260101T120000Z \
  --train-annotations data/train_annotations.csv \
  --out data/interim/super_dictionary_full.tsv \
  --include-concept-name
```

Output columns:

- `snomed_concept_id`
- `term`
- `source` (`athena`, `snomed_rf2`, `train_span`)
- `source_detail` (`concept_synonym`, `concept_name`, `FSN`, `SYN`, `annotation_span`)

You can disable any source with `--no-athena`, `--no-snomed-rf2`, or `--no-train-spans`.

## Compare KIRI default vs Super-Dictionary

1) Convert the super dictionary into KIRI’s synonym format:

```bash
python scripts/super_dictionary/export_to_kiri.py \
  --super-dict data/interim/super_dictionary_full.tsv \
  --flattened-terminology "1st Place/data/interim/flattened_terminology.csv" \
  --out "1st Place/data/interim/flattened_terminology_syn_super.csv"
```

2) Run side‑by‑side comparison (same train/test split):

```bash
python scripts/super_dictionary/compare_kiri.py \
  --kiri-data-dir "1st Place/data" \
  --default-synonyms "1st Place/data/interim/flattened_terminology_syn_snomed+omop_v5.csv" \
  --super-synonyms "1st Place/data/interim/flattened_terminology_syn_super.csv"
```

This also prints the runtime-style **macro‑averaged character IoU**.

You can score any `submission.csv` vs. ground truth directly:

```bash
python scripts/super_dictionary/runtime_scoring.py \
  path/to/submission.csv \
  path/to/ground_truth.csv
```

If `train_annotations_cln.csv` is missing, generate it:

```bash
cd "1st Place"
python src/process_data.py make-clean-annotations
```

## Linguistic Rules

The engine includes:

- Coordination splitting (e.g., “fracture of left and right femur”)
- Optional dependency‑parser coordination (spaCy, if installed)
- Abbreviation expansion + simple permutations (e.g., “L femur fx”)
- Stopword-transparent matching (e.g., “fracture femur” matches “fracture of the femur”)

### Optional spaCy setup

To enable dependency‑parser coordination:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Tests

Run unit tests (requires Athena files in `data/athena/`):

```bash
python -m unittest tests/test_super_dictionary.py
```

## Output

TSV with header:

- `snomed_concept_id` (the SNOMED CT concept id, from `CONCEPT.concept_code`)
- `term`
- `source_vocab` (default `CHV`)
- `source_kind` (`concept_name` or `concept_synonym`)

## Limitations vs UMLS MRCONSO

Athena does **not** include UMLS-specific MRCONSO columns (like `TTY`), so you can’t exactly filter down to UMLS “clinician-entered” CHV term types until you ingest `MRCONSO.RRF`.
