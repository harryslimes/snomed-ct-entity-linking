# GLinker Competition Runbook (PR1 Foundation)

This runbook covers the PR1 foundation artifacts:

- terminology/release validation manifest
- allowed concept universe by SNOMED ancestry (+ train safety override)
- scoped super-dictionary JSONL
- leakage-safe folds and fold-local train-span aliases

## 1) Validate terminology coverage

```bash
python scripts/glinker/validate_terminology_release.py \
  --snomed-dir data/SnomedCT_InternationalRF2_PRODUCTION_20260101T120000Z \
  --athena-dir data/athena \
  --train-annotations data/train_annotations.csv \
  --out outputs/glinker/manifests/data_manifest.json \
  --min-train-coverage 0.99
```

## 2) Build allowed concept universe

```bash
python scripts/glinker/build_allowed_concepts.py \
  --snomed-dir data/SnomedCT_InternationalRF2_PRODUCTION_20260101T120000Z \
  --train-annotations data/train_annotations.csv \
  --out-parquet data/interim/glinker/allowed_concepts.parquet \
  --out-csv data/interim/glinker/allowed_concepts.csv
```

If parquet dependencies are unavailable, the script still writes CSV. Downstream
steps auto-fallback from `.parquet` to `.csv`.

## 3) Build scoped super dictionary JSONL

```bash
python scripts/glinker/build_super_dictionary_jsonl.py \
  --super-dict-tsv data/interim/super_dictionary_full.tsv \
  --allowed-concepts data/interim/glinker/allowed_concepts.parquet \
  --out-jsonl data/interim/glinker/super_dictionary_scoped.jsonl \
  --max-aliases-per-concept 256
```

## 4) Create strict leakage-safe folds (+ final full-train mode)

```bash
python scripts/glinker/make_folds.py \
  --notes-csv data/train_notes.csv \
  --annotations-csv data/train_annotations.csv \
  --out-dir data/interim/glinker/folds \
  --n-folds 5 \
  --seed 1337 \
  --write-full-train
```

Outputs:

- `data/interim/glinker/folds/fold_XX/train_notes.csv`
- `data/interim/glinker/folds/fold_XX/val_notes.csv`
- `data/interim/glinker/folds/fold_XX/train_annotations.csv`
- `data/interim/glinker/folds/fold_XX/val_annotations.csv`
- `data/interim/glinker/folds/fold_XX/train_span_aliases.tsv` (train-only)
- `data/interim/glinker/folds/full_train/train_*.csv` (final submission mode)

## 5) Build L2 exact dictionary + Elasticsearch fuzzy index

```bash
python scripts/glinker/build_es_index.py \
  --super-dict-jsonl data/interim/glinker/super_dictionary_scoped.jsonl \
  --exact-out-tsv data/interim/glinker/l2_exact_dictionary.tsv \
  --es-url http://127.0.0.1:9200 \
  --es-index-name snomed_super_dict_v1 \
  --es-index-body configs/es_index.json \
  --recreate-index
```

If you only want exact lookup artifacts and no ES calls:

```bash
python scripts/glinker/build_es_index.py \
  --super-dict-jsonl data/interim/glinker/super_dictionary_scoped.jsonl \
  --exact-out-tsv data/interim/glinker/l2_exact_dictionary.tsv \
  --no-es
```

## 6) Run hybrid L2 candidate generation (exact + ES fallback)

Input is a mentions CSV with at least:

- `mention_id`
- `mention`
- optional `l1_type` (`finding`, `procedure`, `body_structure`)

```bash
python scripts/glinker/run_l2_candidates.py \
  --mentions-csv data/interim/glinker/l2_mentions.csv \
  --mention-id-col mention_id \
  --mention-col mention \
  --l1-type-col l1_type \
  --exact-dict-tsv data/interim/glinker/l2_exact_dictionary.tsv \
  --es-url http://127.0.0.1:9200 \
  --es-index-name snomed_super_dict_v1 \
  --top-k-final 50 \
  --out-jsonl outputs/glinker/l2_candidates.jsonl \
  --out-flat-csv outputs/glinker/l2_candidates.csv
```

For exact-only mode:

```bash
python scripts/glinker/run_l2_candidates.py \
  --mentions-csv data/interim/glinker/l2_mentions.csv \
  --exact-dict-tsv data/interim/glinker/l2_exact_dictionary.tsv \
  --out-jsonl outputs/glinker/l2_candidates_exact_only.jsonl \
  --no-es
```

## 7) Run direct L1->L2 candidate pipeline

Input is an L1 spans CSV with:

- `note_id`
- `start_char`
- `end_char`
- optional `mention` (if missing, provide `--notes-csv` to slice from text)
- optional `l1_type`

```bash
python scripts/glinker/run_l1_l2_pipeline.py \
  --l1-spans-csv outputs/glinker/l1_spans.csv \
  --notes-csv data/test_notes.csv \
  --exact-dict-tsv data/interim/glinker/l2_exact_dictionary.tsv \
  --es-url http://127.0.0.1:9200 \
  --es-index-name snomed_super_dict_v1 \
  --out-jsonl outputs/glinker/l1_l2_candidates.jsonl \
  --out-flat-csv outputs/glinker/l1_l2_candidates.csv
```

For exact-only mode:

```bash
python scripts/glinker/run_l1_l2_pipeline.py \
  --l1-spans-csv outputs/glinker/l1_spans.csv \
  --exact-dict-tsv data/interim/glinker/l2_exact_dictionary.tsv \
  --out-jsonl outputs/glinker/l1_l2_candidates_exact_only.jsonl \
  --no-es
```
