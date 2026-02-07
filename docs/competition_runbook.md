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
