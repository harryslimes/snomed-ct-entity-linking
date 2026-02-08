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

## 8) Run end-to-end notes -> L1 spans -> L2 candidates

```bash
python scripts/glinker/run_end_to_end_l1_l2.py \
  --notes-csv data/test_notes.csv \
  --l1-model-path models/gliner_finetuned_l1 \
  --exact-dict-tsv data/interim/glinker/l2_exact_dictionary.tsv \
  --es-url http://127.0.0.1:9200 \
  --es-index-name snomed_super_dict_v1 \
  --out-jsonl outputs/glinker/e2e_l1_l2_candidates.jsonl \
  --out-flat-csv outputs/glinker/e2e_l1_l2_candidates.csv
```

Exact-only mode (no ES):

```bash
python scripts/glinker/run_end_to_end_l1_l2.py \
  --notes-csv data/test_notes.csv \
  --l1-model-path models/gliner_finetuned_l1 \
  --exact-dict-tsv data/interim/glinker/l2_exact_dictionary.tsv \
  --out-jsonl outputs/glinker/e2e_l1_l2_candidates_exact_only.jsonl \
  --no-es
```

Notes:

- If L1 output already exists, pass `--l1-spans-csv ... --reuse-existing-l1-spans`.
- For long notes, try `--l1-window-chars 12000 --l1-window-overlap-chars 512`.

## 8b) Build and enable true L3 (bi-encoder) + L4 (cross-encoder)

Build L3 alias embedding index:

```bash
python scripts/glinker/build_l3_index.py \
  --super-dict-jsonl data/interim/glinker/super_dictionary_scoped.jsonl \
  --out-index-npz data/interim/glinker/l3_biencoder_index.npz \
  --backend auto \
  --model-path knowledgator/gliner-linker-large-v1.0
```

L3 retrieval search backends:

- `--l3-search-backend auto` (recommended): uses `faiss_hnsw` if `faiss` is installed, else `bruteforce`
- `--l3-search-backend faiss_hnsw|faiss_ivf|faiss_flat|bruteforce`

Build persisted FAISS ANN indexes once (recommended for CV speed):

```bash
python scripts/glinker/build_l3_ann_index.py \
  --index-npz data/interim/glinker/l3_biencoder_index.npz \
  --out-dir data/interim/glinker/l3_ann_faiss \
  --mode faiss_hnsw
```

Run end-to-end with L3 backoff on no-exact and L4 reranking:

```bash
python scripts/glinker/run_end_to_end_l1_l2.py \
  --notes-csv data/test_notes.csv \
  --l1-model-path models/gliner_finetuned_l1 \
  --exact-dict-tsv data/interim/glinker/l2_exact_dictionary.tsv \
  --es-url http://127.0.0.1:9200 \
  --es-index-name snomed_super_dict_v1 \
  --enable-l3 \
  --l3-index-npz data/interim/glinker/l3_biencoder_index.npz \
  --l3-model-path knowledgator/gliner-linker-large-v1.0 \
  --l3-search-backend auto \
  --l3-ann-index-dir data/interim/glinker/l3_ann_faiss \
  --l3-trigger no_exact \
  --enable-l4 \
  --l4-model-path knowledgator/gliner-linker-rerank-v1.0 \
  --l4-trigger ambiguous \
  --l4-min-candidates 2 \
  --resolver-route-min-top1-score none+l3+l4=0.6 \
  --out-jsonl outputs/glinker/e2e_l1_l2_l3_l4_candidates.jsonl \
  --out-resolved-csv outputs/glinker/e2e_l1_l2_l3_l4_resolved.csv \
  --out-decisions-csv outputs/glinker/e2e_l1_l2_l3_l4_decisions.csv
```

## 9) Resolve L2 candidates to one concept per span (submission-ready)

```bash
python scripts/glinker/resolve_l2_links.py \
  --candidates-jsonl outputs/glinker/e2e_l1_l2_candidates.jsonl \
  --out-resolved-csv outputs/glinker/e2e_l1_l2_resolved.csv \
  --out-decisions-csv outputs/glinker/e2e_l1_l2_decisions.csv \
  --min-top1-score-exact 0.2 \
  --min-top1-score-fuzzy 6.0 \
  --min-score-margin 0.0 \
  --max-second-to-first-ratio 1.0
```

Route-aware resolver thresholds are supported for fine-grained gating:

```bash
python scripts/glinker/resolve_l2_links.py \
  --candidates-jsonl outputs/glinker/e2e_l1_l2_l3_l4_candidates.jsonl \
  --out-resolved-csv outputs/glinker/e2e_l1_l2_l3_l4_resolved.csv \
  --route-min-top1-score none+l3+l4=0.40 \
  --route-min-top1-score exact_plus_fuzzy+l4=0.20 \
  --route-min-score-margin exact_plus_fuzzy+l4=0.02 \
  --route-max-second-to-first-ratio none+l3+l4=0.98
```

The same route override flags are available in `run_end_to_end_l1_l2.py` and `eval_cv.py`
with the `resolver-` prefix (for example `--resolver-route-min-top1-score ...`).

Current default recommendation from 5-fold CV on this branch:

- `--resolver-route-min-top1-score none+l3+l4=0.6`

You can also run this inside the orchestration command:

```bash
python scripts/glinker/run_end_to_end_l1_l2.py \
  --notes-csv data/test_notes.csv \
  --l1-model-path models/gliner_finetuned_l1 \
  --exact-dict-tsv data/interim/glinker/l2_exact_dictionary.tsv \
  --es-url http://127.0.0.1:9200 \
  --es-index-name snomed_super_dict_v1 \
  --out-jsonl outputs/glinker/e2e_l1_l2_candidates.jsonl \
  --out-resolved-csv outputs/glinker/e2e_l1_l2_resolved.csv \
  --out-decisions-csv outputs/glinker/e2e_l1_l2_decisions.csv
```

## 10) Cross-validation evaluation harness

Fast linker-focused baseline (uses gold spans as L1 input, no ES):

```bash
python scripts/glinker/eval_cv.py \
  --folds-dir /tmp/glinker_folds \
  --output-dir outputs/glinker/eval_cv_gold_noes \
  --l1-source gold \
  --exact-dict-tsv /tmp/l2_exact_dictionary.tsv \
  --no-es \
  --resolver-min-top1-score-exact 0.2 \
  --resolver-min-top1-score-fuzzy 6.0
```

Full model path (L1 model + L2):

```bash
python scripts/glinker/eval_cv.py \
  --folds-dir /tmp/glinker_folds \
  --output-dir outputs/glinker/eval_cv_model \
  --l1-source model \
  --l1-model-path models/gliner_finetuned_l1 \
  --exact-dict-tsv /tmp/l2_exact_dictionary.tsv \
  --es-url http://127.0.0.1:9200 \
  --es-index-name snomed_super_dict_v1
```

Outputs:

- `outputs/.../fold_metrics.csv`
- `outputs/.../summary.json`

For ablations, run `eval_cv.py` with:

- `L2-only`: omit `--enable-l3 --enable-l4`
- `L2+L3`: add `--enable-l3 --l3-index-npz ... --l3-model-path ...`
- `L2+L3+L4`: add both `--enable-l3 ...` and `--enable-l4 --l4-model-path ...`
  and `--resolver-route-min-top1-score none+l3+l4=0.6`
