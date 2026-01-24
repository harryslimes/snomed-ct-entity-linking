[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[<img src='https://s3.amazonaws.com/drivendata-prod-public/comp_images/snomed-ct-banner.png'>](https://www.drivendata.org/competitions/258/competition-snomed-ct)

# SNOMED CT Entity Linking Challenge

Much of the world's healthcare data is stored in free-text documents, usually clinical notes taken by doctors. One way to analyze clinical notes is to identify and label the portions of each note that correspond to specific medical concepts. This process is called **entity linking** because it involves identifying candidate spans in the unstructured text (the _entities_) and _linking_ them to a particular concept in a knowledge base of medical terminology. Medical notes are often rife with abbreviations (some of them context-dependent) and assumed knowledge. Furthermore, the target knowledge bases can easily include hundreds of thousands of concepts, many of which occur infrequently leading to a “long tail” effect in the distribution of concepts.

The objective of this competition was to **link spans of text in clinical notes with specific topics in the [SNOMED CT](https://www.snomed.org/) clinical terminology**. Participants trained models based on real-world doctor's notes which have been de-identified and annotated with SNOMED CT concepts by medically trained professionals. This is the largest publicly available dataset of labelled clinical notes, and participants in this competition were among the first to demonstrate it's potential!

## What's in this Repository

This repository contains code from winning competitors in the [SNOMED CT Entity Linking Challenge](https://www.drivendata.org/competitions/258/competition-snomed-ct) DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Public Score | Private Score | Summary of Model
--- | ---            | ---          | ---           | ---
1   | KIRIs          | 0.4452       | 0.4202        | Construct a dictionary that maps (section header, text) to concept IDs by compiling [OMOP](https://www.ohdsi.org/data-standardization/)-based synonyms, openly available medical abbreviations, and simple linguistic rules.
2   | SNOBERT        | 0.4447       | 0.4194        | Fine-tune an ensemble of BERT-based named entity recognition models to extract finding, procedures, body parts, and other entity types. Then classify spans using a pretrained embedding model to find SNOMED CT concepts with the closest embedding distance to the extracted span.
3   | MITEL-UNIUD    | 0.4065       | 0.3777        | First, use a low-rank approximation (LoRA) fine-tuned Large Language Model (LLM) to extract clinical entities from notes. Next classify the extracted spans in two stages: retrieve relevant context using the vector database [Faiss](https://github.com/facebookresearch/faiss), then classify the spans using an LLM.

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Winners Blog Post: [Meet the winners of the SNOMED CT Entity Linking Challenge](https://drivendata.co/blog/snomed-ct-entity-linking-challenge-winners)**

**Benchmark Blog Post: [SNOMED CT Entity Linking Challenge - Benchmark](https://drivendata.co/blog/snomed-ct-entity-linking-benchmark)**

## Training + Scoring (2nd Place / SNOBERT)

This repo includes the full SNOBERT training code under `2nd Place/`. Two helper scripts are provided:

- Train + export a first-stage checkpoint: `scripts/train_2nd_place.py`
- Run inference + compute macro-IoU score: `scripts/evaluate_2nd_place.py`

Example workflow (from the repo root):

```bash
# Install deps (set TORCH_CPU=false if you want GPU training)
TORCH_CPU=false ./install_requirements.sh

# Preprocess + train a single-fold model (adjust epochs as desired)
python scripts/train_2nd_place.py --split 0 --epochs 100

# Score on the training data (defaults to cutmed training files if present)
python scripts/evaluate_2nd_place.py --copy-submission
```

## Diagnostics

After you have a scored eval JSON (e.g. `outputs/2nd_place_train_eval.json`) you can generate more detailed error reports:

- Per-class IoU + SNOMED term lookup: `python scripts/per_class_iou_report.py --eval-json outputs/2nd_place_train_eval.json --data-dir data --limit 0 > outputs/per_class_iou_sorted.tsv`
- Span-level alignment (mismatches + context): `python scripts/span_alignment_report.py --eval-json outputs/2nd_place_train_eval.json --data-dir data --only-mismatches --include-snippets --include-context > outputs/2nd_place_span_alignment.tsv`
- Per-note, LLM-ready diagnosis records (JSONL): `python scripts/llm_note_diagnoser.py --entry 2nd --eval-json outputs/2nd_place_train_eval.json --output outputs/llm_note_diagnosis_2nd.jsonl`

For 1st place, start with the smoke test (the full dictionary matcher can be very slow on the full train set):

- Smoke eval JSON: `python scripts/evaluate_1st_place.py --make-smoke-test --output outputs/1st_place_smoke_eval.json`
- Span-level alignment (like `2nd_place_span_alignment_v2.tsv`): `python scripts/span_alignment_report.py --eval-json outputs/1st_place_smoke_eval.json --data-dir data --only-mismatches --include-snippets --include-context > outputs/1st_place_smoke_span_alignment_v2.tsv`

If you need more than 3 notes, run a small subset with `--note-limit` (this avoids the "full train set takes forever" trap):

- Subset eval JSON: `python scripts/evaluate_1st_place.py --notes data/train_notes.csv --annotations data/train_annotations.csv --note-limit 25`
- Subset span alignment: `python scripts/span_alignment_report.py --eval-json outputs/1st_place_train_n25_eval.json --data-dir data --only-mismatches --include-snippets --include-context > outputs/1st_place_train_n25_span_alignment_v2.tsv`

To speed up 1st-place inference, the submission code supports a few environment variables:

- `KIRI_INDEX=1` (default): enable dictionary prefiltering (big speedup); set `KIRI_INDEX=0` for the original behavior.
- `KIRI_WORKERS=4`: parallelize over notes (Linux only; uses `fork`). Start with 4–8.
- `KIRI_PROGRESS=1`: show a progress bar during submission-mode inference.

### Batch LLM post-mortems (vLLM)

To batch-generate rule-based correction guidance from an entry-specific diagnosis file (example model: MedGemma NVFP4):

- Dry-run (just writes prompts): `python scripts/vllm_batch_rules.py --entry 2nd --dry-run --note-limit 5`
- Run vLLM inference: `python scripts/vllm_batch_rules.py --entry 2nd --model medgemma-1.5-4b-it-nvfp4 --quantization modelopt_fp4`

The output JSONL contains the raw model response plus a best-effort parsed JSON object with keys like `rule_based_instructions` and `ignorance_concepts`.
This requires `vllm` + `transformers` installed; for NVFP4 checkpoints, vLLM expects `--quantization modelopt_fp4` (the script auto-selects this when the model name contains `nvfp4`).

### RTX 5090 / `sm_120` note

If you see errors like `no kernel image is available for execution on the device` (or a warning that your PyTorch build does not support `sm_120`), install a newer CUDA-enabled PyTorch wheel that includes `sm_120` support. One option on Windows is:

```bash
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
```
