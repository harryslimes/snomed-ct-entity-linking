# LLM Postprocess Agent (vLLM / SGLang)

This repo includes a constrained “edit-script” agent for post-processing KIRI `*_pred.csv` outputs:

- delete spans (highest ROI for macro-char IoU)
- shift span boundaries (bounded)

Implementation lives in `scripts/llm_postprocess/`.

## Install

The agent can run against:

- a vLLM OpenAI-compatible endpoint (recommended on DGX Spark for FP8),
- or SGLang (optional fallback).

Local Python dependency:

```bash
pip install -r scripts/llm_postprocess/requirements.txt
```

For vLLM endpoint mode, start `vllm serve` separately and pass `--endpoint-url`.

## Keep Model Loaded (Fast Iteration)

To iterate quickly, run SGLang as a persistent server (loads weights once), then call the agent against the endpoint:

```bash
scripts/llm_postprocess/serve_sglang_fp8.sh Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

## Run on a prediction file

Example (recommended: use an existing vLLM endpoint):

```bash
python scripts/llm_postprocess/run_agent.py \
  --pred-csv outputs/super_dictionary/full_train_20260205/kiri_super_fulltrain_base_pred.csv \
  --notes-csv "1st Place/data/raw/mimic-iv_notes_training_set.csv" \
  --prompt scripts/llm_postprocess/prompts/base.md \
  --backend vllm \
  --endpoint-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --out-jsonl outputs/llm_postprocess/scripts.jsonl \
  --out-pred-csv outputs/llm_postprocess/pred_edited.csv
```

Delete-only mode (recommended while iterating on prompts):

```bash
python scripts/llm_postprocess/run_agent.py \
  --pred-csv outputs/super_dictionary/full_train_20260205/kiri_super_fulltrain_base_pred.csv \
  --notes-csv "1st Place/data/raw/mimic-iv_notes_training_set.csv" \
  --prompt scripts/llm_postprocess/prompts/base.md \
  --delete-only \
  --max-edits 5 \
  --backend sglang \
  --endpoint-url http://127.0.0.1:30000 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --out-jsonl outputs/llm_postprocess/scripts.jsonl \
  --out-pred-csv outputs/llm_postprocess/pred_edited.csv
```

Optional SGLang mode (in-process runtime launch):

```bash
python scripts/llm_postprocess/run_agent.py \
  --pred-csv ... \
  --notes-csv ... \
  --prompt scripts/llm_postprocess/prompts/base.md \
  --backend sglang \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --tp-size 8 \
  --out-jsonl outputs/llm_postprocess/scripts.jsonl \
  --out-pred-csv outputs/llm_postprocess/pred_edited.csv
```

## Apply scripts separately

If you only want to apply a previously generated `scripts.jsonl`:

```bash
python scripts/llm_postprocess/apply_edits.py \
  --pred-csv input_pred.csv \
  --scripts-jsonl outputs/llm_postprocess/scripts.jsonl \
  --notes-csv "1st Place/data/raw/mimic-iv_notes_training_set.csv" \
  --out-csv output_pred.csv
```

## Prompt evaluation

Evaluate all `*.md` prompts in a folder on a random validation subset:

```bash
python scripts/llm_postprocess/optimize_prompt.py \
  --pred-csv outputs/super_dictionary/full_train_20260205/kiri_super_fulltrain_base_pred.csv \
  --prompt-dir scripts/llm_postprocess/prompts \
  --backend vllm \
  --endpoint-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --val-size 64 \
  --out-dir outputs/llm_postprocess_prompt_eval
```
