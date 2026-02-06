# LLM Postprocess Agent Handoff (SGLang + FP8)

This document hands off the current state of the `scripts/llm_postprocess/` “edit-script” postprocess agent and the fast-iteration workflow on DGX Spark / Blackwell (GB10).

## Goal

Improve macro-averaged per-class character IoU by **removing spurious predicted spans** from KIRI `*_pred.csv` outputs.

Current focus is **delete-only** (no boundary shifting).

## Environment Constraints (Important)

- Use **system Python** (`/usr/bin/python3`). Do **not** create a venv or install a generic PyTorch wheel.
- Base container is NVIDIA PyTorch (ARM/aarch64) with working CUDA/PyTorch already present.
- Fast iteration requires a **persistent inference server** so the model stays loaded (cross-request prefix caching, no reload).

## Key Files

- Agent runner: `scripts/llm_postprocess/run_agent.py`
- Prompt template: `scripts/llm_postprocess/prompts/base.md`
- SGLang/vLLM backends: `scripts/llm_postprocess/sglang_runner.py`
- Start persistent server: `scripts/llm_postprocess/serve_sglang_fp8.sh`
- Oracle (uses gold): `scripts/llm_postprocess/oracle_delete.py`
- Apply scripts: `scripts/llm_postprocess/apply_edits.py`
- Prompt building utilities: `scripts/llm_postprocess/prompting.py`

## Persistent Server (Keep Model Loaded)

Start SGLang server in a dedicated terminal and leave it running:

```bash
cd /workspaces/snomed-ct-entity-linking
HOST=0.0.0.0 PORT=30000 scripts/llm_postprocess/serve_sglang_fp8.sh Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
```

Notes:
- Server will bind `http://127.0.0.1:30000`.
- When the server is up, the log line includes `The server is fired up and ready to roll!`.
- Cross-request caching only exists while this server stays alive.

## Run Agent (Delete-Only)

Known-good fast iteration invocation (10-note subset):

```bash
cd /workspaces/snomed-ct-entity-linking
PYTHONNOUSERSITE=1 PYTHONPATH=. /usr/bin/python3 -m scripts.llm_postprocess.run_agent \
  --pred-csv outputs/super_dictionary/kiri_default_pred.csv \
  --notes-csv outputs/runs/1st_place_20260127T151333Z/notes.csv \
  --prompt scripts/llm_postprocess/prompts/base.md \
  --backend sglang \
  --endpoint-url http://127.0.0.1:30000 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --delete-only \
  --max-edits 5 \
  --max-new-tokens 192 \
  --limit 10 \
  --batch-size 1 \
  --context-chars 48 \
  --max-spans 25 \
  --request-timeout-s 600 \
  --out-jsonl outputs/llm_postprocess/run10_deleteonly_max5_scripts.jsonl \
  --out-pred-csv outputs/llm_postprocess/run10_deleteonly_max5_pred.csv \
  --cache-dir outputs/llm_postprocess_cache/run10_deleteonly_max5
```

Important knobs:
- `--max-edits`: strongly affects quality; `5` worked better than `10` on the 10-note subset.
- `--max-new-tokens`: keep small so the model can’t ramble (paired with JSON schema `maxItems`).
- `--max-spans` / `--context-chars`: control prompt size. Large values slow down and increase truncation risk.

Optional:
- `--dry-run` produces prompt previews (truncated to 2000 chars in the JSONL record).
- `--suspicious-only` filters the prompt rows down to spans that match heuristics (see `scripts/llm_postprocess/prompting.py`). This did not consistently help quality; treat as an iteration-speed tool, not a quality tool.

## Scoring (Subset Macro IoU)

Compute macro per-class char IoU for a subset with the existing scorer:

```bash
cd /workspaces/snomed-ct-entity-linking
PYTHONNOUSERSITE=1 /usr/bin/python3 - <<'PY'
import json, sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, '1st Place/src')
from scoring import iou_per_class

scripts = Path('outputs/llm_postprocess/run10_deleteonly_max5_scripts.jsonl')
subset = {json.loads(l)['note_id'] for l in scripts.read_text(encoding='utf-8').splitlines() if l.strip()}

base = pd.read_csv('outputs/super_dictionary/kiri_default_pred.csv')
post = pd.read_csv('outputs/llm_postprocess/run10_deleteonly_max5_pred.csv')
gold = pd.read_csv('outputs/runs/1st_place_20260127T151333Z/gold.csv')

def prep(df):
    out = df[df['note_id'].astype(str).isin(subset)][['note_id','start','end','concept_id']].copy()
    out['note_id'] = out['note_id'].astype(str)
    out['start'] = pd.to_numeric(out['start'], errors='coerce').fillna(0).astype(int)
    out['end'] = pd.to_numeric(out['end'], errors='coerce').fillna(0).astype(int)
    out['concept_id'] = pd.to_numeric(out['concept_id'], errors='coerce').fillna(0).astype(np.int64)
    out = out[out['end'] > out['start']]
    return out

b, p, g = prep(base), prep(post), prep(gold)
bs = float(np.mean(iou_per_class(b, g)))
ps = float(np.mean(iou_per_class(p, g)))
print('base_macro_iou', f'{bs:.6f}')
print('post_macro_iou', f'{ps:.6f}')
print('abs_delta', f'{(ps-bs):+.6f}')
PY
```

## Oracle Deletes (Upper Bound, Uses Gold)

`oracle_delete.py` generates a delete-only script per note intended to maximize per-note macro IoU (greedy search with bounded candidates).

Example (10-note subset, `max-edits=5`):

```bash
cd /workspaces/snomed-ct-entity-linking
PYTHONNOUSERSITE=1 /usr/bin/python3 -m scripts.llm_postprocess.oracle_delete \
  --pred-csv outputs/super_dictionary/kiri_default_pred.csv \
  --gold-csv outputs/runs/1st_place_20260127T151333Z/gold.csv \
  --notes-csv outputs/runs/1st_place_20260127T151333Z/notes.csv \
  --limit 10 \
  --max-edits 5 \
  --out-jsonl outputs/llm_postprocess/oracle_run10_max5.jsonl \
  --out-pred-csv outputs/llm_postprocess/oracle_run10_max5_pred.csv
```

Observed on the same 10-note subset:
- Baseline macro IoU: `0.491132`
- Oracle macro IoU: `0.508069`
- Delta: `+0.016938`

This is the “headroom” the prompt/agent should try to approximate on small subsets.

## Prompt / Agent Behavior

- The model is not “self-prompting”. It fills a template `scripts/llm_postprocess/prompts/base.md`.
- JSON output is constrained via schema:
  - `--delete-only` uses a delete-only schema (`maxItems` = `--max-edits`).
  - The agent also truncates parsed edits to `--max-edits`.
- Prompt defaults were tightened to reduce context explosion:
  - `--context-chars` default `64`
  - `--max-spans` default `80`

## Known Failure Modes

- If you start the agent without a persistent server, it may launch an in-process runtime and take minutes to load weights per run.
- Over-long prompts can exceed server context limits; keep `--max-spans`, `--context-chars`, `--max-new-tokens` small during iteration.
- If the model outputs malformed JSON, it gets captured in `*.jsonl` `errors`. Tighten `--max-edits`/`--max-new-tokens` if this increases.

## Next Work (Not Implemented Yet)

Implement the “prompt improvement loop”:
1. Select a small subset of notes (train subset) and a small held-out subset (eval subset).
2. For each note:
   - Compute oracle deletes via `oracle_delete.py` (ground truth policy for deletes).
   - Run the agent and collect its deletes.
   - Summarize discrepancies (false deletions vs missed deletions) with a few examples + local context.
3. Ask the LLM to propose a prompt update using those summaries, with explicit instruction:
   - Avoid note-specific rules.
   - Prefer general patterns (section headers, lab-panel tokens, boilerplate exam phrases).
4. Re-run on train subset, then check eval subset before accepting the prompt update.

Pragmatic suggestion: keep the optimization driver deterministic, and store:
- prompt version hash
- per-note agent output
- per-note oracle output
- score deltas

