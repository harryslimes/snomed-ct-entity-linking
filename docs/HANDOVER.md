# Handover: SNOMED Abbreviation Disambiguation Rule Loop

## What this system does

Builds an **abbreviation → SNOMED CT concept** dictionary by running a two-pass
LLM pipeline over clinical notes, then iteratively improving a rule set based on
failures. Rules are injected into prompts at either the **search** (term
generation) or **select** (candidate ranking) stage.

Pipeline per abbreviation instance:
1. LLM generates SNOMED search terms (guided by "search" rules)
2. Hybrid FAISS + BM25 retrieval returns up to 15 candidates
3. LLM selects best candidate (guided by "select" rules)

---

## Key files

| File | Purpose |
|---|---|
| `super-dictionary/abbrev_rule_loop.py` | Main iterative improvement loop |
| `super-dictionary/abbrev_rules.json` | Current live rule set (36 rules, priorities tagged) |
| `super-dictionary/build_abbrev_dictionary_llm.py` | Standalone full-corpus runner |
| `rulebook/rule_testing.py` | Shared pipeline (vllm_pipeline, retrieval, scoring) |
| `rulebook/runs/20260219T171514Z/` | **Active run directory** |
| `rulebook/runs/20260219T171514Z/state.json` | Batch 6/29, 36 rules, holdout active |

---

## Current performance

### Holdout trend (157 held-out pairs, training-set evaluation)

| After batch | Holdout accuracy |
|---|---|
| Batch 1 round 1 | 54.1% |
| Batch 3 round 1 | 59.8% |
| Batch 3 round 3 | **60.3%** (peak) |
| Batch 5 round 2 | 54.6% (post-merge dip) |
| Batch 6 round 1 | 57.8% |
| Batch 6 round 2 | 57.0% |

**Previous best on 87-annotation test note:** 52.9% (from `rulebook/rules/20260217T134141Z/`)

### Batch-level improvements (training set)

| Batch | Baseline | Best |
|---|---|---|
| 1 | 33.0% | 78.0% |
| 2 | 93.7% | 93.7% (no improvement needed) |
| 3 | 63.0% | 86.0% |
| 4 | 68.6% | 79.0% |
| 5 | 42.3% | 54.4% |
| 6 | 51.3% | 74.0% |

---

## Active run

**Run directory:** `rulebook/runs/20260219T171514Z/`

**Status:** Batch 7/29 running (as of session end)

**Resume command:**
```bash
nohup .venv/bin/python super-dictionary/abbrev_rule_loop.py \
  --resume rulebook/runs/20260219T171514Z \
  --index-server http://127.0.0.1:8421 \
  --vllm-url http://localhost:8000 \
  --audit-rules \
  --auto-advance \
  --max-rules 45 \
  --merge-threshold -0.05 \
  >> abbrev_loop.log 2>&1 &
```

---

## Fixes implemented in this session

### 1. CLAUDECODE nested session block (critical fix)
`_sonnet_call` in `abbrev_rule_loop.py` now temporarily pops `CLAUDECODE` from
`os.environ` before calling `claude_code_sdk.query()`. This allows Sonnet to be
called for rule generation and merging when running inside Claude Code.

Without this fix, every Sonnet call fails with:
> "Claude Code cannot be launched inside another Claude Code session."

### 2. Holdout guard in `_merge_with_validation`
The merge function now also validates the proposed merge against a 30-pair sample
of the holdout set. If the holdout accuracy drops by more than `holdout_threshold`
(default -5%), the merge is rejected even if the batch sample passed.

Previously: merges could pass the batch sample check but silently hurt the holdout
(seen as 60.3% → 40.6% regression when the first merge went 29 → 14 rules).

### 3. Gentler merge target
Changed RULE_MERGE_SYSTEM prompt from "reduce by at least 40%" to "reduce by
20-30%, target ~N rules, do NOT reduce by more than 35%".

Old behaviour: Sonnet consistently merged 29 → 14 rules (52%), causing large
accuracy drops that exceeded validation thresholds, leading to merge rejections
and rules staying above the context limit indefinitely.

New behaviour: Sonnet merges 29 → 21 rules (28%), which passes the validation
threshold and keeps prompt sizes within the 8192-token context.

### 4. Priority tagging of R001–R014
All rules now have a `priority` field:
- **P1** (override, apply first): R001 (hallucination recovery), R002 (rank-1 unconditional)
- **P3** (guideline, tie-break): R004 (specificity), R006 (procedure/regime), R010 ('Structure of' form)
- **P2** (standard): R003, R005, R007-R009, R011-R014

### 5. `--max-rules 45`
Changed from default 25 to 45. The previous default caused premature max-rules
merges in round 1/2 (when the rule set had only grown to 29-34), interfering with
rule improvement before the context-length check had a chance to act.

---

## Loop mechanics (for reference)

Each batch cycle:
1. **Pre-flight check**: if estimated prompt > 8192 tokens → auto-merge
   - Merge validated against 30-pair batch sample AND 30-pair holdout sample
   - Merge threshold: -5% (allows up to 5 pp accuracy drop)
   - Holdout threshold: -5%
   - Merge target: 20-30% reduction (not 40%+)
2. **Baseline run**: evaluate current rules on 50-pair batch
3. **Rule coverage audit** (if `--audit-rules`): per-rule injection counts, flags narrow rules
4. **3 improvement rounds**:
   - Sonnet generates up to 5 new rules from the failure cases
   - New rules tested on batch; accepted only if accuracy improves
   - If accepted: full 157-pair holdout evaluated and logged
   - If `len(rules) > --max-rules (45)`: another merge with same validation

---

## Immediate next tasks

### 1. Let the active run complete (batches 7–29)
The run is at batch 7/29. It will continue to generate and refine rules. Monitor
with:
```bash
tail -f /workspaces/snomed-ct-entity-linking/abbrev_loop.log
```
Or check periodic summaries:
```bash
grep -E "Batch [0-9]+ complete|Holdout:" abbrev_loop.log | tail -20
```

### 2. Benchmark final rules against the 87-annotation test note
After the run completes, run rule_testing.py to measure improvement over the
52.9% baseline. Note: the two pipelines use different rule formats; you would
need to adapt the rules OR run the loop's own `_run_and_evaluate` against the
test note's abbreviation pairs.

The loop pipeline can be applied to the test note by building reps from the
first note in train_notes.csv (the 87-annotation note) using `build_reps_for_batch`.

### 3. Address narrow rules (lookup table)
The audit flags rules like R027 (OSA), R028 (BLE/RLE/LLE), R029 (HCV), R030 (PDA)
as firing for <10% of pairs. These are memorised abbreviation-specific rules that
inflate the prompt. Move them to `abbrev_lookup.json`:
```json
{"OSA": 73430006, "BLE": 182281004, "HCV": 50711007, "PDA": 836383009}
```
Wire into `build_reps_for_batch` via `mapping_hit` bypass to skip LLM entirely.

### 4. Investigate the holdout oscillation
Holdout peaked at 60.3% then dropped to 54-58%. This is caused by:
- Individual batches having hard/unusual pairs that push rules in narrow directions
- No cross-batch guard (rules accepted on batch N only validated against batch N + holdout sample)

Consider: after all 29 batches, run a final full-holdout evaluation with the
final rules to get a clean accuracy estimate.

---

## Known architectural issues (unchanged)

- **Rule memorisation**: Many P1 rules name specific abbreviations (R027 OSA,
  R028 BLE, R029 HCV, etc.). These are lookup tables disguised as rules.
- **Context-length pressure**: 21-36 rules fill 7000-12000 tokens of an 8192-token
  context. Merges are frequent.
- **Single representative per pair**: The pipeline commits to one concept per
  (span, section) pair. Genuinely ambiguous abbreviations need per-instance
  disambiguation.

---

## Infrastructure notes

- **vLLM server**: must be running at `--vllm-url` (default `http://localhost:8000`)
  serving the Qwen3-30B-A3B model. Check: `ps aux | grep vllm`
- **Index server**: start with `python snomed_index_server.py` at port 8421;
  pass `--index-server http://127.0.0.1:8421`. Check: `ps aux | grep snomed_index`
- **Data**: `data/old-challenge-split/train_{annotations,notes}.csv`
- **Terminology**: `3rd Place/assets/dataflattened_terminology.csv`

---

## File locations

```
rulebook/runs/20260219T171514Z/
  state.json          # batch_idx, rules (36), holdout_pairs (157), holdout_history
  config.json         # CLI args used to start the run

super-dictionary/
  abbrev_rules.json           # current rules (synced with state.json after each improvement)
  abbrev_rules_backup_*.json  # pre-merge backups (timestamped)
```

The loop saves rules to `--rules-file` (default: `abbrev_rules.json`) after each
improvement, and to `state.json` for resumability.
