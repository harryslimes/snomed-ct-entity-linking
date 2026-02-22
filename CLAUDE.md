# Claude Code Notes

## Critical: Long-running tasks

**ALWAYS run scripts that may take more than 30 seconds in the background** using
`run_in_background: true` on the Bash tool, then poll with `tail` or `cat` on the
output file. Never block the conversation waiting for a long command.

Scripts that are typically long-running:
- `abbrev_rule_loop.py` (rule generation loop) — minutes to hours
- `test_cleaned_rules.py` (holdout evaluation) — 1-5 minutes with Qwen3, longer with reasoning models
- `compress_rules.py` (Sonnet-based rule compression) — ~5 min per 70 rules
- Any script calling `claude_code_sdk` / `_sonnet_call` — SDK calls are slow (~30s each)
- vLLM inference with `reasoning_effort=high` — very slow per annotation

## Project context

- vLLM server must be running on port 8000 before any evaluation scripts
- Both `abbrev_rule_loop.py` and `test_cleaned_rules.py` have pre-flight connectivity checks
- Qwen3-30B-A3B is the primary model; gpt-oss-20b was tested but underperforms (~48% vs ~72%)
- Rules are injected into every LLM prompt — keep them concise (<200 words each)
- `testing_config.json` and `vllm_testing_config.json` are NOT loaded by any code; CLI args only
