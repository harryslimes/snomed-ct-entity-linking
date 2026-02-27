# vLLM Batch Invariance Investigation

**Date:** 2026-02-21
**Model:** Qwen3-30B-A3B-Instruct-2507-AWQ-4bit (quantized MoE)
**GPU:** NVIDIA RTX 5090 (compute capability 12.0)

## Problem

Running the same prompt through vLLM at temperature=0 with concurrent requests produces different outputs each time. This is caused by floating-point non-associativity: when requests are batched together differently (which happens non-deterministically with concurrent requests), the order of floating-point operations in attention, matmul, and normalization layers changes, producing slightly different results that cascade into completely different completions.

Test: 20 identical concurrent requests, `temperature=0.0`, `seed=42`.

## What is VLLM_BATCH_INVARIANT?

`VLLM_BATCH_INVARIANT=1` is an environment variable that enables batch-invariant kernels in vLLM. These kernels use fixed reduction strategies (rather than batch-size-dependent ones) so that the output is identical regardless of how requests are batched together. The feature originates from [Thinking Machines Lab](https://github.com/thinking-machines-lab/batch_invariant_ops) and was integrated into vLLM starting around v0.11.

Requirements:
- NVIDIA GPU with compute capability >= 9.0 (Hopper, Blackwell)
- An explicit attention backend: `FLASH_ATTN`, `TRITON_ATTN`, `FLASH_ATTN_MLA`, or `TRITON_MLA`
- ~60% throughput penalty

## What we tested

### 1. vLLM 0.15.1 (our stable version)

`VLLM_BATCH_INVARIANT` is **not recognized at all** in 0.15.1. The env var is silently ignored — it doesn't appear in `vllm.envs` and has no effect.

- FlashInfer backend: 15/20 unique outputs
- With `seed=42`, sequential requests: 5/5 unique outputs (even without batching!)

Result: **No batch invariance support in 0.15.1.**

### 2. vLLM 0.16.0 nightly (0.16.0rc2.dev367+g98b0205c3)

The nightly has `vllm.model_executor.layers.batch_invariant` module and reads `VLLM_BATCH_INVARIANT` via `os.getenv`. The server logs `"Disabling cascade attention when VLLM_BATCH_INVARIANT is enabled"` and registers batch-invariant kernels.

However, the supported attention backends for batch invariance changed between versions:

| Backend | 0.14.1 | 0.15.x | 0.16.0 nightly |
|---------|--------|--------|----------------|
| FLASHINFER | Works ([reported](https://github.com/vllm-project/vllm/issues/33421)) | N/A (no batch invariance) | Explicitly rejected: "requires backend in ['FLASH_ATTN', 'TRITON_ATTN', ...]" |
| FLASH_ATTN | Unknown | N/A | Supported but requires flash-attn compiled against torch 2.10 |
| TRITON_ATTN | Unknown | N/A | Accepted, kernels register, but **does not produce deterministic output** |

#### 2a. FLASHINFER backend
Server refused to start:
```
RuntimeError: VLLM batch_invariant mode requires an attention backend in
['FLASH_ATTN', 'TRITON_ATTN', 'FLASH_ATTN_MLA', 'TRITON_MLA'], but got 'FLASHINFER'.
```

#### 2b. FLASH_ATTN backend
flash-attn 2.8.3 was compiled against torch 2.9.1; the nightly upgraded to torch 2.10.0, causing an ABI mismatch:
```
ImportError: flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol:
_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```
Rebuilding flash-attn from source requires compiling 72 CUDA kernel files across 4 GPU architectures (sm_80/90/100/120). Build was killed after exhausting available RAM.

#### 2c. TRITON_ATTN backend (with flash-attn uninstalled)
Server started successfully. Batch-invariant kernels registered. But:
- **With prefix caching:** 18/20 unique outputs
- **Without prefix caching:** 13/20 unique outputs (7/20 matched one hash)

The batch-invariant kernels don't cover the full codepath for quantized MoE models (AWQ + Qwen3 MoE architecture). Batch invariance has only been verified on dense models (Llama 3, Qwen2.5, DeepSeek).

### 3. vLLM 0.14.1 (not tested)

[Bug #33421](https://github.com/vllm-project/vllm/issues/33421) reports that `VLLM_BATCH_INVARIANT=1` + FlashInfer worked in 0.14.1, but this was tested with Qwen3-4B (a dense model, not MoE). Unknown if it would work with our quantized MoE model.

## Root causes of non-determinism

1. **Batch composition variance**: Different concurrent requests get grouped into different batch sizes, changing floating-point reduction order in attention/matmul. This is what batch invariance is supposed to fix.

2. **Quantized MoE routing**: AWQ quantization and MoE expert routing may have additional non-deterministic codepaths not covered by the current batch-invariant kernel replacements.

3. **Prefix caching**: KV cache block reuse patterns can vary between runs, affecting the computation order.

## Current status

Reverted to **vLLM 0.15.1** (stable). Batch invariance is not achievable with our current model (Qwen3-30B-A3B-AWQ-4bit) on any available vLLM version.

## Mitigation

Instead of relying on deterministic inference, the rule evaluation loop uses **per-rule cost gating**: new rules are only accepted if the net number of predictions that flipped from wrong-to-right exceeds the number of new rules added. This is inherently robust to small prediction noise because it measures actual prediction changes (wrong→right vs right→wrong) rather than aggregate accuracy percentages.

## Future

- Track [vLLM issue #27433](https://github.com/vllm-project/vllm/issues/27433) for MoE batch invariance support
- Track [vLLM issue #33421](https://github.com/vllm-project/vllm/issues/33421) for FlashInfer batch invariance fix
- If determinism becomes critical, consider using a non-quantized dense model (e.g. Qwen2.5-32B) where batch invariance has been verified
