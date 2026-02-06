#!/usr/bin/env python3
from __future__ import annotations

import os

import sglang as sgl


MODEL = os.getenv("SGLANG_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
MODEL_IMPL = os.getenv("SGLANG_MODEL_IMPL", "transformers")
QUANTIZATION = os.getenv("SGLANG_QUANTIZATION", "")
ATTENTION_BACKEND = os.getenv("SGLANG_ATTENTION_BACKEND", "triton")
FP4_GEMM_BACKEND = os.getenv("SGLANG_FP4_GEMM_BACKEND", "auto")
MEM_FRACTION_STATIC = float(os.getenv("SGLANG_MEM_FRACTION_STATIC", "0.7"))
LAUNCH_TIMEOUT_S = float(os.getenv("SGLANG_LAUNCH_TIMEOUT", "1800"))
LOG_LEVEL = os.getenv("SGLANG_LOG_LEVEL", "info")
DISABLE_CUDA_GRAPH = os.getenv("SGLANG_DISABLE_CUDA_GRAPH", "true").lower() in (
    "1",
    "true",
    "yes",
)


@sgl.function
def _chat(s, prompt: str):
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("out", max_new_tokens=64, temperature=0.0, top_p=1.0))


def main() -> int:
    backend = sgl.Runtime(
        log_level=LOG_LEVEL,
        launch_timeout=LAUNCH_TIMEOUT_S,
        model_path=MODEL,
        model_impl=MODEL_IMPL,
        tp_size=1,
        dtype="auto",
        quantization=(QUANTIZATION or None),
        attention_backend=ATTENTION_BACKEND,
        fp4_gemm_runner_backend=FP4_GEMM_BACKEND,
        mem_fraction_static=MEM_FRACTION_STATIC,
        disable_cuda_graph=DISABLE_CUDA_GRAPH,
        trust_remote_code=True,
    )
    try:
        out = _chat.run({"prompt": "Say hello in one short sentence."}, backend=backend)
        print(out["out"])
    finally:
        backend.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
