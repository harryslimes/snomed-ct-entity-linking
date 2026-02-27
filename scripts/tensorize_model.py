#!/usr/bin/env python3
"""Serialize gpt-oss-20b to tensorizer format for fast vLLM loading.

Run once while the vLLM server is stopped. Output is written alongside the
HuggingFace cache so subsequent vllm serve calls can use --load-format tensorizer.

Usage:
    python scripts/tensorize_model.py [--output-dir PATH]
"""
import argparse
import os
import time
from pathlib import Path

MODEL_ID = "openai/gpt-oss-20b"
DEFAULT_OUTPUT = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--gpt-oss-20b" / "tensorized"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Directory to write tensorized model (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorizer_uri = str(output_dir / "model.tensors")

    print(f"Model:       {MODEL_ID}")
    print(f"Output:      {tensorizer_uri}")
    print()

    if Path(tensorizer_uri).exists():
        print(f"WARNING: {tensorizer_uri} already exists — overwriting.")

    # Import here so errors surface cleanly
    try:
        import tensorizer  # noqa: F401
    except ImportError:
        print("ERROR: tensorizer not installed. Run: pip install tensorizer")
        raise SystemExit(1)

    from vllm import LLM
    from vllm.model_executor.model_loader.tensorizer import (
        TensorizerConfig,
        serialize_vllm_model,
    )

    print("Loading model through vLLM (this is the slow part — same as a normal server start)...")
    t0 = time.time()

    # enforce_eager avoids CUDA graph capture which is wasted work for serialization
    llm = LLM(model=MODEL_ID, enforce_eager=True)

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Serializing to {tensorizer_uri} ...")
    t1 = time.time()

    # Navigate to the underlying model module through vLLM internals
    executor = llm.llm_engine.model_executor
    # Handle both single-worker and distributed executor layouts
    if hasattr(executor, "driver_worker"):
        runner = executor.driver_worker.model_runner
    elif hasattr(executor, "workers") and executor.workers:
        runner = executor.workers[0].model_runner
    else:
        raise RuntimeError("Cannot locate model_runner — vLLM internal layout may have changed")

    model = runner.model
    model_config = llm.llm_engine.model_config

    tensorizer_config = TensorizerConfig(
        tensorizer_uri=tensorizer_uri,
        vllm_tensorized=True,
    )

    serialize_vllm_model(model, tensorizer_config, model_config)
    elapsed = time.time() - t1
    size_gb = Path(tensorizer_uri).stat().st_size / 1e9
    print(f"Serialized {size_gb:.1f} GB in {elapsed:.1f}s")
    print()
    print("Done. Start vLLM with:")
    print(f"  VLLM_SERVER_DEV_MODE=1 vllm serve {MODEL_ID} \\")
    print(f"      --load-format tensorizer \\")
    print(f"      --tensorizer-uri {tensorizer_uri} \\")
    print(f"      --enable-prefix-caching \\")
    print(f"      --enable-chunked-prefill \\")
    print(f"      --max-num-seqs 100")


if __name__ == "__main__":
    main()
