#!/usr/bin/env python3
"""Pre-shard gpt-oss-20b into vLLM's sharded_state format for fast loading.

Run once while the vLLM server is stopped. The sharded state skips the
MXFP4 weight repacking step on every subsequent server start.

Usage:
    python scripts/shard_model.py [--output-dir PATH]
"""
import argparse
import os
import time
from pathlib import Path

# Disable multiprocessing so the model stays in-process and model_executor is accessible.
# In vLLM 0.15+ (v1 engine), VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps EngineCore
# in the same process, exposing llm.llm_engine.model_executor for weight access.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODEL_ID = "openai/gpt-oss-20b"
DEFAULT_OUTPUT = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--gpt-oss-20b" / "sharded"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Directory to write sharded model (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print(f"Model:       {MODEL_ID}")
    print(f"Output:      {output_dir}")
    print()

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"WARNING: {output_dir} already exists and is non-empty — overwriting.")

    output_dir.mkdir(parents=True, exist_ok=True)

    from vllm import LLM
    from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader

    print("Loading model through vLLM (slow — runs MXFP4 repacking once)...")
    t0 = time.time()

    # enforce_eager avoids CUDA graph capture which is wasted work for serialization
    llm = LLM(model=MODEL_ID, enforce_eager=True)

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Saving sharded state to {output_dir} ...")
    t1 = time.time()

    executor = llm.llm_engine.model_executor
    # In v1 in-process mode the executor is a UniProcExecutor wrapping a single GPUWorker
    if hasattr(executor, "worker"):
        model = executor.worker.model_runner.model
    elif hasattr(executor, "driver_worker"):
        model = executor.driver_worker.model_runner.model
    elif hasattr(executor, "workers") and executor.workers:
        model = executor.workers[0].model_runner.model
    else:
        raise RuntimeError(f"Cannot locate model — unknown executor type: {type(executor)}")

    ShardedStateLoader.save_model(model, path=str(output_dir))

    elapsed = time.time() - t1
    size_gb = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / 1e9
    print(f"Saved {size_gb:.1f} GB in {elapsed:.1f}s")

    # Copy config/tokenizer files so the sharded dir is self-contained
    import shutil
    src = Path(llm.llm_engine.model_config.model)
    config_files = [
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
    ]
    for fname in config_files:
        src_file = src / fname
        if src_file.exists():
            shutil.copy2(src_file, output_dir / fname)
    print()
    print("Done. Start vLLM with:")
    print(f"  vllm serve {output_dir} \\")
    print(f"      --load-format sharded_state \\")
    print(f"      --enable-prefix-caching \\")
    print(f"      --enable-chunked-prefill \\")
    print(f"      --max-num-seqs 100")


if __name__ == "__main__":
    main()
