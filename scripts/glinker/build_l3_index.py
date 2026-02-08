#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.l3_biencoder import build_l3_index


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build L3 bi-encoder alias index from super_dictionary_scoped.jsonl. "
            "Writes compressed NPZ with alias embeddings and metadata."
        )
    )
    ap.add_argument("--super-dict-jsonl", default="data/interim/glinker/super_dictionary_scoped.jsonl")
    ap.add_argument("--out-index-npz", default="data/interim/glinker/l3_biencoder_index.npz")
    ap.add_argument("--backend", default="auto", help="auto|hf|hash")
    ap.add_argument("--model-path", default="knowledgator/gliner-linker-large-v1.0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--load-dtype", default="auto", help="auto|float16|bfloat16|float32")
    ap.add_argument("--min-alias-len", type=int, default=2)
    ap.add_argument("--max-aliases-per-concept", type=int, default=0)
    args = ap.parse_args(argv)

    stats = build_l3_index(
        super_dict_jsonl=Path(args.super_dict_jsonl),
        out_index_npz=Path(args.out_index_npz),
        backend=str(args.backend),
        model_path=str(args.model_path),
        device=str(args.device),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        min_alias_len=int(args.min_alias_len),
        max_aliases_per_concept=int(args.max_aliases_per_concept),
        load_dtype=str(args.load_dtype),
    )
    print(f"rows: {stats['rows']:,}")
    print(f"concepts: {stats['concepts']:,}")
    print(f"embedding_dim: {stats['dim']}")
    print(f"backend: {stats['backend']}")
    print(f"out_index_npz: {args.out_index_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
