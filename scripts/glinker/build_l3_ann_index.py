#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.glinker.l3_biencoder import build_l3_faiss_ann_index


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build persisted FAISS ANN indexes for L3 retrieval from l3_biencoder_index.npz. "
            "This avoids rebuilding ANN indexes during every eval fold."
        )
    )
    ap.add_argument("--index-npz", default="data/interim/glinker/l3_biencoder_index.npz")
    ap.add_argument("--out-dir", default="data/interim/glinker/l3_ann_faiss")
    ap.add_argument("--mode", default="faiss_hnsw", help="faiss_hnsw|faiss_ivf|faiss_flat")
    ap.add_argument("--ivf-nlist", type=int, default=4096)
    ap.add_argument("--ivf-nprobe", type=int, default=16)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--hnsw-ef-search", type=int, default=64)
    args = ap.parse_args(argv)

    stats = build_l3_faiss_ann_index(
        index_npz=Path(args.index_npz),
        out_dir=Path(args.out_dir),
        mode=str(args.mode),
        ivf_nlist=int(args.ivf_nlist),
        ivf_nprobe=int(args.ivf_nprobe),
        hnsw_m=int(args.hnsw_m),
        hnsw_ef_search=int(args.hnsw_ef_search),
    )
    print(f"rows: {stats['rows']:,}")
    print(f"dim: {stats['dim']}")
    print(f"l1_indexes: {stats['l1_indexes']}")
    print(f"mode: {stats['mode']}")
    print(f"out_dir: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
