#!/usr/bin/env python3
"""Rebuild the on-disk FAISS index in multi-vector mode.

Instead of one concept-averaged vector per SCTID (~230k), stores one vector
per description (~402k), with faiss_sctids.pkl mapping each vector back to
its SCTID. After rebuilding, dense_search() deduplicates per SCTID using the
first (highest-scoring) hit for each concept.

Old files are backed up with a .centroid_backup suffix before overwriting.

Usage:
    python scripts/rebuild_index_multivector.py
    python scripts/rebuild_index_multivector.py --restore   # revert to centroid backup
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = REPO_ROOT / "snomed_index"


def rebuild(index_dir: Path) -> None:
    cache_path = index_dir / "raw_embeddings.npz"
    if not cache_path.exists():
        print(f"ERROR: {cache_path} not found — run the main index build first.")
        sys.exit(1)

    print(f"Loading raw description embeddings from {cache_path} ...")
    data = np.load(cache_path)
    raw_embs: np.ndarray = data["embeddings"]       # (N_desc, 768), L2-normed
    raw_sctids: list[int] = data["sctids"].tolist() # one per description
    n_desc = len(raw_embs)
    n_concepts = len(set(raw_sctids))
    print(f"  {n_desc:,} descriptions → {n_concepts:,} unique concepts")

    # Back up old files
    for fname in ("vectors.index", "faiss_sctids.pkl"):
        src = index_dir / fname
        dst = index_dir / f"{fname}.centroid_backup"
        if src.exists() and not dst.exists():
            src.rename(dst)
            print(f"  Backed up {fname} → {fname}.centroid_backup")

    # Build multi-vector FAISS index
    print(f"Building FAISS IndexFlatIP with {n_desc:,} description vectors ...")
    embs = np.ascontiguousarray(raw_embs, dtype=np.float32)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    print(f"  FAISS index: {index.ntotal:,} vectors, dim={dim}")

    # Write
    out_index = index_dir / "vectors.index"
    out_sctids = index_dir / "faiss_sctids.pkl"
    faiss.write_index(index, str(out_index))
    with open(out_sctids, "wb") as f:
        pickle.dump(raw_sctids, f, protocol=4)

    print(f"Saved {out_index} ({out_index.stat().st_size / 1e6:.0f} MB)")
    print(f"Saved {out_sctids} ({out_sctids.stat().st_size / 1e6:.1f} MB)")
    print("Done. Restart the index server if it is running.")


def restore(index_dir: Path) -> None:
    for fname in ("vectors.index", "faiss_sctids.pkl"):
        backup = index_dir / f"{fname}.centroid_backup"
        dst = index_dir / fname
        if not backup.exists():
            print(f"No backup found for {fname}, skipping.")
            continue
        dst.unlink(missing_ok=True)
        backup.rename(dst)
        print(f"Restored {fname} from backup.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--restore", action="store_true",
                        help="Restore centroid backup instead of rebuilding.")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR,
                        help=f"Index directory (default: {INDEX_DIR})")
    args = parser.parse_args()

    if args.restore:
        restore(args.index_dir)
    else:
        rebuild(args.index_dir)
