"""Build and persist FAISS dense index and BM25S sparse index for SNOMED descriptions."""

import json
import os
import pickle
import time

import bm25s
import faiss
import numpy as np
import pandas as pd

from .config import Config
from .encoder import SapBERTEncoder


def build_indexes(
    descriptions_df: pd.DataFrame,
    cfg: Config,
    preferred_weight: float = 2.0,
) -> tuple[faiss.Index, list[int], bm25s.BM25, list[int], list[str]]:
    """
    Build FAISS (concept-averaged) and BM25S (description-level) indexes.

    Returns:
        faiss_index: FAISS IndexFlatIP with one concept-averaged vector per SCTID.
        faiss_sctids: List mapping FAISS vector index -> SCTID (one per concept).
        bm25_index: bm25s.BM25 index (one document per description).
        bm25_sctids: List mapping BM25 doc index -> SCTID (one per description).
        bm25_terms: Parallel list of description terms for BM25.
    """
    terms = descriptions_df["index_term"].tolist()
    bm25_sctids = descriptions_df["sctid"].tolist()

    # --- Dense Index (concept-averaged, preferred-term-weighted) ---
    faiss_index, faiss_sctids = _build_concept_averaged_faiss_index(
        descriptions_df, cfg, preferred_weight,
    )

    # --- Sparse Index (description-level for maximum keyword coverage) ---
    print("Building BM25S index ...")
    t0 = time.time()
    corpus_tokens = bm25s.tokenize(terms, show_progress=True)
    bm25_index = bm25s.BM25()
    bm25_index.index(corpus_tokens, show_progress=True)
    print(f"  BM25S index built in {time.time() - t0:.1f}s ({len(terms):,} documents)")

    return faiss_index, faiss_sctids, bm25_index, bm25_sctids, terms


def _encode_and_cache_embeddings(
    descriptions_df: pd.DataFrame,
    cfg: Config,
) -> np.ndarray:
    """
    Encode all descriptions and cache to disk. Returns cached if available.

    Saves raw_embeddings.npz with: embeddings, sctids, is_preferred arrays.
    """
    cache_path = cfg.index_dir / "raw_embeddings.npz"

    if cache_path.exists():
        print(f"Loading cached description embeddings from {cache_path} ...")
        data = np.load(cache_path)
        cached_n = len(data["sctids"])
        current_n = len(descriptions_df)
        if cached_n == current_n:
            print(f"  Loaded {cached_n:,} cached embeddings (skipping re-encoding)")
            return data["embeddings"]
        else:
            print(f"  Cache size mismatch ({cached_n:,} vs {current_n:,}), re-encoding ...")

    terms = descriptions_df["index_term"].tolist()
    sctids = descriptions_df["sctid"].values
    is_preferred = descriptions_df["is_preferred"].values

    encoder = SapBERTEncoder(cfg.embedding_model)
    print(f"Encoding {len(terms):,} descriptions (batch_size={cfg.index_batch_size}) ...")
    t0 = time.time()
    all_embeddings = encoder.encode(terms, batch_size=cfg.index_batch_size, show_progress=True)
    elapsed = time.time() - t0
    print(f"  Encoded in {elapsed:.1f}s ({len(terms) / elapsed:.0f} desc/s)")
    encoder.close()

    # Cache to disk
    os.makedirs(cfg.index_dir, exist_ok=True)
    np.savez(
        cache_path,
        embeddings=all_embeddings,
        sctids=sctids,
        is_preferred=is_preferred,
    )
    print(f"  Raw embeddings cached to {cache_path}")

    return all_embeddings


def _build_concept_averaged_faiss_index(
    descriptions_df: pd.DataFrame,
    cfg: Config,
    preferred_weight: float = 2.0,
) -> tuple[faiss.Index, list[int]]:
    """
    Build FAISS index with one concept-averaged vector per SCTID.

    Each concept's embedding = weighted average of its description embeddings,
    with preferred terms weighted by preferred_weight. Re-normalized to unit length.
    """
    all_embeddings = _encode_and_cache_embeddings(descriptions_df, cfg)

    sctids = descriptions_df["sctid"].tolist()
    is_preferred = descriptions_df["is_preferred"].tolist()

    # Group embeddings by concept
    print(f"Computing concept-averaged embeddings (preferred weight={preferred_weight:.1f}x) ...")
    concept_groups: dict[int, tuple[list[int], list[float]]] = {}
    for i, (sctid, pref) in enumerate(zip(sctids, is_preferred)):
        if sctid not in concept_groups:
            concept_groups[sctid] = ([], [])
        concept_groups[sctid][0].append(i)
        concept_groups[sctid][1].append(preferred_weight if pref else 1.0)

    # Compute weighted average per concept and re-normalize
    faiss_sctids = []
    concept_embeddings = []
    for sctid, (indices, weights) in concept_groups.items():
        embeds = all_embeddings[indices]
        avg = np.average(embeds, axis=0, weights=weights)
        avg = avg / np.linalg.norm(avg)  # Re-normalize for cosine similarity
        concept_embeddings.append(avg)
        faiss_sctids.append(sctid)

    concept_embeddings = np.ascontiguousarray(
        np.array(concept_embeddings, dtype=np.float32)
    )

    dim = concept_embeddings.shape[1]
    print(f"  Concept-averaged: {len(faiss_sctids):,} concepts "
          f"(from {len(sctids):,} descriptions, dim={dim})")

    index = faiss.IndexFlatIP(dim)
    index.add(concept_embeddings)
    print(f"  FAISS index built: {index.ntotal:,} vectors (1 per concept)")

    return index, faiss_sctids


def rebuild_faiss_from_cache(
    cfg: Config,
    preferred_weight: float = 2.0,
) -> tuple[faiss.Index, list[int]] | None:
    """
    Rebuild FAISS index from cached raw embeddings with a new preferred weight.
    Returns None if no cached embeddings exist.
    """
    cache_path = cfg.index_dir / "raw_embeddings.npz"
    if not cache_path.exists():
        return None

    print(f"Rebuilding FAISS from cached embeddings (preferred_weight={preferred_weight:.1f}x) ...")
    data = np.load(cache_path)
    all_embeddings = data["embeddings"]
    sctids = data["sctids"].tolist()
    is_preferred = data["is_preferred"].tolist()

    concept_groups: dict[int, tuple[list[int], list[float]]] = {}
    for i, (sctid, pref) in enumerate(zip(sctids, is_preferred)):
        if sctid not in concept_groups:
            concept_groups[sctid] = ([], [])
        concept_groups[sctid][0].append(i)
        concept_groups[sctid][1].append(preferred_weight if pref else 1.0)

    faiss_sctids = []
    concept_embeddings = []
    for sctid, (indices, weights) in concept_groups.items():
        embeds = all_embeddings[indices]
        avg = np.average(embeds, axis=0, weights=weights)
        avg = avg / np.linalg.norm(avg)
        concept_embeddings.append(avg)
        faiss_sctids.append(sctid)

    concept_embeddings = np.ascontiguousarray(
        np.array(concept_embeddings, dtype=np.float32)
    )

    index = faiss.IndexFlatIP(concept_embeddings.shape[1])
    index.add(concept_embeddings)
    print(f"  FAISS rebuilt: {index.ntotal:,} vectors (preferred_weight={preferred_weight:.1f}x)")

    return index, faiss_sctids


def save_indexes(
    faiss_index: faiss.Index,
    faiss_sctids: list[int],
    bm25_index: bm25s.BM25,
    bm25_sctids: list[int],
    bm25_terms: list[str],
    cfg: Config,
) -> None:
    """Persist indexes to disk."""
    folder = cfg.index_dir
    os.makedirs(folder, exist_ok=True)

    faiss.write_index(faiss_index, str(folder / "vectors.index"))
    with open(folder / "faiss_sctids.pkl", "wb") as f:
        pickle.dump(faiss_sctids, f)
    with open(folder / "bm25_sctids.pkl", "wb") as f:
        pickle.dump(bm25_sctids, f)

    bm25_index.save(str(folder / "bm25s"), corpus=bm25_terms)
    print(f"Indexes saved to {folder}")


def load_indexes(
    cfg: Config,
) -> tuple[faiss.Index, list[int], bm25s.BM25, list[int], list[str]]:
    """Load persisted indexes from disk."""
    folder = cfg.index_dir
    print(f"Loading indexes from {folder} ...")

    faiss_index = faiss.read_index(str(folder / "vectors.index"))
    with open(folder / "faiss_sctids.pkl", "rb") as f:
        faiss_sctids = pickle.load(f)
    with open(folder / "bm25_sctids.pkl", "rb") as f:
        bm25_sctids = pickle.load(f)

    bm25_index = bm25s.BM25.load(str(folder / "bm25s"), load_corpus=False)
    corpus_path = folder / "bm25s" / "corpus.jsonl"
    bm25_terms = []
    with open(corpus_path) as f:
        for line in f:
            bm25_terms.append(json.loads(line)["text"])

    print(f"  FAISS: {faiss_index.ntotal:,} concept vectors | "
          f"BM25S: {len(bm25_terms):,} descriptions")
    return faiss_index, faiss_sctids, bm25_index, bm25_sctids, bm25_terms


def load_raw_embeddings(cfg: Config) -> tuple[np.ndarray, list[int]] | None:
    """Load raw per-description embeddings for max-synonym reranking."""
    cache_path = cfg.index_dir / "raw_embeddings.npz"
    if not cache_path.exists():
        return None
    data = np.load(cache_path)
    return data["embeddings"], data["sctids"].tolist()


def indexes_exist(cfg: Config) -> bool:
    """Check if all index files exist on disk."""
    folder = cfg.index_dir
    return all([
        (folder / "vectors.index").exists(),
        (folder / "faiss_sctids.pkl").exists(),
        (folder / "bm25_sctids.pkl").exists(),
        (folder / "bm25s").exists(),
    ])
