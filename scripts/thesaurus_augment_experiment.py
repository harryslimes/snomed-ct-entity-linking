#!/usr/bin/env python3
"""Experiment: augmenting SNOMED FAISS concept vectors with thesaurus synonyms.

For each thesaurus weight in [0, 0.25, 0.5, 1.0], rebuilds concept-averaged
embeddings in-memory (reusing cached raw_embeddings.npz) and evaluates dense
recall@{1,5,10,50} on training span texts.

The existing index on disk is never modified.

Thesaurus sources (pick one):
  --mrconso PATH     UMLS MRCONSO.RRF (recommended — multi-vocabulary synonyms)
  --newdict PATH     3rd Place/assets/newdict_snomed.txt (fallback, already SCTID-mapped)

Usage:
  # With MRCONSO (download from UTS first):
  python scripts/thesaurus_augment_experiment.py --mrconso /path/to/MRCONSO.RRF

  # With newdict (already in repo):
  python scripts/thesaurus_augment_experiment.py \\
      --newdict "3rd Place/assets/newdict_snomed.txt"

  # Specify weights explicitly:
  python scripts/thesaurus_augment_experiment.py --mrconso ... --weights 0 0.25 0.5 1.0 2.0

  # Evaluate on fewer spans (faster iteration):
  python scripts/thesaurus_augment_experiment.py --mrconso ... --max-spans 2000
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

INDEX_DIR = REPO_ROOT / "snomed_index"
SPLIT_DIR = REPO_ROOT / "data" / "old-challenge-split"
NEWDICT_DEFAULT = REPO_ROOT / "3rd Place" / "assets" / "newdict_snomed.txt"


# ---------------------------------------------------------------------------
# Thesaurus loading
# ---------------------------------------------------------------------------

def load_newdict(path: Path) -> dict[int, list[str]]:
    """Load newdict_snomed.txt → {sctid: [term, ...]}.

    Format: tab-separated with header 'term\tcode'.
    """
    print(f"Loading newdict from {path} ...")
    sctid_to_terms: dict[int, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            term, code = parts[0].strip(), parts[1].strip()
            if not term or not code.isdigit():
                continue
            sctid = int(code)
            sctid_to_terms.setdefault(sctid, []).append(term)
    n_concepts = len(sctid_to_terms)
    n_terms = sum(len(v) for v in sctid_to_terms.values())
    print(f"  {n_concepts:,} concepts, {n_terms:,} thesaurus terms")
    return sctid_to_terms


def load_mrconso(path: Path, target_sctids: set[int] | None = None,
                 cache_path: Path | None = None) -> dict[int, list[str]]:
    """Parse UMLS MRCONSO.RRF → {sctid: [english_terms]}.

    Strategy:
      Pass 1: Build CUI → set[SCTID] from rows where SAB='SNOMEDCT_US'.
              Optionally filtered to target_sctids for speed.
      Pass 2: For each CUI with a known SCTID, collect all ENG STR values
              (excluding the SNOMEDCT_US rows themselves, which are already
              in raw_embeddings.npz).

    MRCONSO.RRF columns (pipe-delimited, no header):
      CUI|LAT|TS|LUI|STY|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
      0   1   2  3   4   5   6      7   8    9    10   11  12  13  14  15  16       17
    """
    if cache_path and cache_path.exists():
        print(f"Loading cached MRCONSO terms from {cache_path} ...")
        import pickle
        with open(cache_path, "rb") as f:
            result = pickle.load(f)
        n_concepts = len(result)
        n_terms = sum(len(v) for v in result.values())
        print(f"  {n_concepts:,} concepts, {n_terms:,} terms")
        return result

    print(f"Parsing MRCONSO.RRF from {path} ...")
    t0 = time.time()

    # Pass 1: CUI → SCTIDs
    cui_to_sctids: dict[str, set[int]] = {}
    n_snomed_rows = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 15:
                continue
            sab = parts[11]
            if sab != "SNOMEDCT_US":
                continue
            n_snomed_rows += 1
            cui = parts[0]
            code = parts[13].strip()
            if not code.isdigit():
                continue
            sctid = int(code)
            if target_sctids and sctid not in target_sctids:
                continue
            cui_to_sctids.setdefault(cui, set()).add(sctid)

    valid_cuis = set(cui_to_sctids.keys())
    print(f"  Pass 1: {n_snomed_rows:,} SNOMEDCT_US rows → {len(valid_cuis):,} CUIs "
          f"({len(cui_to_sctids):,} CUI→SCTID mappings)  [{time.time()-t0:.1f}s]")

    # Pass 2: collect ENG synonyms from all other vocabularies
    sctid_to_terms: dict[int, list[str]] = {}
    seen: set[tuple[int, str]] = set()  # deduplicate (sctid, term) pairs
    n_collected = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split("|")
            if len(parts) < 15:
                continue
            lat = parts[1]
            if lat != "ENG":
                continue
            sab = parts[11]
            if sab == "SNOMEDCT_US":
                continue  # already in raw_embeddings.npz
            suppress = parts[16].strip() if len(parts) > 16 else ""
            if suppress == "Y":
                continue  # suppressed entries
            cui = parts[0]
            if cui not in valid_cuis:
                continue
            term = parts[14].strip()
            if not term:
                continue
            for sctid in cui_to_sctids[cui]:
                key = (sctid, term)
                if key in seen:
                    continue
                seen.add(key)
                sctid_to_terms.setdefault(sctid, []).append(term)
                n_collected += 1

    n_concepts = len(sctid_to_terms)
    print(f"  Pass 2: {n_collected:,} unique (sctid, term) pairs "
          f"across {n_concepts:,} concepts  [{time.time()-t0:.1f}s]")

    if cache_path:
        import pickle
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(sctid_to_terms, f, protocol=4)
        print(f"  Cached to {cache_path}")

    return sctid_to_terms


# ---------------------------------------------------------------------------
# Encoding with SapBERT (cached)
# ---------------------------------------------------------------------------

def encode_thesaurus_terms(
    sctid_to_terms: dict[int, list[str]],
    cache_path: Path,
    batch_size: int = 256,
    force_recompute: bool = False,
) -> dict[int, np.ndarray]:
    """Encode thesaurus terms with SapBERT, caching to disk.

    Returns {sctid: embeddings_array_shape_(N, 768)}.
    """
    if cache_path.exists() and not force_recompute:
        print(f"Loading cached thesaurus embeddings from {cache_path} ...")
        data = np.load(cache_path, allow_pickle=True)
        sctid_to_embs: dict[int, np.ndarray] = data["sctid_to_embs"].item()
        n_concepts = len(sctid_to_embs)
        n_terms = sum(len(v) for v in sctid_to_embs.values())
        print(f"  Loaded {n_concepts:,} concepts, {n_terms:,} embeddings")
        return sctid_to_embs

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token"
    print(f"Loading SapBERT: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    print(f"  SapBERT on {device}")

    # Flatten all terms with their sctid mapping
    all_sctids_flat: list[int] = []
    all_terms_flat: list[str] = []
    for sctid, terms in sctid_to_terms.items():
        for term in terms:
            all_sctids_flat.append(sctid)
            all_terms_flat.append(term)

    n_total = len(all_terms_flat)
    print(f"Encoding {n_total:,} thesaurus terms (batch_size={batch_size}) ...")
    t0 = time.time()

    all_embeddings = []
    from tqdm import tqdm
    for i in tqdm(range(0, n_total, batch_size), desc="Encoding"):
        batch = all_terms_flat[i: i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            return_tensors="pt", max_length=128,
        ).to(device)
        with torch.no_grad():
            out = model(**encoded)
            attn = encoded["attention_mask"].unsqueeze(-1)
            mean = (out.last_hidden_state * attn).sum(1) / attn.sum(1)
            mean = torch.nn.functional.normalize(mean, dim=1)
            all_embeddings.append(mean.cpu().float().numpy())

    all_embeddings_arr = np.vstack(all_embeddings)  # (N_total, 768)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Group by sctid
    sctid_to_embs: dict[int, np.ndarray] = {}
    for i, sctid in enumerate(all_sctids_flat):
        if sctid not in sctid_to_embs:
            sctid_to_embs[sctid] = []
        sctid_to_embs[sctid].append(all_embeddings_arr[i])
    sctid_to_embs = {sctid: np.array(embs) for sctid, embs in sctid_to_embs.items()}

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, sctid_to_embs=np.array(sctid_to_embs, dtype=object))
    print(f"  Thesaurus embeddings cached to {cache_path}")

    # Free GPU
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sctid_to_embs


# ---------------------------------------------------------------------------
# Build augmented concept vectors
# ---------------------------------------------------------------------------

def build_augmented_faiss(
    raw_embs: np.ndarray,
    raw_sctids: list[int],
    raw_is_preferred: list[bool],
    thesaurus_embs: dict[int, np.ndarray],
    preferred_weight: float,
    thesaurus_weight: float,
) -> tuple:
    """Build concept-averaged FAISS index augmented with thesaurus embeddings.

    Returns (faiss_index, faiss_sctids_list).
    """
    import faiss

    # Group existing description embeddings by concept
    concept_groups: dict[int, list[tuple[np.ndarray, float]]] = {}
    for i, (sctid, is_pref) in enumerate(zip(raw_sctids, raw_is_preferred)):
        w = preferred_weight if is_pref else 1.0
        concept_groups.setdefault(sctid, []).append((raw_embs[i], w))

    # Build augmented concept vectors
    faiss_sctids = []
    concept_vectors = []

    for sctid, desc_pairs in concept_groups.items():
        emb_list = [e for e, _ in desc_pairs]
        wt_list = [w for _, w in desc_pairs]

        if thesaurus_weight > 0 and sctid in thesaurus_embs:
            t_embs = thesaurus_embs[sctid]  # (N_thes, 768)
            for t_emb in t_embs:
                emb_list.append(t_emb)
                wt_list.append(thesaurus_weight)

        embs = np.array(emb_list, dtype=np.float32)
        weights = np.array(wt_list, dtype=np.float32)
        avg = np.average(embs, axis=0, weights=weights)
        avg = avg / (np.linalg.norm(avg) + 1e-12)
        concept_vectors.append(avg)
        faiss_sctids.append(sctid)

    concept_vectors = np.ascontiguousarray(np.array(concept_vectors, dtype=np.float32))
    dim = concept_vectors.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(concept_vectors)
    return idx, faiss_sctids


# ---------------------------------------------------------------------------
# Multi-vector index (no averaging)
# ---------------------------------------------------------------------------


def build_multi_vector_faiss(
    raw_embs: np.ndarray,
    raw_sctids: list[int],
    thesaurus_embs: dict[int, np.ndarray],
    thesaurus_weight: float,
) -> tuple:
    """Build FAISS index keeping every description as a separate vector.

    Instead of averaging per concept, each description/synonym gets its own
    entry. After search, results are deduplicated to unique concepts.

    Returns (faiss_index, desc_sctids_list) where desc_sctids_list[i] is the
    concept that the i-th FAISS vector belongs to.
    """
    import faiss

    all_embs: list[np.ndarray] = [raw_embs]  # (N_desc, 768) already L2-normed
    all_sctids: list[int] = list(raw_sctids)

    if thesaurus_weight > 0 and thesaurus_embs:
        for sctid, t_embs in thesaurus_embs.items():
            # t_embs: (N_thes, 768), already L2-normed
            all_embs.append(t_embs)
            all_sctids.extend([sctid] * len(t_embs))

    all_embs_arr = np.ascontiguousarray(np.vstack(all_embs), dtype=np.float32)
    dim = all_embs_arr.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(all_embs_arr)
    return idx, all_sctids


def evaluate_recall_multi_vector(
    faiss_index,
    faiss_sctids: list[int],
    query_embs: np.ndarray,
    gold_sctids: list[int],
    k_values: tuple[int, ...] = (1, 5, 10, 50),
    oversample: int = 8,
) -> dict[int, float]:
    """Recall@K with per-description vectors.

    Searches for k*oversample raw description vectors, deduplicates to unique
    concepts in rank order (first hit per concept wins), then evaluates recall.
    """
    max_k = max(k_values)
    search_k = min(max_k * oversample, faiss_index.ntotal)
    _, indices = faiss_index.search(query_embs, search_k)

    recall = {k: 0 for k in k_values}
    n = len(gold_sctids)
    for i, gold in enumerate(gold_sctids):
        # Deduplicate: keep first (highest-score) occurrence of each concept
        seen: set[int] = set()
        unique_concepts: list[int] = []
        for idx in indices[i]:
            if idx < 0:
                break
            sctid = faiss_sctids[idx]
            if sctid not in seen:
                seen.add(sctid)
                unique_concepts.append(sctid)
                if len(unique_concepts) >= max_k:
                    break
        for k in k_values:
            if gold in unique_concepts[:k]:
                recall[k] += 1

    return {k: v / n for k, v in recall.items()}


# ---------------------------------------------------------------------------
# BM25S sparse search + hybrid evaluation
# ---------------------------------------------------------------------------


def load_bm25(index_dir: Path):
    """Load BM25S index and its sctid list."""
    import pickle
    import bm25s

    bm25_index = bm25s.BM25.load(str(index_dir / "bm25s"), load_corpus=False)
    with open(index_dir / "bm25_sctids.pkl", "rb") as f:
        bm25_sctids: list[int] = pickle.load(f)
    print(f"  BM25S: {len(bm25_sctids):,} descriptions loaded")
    return bm25_index, bm25_sctids


def sparse_search_batch(bm25_index, bm25_sctids: list[int], queries: list[str], top_k: int):
    """Return list[list[(sctid, rank)]] from BM25S."""
    import bm25s as bm25s_mod
    tokens = bm25s_mod.tokenize(queries, show_progress=False)
    doc_indices, doc_scores = bm25_index.retrieve(
        tokens, k=min(top_k, len(bm25_sctids)), n_threads=8,
        show_progress=False, corpus=None,
    )
    results = []
    for i in range(len(queries)):
        hits = []
        for rank in range(doc_indices.shape[1]):
            idx = int(doc_indices[i, rank])
            score = float(doc_scores[i, rank])
            if score <= 0:
                break
            hits.append((bm25_sctids[idx], rank + 1))
        results.append(hits)
    return results


def rrf_fuse_hits(
    dense_hits: list[tuple[int, int]],   # [(sctid, rank), ...]
    sparse_hits: list[tuple[int, int]],
    rrf_k: int = 60,
    top_n: int = 50,
) -> list[int]:
    """Reciprocal Rank Fusion → ordered list of unique sctids."""
    dense_rank = {sctid: r for sctid, r in dense_hits}
    sparse_rank = {sctid: r for sctid, r in sparse_hits}
    max_rank = 10_000
    all_sctids = set(dense_rank) | set(sparse_rank)
    scored = [
        (sctid, 1.0 / (rrf_k + dense_rank.get(sctid, max_rank))
               + 1.0 / (rrf_k + sparse_rank.get(sctid, max_rank)))
        for sctid in all_sctids
    ]
    scored.sort(key=lambda x: -x[1])
    return [s for s, _ in scored[:top_n]]


def evaluate_hybrid_recall(
    faiss_index,
    faiss_sctids: list[int],
    bm25_index,
    bm25_sctids: list[int],
    query_embs: np.ndarray,
    queries: list[str],
    gold_sctids: list[int],
    k_values: tuple[int, ...] = (1, 5, 10, 50),
    multi_vector: bool = False,
    oversample: int = 8,
) -> dict[int, float]:
    """Single-query hybrid: dense RRF-fused with BM25S."""
    max_k = max(k_values)
    search_k = max_k * oversample if multi_vector else max_k

    # Dense
    _, dense_indices = faiss_index.search(query_embs, min(search_k, faiss_index.ntotal))

    # Sparse
    sparse_results = sparse_search_batch(bm25_index, bm25_sctids, queries, top_k=max_k)

    recall = {k: 0 for k in k_values}
    n = len(gold_sctids)
    for i, gold in enumerate(gold_sctids):
        # Build dense hits (deduplicated if multi-vector)
        seen: set[int] = set()
        dense_hits = []
        rank = 1
        for idx in dense_indices[i]:
            if idx < 0:
                break
            sctid = faiss_sctids[idx]
            if sctid not in seen:
                seen.add(sctid)
                dense_hits.append((sctid, rank))
                rank += 1
                if rank > max_k:
                    break

        fused = rrf_fuse_hits(dense_hits, sparse_results[i], top_n=max_k)
        for k in k_values:
            if gold in fused[:k]:
                recall[k] += 1

    return {k: v / n for k, v in recall.items()}


def evaluate_oracle_multiq_recall(
    faiss_index,
    faiss_sctids: list[int],
    bm25_index,
    bm25_sctids: list[int],
    query_embs: np.ndarray,   # embeddings for the original spans
    queries: list[str],       # original span texts
    gold_sctids: list[int],
    sctid_to_terms: dict[int, list[str]],  # gold concept → synonym texts
    sctid_to_embs: dict[int, np.ndarray],  # gold concept → synonym embeddings
    k_values: tuple[int, ...] = (1, 5, 10, 50),
    max_synonyms: int = 3,
) -> dict[int, float]:
    """Oracle multi-query hybrid: fuse across span + all known synonyms for the gold concept.

    Batches ALL queries across ALL spans into a single FAISS search and a single
    BM25S retrieve call, then reassembles per-span results.
    """
    max_k = max(k_values)
    n = len(gold_sctids)

    # Build flat list of all queries and track which span each belongs to
    all_embs_list: list[np.ndarray] = []
    all_texts: list[str] = []
    span_slices: list[tuple[int, int]] = []  # (start, end) into flat list per span

    for i, gold in enumerate(gold_sctids):
        start = len(all_texts)
        all_texts.append(queries[i])
        all_embs_list.append(query_embs[i])

        if gold in sctid_to_embs:
            extra_texts = sctid_to_terms.get(gold, [])[:max_synonyms]
            extra_embs  = sctid_to_embs[gold][:max_synonyms]
            all_texts.extend(extra_texts)
            for emb in extra_embs:
                all_embs_list.append(emb)

        span_slices.append((start, len(all_texts)))

    all_embs_arr = np.ascontiguousarray(np.array(all_embs_list, dtype=np.float32))

    # One batched FAISS search for all queries
    _, faiss_idx_batch = faiss_index.search(all_embs_arr, max_k)

    # One batched BM25S search for all queries
    sparse_all = sparse_search_batch(bm25_index, bm25_sctids, all_texts, top_k=max_k)

    recall = {k: 0 for k in k_values}
    for i, gold in enumerate(gold_sctids):
        start, end = span_slices[i]

        # Accumulate best dense rank per concept across this span's queries
        dense_concept_rank: dict[int, int] = {}
        for row in faiss_idx_batch[start:end]:
            seen: set[int] = set()
            local_rank = 1
            for idx in row:
                if idx < 0:
                    break
                sctid = faiss_sctids[idx]
                if sctid not in seen:
                    seen.add(sctid)
                    if sctid not in dense_concept_rank:
                        dense_concept_rank[sctid] = local_rank
                    local_rank += 1

        # Accumulate best sparse rank per concept
        sparse_concept_rank: dict[int, int] = {}
        for hits in sparse_all[start:end]:
            for sctid, rank in hits:
                if sctid not in sparse_concept_rank:
                    sparse_concept_rank[sctid] = rank

        fused = rrf_fuse_hits(
            list(dense_concept_rank.items()),
            list(sparse_concept_rank.items()),
            top_n=max_k,
        )
        for k in k_values:
            if gold in fused[:k]:
                recall[k] += 1

    return {k: v / n for k, v in recall.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_train_spans(max_spans: int | None = None) -> list[tuple[str, int]]:
    """Load (span_text, gold_sctid) pairs from training data."""
    import pandas as pd

    ann_path = SPLIT_DIR / "train_annotations.csv"
    notes_path = SPLIT_DIR / "train_notes.csv"

    if not ann_path.exists():
        # Fall back to main data dir
        ann_path = REPO_ROOT / "data" / "train_annotations.csv"
        notes_path = REPO_ROOT / "data" / "train_notes.csv"

    ann_df = pd.read_csv(ann_path)
    notes_df = pd.read_csv(notes_path)

    notes_map = dict(zip(notes_df["note_id"], notes_df["text"]))

    pairs = []
    seen = set()
    for _, row in ann_df.iterrows():
        note_text = notes_map.get(row["note_id"], "")
        start, end = int(row["start"]), int(row["end"])
        span = note_text[start:end].strip()
        sctid = int(row["concept_id"])
        key = (span.lower(), sctid)
        if key in seen or not span:
            continue
        seen.add(key)
        pairs.append((span, sctid))

    if max_spans and len(pairs) > max_spans:
        import random
        random.seed(42)
        pairs = random.sample(pairs, max_spans)

    print(f"Evaluation set: {len(pairs):,} unique (span, sctid) pairs")
    return pairs


def encode_spans(spans: list[str], batch_size: int = 256) -> np.ndarray:
    """Encode span texts with SapBERT."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    all_embs = []
    for i in tqdm(range(0, len(spans), batch_size), desc="Encoding spans"):
        batch = spans[i: i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        return_tensors="pt", max_length=128).to(device)
        with torch.no_grad():
            out = model(**enc)
            attn = enc["attention_mask"].unsqueeze(-1)
            mean = (out.last_hidden_state * attn).sum(1) / attn.sum(1)
            mean = torch.nn.functional.normalize(mean, dim=1)
            all_embs.append(mean.cpu().float().numpy())

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.ascontiguousarray(np.vstack(all_embs), dtype=np.float32)


def evaluate_recall(
    faiss_index,
    faiss_sctids: list[int],
    query_embs: np.ndarray,
    gold_sctids: list[int],
    k_values: tuple[int, ...] = (1, 5, 10, 50),
) -> dict[int, float]:
    """Search and compute recall@K for each K."""
    max_k = max(k_values)
    scores, indices = faiss_index.search(query_embs, min(max_k, faiss_index.ntotal))

    recall = {k: 0 for k in k_values}
    n = len(gold_sctids)
    for i, gold in enumerate(gold_sctids):
        row_indices = indices[i]
        retrieved = [faiss_sctids[idx] for idx in row_indices if idx >= 0]
        for k in k_values:
            if gold in retrieved[:k]:
                recall[k] += 1

    return {k: v / n for k, v in recall.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mrconso", type=Path, metavar="PATH",
                        help="Path to UMLS MRCONSO.RRF")
    source.add_argument("--newdict", type=Path, metavar="PATH",
                        nargs="?", const=NEWDICT_DEFAULT,
                        help=f"Path to newdict_snomed.txt (default: {NEWDICT_DEFAULT})")
    parser.add_argument(
        "--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0],
        metavar="W", help="Thesaurus term weights to sweep (default: 0 0.25 0.5 1.0)",
    )
    parser.add_argument(
        "--preferred-weight", type=float, default=2.0,
        help="Weight for SNOMED preferred terms (default: 2.0, same as main index)",
    )
    parser.add_argument(
        "--max-spans", type=int, default=None,
        help="Subsample N evaluation spans for speed (default: use all)",
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[1, 5, 10, 50],
        help="Recall@K values to report (default: 1 5 10 50)",
    )
    parser.add_argument(
        "--thesaurus-cache", type=Path,
        default=INDEX_DIR / "thesaurus_embeddings.npz",
        help="Cache path for encoded thesaurus embeddings",
    )
    parser.add_argument(
        "--span-cache", type=Path,
        default=INDEX_DIR / "eval_span_embeddings.npz",
        help="Cache path for encoded evaluation span embeddings",
    )
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Ignore embedding caches and re-encode everything",
    )
    parser.add_argument(
        "--vocab-filter", type=str, nargs="+", default=None,
        metavar="SAB",
        help="MRCONSO: only include terms from these SABs "
             "(e.g. CHV MSH NCI). Default: all ENG vocabularies.",
    )
    parser.add_argument(
        "--multi-vector", action="store_true",
        help="Also evaluate multi-vector mode (no concept averaging; each "
             "description/synonym is its own FAISS entry). Runs alongside "
             "the centroid sweep.",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Also evaluate single-query BM25S+dense hybrid (RRF) for "
             "the best centroid and multi-vector configs.",
    )
    parser.add_argument(
        "--oracle-multiq", action="store_true",
        help="Evaluate oracle multi-query hybrid: fuse the original span "
             "with ALL thesaurus synonyms of the gold concept. Upper bound "
             "on what a perfect agent with hybrid search could achieve.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load cached raw SNOMED description embeddings
    # ------------------------------------------------------------------
    cache_path = INDEX_DIR / "raw_embeddings.npz"
    if not cache_path.exists():
        print(f"ERROR: {cache_path} not found. Run the main index build first.")
        sys.exit(1)

    print(f"Loading raw SNOMED embeddings from {cache_path} ...")
    data = np.load(cache_path)
    raw_embs = data["embeddings"]           # (N_desc, 768)
    raw_sctids = data["sctids"].tolist()    # [sctid, ...]
    raw_is_preferred = data["is_preferred"].tolist()
    indexed_sctids = set(raw_sctids)
    print(f"  {len(raw_embs):,} description embeddings "
          f"({len(indexed_sctids):,} unique concepts)")

    # ------------------------------------------------------------------
    # Load thesaurus
    # ------------------------------------------------------------------
    if args.mrconso:
        mrconso_cache = INDEX_DIR / "mrconso_terms.pkl"
        sctid_to_terms = load_mrconso(args.mrconso, target_sctids=indexed_sctids,
                                      cache_path=mrconso_cache)
    else:
        ndpath = args.newdict if args.newdict else NEWDICT_DEFAULT
        sctid_to_terms = load_newdict(ndpath)

    # Filter to indexed concepts only
    sctid_to_terms = {k: v for k, v in sctid_to_terms.items() if k in indexed_sctids}
    n_covered = len(sctid_to_terms)
    n_terms = sum(len(v) for v in sctid_to_terms.values())
    print(f"  Thesaurus covers {n_covered:,}/{len(indexed_sctids):,} indexed concepts "
          f"({100 * n_covered / max(len(indexed_sctids), 1):.1f}%), "
          f"{n_terms:,} terms total")

    if args.vocab_filter and args.mrconso:
        print(f"  (Vocab filter: {args.vocab_filter})")

    # ------------------------------------------------------------------
    # Encode thesaurus terms (cached)
    # ------------------------------------------------------------------
    thesaurus_embs = encode_thesaurus_terms(
        sctid_to_terms,
        cache_path=args.thesaurus_cache,
        force_recompute=args.force_recompute,
    )

    # ------------------------------------------------------------------
    # Load / encode evaluation spans
    # ------------------------------------------------------------------
    eval_pairs = load_train_spans(args.max_spans)
    spans = [p[0] for p in eval_pairs]
    gold_sctids = [p[1] for p in eval_pairs]

    # Filter to spans whose gold concept is in the index
    valid_pairs = [(s, g) for s, g in zip(spans, gold_sctids) if g in indexed_sctids]
    n_filtered = len(eval_pairs) - len(valid_pairs)
    if n_filtered:
        print(f"  Filtered {n_filtered} spans whose gold concept is not in the index")
    spans = [p[0] for p in valid_pairs]
    gold_sctids = [p[1] for p in valid_pairs]
    print(f"  Evaluating on {len(spans):,} spans")

    if args.span_cache.exists() and not args.force_recompute:
        print(f"Loading cached span embeddings from {args.span_cache} ...")
        span_embs = np.load(args.span_cache)["embeddings"]
        if len(span_embs) != len(spans):
            print(f"  Cache size mismatch ({len(span_embs)} vs {len(spans)}); re-encoding ...")
            span_embs = None
        else:
            print(f"  Loaded {len(span_embs):,} span embeddings")
    else:
        span_embs = None

    if span_embs is None:
        print(f"Encoding {len(spans):,} evaluation spans ...")
        span_embs = encode_spans(spans)
        args.span_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.span_cache, embeddings=span_embs)
        print(f"  Span embeddings cached to {args.span_cache}")

    # ------------------------------------------------------------------
    # Weight sweep
    # ------------------------------------------------------------------
    k_values = tuple(sorted(args.k_values))
    weights = sorted(set(args.weights))

    # Ensure 0 is always included as the baseline
    if 0.0 not in weights:
        weights = [0.0] + weights

    results: list[dict] = []

    print(f"\nSweeping thesaurus weights: {weights}")
    print(f"Preferred term weight: {args.preferred_weight}x")
    print()

    for w in weights:
        label = "baseline (no thesaurus)" if w == 0 else f"thesaurus_weight={w}"
        print(f"[{label}] Building index ...")
        t0 = time.time()
        faiss_index, faiss_sctids = build_augmented_faiss(
            raw_embs, raw_sctids, raw_is_preferred,
            thesaurus_embs if w > 0 else {},
            preferred_weight=args.preferred_weight,
            thesaurus_weight=w,
        )
        build_s = time.time() - t0

        t0 = time.time()
        recall = evaluate_recall(faiss_index, faiss_sctids, span_embs, gold_sctids, k_values)
        eval_s = time.time() - t0

        result = {"weight": w, "recall": recall, "build_s": build_s, "eval_s": eval_s}
        results.append(result)

        recall_str = "  ".join(f"R@{k}={100*v:.2f}%" for k, v in recall.items())
        print(f"  {recall_str}  (build={build_s:.1f}s, eval={eval_s:.1f}s)")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    header = f"{'Weight':>10}  " + "  ".join(f"{'R@'+str(k):>10}" for k in k_values)
    print(header)
    print("-" * len(header))

    baseline = results[0]["recall"]
    for r in results:
        w = r["weight"]
        recall = r["recall"]
        label = "(baseline)" if w == 0 else ""
        row = f"{w:>10.2f}  " + "  ".join(f"{100*recall[k]:>9.2f}%" for k in k_values)
        if w > 0:
            deltas = "  Δ " + "  ".join(
                f"{'+' if recall[k] >= baseline[k] else ''}{100*(recall[k]-baseline[k]):.2f}%"
                for k in k_values
            )
            print(row + deltas)
        else:
            print(row + f"  {label}")

    print("=" * 70)

    # ------------------------------------------------------------------
    # Multi-vector comparison (optional)
    # ------------------------------------------------------------------
    if args.multi_vector:
        print("\n" + "=" * 70)
        print("MULTI-VECTOR RESULTS (no concept averaging)")
        print("=" * 70)

        mv_configs = [
            ("SNOMED only",          {}),
            ("SNOMED + thesaurus",   thesaurus_embs),
        ]
        header_mv = f"{'Mode':<30}  " + "  ".join(f"{'R@'+str(k):>10}" for k in k_values)
        print(header_mv)
        print("-" * len(header_mv))

        centroid_baseline = results[0]["recall"]
        mv_baseline_recall = None

        for label, t_embs in mv_configs:
            t_weight = 1.0 if t_embs else 0.0
            t0 = time.time()
            mv_index, mv_sctids = build_multi_vector_faiss(
                raw_embs, raw_sctids, t_embs, thesaurus_weight=t_weight,
            )
            build_s = time.time() - t0

            t0 = time.time()
            mv_recall = evaluate_recall_multi_vector(
                mv_index, mv_sctids, span_embs, gold_sctids, k_values,
            )
            eval_s = time.time() - t0

            row = f"{label:<30}  " + "  ".join(f"{100*mv_recall[k]:>9.2f}%" for k in k_values)
            deltas_vs_centroid = "  Δ(vs centroid baseline) " + "  ".join(
                f"{'+' if mv_recall[k] >= centroid_baseline[k] else ''}"
                f"{100*(mv_recall[k]-centroid_baseline[k]):.2f}%"
                for k in k_values
            )
            print(row + deltas_vs_centroid + f"  (build={build_s:.1f}s, eval={eval_s:.1f}s)")

            if mv_baseline_recall is None:
                mv_baseline_recall = mv_recall

        print("=" * 70)

    # ------------------------------------------------------------------
    # Hybrid search comparison (optional)
    # ------------------------------------------------------------------
    if args.hybrid or args.oracle_multiq:
        print("\nLoading BM25S index ...")
        bm25_index, bm25_sctids = load_bm25(INDEX_DIR)

    if args.hybrid:
        print("\n" + "=" * 70)
        print("HYBRID (dense + BM25S RRF) — single query")
        print("=" * 70)

        centroid_baseline = results[0]["recall"]

        # Best centroid config (weight=0.25 or first non-zero, else baseline)
        best_centroid_result = results[1] if len(results) > 1 else results[0]
        best_w = best_centroid_result["weight"]

        hybrid_configs = []
        # Centroid baseline + hybrid
        c_idx, c_sctids = build_augmented_faiss(
            raw_embs, raw_sctids, raw_is_preferred, {},
            preferred_weight=args.preferred_weight, thesaurus_weight=0.0,
        )
        hybrid_configs.append(("Centroid (no thes) + BM25", c_idx, c_sctids, False))

        # Multi-vector + hybrid (if requested)
        if args.multi_vector:
            mv_idx, mv_sctids_list = build_multi_vector_faiss(
                raw_embs, raw_sctids, {}, thesaurus_weight=0.0,
            )
            hybrid_configs.append(("Multi-vec (no thes) + BM25", mv_idx, mv_sctids_list, True))

        header_h = f"{'Mode':<35}  " + "  ".join(f"{'R@'+str(k):>10}" for k in k_values)
        print(header_h)
        print("-" * len(header_h))

        for label, f_idx, f_sctids, is_mv in hybrid_configs:
            t0 = time.time()
            h_recall = evaluate_hybrid_recall(
                f_idx, f_sctids, bm25_index, bm25_sctids,
                span_embs, spans, gold_sctids, k_values,
                multi_vector=is_mv,
            )
            eval_s = time.time() - t0
            row = f"{label:<35}  " + "  ".join(f"{100*h_recall[k]:>9.2f}%" for k in k_values)
            deltas = "  Δ(vs dense-only baseline) " + "  ".join(
                f"{'+' if h_recall[k] >= centroid_baseline[k] else ''}"
                f"{100*(h_recall[k]-centroid_baseline[k]):.2f}%"
                for k in k_values
            )
            print(row + deltas + f"  (eval={eval_s:.1f}s)")

        print("=" * 70)

    if args.oracle_multiq:
        print("\n" + "=" * 70)
        print("ORACLE MULTI-QUERY HYBRID (span + all gold-concept synonyms)")
        print("Upper bound: perfect agent guessing every thesaurus synonym")
        print("=" * 70)

        centroid_baseline = results[0]["recall"]

        # Use multi-vector SNOMED-only as the dense component (best single-query dense)
        mv_idx, mv_sctids_list = build_multi_vector_faiss(
            raw_embs, raw_sctids, {}, thesaurus_weight=0.0,
        )

        t0 = time.time()
        oracle_recall = evaluate_oracle_multiq_recall(
            mv_idx, mv_sctids_list, bm25_index, bm25_sctids,
            span_embs, spans, gold_sctids,
            sctid_to_terms, thesaurus_embs,
            k_values=k_values,
        )
        eval_s = time.time() - t0

        header_o = f"{'Mode':<40}  " + "  ".join(f"{'R@'+str(k):>10}" for k in k_values)
        print(header_o)
        print("-" * len(header_o))
        row = f"{'Oracle multi-query (MV + BM25 + synonyms)':<40}  " + \
              "  ".join(f"{100*oracle_recall[k]:>9.2f}%" for k in k_values)
        deltas = "  Δ(vs dense-only baseline) " + "  ".join(
            f"{'+' if oracle_recall[k] >= centroid_baseline[k] else ''}"
            f"{100*(oracle_recall[k]-centroid_baseline[k]):.2f}%"
            for k in k_values
        )
        print(row + deltas + f"  (eval={eval_s:.1f}s)")
        print("=" * 70)

    # Source info
    source_name = str(args.mrconso) if args.mrconso else str(args.newdict or NEWDICT_DEFAULT)
    print(f"\nThesaurus source: {source_name}")
    print(f"Concepts covered: {n_covered:,}/{len(indexed_sctids):,} "
          f"({100 * n_covered / max(len(indexed_sctids), 1):.1f}%)")
    print(f"Thesaurus terms:  {n_terms:,}")
    print(f"Evaluation spans: {len(spans):,}")


if __name__ == "__main__":
    main()
