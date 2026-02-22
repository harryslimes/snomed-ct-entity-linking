"""FastAPI server for the SNOMED CT index.

Keeps FAISS, BM25S, SapBERT, and the subsumption index hot in memory so
batch jobs can query without paying per-process startup cost.

Usage:
    python snomed_index_server.py              # default port 8421
    python snomed_index_server.py --port 9000

Endpoints:
    GET  /health
    POST /search/hybrid
    POST /search/batch_hybrid
    POST /search/dense
    POST /search/sparse
    POST /subsumption/ancestors
    POST /subsumption/parents
    POST /subsumption/is_descendant
    POST /subsumption/lca
    GET  /subsumption/name/{concept_id}
"""
from __future__ import annotations

import argparse
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import bm25s
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent
INDEX_DIR = REPO_ROOT / "snomed_index"

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


def _load_encoder():
    """Load SapBERT onto GPU (or CPU if CUDA unavailable)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token"
    print(f"Loading SapBERT: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    print(f"  SapBERT loaded on {device}.")
    return tokenizer, model, device


def _encode(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Encode texts with SapBERT (mean pooling + L2 norm)."""
    import torch

    tokenizer = _state["tokenizer"]
    model = _state["model"]
    device = _state["device"]

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            return_tensors="pt", max_length=128,
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
            attn = encoded["attention_mask"].unsqueeze(-1)
            embeds = output.last_hidden_state
            mean = (embeds * attn).sum(dim=1) / attn.sum(dim=1)
            mean = torch.nn.functional.normalize(mean, dim=1)
            all_embeddings.append(mean.cpu().float().numpy())

    return np.ascontiguousarray(np.vstack(all_embeddings), dtype=np.float32)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading SNOMED index ...")

    # FAISS
    faiss_index = faiss.read_index(str(INDEX_DIR / "vectors.index"))
    with open(INDEX_DIR / "faiss_sctids.pkl", "rb") as f:
        faiss_sctids: list[int] = pickle.load(f)
    print(f"  FAISS: {faiss_index.ntotal:,} concept vectors")

    # BM25S
    bm25_index = bm25s.BM25.load(str(INDEX_DIR / "bm25s"), load_corpus=False)
    with open(INDEX_DIR / "bm25_sctids.pkl", "rb") as f:
        bm25_sctids: list[int] = pickle.load(f)
    print(f"  BM25S: {len(bm25_sctids):,} descriptions")

    # Subsumption
    with open(INDEX_DIR / "subsumption.pkl", "rb") as f:
        sub_data = pickle.load(f)
    ancestors: dict[int, set[int]] = sub_data["ancestors"]
    parents: dict[int, set[int]] = sub_data["parents"]
    concept_names: dict[int, str] = sub_data["concept_names"]
    print(f"  Subsumption: {len(concept_names):,} concepts")

    # SapBERT
    tokenizer, model, device = _load_encoder()

    _state.update(
        faiss_index=faiss_index,
        faiss_sctids=faiss_sctids,
        bm25_index=bm25_index,
        bm25_sctids=bm25_sctids,
        ancestors=ancestors,
        parents=parents,
        concept_names=concept_names,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    print("SNOMED index server ready.")
    yield
    _state.clear()


app = FastAPI(title="SNOMED CT Index Server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    queries: list[str]
    top_k: int = 100


class HybridSearchRequest(BaseModel):
    queries: list[str]
    dense_top_k: int = 100
    sparse_top_k: int = 100
    rrf_k: int = 60
    fusion_top_k: int = 50


class SearchHit(BaseModel):
    sctid: int
    score: float
    rank: int
    name: str | None = None


class SearchResponse(BaseModel):
    results: list[list[SearchHit]]


class HybridHit(BaseModel):
    sctid: int
    rrf_score: float
    rank: int
    name: str | None = None


class HybridSearchResponse(BaseModel):
    results: list[list[HybridHit]]


class BatchItem(BaseModel):
    id: str
    queries: list[str]
    top_k: int = 10


class BatchHybridRequest(BaseModel):
    items: list[BatchItem]
    dense_top_k: int = 100
    sparse_top_k: int = 100
    rrf_k: int = 60


class BatchHybridResult(BaseModel):
    id: str
    results: list[HybridHit]


class BatchHybridResponse(BaseModel):
    results: list[BatchHybridResult]


class AncestorsRequest(BaseModel):
    concept_id: int


class AncestorsResponse(BaseModel):
    concept_id: int
    name: str | None
    ancestors: list[int]


class ParentsResponse(BaseModel):
    concept_id: int
    name: str | None
    parents: list[int]


class IsDescendantRequest(BaseModel):
    concept_id: int
    ancestor_id: int


class IsDescendantResponse(BaseModel):
    concept_id: int
    ancestor_id: int
    is_descendant: bool


class LCARequest(BaseModel):
    concept_ids: list[int]
    min_depth: int = 0


class LCAResponse(BaseModel):
    lca: int | None
    name: str | None
    candidates: list[dict]  # [{concept_id, name, depth}] sorted most-specific first


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _name(sctid: int) -> str | None:
    return _state["concept_names"].get(sctid)


def _dense_search(query_embeddings: np.ndarray, top_k: int, oversample: int = 4) -> list[list[SearchHit]]:
    faiss_index: faiss.Index = _state["faiss_index"]
    faiss_sctids: list[int] = _state["faiss_sctids"]

    search_k = min(top_k * oversample, faiss_index.ntotal)
    scores, indices = faiss_index.search(query_embeddings, search_k)
    results = []
    for i in range(len(query_embeddings)):
        seen: set[int] = set()
        hits = []
        rank = 1
        for idx, score in zip(indices[i], scores[i]):
            if idx == -1:
                break
            sctid = faiss_sctids[idx]
            if sctid not in seen:
                seen.add(sctid)
                hits.append(SearchHit(sctid=sctid, score=float(score), rank=rank, name=_name(sctid)))
                rank += 1
                if rank > top_k:
                    break
        results.append(hits)
    return results


def _sparse_search(queries: list[str], top_k: int) -> list[list[SearchHit]]:
    bm25_index: bm25s.BM25 = _state["bm25_index"]
    bm25_sctids: list[int] = _state["bm25_sctids"]

    query_tokens = bm25s.tokenize(queries, show_progress=False)
    doc_indices, doc_scores = bm25_index.retrieve(
        query_tokens, k=top_k, n_threads=16, show_progress=False, corpus=None,
    )

    results = []
    for i in range(len(queries)):
        hits = []
        for rank in range(doc_indices.shape[1]):
            idx = int(doc_indices[i, rank])
            score = float(doc_scores[i, rank])
            if score <= 0:
                break
            sctid = bm25_sctids[idx]
            hits.append(SearchHit(sctid=sctid, score=score, rank=rank + 1, name=_name(sctid)))
        results.append(hits)
    return results


def _rrf_fuse(
    dense: list[SearchHit],
    sparse: list[SearchHit],
    rrf_k: int,
    top_n: int,
) -> list[HybridHit]:
    dense_rank = {h.sctid: h.rank for h in dense}
    sparse_rank = {h.sctid: h.rank for h in sparse}
    all_sctids = set(dense_rank) | set(sparse_rank)
    max_rank = 10_000

    scored = []
    for sctid in all_sctids:
        dr = dense_rank.get(sctid, max_rank)
        sr = sparse_rank.get(sctid, max_rank)
        rrf_score = 1.0 / (rrf_k + dr) + 1.0 / (rrf_k + sr)
        scored.append((sctid, rrf_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        HybridHit(sctid=sctid, rrf_score=score, rank=rank + 1, name=_name(sctid))
        for rank, (sctid, score) in enumerate(scored[:top_n])
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    if not _state:
        raise HTTPException(503, "Index not loaded")
    return {
        "status": "ok",
        "faiss_vectors": _state["faiss_index"].ntotal,
        "bm25_docs": len(_state["bm25_sctids"]),
        "subsumption_concepts": len(_state["concept_names"]),
        "device": _state["device"],
    }


@app.post("/search/dense", response_model=SearchResponse)
def search_dense(req: SearchRequest):
    embeddings = _encode(req.queries)
    return SearchResponse(results=_dense_search(embeddings, req.top_k))


@app.post("/search/sparse", response_model=SearchResponse)
def search_sparse(req: SearchRequest):
    return SearchResponse(results=_sparse_search(req.queries, req.top_k))


@app.post("/search/hybrid", response_model=HybridSearchResponse)
def search_hybrid(req: HybridSearchRequest):
    embeddings = _encode(req.queries)
    dense_results = _dense_search(embeddings, req.dense_top_k)
    sparse_results = _sparse_search(req.queries, req.sparse_top_k)

    fused = [
        _rrf_fuse(d, s, req.rrf_k, req.fusion_top_k)
        for d, s in zip(dense_results, sparse_results)
    ]
    return HybridSearchResponse(results=fused)


@app.post("/search/batch_hybrid", response_model=BatchHybridResponse)
def search_batch_hybrid(req: BatchHybridRequest):
    """Multi-query batch search: one encode/search pass, per-item RRF fusion.

    Each item has an id and multiple queries (e.g. abbreviation expansions).
    All queries are flattened into a single SapBERT + FAISS + BM25 pass,
    then results are grouped by item and fused with cross-query RRF.
    """
    if not req.items:
        return BatchHybridResponse(results=[])

    # Flatten all queries, tracking which item each belongs to
    all_queries: list[str] = []
    item_slices: list[tuple[int, int]] = []  # (start, end) per item
    for item in req.items:
        start = len(all_queries)
        all_queries.extend(item.queries)
        item_slices.append((start, len(all_queries)))

    if not all_queries:
        return BatchHybridResponse(
            results=[BatchHybridResult(id=item.id, results=[]) for item in req.items]
        )

    # Single batched encode + search
    embeddings = _encode(all_queries)
    dense_results = _dense_search(embeddings, req.dense_top_k)
    sparse_results = _sparse_search(all_queries, req.sparse_top_k)

    # Per-item: fuse across queries with RRF
    batch_results = []
    for item, (start, end) in zip(req.items, item_slices):
        # Collect all hits across this item's queries
        sctid_best_dense_rank: dict[int, int] = {}
        sctid_best_sparse_rank: dict[int, int] = {}

        for qi in range(start, end):
            for hit in dense_results[qi]:
                prev = sctid_best_dense_rank.get(hit.sctid)
                if prev is None or hit.rank < prev:
                    sctid_best_dense_rank[hit.sctid] = hit.rank
            for hit in sparse_results[qi]:
                prev = sctid_best_sparse_rank.get(hit.sctid)
                if prev is None or hit.rank < prev:
                    sctid_best_sparse_rank[hit.sctid] = hit.rank

        # RRF over best ranks from any query
        all_sctids = set(sctid_best_dense_rank) | set(sctid_best_sparse_rank)
        max_rank = 10_000
        scored = []
        for sctid in all_sctids:
            dr = sctid_best_dense_rank.get(sctid, max_rank)
            sr = sctid_best_sparse_rank.get(sctid, max_rank)
            rrf_score = 1.0 / (req.rrf_k + dr) + 1.0 / (req.rrf_k + sr)
            scored.append((sctid, rrf_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        hits = [
            HybridHit(sctid=sctid, rrf_score=score, rank=rank + 1, name=_name(sctid))
            for rank, (sctid, score) in enumerate(scored[:item.top_k])
        ]
        batch_results.append(BatchHybridResult(id=item.id, results=hits))

    return BatchHybridResponse(results=batch_results)


@app.post("/subsumption/ancestors", response_model=AncestorsResponse)
def subsumption_ancestors(req: AncestorsRequest):
    anc = _state["ancestors"].get(req.concept_id, set())
    return AncestorsResponse(
        concept_id=req.concept_id,
        name=_name(req.concept_id),
        ancestors=sorted(anc),
    )


@app.post("/subsumption/parents", response_model=ParentsResponse)
def subsumption_parents(req: AncestorsRequest):
    par = _state["parents"].get(req.concept_id, set())
    return ParentsResponse(
        concept_id=req.concept_id,
        name=_name(req.concept_id),
        parents=sorted(par),
    )


@app.post("/subsumption/is_descendant", response_model=IsDescendantResponse)
def subsumption_is_descendant(req: IsDescendantRequest):
    if req.concept_id == req.ancestor_id:
        is_desc = True
    else:
        is_desc = req.ancestor_id in _state["ancestors"].get(req.concept_id, set())
    return IsDescendantResponse(
        concept_id=req.concept_id,
        ancestor_id=req.ancestor_id,
        is_descendant=is_desc,
    )


@app.post("/subsumption/lca", response_model=LCAResponse)
def subsumption_lca(req: LCARequest):
    concept_ids = req.concept_ids
    if not concept_ids:
        return LCAResponse(lca=None, name=None, candidates=[])

    ancestors: dict[int, set[int]] = _state["ancestors"]

    common = ancestors.get(concept_ids[0], set()).copy()
    common.add(concept_ids[0])
    for cid in concept_ids[1:]:
        anc = ancestors.get(cid, set()).copy()
        anc.add(cid)
        common &= anc

    if not common:
        return LCAResponse(lca=None, name=None, candidates=[])

    candidates = []
    for c in common:
        depth = len(ancestors.get(c, set()))
        if depth >= req.min_depth:
            candidates.append({"concept_id": c, "name": _name(c), "depth": depth})
    candidates.sort(key=lambda x: -x["depth"])

    lca = candidates[0]["concept_id"] if candidates else None
    return LCAResponse(lca=lca, name=_name(lca) if lca else None, candidates=candidates)


@app.get("/subsumption/name/{concept_id}")
def subsumption_name(concept_id: int):
    name = _state["concept_names"].get(concept_id)
    if name is None:
        raise HTTPException(404, f"Concept {concept_id} not found")
    return {"concept_id": concept_id, "name": name}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
