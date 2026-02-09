"""Reranking strategies for SNOMED CT entity linking."""

import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import Config


class SapBERTReranker:
    """
    Rerank candidates using max-synonym cosine similarity.

    Instead of comparing the mention against the concept-averaged embedding
    (which the FAISS bi-encoder does), this compares against each individual
    synonym embedding and takes the max. This gives a tighter match.
    """

    def __init__(self, raw_embeddings: np.ndarray, raw_sctids: list[int]):
        print("Building SapBERT max-synonym reranker ...")
        t0 = time.time()

        # Build sctid -> array of embedding indices
        self.raw_embeddings = raw_embeddings
        self.sctid_to_indices: dict[int, list[int]] = {}
        for i, sctid in enumerate(raw_sctids):
            if sctid not in self.sctid_to_indices:
                self.sctid_to_indices[sctid] = []
            self.sctid_to_indices[sctid].append(i)

        n_concepts = len(self.sctid_to_indices)
        avg_syns = len(raw_sctids) / n_concepts
        print(f"  {n_concepts:,} concepts, {len(raw_sctids):,} synonym embeddings "
              f"(avg {avg_syns:.1f} per concept, built in {time.time() - t0:.1f}s)")

    def rerank(
        self,
        query_embeddings: np.ndarray,
        candidate_sctids_per_query: list[list[int]],
    ) -> list[list[tuple[int, float]]]:
        """
        Rerank candidates using max cosine similarity over all synonyms.

        For each (query, candidate_concept), computes:
            score = max(cos(query_emb, syn_emb) for syn_emb in concept_synonyms)

        Args:
            query_embeddings: Pre-computed query embeddings (N x dim), L2-normalized.
            candidate_sctids_per_query: Per-query list of candidate SCTIDs.

        Returns:
            Per-query list of (sctid, score) sorted by score descending.
        """
        print(f"SapBERT max-synonym reranking {len(query_embeddings):,} queries ...")
        t0 = time.time()
        n_pairs = 0
        all_reranked = []

        for q_emb, candidates in zip(query_embeddings, candidate_sctids_per_query):
            scored = []
            for sctid in candidates:
                indices = self.sctid_to_indices.get(sctid)
                if indices:
                    # Cosine similarity = dot product (both L2-normalized)
                    syn_embeddings = self.raw_embeddings[indices]
                    similarities = syn_embeddings @ q_emb
                    max_sim = float(similarities.max())
                    n_pairs += len(indices)
                else:
                    max_sim = 0.0
                scored.append((sctid, max_sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            all_reranked.append(scored)

        elapsed = time.time() - t0
        print(f"  Reranked {n_pairs:,} synonym comparisons in {elapsed:.1f}s "
              f"({n_pairs / elapsed:.0f} comparisons/s)")
        return all_reranked

    def close(self):
        """Free memory."""
        del self.raw_embeddings
        del self.sctid_to_indices


class CrossEncoderReranker:
    """Reranker using a cross-encoder model (MedCPT-Cross-Encoder)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        print(f"Loading cross-encoder: {cfg.cross_encoder_model} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.cross_encoder_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.cross_encoder_model
        ).cuda().eval()
        print("  Cross-encoder loaded on GPU.")

    def rerank(
        self,
        mention_spans: list[str],
        candidate_sctids_per_query: list[list[int]],
        sctid_to_fsn: dict[int, str],
    ) -> list[list[tuple[int, float]]]:
        """
        Rerank candidates for each query using the cross-encoder.

        Args:
            mention_spans: List of mention span texts (short clinical phrases).
            candidate_sctids_per_query: Per-query list of candidate SCTIDs.
            sctid_to_fsn: SCTID -> FSN mapping for constructing pair text.

        Returns:
            Per-query list of (sctid, score) sorted by cross-encoder score descending.
        """
        print(f"Cross-encoder reranking {len(mention_spans):,} queries ...")
        t0 = time.time()
        all_reranked = []

        # Flatten all pairs for batched inference
        # MedCPT expects (short_query, document) — use mention span as query
        flat_pairs = []
        pair_indices = []  # (query_idx, candidate_idx_within_query)

        for q_idx, (span, sctids) in enumerate(zip(mention_spans, candidate_sctids_per_query)):
            for c_idx, sctid in enumerate(sctids):
                fsn = sctid_to_fsn.get(sctid, "Unknown concept")
                flat_pairs.append((span, fsn))
                pair_indices.append((q_idx, c_idx))

        # Batched inference
        flat_scores = self._batch_predict(flat_pairs)

        # Reconstruct per-query results
        query_scores: dict[int, list[tuple[int, float]]] = {}
        for (q_idx, c_idx), score in zip(pair_indices, flat_scores):
            if q_idx not in query_scores:
                query_scores[q_idx] = []
            sctid = candidate_sctids_per_query[q_idx][c_idx]
            query_scores[q_idx].append((sctid, score))

        for q_idx in range(len(mention_spans)):
            candidates = query_scores.get(q_idx, [])
            candidates.sort(key=lambda x: x[1], reverse=True)
            all_reranked.append(candidates)

        elapsed = time.time() - t0
        print(f"  Reranked {len(flat_pairs):,} pairs in {elapsed:.1f}s "
              f"({len(flat_pairs) / elapsed:.0f} pairs/s)")
        return all_reranked

    def _batch_predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run cross-encoder inference in batches."""
        scores = []
        bs = self.cfg.rerank_batch_size

        for i in range(0, len(pairs), bs):
            batch = pairs[i : i + bs]
            queries = [p[0] for p in batch]
            docs = [p[1] for p in batch]

            encoded = self.tokenizer(
                queries, docs,
                truncation=True, padding=True,
                return_tensors="pt", max_length=512,
            ).to("cuda")

            with torch.no_grad():
                logits = self.model(**encoded).logits.squeeze(dim=-1)
                scores.extend(logits.cpu().tolist())

        return scores

    def close(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
