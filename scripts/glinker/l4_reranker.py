#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.glinker.io import normalize_alias


def _resolve_device(device: str) -> str:
    d = str(device).strip().lower()
    if d and d != "auto":
        return device
    try:
        import torch  # type: ignore

        if bool(torch.cuda.is_available()):
            return "cuda"
    except Exception:
        pass
    return "cpu"


class _PairScorer:
    def score_pairs(self, mentions: list[str], aliases: list[str]) -> np.ndarray:
        raise NotImplementedError


class _HashPairScorer(_PairScorer):
    @staticmethod
    def _score_one(m: str, a: str) -> float:
        m_toks = set(normalize_alias(m).split())
        a_toks = set(normalize_alias(a).split())
        if not m_toks and not a_toks:
            return 0.0
        if not m_toks or not a_toks:
            return 0.0
        inter = len(m_toks & a_toks)
        union = len(m_toks | a_toks)
        return float(inter / max(1, union))

    def score_pairs(self, mentions: list[str], aliases: list[str]) -> np.ndarray:
        vals = [self._score_one(m, a) for m, a in zip(mentions, aliases)]
        return np.array(vals, dtype=np.float32)


class _HFPairScorer(_PairScorer):
    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        batch_size: int,
        max_length: int,
        load_dtype: str = "auto",
    ):
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(8, int(max_length))
        self.device = _resolve_device(device)

        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Transformers + torch are required for HF L4 reranker. "
                "Install them or use backend=hash."
            ) from e

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_kwargs = {}
        raw_dtype = str(load_dtype).strip().lower()
        if raw_dtype in {"float16", "fp16"}:
            model_kwargs["torch_dtype"] = torch.float16
        elif raw_dtype in {"bfloat16", "bf16"}:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif raw_dtype in {"float32", "fp32"}:
            model_kwargs["torch_dtype"] = torch.float32
        elif raw_dtype in {"auto", "", "none"}:
            pass
        else:
            raise ValueError(f"Unsupported load_dtype: {load_dtype}")

        self._model = AutoModelForSequenceClassification.from_pretrained(model_path, **model_kwargs)
        self._model.eval()
        self._model.to(self.device)

    def score_pairs(self, mentions: list[str], aliases: list[str]) -> np.ndarray:
        if not mentions:
            return np.zeros((0,), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(mentions), self.batch_size):
            m_batch = mentions[i : i + self.batch_size]
            a_batch = aliases[i : i + self.batch_size]
            tok = self._tokenizer(
                m_batch,
                a_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with self._torch.no_grad():
                logits = self._model(**tok).logits
            if logits.ndim == 2 and logits.shape[1] > 1:
                scores = logits[:, 1]
            else:
                scores = logits.squeeze(-1)
            out.append(scores.detach().cpu().float().numpy())
        return np.concatenate(out, axis=0).astype(np.float32, copy=False)


def _make_pair_scorer(
    *,
    backend: str,
    model_path: str,
    device: str,
    batch_size: int,
    max_length: int,
    load_dtype: str,
) -> _PairScorer:
    b = str(backend).strip().lower()
    if b in {"hash", "hashed"}:
        return _HashPairScorer()
    if b in {"hf", "transformers", "auto"}:
        return _HFPairScorer(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            load_dtype=load_dtype,
        )
    raise ValueError(f"Unsupported L4 backend: {backend}")


@dataclass(frozen=True)
class L4RerankConfig:
    top_n: int = 1


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_path: str,
        backend: str = "auto",
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 128,
        load_dtype: str = "auto",
        config: L4RerankConfig | None = None,
    ):
        self.config = config or L4RerankConfig()
        self.scorer = _make_pair_scorer(
            backend=backend,
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            load_dtype=load_dtype,
        )

    def rerank(
        self,
        mention: str,
        candidates: list[dict],
        *,
        top_n: int | None = None,
    ) -> list[dict]:
        if not candidates:
            return []
        use_top_n = int(top_n if top_n is not None else self.config.top_n)
        if use_top_n <= 0:
            use_top_n = len(candidates)

        aliases = []
        for c in candidates:
            alias = str(c.get("matched_alias") or "").strip()
            if not alias:
                alias = str(c.get("canonical_name") or c.get("concept_name") or c.get("concept_id") or "")
            aliases.append(alias)
        mentions = [str(mention)] * len(candidates)
        scores = self.scorer.score_pairs(mentions, aliases)

        ranked = []
        for i, c in enumerate(candidates):
            row = dict(c)
            row["base_score"] = float(c.get("score", 0.0))
            row["base_method"] = str(c.get("method") or "")
            row["score"] = float(scores[i])
            row["method"] = "l4_cross_rerank"
            ranked.append(row)
        ranked.sort(key=lambda x: (-float(x.get("score", 0.0)), -float(x.get("base_score", 0.0))))
        return ranked[:use_top_n]
