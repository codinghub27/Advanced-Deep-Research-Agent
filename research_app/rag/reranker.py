"""Rerankers behind the ``Reranker`` protocol (Phase 7).

Swapping the provider (Cohere, Jina, ...) is one new class plus one entry in ``RERANKERS``,
selected by ``RERANK_PROVIDER``. Nothing else changes. The default is a local sentence-transformers
CrossEncoder that loads lazily on the first ``rerank`` call; when it cannot load or score, the
chunks come back in their incoming order (logged once), so retrieval never fails because of it.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Sequence

from research_app.domain import Reranker, RetrievedChunk
from research_app.rag.settings import RagSettings

logger = logging.getLogger(__name__)


def _reordered(chunks: Sequence[RetrievedChunk], scores: Sequence[float], top_k: int) -> list[RetrievedChunk]:
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [
        RetrievedChunk(
            chunk=item.chunk,
            score=min(1.0, max(0.0, float(score))),
            rank=rank,
            fused_score=item.fused_score if item.fused_score is not None else item.score,
            reranked=True,
        )
        for rank, (item, score) in enumerate(ranked)
    ]


class NoopReranker:
    """Keeps the incoming order and scores."""

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return list(chunks)[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512, batch_size: int = 16) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._batch_size = batch_size
        self._model = None
        self._activation: Any = None
        self._failed = False
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None and not self._failed:
            with self._lock:
                if self._model is None and not self._failed:
                    try:
                        from sentence_transformers import CrossEncoder
                        from torch import nn

                        self._model = CrossEncoder(self._model_name, max_length=self._max_length)
                        # Some checkpoints (e.g. ms-marco MiniLM) declare an identity activation and
                        # return raw logits; force a sigmoid so every model scores 0-1.
                        self._activation = nn.Sigmoid()
                        logger.info("Reranker loaded: %s", self._model_name)
                    except Exception as exc:
                        self._failed = True
                        logger.warning("Reranker %s unavailable (%s); keeping the fused order",
                                       self._model_name, type(exc).__name__)
        return self._model

    def rerank(self, query: str, chunks: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        model = self._load()
        if model is None:
            return list(chunks)[:top_k]
        try:
            options: dict[str, Any] = {"activation_fn": self._activation} if self._activation is not None else {}
            scores = model.predict([(query, c.chunk.text) for c in chunks], batch_size=self._batch_size, **options)
            return _reordered(chunks, [float(s) for s in scores], top_k)
        except Exception as exc:
            logger.warning("Reranking failed (%s); keeping the fused order", type(exc).__name__)
            return list(chunks)[:top_k]


RerankerFactory = Callable[[RagSettings], Reranker]

RERANKERS: dict[str, RerankerFactory] = {
    "cross_encoder": lambda s: CrossEncoderReranker(s.rerank_model, s.rerank_max_length, s.rerank_batch_size),
    "none": lambda s: NoopReranker(),
}


def build_reranker(settings: RagSettings) -> Reranker:
    if not settings.rerank_enabled:
        return NoopReranker()
    factory = RERANKERS.get(settings.rerank_provider)
    if factory is None:
        logger.warning("Unknown RERANK_PROVIDER %r (known: %s); reranking disabled",
                       settings.rerank_provider, sorted(RERANKERS))
        return NoopReranker()
    return factory(settings)
