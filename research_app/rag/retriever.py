"""``HybridRetriever``: the one entry point the graph uses to read stored sources (Phase 7).

    docs = await retriever.retrieve(query, filters, top_k)   # -> list[SourceDocument]

dense + BM25 search fused with RRF, then optionally reranked (``research_app.rag.reranker``).
Never raises for ordinary failures: a Qdrant outage, a timeout or a bad model returns ``[]``, so
the run falls back to live research. ``RAG_MODE=dense`` is the dense-only baseline path.

Each returned document is one chunk. Scores travel in ``metadata``: ``rag_score`` (0-1),
``rag_rank``, ``rag_reranked``, ``rag_fused_score``, ``chunk_id``, ``chunk_index``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from research_app.domain import (
    Reranker,
    RetrievalFilter,
    RetrievedChunk,
    SourceCredibility,
    SourceDocument,
    SourceType,
)
from research_app.rag.embeddings import DenseEmbedder, FastEmbedDense, FastEmbedSparse, SparseEmbedder
from research_app.rag.reranker import build_reranker
from research_app.rag.settings import RagSettings
from research_app.rag.store import SourceChunkStore

logger = logging.getLogger(__name__)

PROVIDER = "source_chunks"
SNIPPET_CHARS = 300


def chunk_to_document(hit: RetrievedChunk, query: str) -> SourceDocument:
    chunk = hit.chunk
    return SourceDocument(
        source_id=chunk.source_id,
        source_type=chunk.source_type,
        title=chunk.title,
        url=chunk.url,
        domain=chunk.domain or None,
        published_at=chunk.published_at,
        retrieved_at=chunk.retrieved_at,
        provider=PROVIDER,
        snippet=chunk.text[:SNIPPET_CHARS],
        content=chunk.text,
        technology=chunk.technology,
        version=chunk.version,
        credibility=SourceCredibility(is_official=chunk.source_type is SourceType.OFFICIAL_DOCS),
        query=query,
        metadata={
            "rag_score": hit.score,
            "rag_rank": hit.rank,
            "rag_reranked": hit.reranked,
            "rag_fused_score": hit.fused_score,
            "chunk_id": chunk.chunk_id,
            "chunk_index": chunk.chunk_index,
            "original_provider": chunk.provider,
        },
    )


class HybridRetriever:
    def __init__(self, settings: Optional[RagSettings] = None, store: Optional[SourceChunkStore] = None,
                 reranker: Optional[Reranker] = None) -> None:
        self._settings = settings or RagSettings.from_env()
        self._store = store
        self._reranker = reranker

    @property
    def store(self) -> SourceChunkStore:
        if self._store is None:
            self._store = build_store(self._settings)
        return self._store

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = build_reranker(self._settings)
        return self._reranker

    async def retrieve(self, query: str, filters: Optional[RetrievalFilter] = None,
                       top_k: Optional[int] = None) -> list[SourceDocument]:
        query = (query or "").strip()
        if not query:
            return []
        started = time.perf_counter()
        try:
            hits = await asyncio.wait_for(
                asyncio.to_thread(self._retrieve_sync, query, filters, top_k), self._settings.timeout_s)
        except asyncio.TimeoutError:
            logger.warning("Source retrieval timed out after %ss; continuing without it", self._settings.timeout_s)
            return []
        except Exception as exc:
            logger.warning("Source retrieval failed (%s: %s); continuing without it", type(exc).__name__, exc)
            return []
        logger.info("Source retrieval: mode=%s hits=%d top_score=%.2f reranked=%s %.0f ms",
                    self._settings.mode.value, len(hits), hits[0].score if hits else 0.0,
                    bool(hits and hits[0].reranked), (time.perf_counter() - started) * 1000)
        return [chunk_to_document(hit, query) for hit in hits]

    def _retrieve_sync(self, query: str, filters: Optional[RetrievalFilter],
                       top_k: Optional[int]) -> list[RetrievedChunk]:
        s = self._settings
        rerank = s.rerank_enabled and s.rerank_provider != "none"
        final = top_k or (s.rerank_top_k if rerank else s.top_k)
        candidates = self.store.search(query, flt=filters, limit=max(s.top_k, final))
        if not candidates:
            return []
        if rerank:
            return self.reranker.rerank(query, candidates, final)
        return candidates[:final]


def build_store(settings: RagSettings) -> SourceChunkStore:
    dense: DenseEmbedder = FastEmbedDense(settings.dense_model, settings.embed_batch_size)
    sparse: SparseEmbedder = FastEmbedSparse(settings.sparse_model, settings.embed_batch_size)
    return SourceChunkStore(settings, dense, sparse)
