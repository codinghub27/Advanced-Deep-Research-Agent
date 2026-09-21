"""Index a run's source documents into ``source_chunks`` (Phase 7).

Never raises: a failure is logged and reported as 0 chunks, because indexing is a side effect of
answering, not part of it. Runs in a worker thread under ``RAG_TIMEOUT_S``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Iterable, Optional

from research_app.domain import SourceDocument
from research_app.rag.chunking import chunk_document
from research_app.rag.retriever import PROVIDER, build_store
from research_app.rag.settings import RagSettings
from research_app.rag.store import SourceChunkStore

logger = logging.getLogger(__name__)


def indexable(documents: Iterable[SourceDocument]) -> list[SourceDocument]:
    """Documents with text that did not themselves come out of ``source_chunks``."""
    return [d for d in documents if d.provider != PROVIDER and (d.content or d.snippet or "").strip()]


class SourceIndexer:
    def __init__(self, settings: Optional[RagSettings] = None, store: Optional[SourceChunkStore] = None) -> None:
        self._settings = settings or RagSettings.from_env()
        self._store = store

    @property
    def store(self) -> SourceChunkStore:
        if self._store is None:
            self._store = build_store(self._settings)
        return self._store

    def _index_sync(self, documents: list[SourceDocument]) -> int:
        s = self._settings
        chunks = [c for d in documents for c in chunk_document(d, s.chunk_size, s.chunk_overlap)]
        return self.store.upsert(chunks)

    async def index(self, documents: Iterable[SourceDocument]) -> int:
        docs = indexable(documents)
        if not docs:
            return 0
        started = time.perf_counter()
        try:
            written = await asyncio.wait_for(
                asyncio.to_thread(self._index_sync, docs), self._settings.timeout_s)
        except asyncio.TimeoutError:
            logger.warning("Source indexing timed out after %ss", self._settings.timeout_s)
            return 0
        except Exception as exc:
            logger.warning("Source indexing failed (%s: %s)", type(exc).__name__, exc)
            return 0
        logger.info("Indexed %d chunks from %d documents into %s in %.0f ms", written, len(docs),
                    self._settings.collection, (time.perf_counter() - started) * 1000)
        return written
