"""Deterministic fake embedders and document builders for the Phase 7 tests (no models, no server)."""
from __future__ import annotations

import math
import re
import zlib
from dataclasses import replace
from datetime import datetime
from typing import Any, Optional, Sequence

from research_app.domain import SourceDocument, SourceType, utc_now
from research_app.rag.settings import RagSettings
from research_app.rag.store import SourceChunkStore

_WORD = re.compile(r"[a-z0-9]+")
DIM = 64


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bucket(word: str, size: int) -> int:
    return zlib.crc32(word.encode()) % size


class FakeDense:
    """Hashed bag of words, L2-normalised: texts that share words are close."""

    dim = DIM

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for w in words(text):
            vec[_bucket(w, DIM)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeSparse:
    """Term counts keyed by word hash: only exact words match."""

    def _vec(self, text: str) -> tuple[list[int], list[float]]:
        counts: dict[int, float] = {}
        for w in words(text):
            counts[_bucket(w, 100_000)] = counts.get(_bucket(w, 100_000), 0.0) + 1.0
        return list(counts), list(counts.values())

    def embed_documents(self, texts: Sequence[str]):
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str):
        return self._vec(text)


def settings(**overrides: Any) -> RagSettings:
    return replace(RagSettings(qdrant_url=":memory:", rerank_enabled=False, timeout_s=5), **overrides)


def make_store(**overrides) -> SourceChunkStore:
    return SourceChunkStore(settings(**overrides), FakeDense(), FakeSparse())


def doc(url: str, content: str, *, source_type: SourceType = SourceType.WEB, technology: Optional[str] = None,
        title: Optional[str] = None, retrieved_at: Optional[datetime] = None,
        provider: str = "tavily") -> SourceDocument:
    return SourceDocument(url=url, content=content, source_type=source_type, technology=technology,
                          title=title or url, provider=provider, retrieved_at=retrieved_at or utc_now())


CORPUS = [
    doc("https://fastapi.tiangolo.com/tutorial/first-steps/",
        "Install FastAPI with pip install fastapi uvicorn. Then create main.py and run the development server.",
        source_type=SourceType.OFFICIAL_DOCS, technology="FastAPI"),
    doc("https://www.postgresql.org/docs/current/sql-createindex.html",
        "CREATE INDEX CONCURRENTLY builds an index without locking writes to the table.",
        source_type=SourceType.OFFICIAL_DOCS, technology="PostgreSQL"),
    doc("https://www.reddit.com/r/Python/comments/abc/slow_uvicorn/",
        "My uvicorn server feels slow under load, workers helped a lot in production deployments.",
        source_type=SourceType.REDDIT, technology="FastAPI"),
    doc("https://example.com/blog/cooking",
        "A simple recipe for tomato soup with basil and garlic.", source_type=SourceType.WEB),
]
