"""Environment-driven settings for hybrid source retrieval (Phase 7).

Feature flags (both default false: with them off the graph is exactly the Phase 6 graph)
``SOURCE_RAG_ENABLED``         retrieve from ``source_chunks`` after a cache miss, before research
``SOURCE_RAG_INGEST_ENABLED``  index each run's source documents into ``source_chunks``

Qdrant
``QDRANT_URL``                 default http://localhost:6333 (":memory:" or a folder path runs the
                               embedded local engine, used by tests and the seed script)
``QDRANT_API_KEY``             optional secret, never logged
``QDRANT_TIMEOUT_S``           default 5, allowed 1-120: per Qdrant request
``SOURCE_CHUNKS_COLLECTION``   default source_chunks

Retrieval
``RAG_MODE``                   hybrid (default) | dense | sparse. ``dense`` is the baseline path
``RAG_DENSE_MODEL`` / ``RAG_SPARSE_MODEL``   fastembed model names
``RAG_DENSE_TOP_K`` / ``RAG_SPARSE_TOP_K``   default 30 each: candidates per leg before fusion
``RAG_TOP_K``                  default 8: chunks returned after fusion
``RAG_RRF_K``                  default 60: reciprocal-rank-fusion constant
``RAG_RELEVANCE_THRESHOLD``    default 0.5: top score needed to answer from stored sources
``RAG_MIN_CHUNKS``             default 2: chunks at or above the threshold needed to do so
``RAG_TIMEOUT_S``              default 30: whole retrieval (embedding + search + rerank)
``RAG_EMBED_BATCH_SIZE``       default 32
``RAG_CHUNK_SIZE`` / ``RAG_CHUNK_OVERLAP``   default 1200 / 150 characters

Reranking
``RERANK_ENABLED``             default true (falls back to the fused order if it cannot load)
``RERANK_PROVIDER``            default cross_encoder; ``none`` disables. See ``rag.reranker``
``RERANK_MODEL``               default BAAI/bge-reranker-v2-m3
``RERANK_TOP_K``               default 5
``RERANK_MAX_LENGTH``          default 512 tokens
``RERANK_BATCH_SIZE``          default 16

Read at call time; malformed values fall back to their defaults. Pure: no Qdrant / model imports.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

from research_app.conversation.settings import read_bool, read_float, read_int
from research_app.domain.enums import RetrievalMode

logger = logging.getLogger(__name__)


def read_str(env: Mapping[str, str], name: str, default: str) -> str:
    raw = env.get(name, "").strip()
    return raw or default


def read_mode(env: Mapping[str, str], name: str, default: RetrievalMode) -> RetrievalMode:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    try:
        return RetrievalMode(raw)
    except ValueError:
        logger.warning("%s must be one of %s; using %s", name, [m.value for m in RetrievalMode], default.value)
        return default


@dataclass(frozen=True)
class RagSettings:
    source_rag_enabled: bool = False
    ingest_enabled: bool = False
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = field(default=None, repr=False)
    qdrant_timeout_s: int = 5
    collection: str = "source_chunks"
    mode: RetrievalMode = RetrievalMode.HYBRID
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    dense_top_k: int = 30
    sparse_top_k: int = 30
    top_k: int = 8
    rrf_k: int = 60
    relevance_threshold: float = 0.5
    min_chunks: int = 2
    timeout_s: int = 30
    embed_batch_size: int = 32
    chunk_size: int = 1200
    chunk_overlap: int = 150
    rerank_enabled: bool = True
    rerank_provider: str = "cross_encoder"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 5
    rerank_max_length: int = 512
    rerank_batch_size: int = 16

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "RagSettings":
        env = os.environ if env is None else env
        chunk_size = read_int(env, "RAG_CHUNK_SIZE", 1200, 200, 8000)
        overlap = read_int(env, "RAG_CHUNK_OVERLAP", 150, 0, 2000)
        if overlap >= chunk_size:
            logger.warning("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE; using %d", chunk_size // 8)
            overlap = chunk_size // 8
        return cls(
            source_rag_enabled=read_bool(env, "SOURCE_RAG_ENABLED", False),
            ingest_enabled=read_bool(env, "SOURCE_RAG_INGEST_ENABLED", False),
            qdrant_url=read_str(env, "QDRANT_URL", cls.qdrant_url),
            qdrant_api_key=env.get("QDRANT_API_KEY", "").strip() or None,
            qdrant_timeout_s=read_int(env, "QDRANT_TIMEOUT_S", 5, 1, 120),
            collection=read_str(env, "SOURCE_CHUNKS_COLLECTION", cls.collection),
            mode=read_mode(env, "RAG_MODE", RetrievalMode.HYBRID),
            dense_model=read_str(env, "RAG_DENSE_MODEL", cls.dense_model),
            sparse_model=read_str(env, "RAG_SPARSE_MODEL", cls.sparse_model),
            dense_top_k=read_int(env, "RAG_DENSE_TOP_K", 30, 1, 200),
            sparse_top_k=read_int(env, "RAG_SPARSE_TOP_K", 30, 1, 200),
            top_k=read_int(env, "RAG_TOP_K", 8, 1, 50),
            rrf_k=read_int(env, "RAG_RRF_K", 60, 1, 1000),
            relevance_threshold=read_float(env, "RAG_RELEVANCE_THRESHOLD", 0.5, 0.0, 1.0),
            min_chunks=read_int(env, "RAG_MIN_CHUNKS", 2, 1, 50),
            timeout_s=read_int(env, "RAG_TIMEOUT_S", 30, 1, 600),
            embed_batch_size=read_int(env, "RAG_EMBED_BATCH_SIZE", 32, 1, 256),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            rerank_enabled=read_bool(env, "RERANK_ENABLED", True),
            rerank_provider=read_str(env, "RERANK_PROVIDER", cls.rerank_provider).lower(),
            rerank_model=read_str(env, "RERANK_MODEL", cls.rerank_model),
            rerank_top_k=read_int(env, "RERANK_TOP_K", 5, 1, 50),
            rerank_max_length=read_int(env, "RERANK_MAX_LENGTH", 512, 32, 8192),
            rerank_batch_size=read_int(env, "RERANK_BATCH_SIZE", 16, 1, 128),
        )
