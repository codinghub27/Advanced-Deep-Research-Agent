"""The ``source_chunks`` Qdrant collection (Phase 7).

Named vectors: ``dense`` (cosine) and ``sparse`` (BM25 term weights, IDF applied by Qdrant), one
payload per chunk. Hybrid search is two prefetch legs fused with Reciprocal Rank Fusion
(``RAG_RRF_K``); a server that rejects the RRF-with-k query gets the same fusion done client-side.

This is separate from the answer cache (``agent/vectordb.py``); nothing here touches it.
Synchronous by design (the retriever runs it in a worker thread under a timeout).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Sequence

from qdrant_client import QdrantClient, models
from qdrant_client.hybrid.fusion import reciprocal_rank_fusion

from research_app.domain import RetrievalFilter, RetrievalMode, RetrievedChunk, SourceChunk
from research_app.rag.embeddings import DenseEmbedder, SparseEmbedder
from research_app.rag.settings import RagSettings

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
_LOCAL_URLS = (":memory:",)


def _is_local(url: str) -> bool:
    """":memory:" or a folder path: the embedded engine (tests, seed script), not a server."""
    return url in _LOCAL_URLS or not url.lower().startswith(("http://", "https://"))


def make_client(settings: RagSettings) -> QdrantClient:
    url = settings.qdrant_url
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if _is_local(url):
        return QdrantClient(path=url)
    return QdrantClient(url=url, api_key=settings.qdrant_api_key, timeout=settings.qdrant_timeout_s)


def build_filter(flt: Optional[RetrievalFilter]) -> Optional[models.Filter]:
    if flt is None or flt.is_empty:
        return None
    must: list[Any] = []
    if flt.source_types:
        must.append(models.FieldCondition(
            key="source_type", match=models.MatchAny(any=[t.value for t in flt.source_types])))
    if flt.domains:
        must.append(models.FieldCondition(
            key="domain", match=models.MatchAny(any=[d.lower() for d in flt.domains])))
    if flt.technology:
        must.append(models.FieldCondition(
            key="technology", match=models.MatchValue(value=flt.technology.strip().lower())))
    if flt.retrieved_after or flt.retrieved_before:
        must.append(models.FieldCondition(
            key="retrieved_at",
            range=models.DatetimeRange(gte=flt.retrieved_after, lte=flt.retrieved_before)))
    return models.Filter(must=must)


def _payload(chunk: SourceChunk) -> dict[str, Any]:
    payload = chunk.model_dump(mode="json", exclude={"chunk_id"})
    payload["chunk_id"] = chunk.chunk_id
    if payload.get("technology"):
        payload["technology"] = payload["technology"].strip().lower()
    return payload


def _chunk_from(point: Any) -> SourceChunk:
    return SourceChunk.model_validate(point.payload)


class SourceChunkStore:
    def __init__(self, settings: RagSettings, dense: DenseEmbedder, sparse: SparseEmbedder,
                 client: Optional[QdrantClient] = None) -> None:
        self._settings = settings
        self._dense = dense
        self._sparse = sparse
        self._client = client
        self._ready = False
        self._lock = threading.Lock()

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = make_client(self._settings)
        return self._client

    @property
    def collection(self) -> str:
        return self._settings.collection

    # ------------------------------------------------------------------ collection

    def ensure_collection(self) -> None:
        """Create ``source_chunks`` and its payload indexes if missing. Idempotent."""
        if self._ready:
            return
        client = self.client
        if not client.collection_exists(self.collection):
            client.create_collection(
                collection_name=self.collection,
                vectors_config={DENSE_VECTOR: models.VectorParams(
                    size=self._dense.dim, distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            if not _is_local(self._settings.qdrant_url):  # indexes have no effect in the embedded engine
                for field, schema in (
                    ("source_type", models.PayloadSchemaType.KEYWORD),
                    ("domain", models.PayloadSchemaType.KEYWORD),
                    ("technology", models.PayloadSchemaType.KEYWORD),
                    ("retrieved_at", models.PayloadSchemaType.DATETIME),
                ):
                    client.create_payload_index(self.collection, field_name=field, field_schema=schema)
            logger.info("Created Qdrant collection %s", self.collection)
        self._ready = True

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    # ------------------------------------------------------------------ write

    def upsert(self, chunks: Sequence[SourceChunk]) -> int:
        """Embed and store ``chunks`` (same chunk id = overwrite). Returns how many were written."""
        if not chunks:
            return 0
        self.ensure_collection()
        batch = self._settings.embed_batch_size
        written = 0
        for start in range(0, len(chunks), batch):
            part = list(chunks[start:start + batch])
            texts = [c.text for c in part]
            dense = self._dense.embed_documents(texts)
            sparse = self._sparse.embed_documents(texts)
            points = [
                models.PointStruct(
                    id=chunk.chunk_id,
                    vector={DENSE_VECTOR: d, SPARSE_VECTOR: models.SparseVector(indices=s[0], values=s[1])},
                    payload=_payload(chunk),
                )
                for chunk, d, s in zip(part, dense, sparse)
            ]
            self.client.upsert(self.collection, points=points, wait=True)
            written += len(points)
        return written

    # ------------------------------------------------------------------ read

    def search(self, query: str, *, mode: Optional[RetrievalMode] = None,
               flt: Optional[RetrievalFilter] = None, limit: Optional[int] = None) -> list[RetrievedChunk]:
        """Top ``limit`` chunks for ``query``, best first, scored 0-1 (see ``_scored``)."""
        s = self._settings
        mode = mode or s.mode
        limit = limit or s.top_k
        qfilter = build_filter(flt)
        if mode is RetrievalMode.DENSE:
            points = self._dense_points(self._dense.embed_query(query), qfilter, limit)
        elif mode is RetrievalMode.SPARSE:
            points = self._sparse_points(self._sparse_vector(query), qfilter, limit)
        else:
            points = self._hybrid_points(query, qfilter, limit)
        return self._scored(points, mode)

    def _sparse_vector(self, query: str) -> models.SparseVector:
        indices, values = self._sparse.embed_query(query)
        return models.SparseVector(indices=indices, values=values)

    def _dense_points(self, vec: list[float], qfilter: Optional[models.Filter], limit: int) -> list[Any]:
        return self.client.query_points(
            self.collection, query=vec, using=DENSE_VECTOR, query_filter=qfilter, limit=limit,
            with_payload=True).points

    def _sparse_points(self, vec: models.SparseVector, qfilter: Optional[models.Filter], limit: int) -> list[Any]:
        return self.client.query_points(
            self.collection, query=vec, using=SPARSE_VECTOR, query_filter=qfilter, limit=limit,
            with_payload=True).points

    def _hybrid_points(self, query: str, qfilter: Optional[models.Filter], limit: int) -> list[Any]:
        s = self._settings
        dense_vec = self._dense.embed_query(query)
        sparse_vec = self._sparse_vector(query)
        try:
            return self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense_vec, using=DENSE_VECTOR, filter=qfilter, limit=s.dense_top_k),
                    models.Prefetch(query=sparse_vec, using=SPARSE_VECTOR, filter=qfilter, limit=s.sparse_top_k),
                ],
                query=models.RrfQuery(rrf=models.Rrf(k=s.rrf_k)),
                limit=limit, with_payload=True).points
        except Exception as exc:
            # Older servers do not know RRF-with-k: fuse the two legs here instead.
            logger.warning("Server-side RRF unavailable (%s); fusing client-side", type(exc).__name__)
            dense = self._dense_points(dense_vec, qfilter, s.dense_top_k)
            sparse = self._sparse_points(sparse_vec, qfilter, s.sparse_top_k)
            return reciprocal_rank_fusion([dense, sparse], limit=limit, ranking_constant_k=s.rrf_k)

    def _scored(self, points: Sequence[Any], mode: RetrievalMode) -> list[RetrievedChunk]:
        """0-1 scores so one threshold works for every mode. Dense: cosine. Hybrid: the RRF score
        as a share of the best possible one (rank 1 in both legs = 2/k). Sparse: relative to the
        best hit (BM25 is unbounded), so ``sparse`` is a debugging mode, not a gating one."""
        if not points:
            return []
        raw = [float(p.score) for p in points]
        if mode is RetrievalMode.HYBRID:
            best = 2.0 / self._settings.rrf_k
            norm = [v / best for v in raw]
        elif mode is RetrievalMode.SPARSE:
            top = max(raw) or 1.0
            norm = [v / top for v in raw]
        else:
            norm = raw
        return [
            RetrievedChunk(chunk=_chunk_from(p), score=min(1.0, max(0.0, n)), rank=rank, fused_score=v)
            for rank, (p, v, n) in enumerate(zip(points, raw, norm))
        ]
