"""Dense and sparse (BM25) text embedders behind small protocols (Phase 7).

fastembed runs on ONNX Runtime (no torch). It is imported inside the constructors, so importing
this module never loads a model or fails when fastembed is missing; the model itself loads on
first use. Tests and the seed script can inject their own ``DenseEmbedder`` / ``SparseEmbedder``.
"""
from __future__ import annotations

import threading
from typing import Protocol, Sequence, runtime_checkable

# (indices, values) of a sparse vector
SparseVec = tuple[list[int], list[float]]


@runtime_checkable
class DenseEmbedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class SparseEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[SparseVec]: ...

    def embed_query(self, text: str) -> SparseVec: ...


class FastEmbedDense:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        return int(self._load().embedding_size)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        return [vec.tolist() for vec in model.embed(list(texts), batch_size=self._batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().query_embed(text))).tolist()


class FastEmbedSparse:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed import SparseTextEmbedding

                    self._model = SparseTextEmbedding(model_name=self._model_name)
        return self._model

    @staticmethod
    def _vec(embedding) -> SparseVec:
        return embedding.indices.tolist(), embedding.values.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVec]:
        model = self._load()
        return [self._vec(e) for e in model.embed(list(texts), batch_size=self._batch_size)]

    def embed_query(self, text: str) -> SparseVec:
        return self._vec(next(iter(self._load().query_embed(text))))
