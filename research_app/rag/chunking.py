"""Split a ``SourceDocument`` into ``SourceChunk``s (Phase 7).

Character windows that prefer to end on a paragraph, line or sentence boundary. Chunk ids are
uuid5(source_id, index), so re-indexing a document overwrites its chunks instead of duplicating.
"""
from __future__ import annotations

import uuid
from typing import Optional

from research_app.domain import SourceChunk, SourceDocument

_CHUNK_NAMESPACE = uuid.UUID("6f1d3c7e-52b4-4a0e-9d0a-7c5b0b6a4e21")
_BOUNDARIES = ("\n\n", "\n", ". ", " ")
_MIN_FRACTION = 0.5  # a boundary in the first half of the window is ignored: chunks stay substantial


def chunk_id_for(source_id: str, index: int) -> str:
    return str(uuid.uuid5(_CHUNK_NAMESPACE, f"{source_id}:{index}"))


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap smaller than size")
    text = text.strip()
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            floor = int(size * _MIN_FRACTION)
            for boundary in _BOUNDARIES:
                cut = window.rfind(boundary)
                if cut >= floor:
                    end = start + cut + len(boundary)
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_document(doc: SourceDocument, size: int, overlap: int) -> list[SourceChunk]:
    """Chunks of ``doc.content`` (or its snippet when there is no content). Empty when neither has text."""
    body: Optional[str] = doc.content or doc.snippet
    if not body or not body.strip():
        return []
    source_id = doc.source_id or ""
    return [
        SourceChunk(
            chunk_id=chunk_id_for(source_id, index),
            source_id=source_id,
            chunk_index=index,
            text=piece,
            url=doc.url,
            title=doc.title,
            domain=doc.domain or "",
            source_type=doc.source_type,
            technology=doc.technology,
            version=doc.version,
            retrieved_at=doc.retrieved_at,
            published_at=doc.published_at,
            provider=doc.provider,
            query=doc.query,
        )
        for index, piece in enumerate(split_text(body, size, overlap))
    ]
