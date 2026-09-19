"""Service interfaces (structural ``Protocol``s, no inheritance required).

Only the two seams with a near-term consumer are defined. Critic, gap detection
and the evidence store get their interfaces in the phases that build them.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from research_app.domain.enums import SourceType
from research_app.domain.models import ResearchResult, SearchRequest


@runtime_checkable
class SourceAdapter(Protocol):
    """One external source (web, official docs, GitHub, Reddit, ...).

    Implementations must not raise for ordinary source failures: they return a
    ``ResearchResult`` with ``status`` FAILED/TIMEOUT and an ``error`` so one
    failing source cannot kill a research run.
    """

    source_type: SourceType

    async def search(self, request: SearchRequest) -> ResearchResult: ...


@runtime_checkable
class AnswerCache(Protocol):
    """Answer cache. Signatures match today's ``lookup_cache``/``store_cache``
    seam in ``agent/vectordb.py``; Phase 12 is expected to widen them
    (freshness, source-aware TTL)."""

    def lookup(self, question: str) -> tuple[bool, str]: ...

    def store(self, question: str, answer: str) -> None: ...
