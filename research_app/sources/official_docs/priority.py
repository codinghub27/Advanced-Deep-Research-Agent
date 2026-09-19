"""Source priority for documentation questions.

Deliberately tiny and NOT a general ranking (that is Phase 8): official documentation
sorts before everything else, and everything else keeps its existing order (the sort is
stable). Only documents the adapter verified against the registry qualify; the check
needs both the type and the ``is_official`` flag.
"""
from __future__ import annotations

from typing import Iterable

from research_app.domain import SourceDocument, SourceType


def is_official(doc: SourceDocument) -> bool:
    return doc.source_type == SourceType.OFFICIAL_DOCS and doc.credibility.is_official


def authority_rank(doc: SourceDocument) -> int:
    return 0 if is_official(doc) else 1


def prioritize(docs: Iterable[SourceDocument]) -> list[SourceDocument]:
    return sorted(docs, key=authority_rank)
