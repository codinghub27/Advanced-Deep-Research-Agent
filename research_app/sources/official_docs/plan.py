"""The single seam between the agent and official documentation.

``plan_official_docs`` answers "does this question need official docs, and if so what
exactly should be searched?". The current integration (``agent/state.py``) calls it from
inside the existing search nodes. Phase 6's source router can call it, or replace it,
without touching detection, the registry or the adapter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from research_app.domain import SearchRequest
from research_app.sources.official_docs.adapter import build_docs_request
from research_app.sources.official_docs.detector import detect_docs_intent
from research_app.sources.official_docs.registry import DocsRegistry, get_registry
from research_app.sources.official_docs.settings import DocsSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocsPlan:
    request: SearchRequest
    registry: DocsRegistry


def plan_official_docs(
    question: str, query: str, settings: Optional[DocsSettings] = None
) -> Optional[DocsPlan]:
    """``question`` decides whether this is a documentation query; ``query`` is the text
    actually searched (a sub-question, or the question itself). ``None`` means "web
    search only". May raise; callers isolate failures."""
    settings = settings or DocsSettings.from_env()
    if not settings.enabled:
        return None
    registry = get_registry(settings.registry_path)
    intent = detect_docs_intent(question, registry)
    if not intent.is_documentation_query:
        logger.debug("Not a documentation query (%s)", intent.reason)
        return None
    request = build_docs_request(intent, query, timeout_s=settings.timeout_s)
    if request is None:
        return None
    logger.info("Documentation query: technologies=%s cues=%s", ",".join(intent.technology_ids), ",".join(intent.cues))
    return DocsPlan(request=request, registry=registry)
