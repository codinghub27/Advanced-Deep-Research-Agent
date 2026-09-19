"""Tavily search response -> ``ResearchResult`` of normalised ``SourceDocument``s.

Pure data conversion: it does not call Tavily. The response is what
``TavilySearch.ainvoke`` returns: ``{"query", "answer", "results": [{"title",
"url", "content", "score", ...}], ...}``, ``{"error": <exception>}`` when the
wrapper swallowed a failure, or a plain string when it raised a ``ToolException``
(zero results). Per-result fields Tavily sends beyond
title/url/content (e.g. ``score``) end up in ``SourceDocument.metadata``.
Response-level fields (``answer``, ``response_time``, ...) are not carried.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional

from research_app.domain import (
    ResearchError,
    ResearchResult,
    ResultStatus,
    SourceType,
    utc_now,
)
from research_app.sources.normalizer import normalize_sources

PROVIDER = "tavily"


def _failed(code: str, message: str, task_id: Optional[str]) -> ResearchResult:
    return ResearchResult(
        task_id=task_id,
        source_type=SourceType.WEB,
        provider=PROVIDER,
        status=ResultStatus.FAILED,
        error=ResearchError(code=code, message=message, source_type=SourceType.WEB),
    )


def normalize_tavily_response(
    response: Any,
    *,
    query: str,
    task_id: Optional[str] = None,
    max_results: Optional[int] = None,
    retrieved_at: Optional[datetime] = None,
) -> ResearchResult:
    """Never raises for bad provider output; returns a FAILED result instead.

    - OK: every returned result was normalised (an empty list is a valid, empty OK).
    - PARTIAL: some results were unusable and skipped.
    - FAILED: not a mapping, provider error, no ``results`` list, or every result unusable.

    ``max_results`` caps how many raw results are considered (in provider order).
    """
    if isinstance(response, str):
        # TavilySearch has handle_tool_error=True: a ToolException (e.g. "No search
        # results found for ...") is handed back as its message string.
        return _failed("provider_error", response[:200], task_id)
    if not isinstance(response, Mapping):
        return _failed("invalid_response", f"expected a mapping, got {type(response).__name__}", task_id)
    results = response.get("results")
    if not isinstance(results, list):
        if "error" in response:
            err = response["error"]
            detail = f"{type(err).__name__}: {err}" if isinstance(err, BaseException) else str(err)
            return _failed("provider_error", detail[:200], task_id)
        return _failed("invalid_response", "response has no results list", task_id)

    raw_items = results if max_results is None else results[:max_results]
    docs = normalize_sources(
        raw_items,
        source_type=SourceType.WEB,
        provider=PROVIDER,
        query=query,
        task_id=task_id,
        retrieved_at=retrieved_at or utc_now(),
    )
    skipped = len(raw_items) - len(docs)
    if raw_items and not docs:
        return _failed("no_valid_results", f"{skipped} result(s) could not be normalized", task_id)
    return ResearchResult(
        task_id=task_id,
        source_type=SourceType.WEB,
        provider=PROVIDER,
        status=ResultStatus.PARTIAL if skipped else ResultStatus.OK,
        documents=docs,
    )
