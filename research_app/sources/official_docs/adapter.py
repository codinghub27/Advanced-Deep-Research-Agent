"""Official-documentation source adapter (implements ``domain.SourceAdapter``).

Flow: ``SearchRequest`` -> injected search callable (Tavily, restricted with
``include_domains``) -> ``normalize_tavily_response`` (Phase 3) -> registry verification
of every result URL -> ``SourceDocument(source_type=official_docs, ...)``.

A result becomes ``official_docs`` ONLY if ``DocsRegistry.match_url`` accepts its URL
against the hosts the request was restricted to. Results that fail are dropped, never
downgraded to another type, so an off-domain page can never carry the official label
no matter what the search provider returns.

The adapter never raises for source failures (timeouts, provider errors, malformed
output): it returns a FAILED/TIMEOUT ``ResearchResult``. ``asyncio.CancelledError`` is
not caught, so cancellation still works.

It takes the search callable as a dependency, so this module needs no LangChain import
and unit tests need no network.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

from research_app.domain import (
    ResearchError,
    ResearchResult,
    ResultStatus,
    SearchRequest,
    SourceCredibility,
    SourceDocument,
    SourceType,
    utc_now,
)
from research_app.sources.official_docs.detector import DocsIntent
from research_app.sources.official_docs.registry import DocsMatch, DocsRegistry
from research_app.sources.tavily import normalize_tavily_response

logger = logging.getLogger(__name__)

PROVIDER = "tavily"  # documentation search goes through Tavily, restricted to official hosts
DEFAULT_MAX_RESULTS = 3  # same as the Tavily tool's max_results

SearchFn = Callable[[dict[str, Any]], Awaitable[Any]]


def build_docs_request(
    intent: DocsIntent,
    query: str,
    *,
    timeout_s: Optional[float] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    run_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Optional[SearchRequest]:
    """A domain-restricted request for the detected technologies, or ``None`` when
    there is nothing to search (not a docs query, blank query)."""
    if not intent.is_documentation_query or not query or not query.strip():
        return None
    hosts = tuple(dict.fromkeys(h for t in intent.technologies for h in t.search_hosts()))
    if not hosts:
        return None
    return SearchRequest(
        query=query.strip(),
        source_type=SourceType.OFFICIAL_DOCS,
        include_domains=list(hosts),
        max_results=max_results,
        timeout_s=timeout_s,
        run_id=run_id,
        task_id=task_id,
    )


def _as_official(doc: SourceDocument, match: DocsMatch) -> SourceDocument:
    """Re-stamp a verified web-normalised document as official documentation."""
    metadata = dict(doc.metadata)
    metadata["official_docs"] = {
        "technology_id": match.entry.id,
        "host": match.site.host,
        "path_prefix": match.matched_prefix,
    }
    return doc.model_copy(update={
        "source_type": SourceType.OFFICIAL_DOCS,
        "technology": match.entry.name,
        "version": match.version,
        "credibility": SourceCredibility(
            is_official=True,
            notes=f"official documentation registry: {match.entry.id} ({match.site.host})",
        ),
        "metadata": metadata,
    })


class OfficialDocsAdapter:
    source_type = SourceType.OFFICIAL_DOCS

    def __init__(self, search_fn: SearchFn, registry: DocsRegistry):
        self._search_fn = search_fn
        self._registry = registry

    def _failure(
        self, request: SearchRequest, status: ResultStatus, code: str, message: str,
        *, retryable: bool, started_at, t0: float,
    ) -> ResearchResult:
        logger.warning("Official docs search %s (%s)", status.value, code)
        return ResearchResult(
            task_id=request.task_id,
            source_type=SourceType.OFFICIAL_DOCS,
            provider=PROVIDER,
            status=status,
            error=ResearchError(code=code, message=message[:200], retryable=retryable,
                                source_type=SourceType.OFFICIAL_DOCS),
            latency_ms=(time.monotonic() - t0) * 1000,
            started_at=started_at,
            completed_at=utc_now(),
        )

    async def search(self, request: SearchRequest) -> ResearchResult:
        started_at, t0 = utc_now(), time.monotonic()
        if not request.include_domains:
            return self._failure(request, ResultStatus.FAILED, "no_domains",
                                 "official docs search needs at least one registered host",
                                 retryable=False, started_at=started_at, t0=t0)

        payload = {"query": request.query, "include_domains": list(request.include_domains)}
        try:
            call = self._search_fn(payload)
            response = await (asyncio.wait_for(call, request.timeout_s) if request.timeout_s else call)
        except asyncio.TimeoutError:
            return self._failure(request, ResultStatus.TIMEOUT, "timeout",
                                 "the documentation search timed out",
                                 retryable=True, started_at=started_at, t0=t0)
        except Exception as exc:  # failure isolation: one source must not kill the run
            return self._failure(request, ResultStatus.FAILED, "provider_error",
                                 f"{type(exc).__name__}: {exc}",
                                 retryable=True, started_at=started_at, t0=t0)

        try:
            return self._build_result(request, response, started_at, t0)
        except Exception as exc:  # defensive: malformed output must not escape
            return self._failure(request, ResultStatus.FAILED, "invalid_response",
                                 f"{type(exc).__name__}: {exc}",
                                 retryable=False, started_at=started_at, t0=t0)

    def _build_result(self, request: SearchRequest, response: Any, started_at, t0: float) -> ResearchResult:
        inner = normalize_tavily_response(
            response, query=request.query, task_id=request.task_id, max_results=request.max_results
        )
        if inner.error:
            return self._failure(request, ResultStatus.FAILED, inner.error.code, inner.error.message,
                                 retryable=inner.error.retryable, started_at=started_at, t0=t0)

        verified: list[SourceDocument] = []
        rejected: list[str] = []
        for doc in inner.documents:
            match = self._registry.match_url(doc.url, allowed_hosts=request.include_domains)
            if match is None:
                rejected.append(doc.domain or "?")
            else:
                verified.append(_as_official(doc, match))
        if rejected:
            logger.info("Official docs: dropped %d result(s) outside the registered official sites: %s",
                        len(rejected), ", ".join(repr(d) for d in sorted(set(rejected))[:5]))

        if inner.documents and not verified:
            return self._failure(
                request, ResultStatus.FAILED, "no_official_results",
                f"{len(rejected)} result(s) were not on a registered official documentation site",
                retryable=False, started_at=started_at, t0=t0)

        partial = bool(rejected) or inner.status == ResultStatus.PARTIAL
        return ResearchResult(
            task_id=request.task_id,
            source_type=SourceType.OFFICIAL_DOCS,
            provider=PROVIDER,
            status=ResultStatus.PARTIAL if partial else ResultStatus.OK,
            documents=verified,
            latency_ms=(time.monotonic() - t0) * 1000,
            started_at=started_at,
            completed_at=utc_now(),
        )
