"""GitHub and Reddit source adapters (implement ``domain.SourceAdapter``).

Both are the same mechanism, parameterised by a ``DomainPolicy``:

    SearchRequest -> injected search callable (Tavily, ``include_domains`` restricted)
      -> ``normalize_tavily_response`` (Phase 3)
      -> EVERY result URL checked against the policy's exact hostname allowlist
      -> ``SourceDocument(source_type=github|reddit, ...)`` with URL-derived metadata

A result gets the GitHub/Reddit type only if ``is_valid_url`` accepts its URL. Anything else
is dropped, never downgraded to another type, so no other site can carry the label whatever
the search provider returns (Tavily's ``include_domains`` is a request, not a guarantee).

No quality score is assigned: ``credibility.is_official`` is False and ``score`` stays unset.
The metadata is only what the URL itself says (owner/repo/issue number, subreddit/post id),
each part validated against a strict pattern, so labels built from it cannot be spoofed by
result text.

Adapters never raise for source failures (timeouts, provider errors, malformed output); they
return a FAILED/TIMEOUT ``ResearchResult``. ``asyncio.CancelledError`` is not caught.

The search callable is injected, so this module needs no LangChain import and tests need no
network. Official documentation keeps its own adapter (Phase 4); web keeps the existing path.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlsplit

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
from research_app.sources.official_docs.registry import normalize_path
from research_app.sources.routing.hosts import hostname_of, is_github_url, is_reddit_url
from research_app.sources.tavily import normalize_tavily_response

logger = logging.getLogger(__name__)

PROVIDER = "tavily"
DEFAULT_MAX_RESULTS = 3  # same as the Tavily tool's max_results

SearchFn = Callable[[dict[str, Any]], Awaitable[Any]]


# --------------------------------------------------------------------------- URL metadata

_GH_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
_GH_RESERVED = frozenset({
    "about", "apps", "codespaces", "collections", "contact", "customer-stories", "enterprise",
    "events", "explore", "features", "issues", "login", "marketplace", "new", "notifications",
    "orgs", "organizations", "pricing", "pulls", "readme", "search", "security", "sessions",
    "settings", "signup", "site", "sponsors", "topics", "trending", "users",
})
_GH_NUMBERED = {"issues": "issue", "pull": "pull_request", "discussions": "discussion"}
_GH_OTHER = {"blob": "file", "tree": "directory", "releases": "release", "wiki": "wiki",
             "commit": "commit", "commits": "commit"}

_REDDIT_SUB = re.compile(r"^[A-Za-z0-9_]{2,21}$")
_REDDIT_ID = re.compile(r"^[a-z0-9]{1,10}$")
_REDDIT_USER = re.compile(r"^[A-Za-z0-9_-]{3,20}$")


def _segments(url: str) -> list[str]:
    path = normalize_path(urlsplit(url).path)
    if path is None:
        return []
    return [s for s in path.split("/") if s]


def github_metadata(url: str) -> dict[str, Any]:
    """Structure the URL itself reveals: owner, repo, kind, number, ref/path."""
    meta: dict[str, Any] = {"host": hostname_of(url), "kind": "other"}
    segs = _segments(url)
    if not segs:
        return meta
    owner = segs[0]
    if owner.lower() in _GH_RESERVED or not _GH_NAME.match(owner):
        return meta
    meta["owner"] = owner
    if len(segs) == 1:
        meta["kind"] = "profile"
        return meta
    repo = segs[1]
    if not _GH_NAME.match(repo):
        return meta
    meta["repo"] = repo
    if len(segs) == 2:
        meta["kind"] = "repository"
        return meta
    section = segs[2]
    if section in _GH_NUMBERED and len(segs) >= 4 and segs[3].isdigit() and len(segs[3]) <= 9:
        meta["kind"] = _GH_NUMBERED[section]
        meta["number"] = int(segs[3])
    elif section in _GH_OTHER:
        meta["kind"] = _GH_OTHER[section]
        if section in ("blob", "tree") and len(segs) >= 4:
            meta["ref"] = segs[3][:100]
            meta["path"] = "/".join(segs[4:])[:300]
    return meta


def reddit_metadata(url: str) -> dict[str, Any]:
    """Structure the URL itself reveals: subreddit, post id, kind."""
    meta: dict[str, Any] = {"host": hostname_of(url), "kind": "other"}
    segs = _segments(url)
    if len(segs) >= 2 and segs[0].lower() == "r" and _REDDIT_SUB.match(segs[1]):
        meta["subreddit"] = segs[1]
        meta["kind"] = "subreddit"
        if len(segs) >= 4 and segs[2] == "comments" and _REDDIT_ID.match(segs[3]):
            meta["post_id"] = segs[3]
            meta["kind"] = "post"
            if len(segs) >= 6 and _REDDIT_ID.match(segs[5]):
                meta["comment_id"] = segs[5]
                meta["kind"] = "comment"
    elif len(segs) >= 2 and segs[0].lower() in ("u", "user") and _REDDIT_USER.match(segs[1]):
        meta["kind"] = "user"
    return meta


# --------------------------------------------------------------------------- policies

@dataclass(frozen=True)
class DomainPolicy:
    source_type: SourceType
    name: str
    include_domains: tuple[str, ...]  # sent to the search provider
    is_valid_url: Callable[[str], bool]  # the real gate, applied to every result
    describe: Callable[[str], dict[str, Any]]
    note: str


GITHUB_POLICY = DomainPolicy(
    SourceType.GITHUB, "GitHub", ("github.com",), is_github_url, github_metadata,
    "community code source: github.com (hostname verified); not official documentation",
)
REDDIT_POLICY = DomainPolicy(
    SourceType.REDDIT, "Reddit", ("reddit.com",), is_reddit_url, reddit_metadata,
    "community discussion: reddit.com (hostname verified); not official documentation",
)
POLICIES = {p.source_type: p for p in (GITHUB_POLICY, REDDIT_POLICY)}


def build_request(
    source_type: SourceType,
    query: str,
    *,
    timeout_s: Optional[float] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    run_id: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Optional[SearchRequest]:
    """A domain-restricted request for GitHub or Reddit, or ``None`` for a blank query.
    Raises ``ValueError`` for a source type without a policy (a caller bug)."""
    policy = POLICIES.get(SourceType(source_type))
    if policy is None:
        raise ValueError(f"no domain policy for source type '{SourceType(source_type).value}'")
    if not isinstance(query, str) or not query.strip():
        return None
    return SearchRequest(
        query=query.strip(),
        source_type=policy.source_type,
        include_domains=list(policy.include_domains),
        max_results=max_results,
        timeout_s=timeout_s,
        run_id=run_id,
        task_id=task_id,
    )


def _as_source(doc: SourceDocument, policy: DomainPolicy) -> SourceDocument:
    """Re-stamp a verified web-normalised document as GitHub/Reddit."""
    metadata = dict(doc.metadata)
    metadata[policy.source_type.value] = policy.describe(doc.url)
    return doc.model_copy(update={
        "source_type": policy.source_type,
        "credibility": SourceCredibility(is_official=False, notes=policy.note),
        "metadata": metadata,
    })


# --------------------------------------------------------------------------- adapter

class DomainSourceAdapter:
    def __init__(self, policy: DomainPolicy, search_fn: SearchFn):
        self._policy = policy
        self._search_fn = search_fn
        self.source_type = policy.source_type

    def _failure(
        self, request: SearchRequest, status: ResultStatus, code: str, message: str,
        *, retryable: bool, started_at, t0: float,
    ) -> ResearchResult:
        logger.warning("%s search %s (%s)", self._policy.name, status.value, code)
        return ResearchResult(
            task_id=request.task_id,
            source_type=self.source_type,
            provider=PROVIDER,
            status=status,
            error=ResearchError(code=code, message=message[:200], retryable=retryable,
                                source_type=self.source_type),
            latency_ms=(time.monotonic() - t0) * 1000,
            started_at=started_at,
            completed_at=utc_now(),
        )

    async def search(self, request: SearchRequest) -> ResearchResult:
        started_at, t0 = utc_now(), time.monotonic()
        payload = {"query": request.query, "include_domains": list(self._policy.include_domains)}
        try:
            call = self._search_fn(payload)
            response = await (asyncio.wait_for(call, request.timeout_s) if request.timeout_s else call)
        except asyncio.TimeoutError:
            return self._failure(request, ResultStatus.TIMEOUT, "timeout",
                                 f"the {self._policy.name} search timed out",
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
        policy = self._policy
        inner = normalize_tavily_response(
            response, query=request.query, task_id=request.task_id, max_results=request.max_results
        )
        if inner.error:
            return self._failure(request, ResultStatus.FAILED, inner.error.code, inner.error.message,
                                 retryable=inner.error.retryable, started_at=started_at, t0=t0)

        verified: list[SourceDocument] = []
        rejected: list[str] = []
        for doc in inner.documents:
            if policy.is_valid_url(doc.url):
                verified.append(_as_source(doc, policy))
            else:
                rejected.append(doc.domain or "?")
        if rejected:
            logger.info("%s: dropped %d result(s) outside %s: %s", policy.name, len(rejected),
                        "/".join(policy.include_domains), ", ".join(repr(d) for d in sorted(set(rejected))[:5]))

        if inner.documents and not verified:
            return self._failure(
                request, ResultStatus.FAILED, "no_domain_results",
                f"{len(rejected)} result(s) were not on {policy.name}",
                retryable=False, started_at=started_at, t0=t0)

        partial = bool(rejected) or inner.status == ResultStatus.PARTIAL
        return ResearchResult(
            task_id=request.task_id,
            source_type=self.source_type,
            provider=PROVIDER,
            status=ResultStatus.PARTIAL if partial else ResultStatus.OK,
            documents=verified,
            latency_ms=(time.monotonic() - t0) * 1000,
            started_at=started_at,
            completed_at=utc_now(),
        )


class GitHubAdapter(DomainSourceAdapter):
    def __init__(self, search_fn: SearchFn):
        super().__init__(GITHUB_POLICY, search_fn)


class RedditAdapter(DomainSourceAdapter):
    def __init__(self, search_fn: SearchFn):
        super().__init__(REDDIT_POLICY, search_fn)


def adapter_for(source_type: SourceType, search_fn: SearchFn) -> DomainSourceAdapter:
    """The adapter for GitHub or Reddit. ``ValueError`` for any other source type."""
    policy = POLICIES.get(SourceType(source_type))
    if policy is None:
        raise ValueError(f"no domain adapter for source type '{SourceType(source_type).value}'")
    return DomainSourceAdapter(policy, search_fn)
