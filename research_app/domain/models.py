"""Domain contracts for the research pipeline.

Pure Pydantic models. This module must not import anything from
``research_app.agent``, ``research_app.db``, ``research_app.main`` or any
LangChain/LangGraph/Qdrant package: those have import-time side effects and the
contracts have to stay importable on their own.

Conventions
- ``extra="forbid"``: a misspelled field is an error, not silent data loss.
- Datetimes are timezone-aware UTC. Naive values are assumed to already be UTC
  (the database columns are naive UTC) and are tagged, not shifted.
- Adding an optional field is a compatible change. Renaming or removing a field
  needs a note in the phase document first.
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from research_app.domain.enums import (
    Complexity,
    CriticIssueKind,
    CriticSeverity,
    CriticVerdict,
    GapKind,
    QueryIntent,
    ResultStatus,
    RunStatus,
    SourceIntent,
    SourceType,
    TaskStatus,
    TimeSensitivity,
)


# --------------------------------------------------------------------------- helpers

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[datetime, AfterValidator(_to_utc)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


def _check_http_url(value: str) -> str:
    value = value.strip()
    parts = urlsplit(value)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError("url must be an absolute http(s) URL")
    return value


HttpUrl = Annotated[str, AfterValidator(_check_http_url)]


def canonical_url(url: str) -> str:
    """Normalise a URL for identity comparison (not for fetching).

    Lower-cases scheme and host, drops default ports, the fragment and trailing
    slashes on the path. The query string is kept exactly as given.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, host, path, parts.query, ""))


def source_id_for(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()[:32]


def _content_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- query / tasks

class HistoryTurn(DomainModel):
    question: str
    answer: str


class ResearchQuery(DomainModel):
    """The user's question plus everything query understanding will learn about it."""

    text: str = Field(min_length=1)
    normalized_text: Optional[str] = None
    user_id: Optional[int] = None
    session_id: Optional[int] = None
    complexity: Optional[Complexity] = None
    intent: QueryIntent = QueryIntent.UNKNOWN
    time_sensitivity: TimeSensitivity = TimeSensitivity.UNKNOWN
    technology: Optional[str] = None
    is_documentation_query: bool = False
    history: list[HistoryTurn] = Field(default_factory=list)


class SearchRequest(DomainModel):
    """One request to one source adapter."""

    query: str = Field(min_length=1)
    source_type: SourceType = SourceType.WEB
    include_domains: list[str] = Field(default_factory=list)
    max_results: int = Field(default=3, ge=1, le=50)  # 3 = current Tavily setting
    search_depth: Literal["basic", "advanced"] = "advanced"
    timeout_s: Optional[float] = Field(default=None, gt=0)
    run_id: Optional[str] = None
    task_id: Optional[str] = None


class ResearchTask(DomainModel):
    """A unit of planned work: a sub-question and where to look for the answer."""

    task_id: str = Field(default_factory=new_id)
    sub_question: str = Field(min_length=1)
    search_query: Optional[str] = None  # defaults to sub_question
    source_types: list[SourceType] = Field(default_factory=lambda: [SourceType.WEB])
    preferred_domains: list[str] = Field(default_factory=list)
    priority: int = Field(default=0, ge=0)
    status: TaskStatus = TaskStatus.PENDING
    # Phase 6. ``source_types`` is what the router DECIDED; the fields below are what the
    # planner PROPOSED. ``suggested_sources`` keeps the raw strings on purpose: the router
    # validates them and reports the ones it drops.
    source_intent: Optional[SourceIntent] = None
    suggested_sources: list[str] = Field(default_factory=list)
    technology: Optional[str] = None

    @model_validator(mode="after")
    def _default_search_query(self) -> "ResearchTask":
        if not self.search_query:
            self.search_query = self.sub_question
        return self


# --------------------------------------------------------------------------- sources / results

class SourceCredibility(DomainModel):
    is_official: bool = False
    score: Optional[Score] = None
    notes: Optional[str] = None


class SourceDocument(DomainModel):
    """One retrieved document in the common internal format."""

    source_id: Optional[str] = None  # derived from the canonical URL when omitted
    source_type: SourceType = SourceType.WEB
    title: Optional[str] = None
    url: HttpUrl
    domain: Optional[str] = None  # derived from the URL when omitted
    author: Optional[str] = None
    published_at: Optional[UtcDatetime] = None
    retrieved_at: UtcDatetime = Field(default_factory=utc_now)
    # Provenance (Phase 3): who returned this and the URL exactly as they returned it.
    # ``source_type``, ``query``/``task_id`` and ``retrieved_at`` cover the rest.
    provider: Optional[str] = None  # e.g. "tavily"
    original_url: Optional[str] = None  # before normalisation; not validated
    snippet: Optional[str] = None
    content: Optional[str] = None
    technology: Optional[str] = None
    version: Optional[str] = None
    credibility: SourceCredibility = Field(default_factory=SourceCredibility)
    query: Optional[str] = None
    task_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)  # source-specific

    @model_validator(mode="after")
    def _derive_identity(self) -> "SourceDocument":
        if not self.domain:
            self.domain = (urlsplit(self.url).hostname or "").lower()
        else:
            self.domain = self.domain.lower()
        if not self.source_id:
            self.source_id = source_id_for(self.url)
        return self


class ResearchError(DomainModel):
    code: str
    message: str
    retryable: bool = False
    source_type: Optional[SourceType] = None


class ResearchResult(DomainModel):
    """What one source adapter returned for one SearchRequest."""

    task_id: Optional[str] = None
    source_type: SourceType = SourceType.WEB
    provider: Optional[str] = None
    status: ResultStatus = ResultStatus.OK
    documents: list[SourceDocument] = Field(default_factory=list)
    error: Optional[ResearchError] = None
    latency_ms: Optional[float] = Field(default=None, ge=0)
    started_at: Optional[UtcDatetime] = None
    completed_at: Optional[UtcDatetime] = None

    @model_validator(mode="after")
    def _failures_need_an_error(self) -> "ResearchResult":
        if self.status in (ResultStatus.FAILED, ResultStatus.TIMEOUT) and self.error is None:
            raise ValueError(f"status '{self.status.value}' requires an error")
        return self

    @property
    def succeeded(self) -> bool:
        return self.status in (ResultStatus.OK, ResultStatus.PARTIAL)


# --------------------------------------------------------------------------- evidence / citations

class Evidence(DomainModel):
    """A piece of text extracted from a source that can support a claim."""

    evidence_id: str = Field(default_factory=new_id)
    run_id: Optional[str] = None
    task_id: Optional[str] = None
    source_id: str
    text: str = Field(min_length=1)
    url: HttpUrl
    source_type: SourceType = SourceType.WEB
    retrieved_at: UtcDatetime = Field(default_factory=utc_now)
    query: Optional[str] = None
    rank: Optional[int] = Field(default=None, ge=0)
    score: Optional[float] = None
    quality: Optional[Score] = None
    confidence: Optional[Score] = None
    content_hash: Optional[str] = None  # derived from the text when omitted (dedup key)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _derive_hash(self) -> "Evidence":
        if not self.content_hash:
            self.content_hash = _content_hash(self.text)
        return self


class Citation(DomainModel):
    """Maps a marker in the final answer (e.g. "[1]") to the evidence behind it."""

    marker: str = Field(min_length=1)
    source_id: str
    evidence_id: Optional[str] = None
    url: HttpUrl
    title: Optional[str] = None
    source_type: SourceType = SourceType.WEB
    excerpt: Optional[str] = None
    # Phase 6 (all optional): the numbered form the UI and the database use.
    index: Optional[int] = Field(default=None, ge=1)
    domain: Optional[str] = None
    snippet: Optional[str] = None
    retrieved_at: Optional[UtcDatetime] = None


# --------------------------------------------------------------------------- gaps / critic

class Gap(DomainModel):
    kind: GapKind
    description: str
    task_id: Optional[str] = None
    evidence_ids: list[str] = Field(default_factory=list)


class GapAnalysis(DomainModel):
    """Result of one evidence-coverage check. Unsupported claims and
    contradictions are gaps with the matching ``kind``."""

    iteration: int = Field(default=0, ge=0)
    sufficient: bool
    covered_task_ids: list[str] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    follow_up_tasks: list[ResearchTask] = Field(default_factory=list)
    rationale: Optional[str] = None


class CriticIssue(DomainModel):
    kind: CriticIssueKind
    description: str
    evidence_ids: list[str] = Field(default_factory=list)
    severity: Optional[CriticSeverity] = None  # Phase 6


class CriticResult(DomainModel):
    """Verdict on a synthesized answer. The legacy ``critic_score``/``critic_feedback``
    state keys are unused and are not mapped."""

    passed: bool
    score: Optional[Score] = None
    issues: list[CriticIssue] = Field(default_factory=list)
    feedback: str = ""
    retry_tasks: list[ResearchTask] = Field(default_factory=list)
    # Phase 6: ``passed`` is True exactly when the verdict is GOOD.
    verdict: Optional[CriticVerdict] = None
    suggestions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- run

class ResearchRun(DomainModel):
    """Everything known about one execution of the research pipeline."""

    run_id: str = Field(default_factory=new_id)
    user_id: Optional[int] = None
    session_id: Optional[int] = None
    query: ResearchQuery
    status: RunStatus = RunStatus.PENDING
    started_at: UtcDatetime = Field(default_factory=utc_now)
    completed_at: Optional[UtcDatetime] = None
    tasks: list[ResearchTask] = Field(default_factory=list)
    results: list[ResearchResult] = Field(default_factory=list)
    sources: list[SourceDocument] = Field(default_factory=list)  # final, de-duplicated list
    evidence: list[Evidence] = Field(default_factory=list)
    gap_analyses: list[GapAnalysis] = Field(default_factory=list)
    critic_results: list[CriticResult] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    final_answer: str = ""
    cache_hit: bool = False
    retry_count: int = Field(default=0, ge=0)
    errors: list[ResearchError] = Field(default_factory=list)
