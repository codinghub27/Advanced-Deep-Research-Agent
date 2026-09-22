"""Enumerations shared by the domain contracts.

All enums are ``str`` based so they serialize to plain strings. Their values are
persisted in later phases (PostgreSQL / Qdrant payloads), so treat renames as
breaking changes.
"""
from enum import Enum


class SourceType(str, Enum):
    WEB = "web"
    OFFICIAL_DOCS = "official_docs"
    GITHUB = "github"
    REDDIT = "reddit"
    RAG_DOCUMENT = "rag_document"
    CACHE = "cache"


class Complexity(str, Enum):
    # Mirrors the legacy ``is_simple`` flag produced by classify_node.
    SIMPLE = "simple"
    COMPLEX = "complex"


class QueryIntent(str, Enum):
    UNKNOWN = "unknown"
    FACTUAL = "factual"
    RESEARCH = "research"
    COMPARISON = "comparison"
    TECHNICAL_HOWTO = "technical_howto"
    TROUBLESHOOTING = "troubleshooting"
    CODE = "code"
    CURRENT_EVENTS = "current_events"
    # A judgment, review or career/learning advice about the user's own plan or project. The
    # technologies it names are background, not the subject.
    ADVICE = "advice"


class TimeSensitivity(str, Enum):
    UNKNOWN = "unknown"
    EVERGREEN = "evergreen"
    RECENT = "recent"
    CURRENT = "current"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResultStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GapKind(str, Enum):
    MISSING_SUBQUESTION = "missing_subquestion"
    WEAK_EVIDENCE = "weak_evidence"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    # Phase 6
    MISSING_COVERAGE = "missing_coverage"
    ORPHAN_CITATION = "orphan_citation"
    WEAK_SOURCE = "weak_source"
    INCOMPLETE = "incomplete"
    CONTRADICTION = "contradiction"
    OTHER = "other"


class CriticIssueKind(str, Enum):
    COVERAGE = "coverage"
    EVIDENCE_SUPPORT = "evidence_support"
    CITATION = "citation"
    SOURCE_QUALITY = "source_quality"
    CONTRADICTION = "contradiction"
    MISSING_INFORMATION = "missing_information"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    # Phase 6
    MISSING_COVERAGE = "missing_coverage"
    ORPHAN_CITATION = "orphan_citation"
    WEAK_SOURCE = "weak_source"
    INCOMPLETE = "incomplete"


class SourceIntent(str, Enum):
    """What kind of source a sub-question needs (Phase 6). Emitted by the planner, mapped to
    sources by ``route_task``. Independent of ``QueryIntent`` (that describes the whole
    question; this describes one sub-question's source needs)."""

    TECHNICAL_HOWTO = "technical_howto"
    LIBRARY_USAGE = "library_usage"
    CODE_IMPLEMENTATION = "code_implementation"
    TROUBLESHOOTING = "troubleshooting"
    COMMUNITY_EXPERIENCE = "community_experience"
    GENERAL_RESEARCH = "general_research"
    COMPARISON = "comparison"
    CURRENT_EVENTS = "current_events"


class CriticVerdict(str, Enum):
    GOOD = "good"
    NEEDS_IMPROVEMENT = "needs_improvement"
    BAD = "bad"


class CriticSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RetrievalMode(str, Enum):
    """How ``source_chunks`` is searched (Phase 7). ``DENSE`` is the rollback / baseline path."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class FreshnessCategory(str, Enum):
    """How a request's answer depends on today's date (P1.1). Extends, and is evaluated
    alongside, ``agent.temporal.is_time_sensitive`` rather than replacing it."""
    STABLE = "stable"
    CURRENT_INFORMATION = "current_information"
    RECENT = "recent"
    AS_OF_DATE = "as_of_date"
    VERSION_DEPENDENT = "version_dependent"
    HISTORICAL = "historical"
    UNKNOWN = "unknown"


class DateConfidence(str, Enum):
    """How a ``SourceDocument`` date was obtained (P1.2). ``UNKNOWN`` means a raw date value
    was present but could not be trusted (kept in ``metadata`` verbatim, never invented)."""
    EXACT = "exact"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


class SourceFreshnessStatus(str, Enum):
    """A source's freshness, evaluated against a ``FreshnessPolicy`` (P1.2/P1.3), never
    against one global threshold. ``UNKNOWN`` when no date is available at all."""
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"
    FUTURE_INVALID = "future_invalid"
    DATE_CONFLICT = "date_conflict"


class ContentClassification(str, Enum):
    """A finer-grained content type than ``SourceType`` (P1.4). ``SourceType`` is the routing
    bucket (web/official_docs/github/reddit/...) and is unchanged and still load-bearing
    everywhere (routing, synthesis labels, citations, the DB); this is an additional,
    optional classification of *what kind of document* a source actually is, so a preprint
    is never presented as peer-reviewed, a blog as an academic paper, or a snippet as a
    complete source."""
    OFFICIAL_DOCUMENTATION = "official_documentation"
    OFFICIAL_ANNOUNCEMENT = "official_announcement"
    RESEARCH_PAPER = "research_paper"
    PREPRINT = "preprint"
    SURVEY = "survey"
    TECHNICAL_REPORT = "technical_report"
    GITHUB_REPOSITORY = "github_repository"
    GITHUB_ISSUE_OR_DISCUSSION = "github_issue_or_discussion"
    REDDIT_POST_OR_DISCUSSION = "reddit_post_or_discussion"
    NEWS_REPORT = "news_report"
    EXPERT_BLOG = "expert_blog"
    SEARCH_SNIPPET = "search_snippet"
    UNKNOWN = "unknown"


class AuthorityLevel(str, Enum):
    """How much weight a source's *type* carries on its own (P1.4), independent of its
    freshness. Not a quality score: a ``COMMUNITY`` source can still be correct, it just
    is not treated as authority for official API/configuration behavior (Phase 4/5 rule,
    now keyed off this field instead of being implicit)."""
    OFFICIAL = "official"
    ACADEMIC = "academic"
    ESTABLISHED = "established"
    COMMUNITY = "community"
    UNKNOWN = "unknown"
