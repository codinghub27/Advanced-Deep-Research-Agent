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
