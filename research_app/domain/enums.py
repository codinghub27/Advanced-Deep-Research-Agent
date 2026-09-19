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
