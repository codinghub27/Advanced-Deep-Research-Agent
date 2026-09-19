"""Stable domain contracts for the research pipeline (Phase 2).

Import from here, e.g. ``from research_app.domain import SourceDocument``.
"""
from research_app.domain.enums import (
    Complexity,
    CriticIssueKind,
    GapKind,
    QueryIntent,
    ResultStatus,
    RunStatus,
    SourceType,
    TaskStatus,
    TimeSensitivity,
)
from research_app.domain.interfaces import AnswerCache, SourceAdapter
from research_app.domain.models import (
    Citation,
    CriticIssue,
    CriticResult,
    Evidence,
    Gap,
    GapAnalysis,
    HistoryTurn,
    ResearchError,
    ResearchQuery,
    ResearchResult,
    ResearchRun,
    ResearchTask,
    SearchRequest,
    SourceCredibility,
    SourceDocument,
    canonical_url,
    new_id,
    source_id_for,
    utc_now,
)

__all__ = [
    "AnswerCache", "Citation", "Complexity", "CriticIssue", "CriticIssueKind",
    "CriticResult", "Evidence", "Gap", "GapAnalysis", "GapKind", "HistoryTurn",
    "QueryIntent", "ResearchError", "ResearchQuery", "ResearchResult",
    "ResearchRun", "ResearchTask", "ResultStatus", "RunStatus", "SearchRequest",
    "SourceAdapter", "SourceCredibility", "SourceDocument", "SourceType",
    "TaskStatus", "TimeSensitivity", "canonical_url", "new_id", "source_id_for",
    "utc_now",
]
