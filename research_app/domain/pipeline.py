"""Contracts for the Phase 6 research pipeline: per-task routing, synthesis output,
conversation context and the final response.

Pure Pydantic, like ``models``: no imports from ``research_app.agent``, ``research_app.db``
or any framework package.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from research_app.domain.enums import SourceIntent, SourceType
from research_app.domain.models import Citation, DomainModel, UtcDatetime

Confidence = Literal["high", "medium", "low"]

# Where a source in a RoutingDecision came from.
ORIGIN_PLANNER = "planner"  # the planner proposed it and the router accepted it
ORIGIN_POLICY = "policy"  # the router added it (intent default or enforcement rule)
ORIGIN_FALLBACK = "fallback"  # the Phase 5 deterministic route (no usable planner hint)


# --------------------------------------------------------------------------- routing

class DroppedSource(DomainModel):
    """A planner suggestion the router did not use, and why (``reason`` is a stable code)."""

    source: str  # raw value as the planner wrote it (may not be a valid source type)
    reason: str


class SourceOutcome(DomainModel):
    """What one source returned for one task, filled in after the search ran."""

    source: SourceType
    status: Literal["ok", "empty", "failed", "timeout", "skipped"]
    result_count: int = Field(default=0, ge=0)
    latency_ms: Optional[float] = Field(default=None, ge=0)
    error_code: Optional[str] = None


class RoutingDecision(DomainModel):
    """The router's decision for one task. Pure function of its inputs (deterministic);
    ``outcomes`` is added afterwards by the executor and is not part of the decision."""

    task_id: str
    sub_question: str
    source_intent: Optional[SourceIntent] = None  # the intent the decision was made under
    sources: list[SourceType]  # ordered: official_docs, github, reddit, web
    origins: dict[str, str] = Field(default_factory=dict)  # source value -> ORIGIN_*
    dropped: list[DroppedSource] = Field(default_factory=list)
    # True when official docs were selected for a technology + problem question (the Phase 4
    # detector did not select them); the executor then plans the documentation search itself.
    docs_from_technology: bool = False
    stage: Literal["search", "gap_search", "retry"] = "search"  # which pass ran it
    outcomes: list[SourceOutcome] = Field(default_factory=list)

    def includes(self, source_type: SourceType) -> bool:
        return source_type in self.sources

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.value for s in self.sources)


# --------------------------------------------------------------------------- synthesis

class SynthesisResult(DomainModel):
    answer_text: str  # full answer with inline [1], [2] markers
    citations: list[Citation] = Field(default_factory=list)  # ordered; index matches the marker
    confidence: Confidence = "low"
    source_types_used: list[str] = Field(default_factory=list)
    subquestions_covered: list[str] = Field(default_factory=list)
    subquestions_missed: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- conversation

class ConversationTurn(DomainModel):
    turn_number: int = Field(ge=1)
    query_text: str
    resolved_query: str
    answer_summary: str  # first N characters only; never the full answer
    source_types_used: list[str] = Field(default_factory=list)
    created_at: Optional[UtcDatetime] = None


class ConversationContext(DomainModel):
    session_id: Optional[int] = None
    turns: list[ConversationTurn] = Field(default_factory=list)  # oldest first
    topics_discussed: list[str] = Field(default_factory=list)
    technologies_mentioned: list[str] = Field(default_factory=list)
    total_turns: int = Field(default=0, ge=0)  # all turns in the session, not just the loaded ones

    @property
    def is_empty(self) -> bool:
        return not self.turns


# --------------------------------------------------------------------------- response

class ResearchMetadata(DomainModel):
    total_sources_found: int = 0
    total_evidence_pieces: int = 0
    gap_search_performed: bool = False
    retry_performed: bool = False
    sources_failed: list[str] = Field(default_factory=list)
    routing_decisions: list[RoutingDecision] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    critic_issues: list[dict[str, Any]] = Field(default_factory=list)
    cache_hit: bool = False
    latency_ms: Optional[float] = None


class ResearchResponse(DomainModel):
    """What a research request returns. The API adds it to the legacy keys
    (``sub_questions``, ``answer``, ``session_id``); nothing existing is renamed."""

    session_id: Optional[int] = None
    turn_number: int = Field(default=1, ge=1)
    is_follow_up: bool = False
    resolved_query: str = ""
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    sources_consulted: list[str] = Field(default_factory=list)  # source types actually used
    subquestions: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"
    critic_verdict: str = ""
    research_metadata: ResearchMetadata = Field(default_factory=ResearchMetadata)
