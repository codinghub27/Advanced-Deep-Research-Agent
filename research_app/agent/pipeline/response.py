"""Build the ``ResearchResponse`` from a finished graph state (Phase 6).

One function for both a researched answer and a cache hit, so the API, the SSE ``done`` event and
the database all describe a run the same way. Pure: reads the state, touches nothing.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from research_app.domain import (
    Citation,
    ResearchMetadata,
    ResearchResponse,
    RoutingDecision,
    canonical_url,
)
from research_app.domain.legacy import CACHE_HIT_PLACEHOLDER

FAILED_STATUSES = {"failed", "timeout"}


def _decisions(items) -> list[RoutingDecision]:
    """Routing decisions from the state, or (cache hit) the JSON stored with the cached answer."""
    decisions = []
    for item in items or []:
        if isinstance(item, RoutingDecision):
            decisions.append(item)
        elif isinstance(item, Mapping):
            try:
                decisions.append(RoutingDecision.model_validate(item))
            except Exception:
                continue
    return decisions


def _citations(items) -> list[Citation]:
    out = []
    for item in items or []:
        if isinstance(item, Citation):
            out.append(item)
            continue
        try:
            source_type = item.get("source_type", "web")
            out.append(Citation(
                marker=item.get("marker") or f"[{item.get('index')}]",
                index=item.get("index"),
                source_id=item.get("source_id") or item.get("url", ""),
                url=item["url"], title=item.get("title") or None, source_type=source_type,
                domain=item.get("domain") or None, snippet=item.get("snippet") or None,
                retrieved_at=item.get("retrieved_at"),
            ))
        except Exception:
            continue
    return out


def sources_failed(decisions) -> list[str]:
    """``source:reason`` for every selected source that failed or timed out, once each."""
    failed: list[str] = []
    for decision in decisions:
        for outcome in decision.outcomes:
            if outcome.status in FAILED_STATUSES:
                label = f"{outcome.source.value}:{outcome.error_code or outcome.status}"
                if label not in failed:
                    failed.append(label)
    return failed


def build_response(
    state: Mapping[str, Any],
    *,
    session_id: Optional[int],
    turn_number: int,
    latency_ms: Optional[float] = None,
) -> ResearchResponse:
    cache_hit = bool(state.get("cache_hit"))
    question = state.get("question", "")
    decisions = _decisions(state.get("cached_routing") if cache_hit else state.get("routing_decisions"))
    citations = _citations(state.get("citations"))
    subquestions = [q for q in state.get("sub_questions") or [] if q != CACHE_HIT_PLACEHOLDER]
    critic = state.get("critic_result")
    synthesis = state.get("synthesis")
    pool = state.get("evidence_pool")

    if cache_hit:
        used = sorted({c.source_type.value for c in citations})
        verdict = "cached"
    else:
        used = list(synthesis.source_types_used) if synthesis is not None else []
        verdict = (critic.verdict.value if critic is not None and critic.verdict is not None
                   else "unavailable")

    urls = {canonical_url(d.url) for d in state.get("source_documents") or []}
    issues = []
    if critic is not None:
        issues = [{"kind": i.kind.value, "severity": i.severity.value if i.severity else None,
                   "description": i.description} for i in critic.issues]
    return ResearchResponse(
        session_id=session_id,
        turn_number=turn_number,
        is_follow_up=bool(state.get("is_follow_up")),
        resolved_query=state.get("resolved_query") or question,
        answer=state.get("final_answer", ""),
        citations=citations,
        sources_consulted=used,
        subquestions=subquestions,
        confidence=state.get("confidence") or (synthesis.confidence if synthesis is not None else "medium"),
        critic_verdict=verdict,
        research_metadata=ResearchMetadata(
            total_sources_found=len(urls),
            total_evidence_pieces=len(pool) if pool is not None else 0,
            gap_search_performed=bool(state.get("gap_search_performed")),
            retry_performed=bool(state.get("retry_performed")),
            sources_failed=sources_failed(decisions),
            routing_decisions=decisions,
            limitations=list(state.get("limitations") or []),
            critic_issues=issues,
            cache_hit=cache_hit,
            latency_ms=round(latency_ms, 1) if latency_ms is not None else None,
        ),
    )
