"""Graph nodes for evidence, gaps, retry and the final response (Phase 6).

Flow (see ``agent/graph.py``):
  evidence_collection -> gap_detection -> [targeted_search] -> synthesis_node -> critic_node
  -> [retry_node] -> format_response

Both bracketed steps happen at most once per run. There is no edge back to an earlier node, so
the bounds are structural, not counters that could be forgotten.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from research_app.agent.pipeline.critic import VERDICT_RANK, run_critic, worth_retrying
from research_app.agent.pipeline.evidence import EvidencePool, build_pool
from research_app.agent.pipeline.gaps import detect_gaps, normalize_query
from research_app.agent.pipeline.search import run_targeted_searches
from research_app.agent.pipeline.settings import PipelineSettings
from research_app.agent.pipeline.synthesis import run_synthesis, strip_orphan_markers
from research_app.domain import (
    CriticResult,
    CriticSeverity,
    CriticVerdict,
    GapAnalysis,
    ResearchTask,
    SynthesisResult,
)
from research_app.domain.legacy import tasks_from_sub_questions

logger = logging.getLogger(__name__)


def _tasks_for(state) -> list[ResearchTask]:
    """The run's tasks; a state that did not go through the planner falls back to its
    sub-question strings."""
    tasks = state.get("research_tasks")
    if tasks:
        return list(tasks)
    return tasks_from_sub_questions(state.get("sub_questions") or [])


def _executed_queries(state) -> set[str]:
    queries: set[str] = set()
    for task in _tasks_for(state):
        queries.add(normalize_query(task.search_query or task.sub_question))
        queries.add(normalize_query(task.sub_question))
    analysis = state.get("gap_analysis")
    for task in getattr(analysis, "follow_up_tasks", None) or []:
        queries.add(normalize_query(task.search_query or task.sub_question))
    return queries


# --------------------------------------------------------------------------- evidence + gaps

def evidence_collection_node(state):
    """Collect every normalised document of the run into an in-memory ``EvidencePool``."""
    settings = PipelineSettings.from_env()
    try:
        tasks = _tasks_for(state)
        pool = build_pool(tasks, state.get("source_documents") or [], settings.relevance_threshold)
        per_task = {t.sub_question[:50]: len(pool.evidence_for(t.task_id)) for t in tasks}
        logger.info("Evidence: %d pieces from %d documents; per sub-question %s",
                    len(pool), len(state.get("source_documents") or []), per_task)
    except Exception as exc:
        logger.warning("Evidence collection failed (%s); continuing with an empty pool", type(exc).__name__)
        pool = EvidencePool()
    return {"evidence_pool": pool}


def gap_detection_node(state):
    """Find sub-questions with no evidence and prepare (at most a few) follow-up queries."""
    settings = PipelineSettings.from_env()
    try:
        analysis = detect_gaps(
            state["evidence_pool"], _tasks_for(state),
            decisions=state.get("routing_decisions") or [],
            understanding=state.get("understanding"),
            max_queries=settings.max_targeted_queries if settings.gap_search_enabled else 0,
            already_run=_executed_queries(state),
        )
    except Exception as exc:
        logger.warning("Gap detection failed (%s); proceeding to synthesis", type(exc).__name__)
        analysis = GapAnalysis(sufficient=True, rationale="gap detection failed")
    logger.info("Gaps: %s; follow-up queries: %s", analysis.rationale,
                [t.search_query for t in analysis.follow_up_tasks])
    return {"gap_analysis": analysis}


def gap_router(state):
    analysis = state.get("gap_analysis")
    if analysis is not None and analysis.follow_up_tasks:
        return "targeted_search"
    return "synthesis_node"


async def targeted_search_node(state):
    """The single gap-filling round: search the follow-up queries, add what comes back to the pool."""
    analysis = state["gap_analysis"]
    update = await run_targeted_searches(
        analysis.follow_up_tasks, question=state["question"], understanding=state.get("understanding"),
        stage="gap_search")
    settings = PipelineSettings.from_env()
    documents = [*(state.get("source_documents") or []), *(update.get("source_documents") or [])]
    pool = build_pool(_tasks_for(state), documents, settings.relevance_threshold)
    logger.info("Gap search: +%d documents, %d evidence pieces now",
                len(update.get("source_documents") or []), len(pool))
    return {**update, "evidence_pool": pool, "gap_search_performed": True}


# --------------------------------------------------------------------------- critic retry

def critic_router(state):
    settings = PipelineSettings.from_env()
    if settings.max_critic_retries < 1 or state.get("retry_performed"):
        return "format_response"
    if not worth_retrying(state.get("critic_result")):
        return "format_response"
    # A retry needs something to work with: a new query to search, or evidence to write from again
    # (an answer whose citations were all wrong is fixed by re-writing it, not by searching).
    pool = state.get("evidence_pool")
    if not _retry_queries(state, settings) and not (pool is not None and len(pool) > 0):
        logger.info("Critic asked for a retry but there is nothing new to try; finishing")
        return "format_response"
    return "retry_node"


def _retry_queries(state, settings: PipelineSettings) -> list[str]:
    done = _executed_queries(state)
    queries: list[str] = []
    for query in state["critic_result"].suggestions:
        key = normalize_query(query)
        if key and key not in done:
            done.add(key)
            queries.append(query)
    return queries[: settings.max_targeted_queries]


def _retry_tasks(state, queries: list[str]) -> list[ResearchTask]:
    """One task per query. A query is filed under a sub-question that has no evidence when there is
    one (its evidence then counts for it), else the first sub-question."""
    tasks = _tasks_for(state)
    _, missing = state["evidence_pool"].coverage()
    homes = [t for t in tasks if t.task_id in missing] or tasks[:1]
    return [homes[i % len(homes)].model_copy(update={"search_query": q}) for i, q in enumerate(queries)]


async def retry_node(state):
    """One retry: targeted searches for what the critic said was missing, synthesize again, review
    again, and keep whichever attempt reviewed better (the retry wins a tie: it has more evidence)."""
    settings = PipelineSettings.from_env()
    first: SynthesisResult = state["synthesis"]
    first_critic: CriticResult = state["critic_result"]
    queries = _retry_queries(state, settings)
    logger.info("Critic retry with queries: %s", queries)
    update = await run_targeted_searches(
        _retry_tasks(state, queries), question=state["question"], understanding=state.get("understanding"),
        stage="retry")
    documents = [*(state.get("source_documents") or []), *(update.get("source_documents") or [])]
    pool = build_pool(_tasks_for(state), documents, settings.relevance_threshold)
    # The critic's findings go to the writer, so a re-write with the same evidence can fix them.
    retry_state = {**state, "evidence_pool": pool, "retry_feedback": first_critic.feedback}

    second = await run_synthesis(retry_state)
    second_critic = await run_critic(retry_state, second)
    use_second = VERDICT_RANK[second_critic.verdict] >= VERDICT_RANK[first_critic.verdict]
    logger.info("Retry verdict %s vs first %s; keeping the %s",
                second_critic.verdict.value if second_critic.verdict else "unavailable",
                first_critic.verdict.value if first_critic.verdict else "unavailable",
                "retry" if use_second else "first attempt")
    best, best_critic = (second, second_critic) if use_second else (first, first_critic)
    return {**update, "evidence_pool": pool, "synthesis": best, "critic_result": best_critic,
            "retry_performed": True}


# --------------------------------------------------------------------------- response

def citation_dict(citation) -> dict[str, Any]:
    """A citation as the UI and the database receive it."""
    return {
        "index": citation.index,
        "marker": citation.marker,
        "source_type": citation.source_type.value,
        "title": citation.title or "",
        "url": citation.url,
        "domain": citation.domain or "",
        "snippet": citation.snippet or "",
        "retrieved_at": citation.retrieved_at.isoformat() if citation.retrieved_at else None,
    }


def format_response_node(state):
    """Finish the run: pick the final answer, note limitations, settle confidence. The ONLY node
    that emits ``final_answer`` (the streaming layer turns each one into tokens)."""
    synthesis: SynthesisResult = state["synthesis"]
    critic: Optional[CriticResult] = state.get("critic_result")
    answer, removed = strip_orphan_markers(synthesis.answer_text, {c.index for c in synthesis.citations})
    confidence = synthesis.confidence
    limitations: list[str] = []
    verdict = critic.verdict if critic else None
    if verdict == CriticVerdict.BAD:
        limitations = [i.description for i in critic.issues if i.severity == CriticSeverity.HIGH][:3] \
            or [critic.feedback or "the answer could not be verified"]
        answer += "\n\n> **Note:** this answer may be incomplete or only partly supported. " + "; ".join(limitations)
        confidence = "low"
    elif verdict == CriticVerdict.NEEDS_IMPROVEMENT and confidence == "high":
        confidence = "medium"
    if removed:
        limitations.append(f"{removed} citation marker(s) that named no source were removed")
    return {
        "final_answer": answer,
        "citations": [citation_dict(c) for c in synthesis.citations],
        "limitations": limitations,
        "confidence": confidence,
    }
