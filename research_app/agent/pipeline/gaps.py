"""Lightweight gap detection and targeted follow-up queries (Phase 6).

Deliberately small: a sub-question with NO evidence is a gap and gets a follow-up query; a
sub-question answered by only one source type when several were selected is reported as a soft
gap but triggers nothing. One round only: the caller never loops.

Follow-up queries are made without an LLM: the sub-question is reduced to its key terms (a
different, shorter query than the one that just came back empty), with the technology
prepended when it is known.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

from research_app.agent.pipeline.evidence import EvidencePool, content_words
from research_app.domain import Gap, GapAnalysis, GapKind, ResearchTask, RoutingDecision
from research_app.sources.routing import task_technology

logger = logging.getLogger(__name__)

MAX_QUERY_WORDS = 8


def keyword_query(text: str, technology: Optional[str] = None) -> str:
    """The key terms of ``text`` in their original order (at most ``MAX_QUERY_WORDS``), with
    ``technology`` first when it is not already in there. Falls back to ``text``."""
    keep = content_words(text)
    seen: set[str] = set()
    words: list[str] = []
    for raw in text.split():
        word = raw.strip("?.,;:!\"'()[]{}").lower()
        if word in keep and word not in seen:
            seen.add(word)
            words.append(word)
    words = words[:MAX_QUERY_WORDS]
    query = " ".join(words) or text.strip()
    if technology and technology.lower() not in query.lower():
        query = f"{technology} {query}"
    return query.strip()


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def follow_up_task(task: ResearchTask, understanding: Optional[dict] = None) -> ResearchTask:
    """A task for the same sub-question (same id, so its evidence lands in the right place)
    with a keyword query. Source hints and intent are kept, so it is routed like the original."""
    return task.model_copy(update={
        "search_query": keyword_query(task.sub_question, task_technology(task, understanding)),
    })


def detect_gaps(
    pool: EvidencePool,
    tasks: Sequence[ResearchTask],
    *,
    decisions: Iterable[RoutingDecision] = (),
    understanding: Optional[dict] = None,
    max_queries: int = 2,
    already_run: Iterable[str] = (),
) -> GapAnalysis:
    """Coverage of ``tasks`` by ``pool``. ``follow_up_tasks`` holds at most ``max_queries`` tasks,
    for sub-questions with zero evidence, whose query differs from anything in ``already_run``."""
    covered, missing = pool.coverage()
    by_id = {t.task_id: t for t in tasks}
    gaps: list[Gap] = []
    follow_ups: list[ResearchTask] = []
    run = {normalize_query(q) for q in already_run}

    for task_id in missing:
        task = by_id.get(task_id) or pool.task(task_id)
        if task is None:
            continue
        gaps.append(Gap(kind=GapKind.MISSING_SUBQUESTION, task_id=task_id,
                        description=f"No evidence found for: {task.sub_question}"))
        if len(follow_ups) >= max_queries:
            continue
        candidate = follow_up_task(task, understanding)
        if normalize_query(candidate.search_query or "") in run:
            continue
        run.add(normalize_query(candidate.search_query or ""))
        follow_ups.append(candidate)

    # Soft gaps (reported only): several sources were selected but one type answered.
    selected = {d.task_id: d for d in decisions}
    for task_id in covered:
        decision = selected.get(task_id)
        if decision is None or len(decision.sources) < 2:
            continue
        if len(pool.source_types_for(task_id)) == 1:
            gaps.append(Gap(kind=GapKind.WEAK_EVIDENCE, task_id=task_id,
                            description="Only one source type returned evidence although "
                                        f"{', '.join(decision.names)} were searched"))

    hard = [g for g in gaps if g.kind == GapKind.MISSING_SUBQUESTION]
    return GapAnalysis(
        iteration=0,
        sufficient=not hard,
        covered_task_ids=covered,
        gaps=gaps,
        follow_up_tasks=follow_ups,
        rationale=f"{len(covered)} covered, {len(hard)} without evidence",
    )
