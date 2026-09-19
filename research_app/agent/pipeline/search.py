"""Targeted searches for the gap round and the critic retry (Phase 6).

Runs a few ``ResearchTask``s through the same routed search the main pass uses
(``state._research_update``): same router, same single Tavily client, same concurrency slots.
A search that fails is logged and skipped; it never fails the run. The results come back as a
state update in the shape the graph already merges (``source_documents`` etc. are add-reducers).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Sequence

import research_app.agent.state as st
from research_app.domain import ResearchTask

logger = logging.getLogger(__name__)

TARGETED_CONTENT_LIMIT = 800


async def run_targeted_searches(
    tasks: Sequence[ResearchTask], *, question: str, understanding: Optional[dict], stage: str = "search"
) -> dict:
    """Search every task concurrently and merge the successful updates. Empty update if none.
    ``stage`` ("gap_search" / "retry") is recorded on the routing decisions of these searches."""
    if not tasks:
        return {}

    async def one(task: ResearchTask):
        query = task.search_query or task.sub_question
        return await st._research_update(
            query, question=question, content_limit=TARGETED_CONTENT_LIMIT,
            task=task, understanding=understanding)

    outcomes = await asyncio.gather(*(one(t) for t in tasks), return_exceptions=True)
    merged: dict[str, list] = {"search_results": [], "sources": [], "source_documents": [],
                               "routing_decisions": []}
    for task, outcome in zip(tasks, outcomes):
        if isinstance(outcome, asyncio.CancelledError):
            raise outcome
        if isinstance(outcome, BaseException):
            logger.warning("Targeted search failed (%s); continuing without it", type(outcome).__name__)
            continue
        for key in merged:
            merged[key].extend(outcome.get(key) or [])
        merged["routing_decisions"] = [d.model_copy(update={"stage": stage}) if d.stage != stage else d
                                       for d in merged["routing_decisions"]]
    return merged
