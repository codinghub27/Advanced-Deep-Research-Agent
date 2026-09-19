"""Run several source adapters concurrently with failure isolation.

Each job is ``(adapter, request)``. A job that raises, returns something that is not a
``ResearchResult``, or times out yields a FAILED/TIMEOUT result for that source only; the
others are unaffected. An empty job list returns an empty list. Concurrency is bounded by the
number of jobs (the router selects at most GitHub and Reddit here).

Cancellation propagates: cancelling the awaiting task cancels every job.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from research_app.domain import (
    ResearchError,
    ResearchResult,
    ResultStatus,
    SearchRequest,
    SourceAdapter,
)

logger = logging.getLogger(__name__)


def _failed(adapter: SourceAdapter, request: SearchRequest, code: str, message: str) -> ResearchResult:
    return ResearchResult(
        task_id=request.task_id,
        source_type=adapter.source_type,
        status=ResultStatus.FAILED,
        error=ResearchError(code=code, message=message[:200], retryable=False,
                            source_type=adapter.source_type),
    )


async def _run_one(adapter: SourceAdapter, request: SearchRequest) -> ResearchResult:
    try:
        result = await adapter.search(request)
    except Exception as exc:  # CancelledError is a BaseException and is not caught
        logger.warning("%s adapter crashed (%s); continuing without it",
                       adapter.source_type.value, type(exc).__name__)
        return _failed(adapter, request, "adapter_crash", f"{type(exc).__name__}: {exc}")
    if not isinstance(result, ResearchResult):
        logger.warning("%s adapter returned %s; ignoring it", adapter.source_type.value, type(result).__name__)
        return _failed(adapter, request, "invalid_result", f"adapter returned {type(result).__name__}")
    return result


async def run_adapters(jobs: Sequence[tuple[SourceAdapter, SearchRequest]]) -> list[ResearchResult]:
    """Results in job order. Never raises for a source failure."""
    tasks = [asyncio.create_task(_run_one(adapter, request)) for adapter, request in jobs]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        raise
