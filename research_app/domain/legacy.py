"""Pure shape adapters between today's loose structures and the domain contracts.

Only structures that exist in the running app are translated:
- the ``{"url", "title", "snippet"}`` source dict (SSE ``sources`` event)
- ``question`` / ``is_simple`` / ``history`` in ``ResearchState``
- ``sub_questions``
- a final ``ResearchState`` snapshot -> ``ResearchRun``

Also (Phase 3): ``search_result_entry`` renders ``SourceDocument``s as the
``"Query: ...\\nurl\\ncontent"`` string held in ``search_results``. Tavily payloads
are parsed in ``research_app/sources``, not here, and ``search_results`` strings are
never parsed back.

Used at runtime by the two search nodes in ``agent/state.py``.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Optional, Sequence

from pydantic import ValidationError

from research_app.domain.enums import Complexity, RunStatus, SourceType
from research_app.domain.models import (
    HistoryTurn,
    ResearchQuery,
    ResearchRun,
    ResearchTask,
    SourceDocument,
    UtcDatetime,
)

logger = logging.getLogger(__name__)

# semantic_cache_node sets sub_questions to this sentinel on a cache hit.
CACHE_HIT_PLACEHOLDER = "(from cache)"


def source_from_legacy(
    item: Mapping[str, Any],
    *,
    source_type: SourceType = SourceType.WEB,
    task_id: Optional[str] = None,
    query: Optional[str] = None,
    retrieved_at: Optional[UtcDatetime] = None,
) -> SourceDocument:
    """``{"url", "title", "snippet"}`` -> ``SourceDocument``. Raises
    ``pydantic.ValidationError`` if the URL is missing or not http(s)."""
    extra: dict[str, Any] = {}
    if retrieved_at is not None:
        extra["retrieved_at"] = retrieved_at
    return SourceDocument(
        url=item.get("url", ""),
        title=item.get("title"),
        snippet=item.get("snippet"),
        source_type=source_type,
        task_id=task_id,
        query=query,
        **extra,
    )


def source_to_legacy(doc: SourceDocument) -> dict[str, str]:
    """``SourceDocument`` -> the exact three-key dict the SSE ``sources`` event carries."""
    return {"url": doc.url, "title": doc.title or "", "snippet": doc.snippet or ""}


def search_result_entry(
    query: str, docs: Iterable[SourceDocument], *, content_limit: int
) -> str:
    """The single ``search_results`` string a search node emits:
    ``Query: <q>\\n<url>\\n<content>`` blocks joined by ``\\n---\\n``. Content is cut
    to ``content_limit`` characters; documents without content are left out, as
    they always were."""
    condensed = [f"{d.url}\n{d.content[:content_limit]}" for d in docs if d.content]
    return f"Query: {query}\n" + "\n---\n".join(condensed)


def sources_from_legacy(
    items: Iterable[Mapping[str, Any]], **kwargs: Any
) -> list[SourceDocument]:
    """Convert many, skipping (and logging) entries that are not valid sources,
    de-duplicated by ``source_id`` with the first occurrence kept."""
    docs: list[SourceDocument] = []
    seen: set[str] = set()
    for item in items:
        try:
            doc = source_from_legacy(item, **kwargs)
        except (ValidationError, AttributeError, TypeError) as exc:
            logger.warning("Skipping invalid legacy source: %s", type(exc).__name__)
            continue
        if doc.source_id in seen:
            continue
        seen.add(doc.source_id)
        docs.append(doc)
    return docs


def query_from_state(
    state: Mapping[str, Any],
    *,
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
    classified: bool = False,
) -> ResearchQuery:
    """Build a ``ResearchQuery`` from graph state.

    ``is_simple`` is initialised to ``False`` before ``classify_node`` runs, so it
    is only read when the caller says the state was classified.
    """
    complexity = None
    if classified and "is_simple" in state:
        complexity = Complexity.SIMPLE if state["is_simple"] else Complexity.COMPLEX

    history: list[HistoryTurn] = []
    for turn in state.get("history") or []:
        try:
            history.append(HistoryTurn(question=turn["question"], answer=turn["answer"]))
        except (ValidationError, KeyError, TypeError):
            logger.warning("Skipping malformed history turn")

    return ResearchQuery(
        text=state.get("question", ""),
        user_id=user_id,
        session_id=session_id,
        complexity=complexity,
        history=history,
    )


def tasks_from_sub_questions(
    sub_questions: Sequence[str],
    *,
    source_types: Optional[Sequence[SourceType]] = None,
) -> list[ResearchTask]:
    """``sub_questions`` -> tasks. Blank entries and the cache-hit sentinel are skipped."""
    tasks: list[ResearchTask] = []
    for text in sub_questions:
        if not isinstance(text, str) or not text.strip() or text == CACHE_HIT_PLACEHOLDER:
            continue
        kwargs: dict[str, Any] = {}
        if source_types:
            kwargs["source_types"] = list(source_types)
        tasks.append(ResearchTask(sub_question=text, **kwargs))
    return tasks


def tasks_to_sub_questions(tasks: Iterable[ResearchTask]) -> list[str]:
    return [t.sub_question for t in tasks]


def run_from_state(
    state: Mapping[str, Any],
    *,
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
    run_id: Optional[str] = None,
    status: RunStatus = RunStatus.COMPLETED,
    classified: bool = True,
) -> ResearchRun:
    """Snapshot a finished ``ResearchState`` as a ``ResearchRun``.

    Legacy sources are not linked to tasks, so they land in ``run.sources``
    (de-duplicated by canonical URL), not in ``run.results``.
    """
    kwargs: dict[str, Any] = {}
    if run_id:
        kwargs["run_id"] = run_id
    cache_hit = bool(state.get("cache_hit"))
    return ResearchRun(
        user_id=user_id,
        session_id=session_id,
        query=query_from_state(
            state, user_id=user_id, session_id=session_id, classified=classified and not cache_hit
        ),
        status=status,
        tasks=tasks_from_sub_questions(state.get("sub_questions") or []),
        sources=[] if cache_hit else sources_from_legacy(state.get("sources") or []),
        final_answer=state.get("final_answer") or "",
        cache_hit=cache_hit,
        **kwargs,
    )
