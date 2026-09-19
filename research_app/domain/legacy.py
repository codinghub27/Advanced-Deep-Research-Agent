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

Phase 4 adds ``official_docs_entries`` (one labelled ``search_results`` entry per
official-documentation document), ``is_official_docs_entry`` and
``prioritize_search_results`` (official entries first, for the synthesis prompt).

Phase 5 adds ``community_entries`` (labelled ``[GITHUB: ...]`` / ``[REDDIT: ...]`` entries),
the ``[WEB]`` label (an optional argument of ``search_result_entry``), ``entry_kind`` and
``select_search_results`` (which entries the synthesis prompt reads).

Used at runtime by the two search nodes and the synthesis step in ``agent/state.py``.
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


WEB_LABEL = "[WEB]"


def search_result_entry(
    query: str, docs: Iterable[SourceDocument], *, content_limit: int, label: Optional[str] = None
) -> str:
    """The single ``search_results`` string a search node emits:
    ``Query: <q>\\n<url>\\n<content>`` blocks joined by ``\\n---\\n``. Content is cut
    to ``content_limit`` characters; documents without content are left out, as
    they always were. ``label`` (Phase 5: ``WEB_LABEL``) goes on its own line after the
    query line; without it the output is exactly what it always was."""
    condensed = [f"{d.url}\n{d.content[:content_limit]}" for d in docs if d.content]
    head = f"Query: {query}\n" + (f"{label}\n" if label else "")
    return head + "\n---\n".join(condensed)


OFFICIAL_DOCS_LABEL = "[OFFICIAL DOCUMENTATION"
# The synthesis step reads at most this many characters of each search_results entry.
SEARCH_ENTRY_MAX_CHARS = 1500


def official_docs_entries(
    query: str,
    docs: Iterable[SourceDocument],
    *,
    content_limit: int = 1000,
    entry_limit: int = SEARCH_ENTRY_MAX_CHARS,
) -> list[str]:
    """One ``search_results`` entry per official-documentation document::

        Query: <q>
        [OFFICIAL DOCUMENTATION: <technology>, version <v>]
        <url>
        <content>

    The second line is the marker (``is_official_docs_entry``). Each entry is at most
    ``entry_limit`` characters (the synthesis step cuts entries there), so the header is
    never sacrificed to a long body. Documents without content are left out."""
    entries: list[str] = []
    for doc in docs:
        if not doc.content:
            continue
        name = doc.technology or doc.domain or "unknown"
        label = f"{OFFICIAL_DOCS_LABEL}: {name}" + (f", version {doc.version}" if doc.version else "") + "]"
        head = f"Query: {query[:200]}\n{label}\n{doc.url}\n"
        budget = max(0, min(content_limit, entry_limit - len(head)))
        entries.append(head + doc.content[:budget])
    return entries


def is_official_docs_entry(entry: str) -> bool:
    lines = entry.split("\n", 2)
    return len(lines) > 1 and lines[1].startswith(OFFICIAL_DOCS_LABEL)


def prioritize_search_results(entries: Iterable[str]) -> list[str]:
    """Official-documentation entries first; everything else keeps its order."""
    entries = list(entries)
    return [e for e in entries if is_official_docs_entry(e)] + [
        e for e in entries if not is_official_docs_entry(e)
    ]


GITHUB_LABEL = "[GITHUB"
REDDIT_LABEL = "[REDDIT"

_GITHUB_NUMBERED = {"issue": "issue", "pull_request": "pull request", "discussion": "discussion"}


def _meta(doc: SourceDocument, key: str) -> Mapping[str, Any]:
    value = doc.metadata.get(key)
    return value if isinstance(value, Mapping) else {}


def _community_label(doc: SourceDocument) -> Optional[str]:
    """``[GITHUB: owner/repo, issue #12]`` / ``[REDDIT: r/sub]``. Built only from the
    URL-derived, pattern-validated metadata the adapters attach, never from result text,
    so page content cannot forge a label."""
    if doc.source_type == SourceType.GITHUB:
        meta = _meta(doc, "github")
        name = "/".join(str(meta[k]) for k in ("owner", "repo") if meta.get(k))
        kind = _GITHUB_NUMBERED.get(meta.get("kind"))
        detail = f", {kind} #{meta['number']}" if kind and isinstance(meta.get("number"), int) else ""
        return f"{GITHUB_LABEL}: {name}{detail}]" if name else f"{GITHUB_LABEL}]"
    if doc.source_type == SourceType.REDDIT:
        sub = _meta(doc, "reddit").get("subreddit")
        return f"{REDDIT_LABEL}: r/{sub}]" if sub else f"{REDDIT_LABEL}]"
    return None


def community_entries(
    query: str,
    docs: Iterable[SourceDocument],
    *,
    content_limit: int = 1000,
    entry_limit: int = SEARCH_ENTRY_MAX_CHARS,
) -> list[str]:
    """One ``search_results`` entry per GitHub/Reddit document::

        Query: <q>
        [GITHUB: owner/repo, issue #12]      (or [REDDIT: r/subreddit])
        <url>
        <content>

    The second line is the label (``entry_kind``). Each entry is at most ``entry_limit``
    characters, so the header is never sacrificed to a long body. Documents of other types
    and documents without content are left out."""
    query_line = " ".join(str(query).split())[:200]  # one line: the label must stay on line 2
    entries: list[str] = []
    for doc in docs:
        label = _community_label(doc)
        if label is None or not doc.content:
            continue
        head = f"Query: {query_line}\n{label}\n{doc.url}\n"
        budget = max(0, min(content_limit, entry_limit - len(head)))
        entries.append(head + doc.content[:budget])
    return entries


def entry_kind(entry: str) -> str:
    """``official`` | ``github`` | ``reddit`` | ``web``, from the entry's second line."""
    lines = entry.split("\n", 2)
    second = lines[1] if len(lines) > 1 else ""
    if second.startswith(OFFICIAL_DOCS_LABEL):
        return "official"
    if second.startswith(GITHUB_LABEL):
        return "github"
    if second.startswith(REDDIT_LABEL):
        return "reddit"
    return "web"


def select_search_results(entries: Iterable[str], limit: int) -> list[str]:
    """The entries the synthesis prompt reads (at most ``limit``).

    Official documentation first, everything else in order (``prioritize_search_results``).
    When there are more than ``limit`` entries AND GitHub/Reddit entries are among them, the
    limit is shared round-robin between official / github / reddit / web, so a source the
    router selected is not silently cut off by the entry cap. Without GitHub/Reddit entries
    this is exactly ``prioritize_search_results(entries)[:limit]`` (Phase 4 behaviour)."""
    ordered = prioritize_search_results(entries)
    if len(ordered) <= limit:
        return ordered
    groups: dict[str, list[str]] = {"official": [], "github": [], "reddit": [], "web": []}
    for entry in ordered:
        groups[entry_kind(entry)].append(entry)
    if not (groups["github"] or groups["reddit"]):
        return ordered[:limit]
    selected: list[str] = []
    for position in range(max(len(g) for g in groups.values())):
        for group in groups.values():
            if position < len(group) and len(selected) < limit:
                selected.append(group[position])
    return selected


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
