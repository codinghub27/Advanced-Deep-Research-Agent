"""Build the ``ConversationContext`` the pipeline reads from stored turns.

Pure: takes turn-like objects (anything with the attributes below), so it needs no database.
Answers are reduced to a short summary on purpose: full answers are thousands of tokens and
would fill the prompt. No LLM call; topics and technologies come from stored text and the
official-documentation registry.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from research_app.conversation.settings import ConversationSettings
from research_app.domain import ConversationContext, ConversationTurn

logger = logging.getLogger(__name__)

MAX_TOPICS = 10


def summarize_answer(answer: Optional[str], length: int) -> str:
    """The first ``length`` characters of ``answer`` on one line (never more)."""
    text = " ".join((answer or "").split())
    return text[:length].rstrip()


def _technologies(texts: Iterable[str]) -> list[str]:
    try:
        from research_app.sources.official_docs.detector import mentioned_technologies
        from research_app.sources.official_docs.registry import get_registry
        from research_app.sources.official_docs.settings import DocsSettings

        registry = get_registry(DocsSettings.from_env().registry_path)
        found: dict[str, str] = {}
        for text in texts:
            for entry in mentioned_technologies(text, registry):
                found.setdefault(entry.id, entry.name)
        return list(found.values())
    except Exception as exc:  # the registry must never break a conversation
        logger.warning("Could not extract technologies (%s)", type(exc).__name__)
        return []


def build_context(
    session_id: Optional[int],
    turns: Iterable[Any],
    *,
    total_turns: Optional[int] = None,
    settings: Optional[ConversationSettings] = None,
) -> ConversationContext:
    """``turns`` oldest first, at most the history limit (the caller loads them)."""
    settings = settings or ConversationSettings.from_env()
    built: list[ConversationTurn] = []
    topics: list[str] = []
    texts: list[str] = []
    for row in turns:
        query = (getattr(row, "query_text", "") or "").strip()
        resolved = (getattr(row, "resolved_query", "") or query).strip()
        summary = summarize_answer(getattr(row, "answer_text", ""), settings.answer_summary_length)
        used = [str(s) for s in (getattr(row, "sources_consulted", None) or [])]
        created = getattr(row, "created_at", None)
        built.append(ConversationTurn(
            turn_number=row.turn_number,
            query_text=query,
            resolved_query=resolved,
            answer_summary=summary,
            source_types_used=used,
            created_at=created,
        ))
        if resolved and resolved not in topics:
            topics.append(resolved)
        texts.extend((query, resolved, summary))
    return ConversationContext(
        session_id=session_id,
        turns=built,
        topics_discussed=topics[-MAX_TOPICS:],
        technologies_mentioned=_technologies(texts),
        total_turns=total_turns if total_turns is not None else len(built),
    )
