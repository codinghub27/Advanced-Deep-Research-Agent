"""Route-layer helpers for a research request (Phase 6).

Kept out of ``main.py`` so the two research endpoints (plain and streaming) share one
implementation: load the conversation, build the graph inputs, describe the finished run, and
save it. Nothing here calls an LLM.

Saving is the one thing that must never hurt the user: ``save_conversation_turn`` catches every
error, logs it, and returns. It runs after the response (a FastAPI background task) with its own
database session, never the request's.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable, Mapping, Optional

from sqlalchemy.orm import Session as DBSession

from research_app.conversation.context import build_context
from research_app.conversation.settings import ConversationSettings
from research_app.db import conversations as repo
from research_app.domain import ConversationContext, ResearchResponse

logger = logging.getLogger(__name__)

# State keys the graph merges with an "add" reducer; the streaming route mirrors that.
ADD_KEYS = ("messages", "sources", "search_results", "source_documents", "routing_decisions")


def merge_update(state: dict, update: Mapping[str, Any]) -> None:
    """Fold one node update into ``state`` the way the graph does (lists add, the rest replace)."""
    for key, value in update.items():
        if key in ADD_KEYS and isinstance(value, list):
            state.setdefault(key, []).extend(value)
        else:
            state[key] = value


def load_conversation(db: DBSession, session_id: int) -> tuple[ConversationContext, int]:
    """The session's recent turns as context, and the number this request's turn will get.
    A failure is logged and degrades to "no history" so research still runs."""
    settings = ConversationSettings.from_env()
    try:
        rows = repo.load_recent_turns(db, session_id, settings.history_limit)
        total = repo.count_turns(db, session_id)
        turn_number = repo.last_turn_number(db, session_id) + 1
        return build_context(session_id, rows, total_turns=total, settings=settings), turn_number
    except Exception as exc:
        logger.error("Could not load conversation history for session %s (%s); continuing without it",
                     session_id, type(exc).__name__)
        return ConversationContext(session_id=session_id), 1


def build_inputs(
    question: str, history: list, *, session_id: int, turn_number: int, context: ConversationContext,
    run_id: Optional[str] = None,
) -> dict:
    """The graph's initial state: exactly what ``main.py`` always passed, plus the Phase 6 keys."""
    return {
        "question": question,
        "messages": [],
        "step_count": 0,
        "final_answer": "",
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "cache_hit": False,
        "api_limit_reached": False,
        "critic_score": 0,
        "critic_feedback": "",
        "is_simple": False,
        "history": history,
        "run_id": run_id or uuid.uuid4().hex,
        "session_id": session_id,
        "turn_number": turn_number,
        "conversation_context": context,
        "started_at": time.perf_counter(),
    }


def elapsed_ms(state: Mapping[str, Any]) -> Optional[float]:
    started = state.get("started_at")
    return (time.perf_counter() - started) * 1000 if isinstance(started, (int, float)) else None


def response_body(response: ResearchResponse, legacy_sub_questions: list) -> dict:
    """The JSON of ``/api/research``: the three keys it always had, plus the Phase 6 fields."""
    body = response.model_dump(mode="json")
    body["sub_questions"] = legacy_sub_questions
    return body


def legacy_sources_payload(response: ResearchResponse) -> dict:
    """What goes into the ``queries.sources`` JSON column (always ``{}`` before): the numbered
    citations and the routing report, so a reloaded chat can show them without another lookup."""
    dump = response.model_dump(mode="json")
    return {
        "citations": dump["citations"],
        "routing": dump["research_metadata"]["routing_decisions"],
    }


def turn_payload(question: str, state: Mapping[str, Any], response: ResearchResponse) -> dict:
    understanding = state.get("understanding") or {}
    query_type = "cache" if response.research_metadata.cache_hit else (understanding.get("intent") or "unknown")
    dump = response.model_dump(mode="json")
    return {
        "query_text": question,
        "query_type": query_type,
        "is_follow_up": response.is_follow_up,
        "resolved_query": response.resolved_query,
        "answer_text": response.answer,
        "confidence": response.confidence,
        "critic_verdict": response.critic_verdict,
        "sources_consulted": response.sources_consulted,
        "subquestions": response.subquestions,
        "research_metadata": dump["research_metadata"],
    }


def save_conversation_turn(
    session_factory: Callable[[], DBSession],
    *,
    session_id: int,
    turn_number: int,
    payload: Mapping[str, Any],
    citations: list,
    run_id: str = "",
) -> bool:
    """Store the turn and its citations. Never raises; returns whether it saved."""
    try:
        with session_factory() as db:
            repo.save_turn(db, session_id=session_id, turn_number=turn_number, data=payload, citations=citations)
        logger.info("Conversation saved: run=%s session=%s turn=%s citations=%d",
                    run_id, session_id, turn_number, len(citations))
        return True
    except Exception as exc:
        logger.error("Conversation save FAILED: run=%s session=%s turn=%s (%s: %s)",
                     run_id, session_id, turn_number, type(exc).__name__, str(exc)[:200])
        return False
