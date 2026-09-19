"""Conversation persistence (Phase 6): sessions listing, turns and citations.

Same style as ``crud.py``: plain functions over a SQLAlchemy ``Session``. Nothing here calls
an LLM. Callers own the ownership check (``crud.get_user_session``); these functions trust
the ``session_id`` they are given.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from research_app.db.models import ResearchCitation, ResearchConversation, Session as ChatSession, _utc_now

logger = logging.getLogger(__name__)

# Titles the UI/API give a session before the first question arrives.
DEFAULT_TITLES = frozenset({"New Research Session", "Research Session"})


def make_session_title(text: str, max_length: int = 80) -> str:
    """First ``max_length`` characters of the question on one line. No LLM."""
    title = " ".join((text or "").split())
    if len(title) > max_length:
        title = title[: max(1, max_length - 1)].rstrip() + "…"
    return title or "New Research Session"


# --------------------------------------------------------------------------- reading

def count_turns(db: Session, session_id: int) -> int:
    return db.query(func.count(ResearchConversation.id)).filter(
        ResearchConversation.session_id == session_id).scalar() or 0


def last_turn_number(db: Session, session_id: int) -> int:
    return db.query(func.max(ResearchConversation.turn_number)).filter(
        ResearchConversation.session_id == session_id).scalar() or 0


def load_recent_turns(db: Session, session_id: int, limit: int) -> list[ResearchConversation]:
    """The last ``limit`` turns of the session, oldest first."""
    rows = (
        db.query(ResearchConversation)
        .filter(ResearchConversation.session_id == session_id)
        .order_by(ResearchConversation.turn_number.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def list_user_sessions(db: Session, user_id: int, limit: int, offset: int) -> tuple[list[ChatSession], int]:
    """Active sessions of the user, most recently updated first, and the total count."""
    base = db.query(ChatSession).filter(ChatSession.user_id == user_id, ChatSession.is_active.is_(True))
    total = base.count()
    rows = (
        base.order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return rows, total


def turn_counts(db: Session, session_ids: Sequence[int]) -> dict[int, int]:
    """Number of stored turns per session, for a page of sessions."""
    if not session_ids:
        return {}
    rows = (
        db.query(ResearchConversation.session_id, func.count(ResearchConversation.id))
        .filter(ResearchConversation.session_id.in_(list(session_ids)))
        .group_by(ResearchConversation.session_id)
        .all()
    )
    return {session_id: count for session_id, count in rows}


def load_all_turns(db: Session, session_id: int) -> list[ResearchConversation]:
    return (
        db.query(ResearchConversation)
        .filter(ResearchConversation.session_id == session_id)
        .order_by(ResearchConversation.turn_number.asc())
        .all()
    )


# --------------------------------------------------------------------------- writing

def _naive_utc(value: Any) -> Optional[datetime]:
    """A datetime, or the ISO string the citation JSON carries, as the naive UTC the columns hold."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _build_turn(session_id: int, turn_number: int, data: Mapping[str, Any],
                citations: Sequence[Mapping[str, Any]]) -> ResearchConversation:
    conversation = ResearchConversation(
        id=data.get("id") or uuid.uuid4(),
        session_id=session_id,
        turn_number=turn_number,
        query_text=data.get("query_text", ""),
        query_type=(data.get("query_type") or "")[:50],
        is_follow_up=bool(data.get("is_follow_up", False)),
        resolved_query=data.get("resolved_query") or data.get("query_text", ""),
        answer_text=data.get("answer_text", ""),
        confidence=(data.get("confidence") or "")[:16],
        critic_verdict=(data.get("critic_verdict") or "")[:32],
        sources_consulted=list(data.get("sources_consulted") or []),
        subquestions=list(data.get("subquestions") or []),
        research_metadata=dict(data.get("research_metadata") or {}),
    )
    for position, item in enumerate(citations, start=1):
        conversation.citations.append(ResearchCitation(
            citation_index=int(item.get("index") or position),
            source_type=str(item.get("source_type") or "web")[:32],
            title=item.get("title") or "",
            url=item.get("url") or "",
            domain=(item.get("domain") or "")[:255],
            snippet=item.get("snippet") or "",
            retrieved_at=_naive_utc(item.get("retrieved_at")),
        ))
    return conversation


def save_turn(
    db: Session,
    *,
    session_id: int,
    turn_number: int,
    data: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]] = (),
) -> ResearchConversation:
    """Store one turn and its citations. ``turn_number`` was chosen when the request began; if
    another request took it meanwhile (UNIQUE violation) the next free number is used, once.
    Raises on any other failure; callers log it (a failed save never fails the response)."""
    for attempt in (1, 2):
        conversation = _build_turn(session_id, turn_number, data, citations)
        db.add(conversation)
        try:
            session = db.get(ChatSession, session_id)
            if session is not None:
                session.updated_at = _utc_now()
                if turn_number == 1 and (session.title or "") in DEFAULT_TITLES:
                    session.title = make_session_title(data.get("query_text", ""))
            db.commit()
            return conversation
        except IntegrityError:
            db.rollback()
            if attempt == 2:
                raise
            turn_number = last_turn_number(db, session_id) + 1
            logger.warning("Turn number taken in session %s; retrying as %s", session_id, turn_number)
    raise RuntimeError("unreachable")  # pragma: no cover


def soft_delete_session(db: Session, session: ChatSession) -> None:
    session.is_active = False
    db.commit()
