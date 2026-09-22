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

from research_app.db.models import (
    ResearchCitation,
    ResearchClaim,
    ResearchConversation,
    ResearchEvidence,
    Session as ChatSession,
    _utc_now,
)

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


def load_evidence(db: Session, conversation_id: "uuid.UUID | str") -> list[ResearchEvidence]:
    """All evidence collected for a turn (P1.5), cited and rejected alike, retrieval order."""
    return (
        db.query(ResearchEvidence)
        .filter(ResearchEvidence.conversation_id == conversation_id)
        .order_by(ResearchEvidence.created_at.asc())
        .all()
    )


def load_claims(db: Session, conversation_id: "uuid.UUID | str") -> list[ResearchClaim]:
    """Every claim recorded for a turn (P1.5/P1.6), in the order they were saved."""
    return (
        db.query(ResearchClaim)
        .filter(ResearchClaim.conversation_id == conversation_id)
        .order_by(ResearchClaim.created_at.asc())
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


def _build_evidence(item: Mapping[str, Any]) -> ResearchEvidence:
    """One ``ResearchEvidence`` row from an evidence-item mapping. Field names match the
    domain ``SourceDocument`` (P1.2/P1.4) plus ``included``/``rejection_reason`` (P1.5)."""
    return ResearchEvidence(
        task_id=item.get("task_id"),
        query=item.get("query"),
        source_id=item.get("source_id"),
        source_type=str(item.get("source_type") or "web")[:32],
        provider=item.get("provider"),
        url=item.get("url") or "",
        domain=(item.get("domain") or "")[:255],
        title=item.get("title") or "",
        excerpt=item.get("excerpt") or item.get("snippet") or "",
        published_at=_naive_utc(item.get("published_at")),
        content_updated_at=_naive_utc(item.get("updated_at")),
        date_confidence=item.get("date_confidence"),
        date_source=item.get("date_source"),
        freshness_status=item.get("freshness_status"),
        content_classification=item.get("content_classification"),
        classification_confidence=item.get("classification_confidence"),
        authority_level=item.get("authority_level"),
        is_primary_source=item.get("is_primary_source"),
        independently_verified=bool(item.get("independently_verified", False)),
        included=bool(item.get("included", True)),
        rejection_reason=item.get("rejection_reason"),
        retrieved_at=_naive_utc(item.get("retrieved_at")),
    )


def _build_claim(item: Mapping[str, Any]) -> ResearchClaim:
    """One ``ResearchClaim`` row from a claim mapping, mirroring the domain ``Claim`` (P1.6)."""
    return ResearchClaim(
        claim_text=item.get("text") or item.get("claim_text") or "",
        claim_type=str(item.get("claim_type") or "fact")[:32],
        support_status=str(item.get("support_status") or "unsupported")[:32],
        evidence_ids=list(item.get("evidence_ids") or []),
        source_ids=list(item.get("source_ids") or []),
        excerpts=list(item.get("excerpts") or []),
        citation_id=item.get("citation_id"),
        evidence_strength=item.get("evidence_strength"),
        freshness_status=item.get("freshness_status"),
        conflict_status=bool(item.get("conflict_status", False)),
        verification_status=str(item.get("verification_status") or "unverified")[:16],
    )


def _build_turn(
    session_id: int,
    turn_number: int,
    data: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]] = (),
    claims: Sequence[Mapping[str, Any]] = (),
) -> ResearchConversation:
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
    for item in evidence:
        conversation.evidence.append(_build_evidence(item))
    for item in claims:
        conversation.claims.append(_build_claim(item))
    return conversation


def save_turn(
    db: Session,
    *,
    session_id: int,
    turn_number: int,
    data: Mapping[str, Any],
    citations: Sequence[Mapping[str, Any]] = (),
    evidence: Sequence[Mapping[str, Any]] = (),
    claims: Sequence[Mapping[str, Any]] = (),
) -> ResearchConversation:
    """Store one turn, its citations, and (P1.5) the evidence collected and claims made while
    answering it. ``turn_number`` was chosen when the request began; if another request took it
    meanwhile (UNIQUE violation) the next free number is used, once. Raises on any other
    failure; callers log it (a failed save never fails the response)."""
    for attempt in (1, 2):
        conversation = _build_turn(session_id, turn_number, data, citations, evidence, claims)
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
