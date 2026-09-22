from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
Integer,String,DateTime,ForeignKey,JSON,Boolean,Float,Text,Uuid,UniqueConstraint,true,false
)
from datetime import datetime, timezone
from typing import Optional
import uuid

from research_app.db.database import Base

def _utc_now() -> datetime:
    # Timezone-aware UTC clock, stored as naive UTC to match the existing
    # 'timestamp without time zone' columns (same values utcnow() produced).
    return datetime.now(timezone.utc).replace(tzinfo=None)

class User(Base):
    __tablename__ = "users"
    id:Mapped[int]=mapped_column(
        Integer,
        primary_key=True,
        index=True
    )
    username:Mapped[str]=mapped_column(
        String(100),
        nullable=False,
    )
    hashed_password:Mapped[int]=mapped_column(
        String(255),
        nullable=False,
    )
    created_at:Mapped[datetime]=mapped_column(
        DateTime,
        default=_utc_now,
    )
    sessions=relationship(
        "Session",
        back_populates="user",
        cascade="all, delete",
    )

class Session(Base):
    __tablename__ =  "sessions"
    id:Mapped[int]=mapped_column(
        Integer,
        primary_key=True,
    )
    user_id:Mapped[int]=mapped_column(
        ForeignKey("users.id"),
        nullable=False,
    )
    title:Mapped[str]=mapped_column(
        String(255),
    )
    created_at:Mapped[datetime]=mapped_column(
        DateTime,
        default=_utc_now,
    )
    # Phase 6. Nullable: rows that predate the column are backfilled by the migration.
    updated_at:Mapped[datetime]=mapped_column(
        DateTime,
        default=_utc_now,
        onupdate=_utc_now,
        nullable=True,
    )
    # Soft delete (DELETE /sessions/{id}); inactive sessions are treated as not found.
    is_active:Mapped[bool]=mapped_column(
        Boolean,
        default=True,
        server_default=true(),
        nullable=False,
    )
    user=relationship(
        "User",
        back_populates="sessions"
    )
    queries=relationship(
        "Query",
        back_populates="session",
        cascade="all, delete"
    )
    conversations=relationship(
        "ResearchConversation",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ResearchConversation.turn_number",
    )

class Query(Base):
    __tablename__ = "queries"
    id:Mapped[int]=mapped_column(
        Integer,
        primary_key=True
    )
    session_id:Mapped[int]=mapped_column(
        ForeignKey("sessions.id"),
        nullable=False,
    )
    question:Mapped[str]=mapped_column(
        String,
        nullable=False,
    )
    answer:Mapped[str]=mapped_column(
        String,
        nullable=False
    )
    sources:Mapped[str]=mapped_column(
        JSON
    )
    created_at:Mapped[datetime]=mapped_column(
    DateTime,
        default=_utc_now
    )
    session=relationship(
        "Session",
        back_populates="queries",
    )


class ResearchConversation(Base):
    """One turn (question + cited answer) of a session. ``queries`` keeps the legacy
    question/answer pair the UI reloads from; this table holds the research detail."""
    __tablename__ = "research_conversations"
    __table_args__ = (
        UniqueConstraint("session_id", "turn_number", name="uq_research_conversations_session_turn"),
    )
    id:Mapped[uuid.UUID]=mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id:Mapped[int]=mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    turn_number:Mapped[int]=mapped_column(
        Integer,
        nullable=False,
    )
    query_text:Mapped[str]=mapped_column(
        Text,
        nullable=False,
    )
    query_type:Mapped[str]=mapped_column(
        String(50),
        nullable=False,
        default="",
    )
    is_follow_up:Mapped[bool]=mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    resolved_query:Mapped[str]=mapped_column(
        Text,
        nullable=False,
        default="",
    )
    answer_text:Mapped[str]=mapped_column(
        Text,
        nullable=False,
        default="",
    )
    confidence:Mapped[str]=mapped_column(
        String(16),
        nullable=False,
        default="",
    )
    critic_verdict:Mapped[str]=mapped_column(
        String(32),
        nullable=False,
        default="",
    )
    sources_consulted:Mapped[list]=mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    subquestions:Mapped[list]=mapped_column(
        JSON,
        nullable=False,
        default=list,
    )
    research_metadata:Mapped[dict]=mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    created_at:Mapped[datetime]=mapped_column(
        DateTime,
        default=_utc_now,
    )
    session=relationship(
        "Session",
        back_populates="conversations",
    )
    citations=relationship(
        "ResearchCitation",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ResearchCitation.citation_index",
    )
    evidence=relationship(
        "ResearchEvidence",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )
    claims=relationship(
        "ResearchClaim",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class ResearchCitation(Base):
    __tablename__ = "research_citations"
    id:Mapped[uuid.UUID]=mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    conversation_id:Mapped[uuid.UUID]=mapped_column(
        ForeignKey("research_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    citation_index:Mapped[int]=mapped_column(
        Integer,
        nullable=False,
    )
    source_type:Mapped[str]=mapped_column(
        String(32),
        nullable=False,
    )
    title:Mapped[str]=mapped_column(
        Text,
        nullable=False,
        default="",
    )
    url:Mapped[str]=mapped_column(
        Text,
        nullable=False,
    )
    domain:Mapped[str]=mapped_column(
        String(255),
        nullable=False,
        default="",
    )
    snippet:Mapped[str]=mapped_column(
        Text,
        nullable=False,
        default="",
    )
    retrieved_at:Mapped[datetime]=mapped_column(
        DateTime,
        nullable=True,
    )
    conversation=relationship(
        "ResearchConversation",
        back_populates="citations",
    )


class ResearchEvidence(Base):
    """Every source collected for one turn (P1.5), cited or not. ``included``/
    ``rejection_reason`` distinguish what the answer used from what was retrieved and dropped.
    Freshness (P1.2) and content-classification (P1.4) metadata travel with each row so a
    stored answer can be audited without re-fetching anything."""
    __tablename__ = "research_evidence"
    id:Mapped[uuid.UUID]=mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    conversation_id:Mapped[uuid.UUID]=mapped_column(
        ForeignKey("research_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_id:Mapped[Optional[str]]=mapped_column(String(64), nullable=True)
    query:Mapped[Optional[str]]=mapped_column(Text, nullable=True)
    source_id:Mapped[Optional[str]]=mapped_column(String(64), nullable=True)
    source_type:Mapped[str]=mapped_column(String(32), nullable=False)
    provider:Mapped[Optional[str]]=mapped_column(String(64), nullable=True)
    url:Mapped[str]=mapped_column(Text, nullable=False)
    domain:Mapped[str]=mapped_column(String(255), nullable=False, default="")
    title:Mapped[str]=mapped_column(Text, nullable=False, default="")
    excerpt:Mapped[str]=mapped_column(Text, nullable=False, default="")
    published_at:Mapped[Optional[datetime]]=mapped_column(DateTime, nullable=True)
    content_updated_at:Mapped[Optional[datetime]]=mapped_column(DateTime, nullable=True)
    date_confidence:Mapped[Optional[str]]=mapped_column(String(16), nullable=True)
    date_source:Mapped[Optional[str]]=mapped_column(String(64), nullable=True)
    freshness_status:Mapped[Optional[str]]=mapped_column(String(16), nullable=True)
    content_classification:Mapped[Optional[str]]=mapped_column(String(48), nullable=True)
    classification_confidence:Mapped[Optional[float]]=mapped_column(Float, nullable=True)
    authority_level:Mapped[Optional[str]]=mapped_column(String(16), nullable=True)
    is_primary_source:Mapped[Optional[bool]]=mapped_column(Boolean, nullable=True)
    independently_verified:Mapped[bool]=mapped_column(Boolean, nullable=False, default=False, server_default=false())
    included:Mapped[bool]=mapped_column(Boolean, nullable=False, default=True, server_default=true())
    rejection_reason:Mapped[Optional[str]]=mapped_column(Text, nullable=True)
    retrieved_at:Mapped[Optional[datetime]]=mapped_column(DateTime, nullable=True)
    created_at:Mapped[datetime]=mapped_column(DateTime, default=_utc_now)
    conversation=relationship(
        "ResearchConversation",
        back_populates="evidence",
    )


class ResearchClaim(Base):
    """One claim a turn's answer makes and how well it is backed (P1.5/P1.6), mirroring the
    domain ``Claim`` model field-for-field. ``evidence_ids``/``source_ids`` reference the
    domain-level ids (stable across runs), not ``ResearchEvidence.id`` primary keys."""
    __tablename__ = "research_claims"
    id:Mapped[uuid.UUID]=mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    conversation_id:Mapped[uuid.UUID]=mapped_column(
        ForeignKey("research_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    claim_text:Mapped[str]=mapped_column(Text, nullable=False)
    claim_type:Mapped[str]=mapped_column(String(32), nullable=False, default="fact")
    support_status:Mapped[str]=mapped_column(String(32), nullable=False, default="unsupported")
    evidence_ids:Mapped[list]=mapped_column(JSON, nullable=False, default=list)
    source_ids:Mapped[list]=mapped_column(JSON, nullable=False, default=list)
    excerpts:Mapped[list]=mapped_column(JSON, nullable=False, default=list)
    citation_id:Mapped[Optional[str]]=mapped_column(String(64), nullable=True)
    evidence_strength:Mapped[Optional[float]]=mapped_column(Float, nullable=True)
    freshness_status:Mapped[Optional[str]]=mapped_column(String(16), nullable=True)
    conflict_status:Mapped[bool]=mapped_column(Boolean, nullable=False, default=False, server_default=false())
    verification_status:Mapped[str]=mapped_column(String(16), nullable=False, default="unverified")
    created_at:Mapped[datetime]=mapped_column(DateTime, default=_utc_now)
    conversation=relationship(
        "ResearchConversation",
        back_populates="claims",
    )
