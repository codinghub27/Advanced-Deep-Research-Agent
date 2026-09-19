from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
Integer,String,DateTime,ForeignKey,JSON,Boolean,Text,Uuid,UniqueConstraint,true
)
from datetime import datetime, timezone
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
