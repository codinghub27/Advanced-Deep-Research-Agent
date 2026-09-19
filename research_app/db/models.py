from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import (
Integer,String,DateTime,ForeignKey,JSON
)
from datetime import datetime

from research_app.db.database import Base

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
        default=datetime.utcnow(),
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
        default=datetime.utcnow(),
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
        default=datetime.utcnow()
    )
    session=relationship(
        "Session",
        back_populates="queries",
    )

