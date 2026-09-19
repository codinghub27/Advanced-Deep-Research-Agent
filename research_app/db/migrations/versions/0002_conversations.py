"""conversation persistence (Phase 6)

- sessions: ``updated_at`` (backfilled from ``created_at``), ``is_active`` (soft delete)
- research_conversations: one row per turn, UNIQUE (session_id, turn_number)
- research_citations: the numbered sources of a turn

Guarded and additive: safe on a database that already has some of it, and running it twice
changes nothing.

Revision ID: 0002_conversations
Revises: 0001_baseline
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_conversations"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def _inspector():
    return sa.inspect(op.get_bind())


def _has_table(name: str) -> bool:
    return _inspector().has_table(name)


def _has_column(table: str, column: str) -> bool:
    return any(c["name"] == column for c in _inspector().get_columns(table))


def _has_index(table: str, name: str) -> bool:
    return any(i["name"] == name for i in _inspector().get_indexes(table))


def upgrade() -> None:
    if not _has_column("sessions", "updated_at"):
        op.add_column("sessions", sa.Column("updated_at", sa.DateTime(), nullable=True))
        op.execute("UPDATE sessions SET updated_at = created_at WHERE updated_at IS NULL")
    if not _has_column("sessions", "is_active"):
        op.add_column(
            "sessions",
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        )

    if not _has_table("research_conversations"):
        op.create_table(
            "research_conversations",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("session_id", sa.Integer(), sa.ForeignKey("sessions.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("turn_number", sa.Integer(), nullable=False),
            sa.Column("query_text", sa.Text(), nullable=False),
            sa.Column("query_type", sa.String(50), nullable=False),
            sa.Column("is_follow_up", sa.Boolean(), nullable=False),
            sa.Column("resolved_query", sa.Text(), nullable=False),
            sa.Column("answer_text", sa.Text(), nullable=False),
            sa.Column("confidence", sa.String(16), nullable=False),
            sa.Column("critic_verdict", sa.String(32), nullable=False),
            sa.Column("sources_consulted", sa.JSON(), nullable=False),
            sa.Column("subquestions", sa.JSON(), nullable=False),
            sa.Column("research_metadata", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime()),
            sa.UniqueConstraint("session_id", "turn_number", name="uq_research_conversations_session_turn"),
        )
    if not _has_index("research_conversations", "ix_research_conversations_session_id"):
        op.create_index("ix_research_conversations_session_id", "research_conversations", ["session_id"])

    if not _has_table("research_citations"):
        op.create_table(
            "research_citations",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("conversation_id", sa.Uuid(),
                      sa.ForeignKey("research_conversations.id", ondelete="CASCADE"), nullable=False),
            sa.Column("citation_index", sa.Integer(), nullable=False),
            sa.Column("source_type", sa.String(32), nullable=False),
            sa.Column("title", sa.Text(), nullable=False),
            sa.Column("url", sa.Text(), nullable=False),
            sa.Column("domain", sa.String(255), nullable=False),
            sa.Column("snippet", sa.Text(), nullable=False),
            sa.Column("retrieved_at", sa.DateTime(), nullable=True),
        )
    if not _has_index("research_citations", "ix_research_citations_conversation_id"):
        op.create_index("ix_research_citations_conversation_id", "research_citations", ["conversation_id"])


def downgrade() -> None:
    if _has_table("research_citations"):
        op.drop_table("research_citations")
    if _has_table("research_conversations"):
        op.drop_table("research_conversations")
    with op.batch_alter_table("sessions") as batch:
        if _has_column("sessions", "is_active"):
            batch.drop_column("is_active")
        if _has_column("sessions", "updated_at"):
            batch.drop_column("updated_at")
