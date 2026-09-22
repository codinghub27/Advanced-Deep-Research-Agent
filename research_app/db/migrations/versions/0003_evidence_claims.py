"""minimal durable evidence + claim persistence (P1.5)

- research_evidence: every source collected for a turn (both cited and rejected), with the
  freshness (P1.2) and content-classification (P1.4) metadata attached. One row per source
  per turn; ``included``/``rejection_reason`` distinguish what made it into the answer from
  what was retrieved and dropped.
- research_claims: the claims a turn's answer makes and how well each is backed, mirroring the
  domain ``Claim`` model (P1.6) field-for-field.

Both reference ``research_conversations.id`` (Phase 6) as the run id -- no separate "runs"
table: a conversation turn already IS one research run's durable record.

Guarded and additive, same style as 0002_conversations: safe on a database that already has
some of it, and running it twice changes nothing.

Revision ID: 0003_evidence_claims
Revises: 0002_conversations
"""
from alembic import op
import sqlalchemy as sa

revision = "0003_evidence_claims"
down_revision = "0002_conversations"
branch_labels = None
depends_on = None


def _inspector():
    return sa.inspect(op.get_bind())


def _has_table(name: str) -> bool:
    return _inspector().has_table(name)


def _has_index(table: str, name: str) -> bool:
    return any(i["name"] == name for i in _inspector().get_indexes(table))


def upgrade() -> None:
    if not _has_table("research_evidence"):
        op.create_table(
            "research_evidence",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("conversation_id", sa.Uuid(),
                      sa.ForeignKey("research_conversations.id", ondelete="CASCADE"), nullable=False),
            sa.Column("task_id", sa.String(64), nullable=True),
            sa.Column("query", sa.Text(), nullable=True),
            sa.Column("source_id", sa.String(64), nullable=True),
            sa.Column("source_type", sa.String(32), nullable=False),
            sa.Column("provider", sa.String(64), nullable=True),
            sa.Column("url", sa.Text(), nullable=False),
            sa.Column("domain", sa.String(255), nullable=False, server_default=""),
            sa.Column("title", sa.Text(), nullable=False, server_default=""),
            sa.Column("excerpt", sa.Text(), nullable=False, server_default=""),
            sa.Column("published_at", sa.DateTime(), nullable=True),
            sa.Column("content_updated_at", sa.DateTime(), nullable=True),
            sa.Column("date_confidence", sa.String(16), nullable=True),
            sa.Column("date_source", sa.String(64), nullable=True),
            sa.Column("freshness_status", sa.String(16), nullable=True),
            sa.Column("content_classification", sa.String(48), nullable=True),
            sa.Column("classification_confidence", sa.Float(), nullable=True),
            sa.Column("authority_level", sa.String(16), nullable=True),
            sa.Column("is_primary_source", sa.Boolean(), nullable=True),
            sa.Column("independently_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("included", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("rejection_reason", sa.Text(), nullable=True),
            sa.Column("retrieved_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
    if not _has_index("research_evidence", "ix_research_evidence_conversation_id"):
        op.create_index("ix_research_evidence_conversation_id", "research_evidence", ["conversation_id"])

    if not _has_table("research_claims"):
        op.create_table(
            "research_claims",
            sa.Column("id", sa.Uuid(), primary_key=True),
            sa.Column("conversation_id", sa.Uuid(),
                      sa.ForeignKey("research_conversations.id", ondelete="CASCADE"), nullable=False),
            sa.Column("claim_text", sa.Text(), nullable=False),
            sa.Column("claim_type", sa.String(32), nullable=False, server_default="fact"),
            sa.Column("support_status", sa.String(32), nullable=False, server_default="unsupported"),
            sa.Column("evidence_ids", sa.JSON(), nullable=False),
            sa.Column("source_ids", sa.JSON(), nullable=False),
            sa.Column("excerpts", sa.JSON(), nullable=False),
            sa.Column("citation_id", sa.String(64), nullable=True),
            sa.Column("evidence_strength", sa.Float(), nullable=True),
            sa.Column("freshness_status", sa.String(16), nullable=True),
            sa.Column("conflict_status", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("verification_status", sa.String(16), nullable=False, server_default="unverified"),
            sa.Column("created_at", sa.DateTime(), nullable=True),
        )
    if not _has_index("research_claims", "ix_research_claims_conversation_id"):
        op.create_index("ix_research_claims_conversation_id", "research_claims", ["conversation_id"])


def downgrade() -> None:
    if _has_table("research_claims"):
        op.drop_table("research_claims")
    if _has_table("research_evidence"):
        op.drop_table("research_evidence")
