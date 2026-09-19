"""baseline: users, sessions, queries

The three tables the application created with ``create_all`` before migrations existed.
Every step is guarded, so a database that already has them (all existing installs) is
adopted as it is, and running this twice changes nothing.

Revision ID: 0001_baseline
Revises:
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    if not _has_table("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("username", sa.String(100), nullable=False),
            sa.Column("hashed_password", sa.String(255), nullable=False),
            sa.Column("created_at", sa.DateTime()),
        )
        op.create_index("ix_users_id", "users", ["id"])
    if not _has_table("sessions"):
        op.create_table(
            "sessions",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("title", sa.String(255)),
            sa.Column("created_at", sa.DateTime()),
        )
    if not _has_table("queries"):
        op.create_table(
            "queries",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("session_id", sa.Integer(), sa.ForeignKey("sessions.id"), nullable=False),
            sa.Column("question", sa.String(), nullable=False),
            sa.Column("answer", sa.String(), nullable=False),
            sa.Column("sources", sa.JSON()),
            sa.Column("created_at", sa.DateTime()),
        )


def downgrade() -> None:
    # The baseline tables hold user data that predates migrations; never drop them here.
    pass
