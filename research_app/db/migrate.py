"""Apply database migrations (Alembic) from application code.

``upgrade_to_head`` replaces the old ``Base.metadata.create_all`` at startup: it creates the
tables that are missing AND adds the columns an existing install lacks, which ``create_all``
never did. The migration scripts contain all schema changes; nothing here issues DDL.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations")


def _config() -> Config:
    config = Config()
    config.set_main_option("script_location", MIGRATIONS_DIR)
    return config


def upgrade_to_head(engine: Optional[Engine] = None, revision: str = "head") -> None:
    """Migrate ``engine`` (default: the application's) to ``revision``. Idempotent."""
    if engine is None:
        from research_app.db.database import engine as app_engine

        engine = app_engine
    config = _config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, revision)
    logger.info("Database migrated to %s", revision)


def downgrade(engine: Engine, revision: str) -> None:
    config = _config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, revision)
