"""Alembic environment. Online only.

The application passes an open connection through ``config.attributes["connection"]``
(``research_app.db.migrate``), which is also how tests migrate an in-memory database. From
the ``alembic`` CLI there is no connection, so the application engine (DATABASE_URL) is used.
"""
from alembic import context

from research_app.db.database import Base
import research_app.db.models  # noqa: F401  (registers the tables on Base.metadata)

config = context.config
target_metadata = Base.metadata


def _run(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is None:
        from research_app.db.database import engine

        with engine.connect() as connection:
            _run(connection)
    else:
        _run(connection)


if context.is_offline_mode():
    raise RuntimeError("Offline migrations are not supported; run against a database.")
run_migrations_online()
