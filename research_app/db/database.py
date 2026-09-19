from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker,DeclarativeBase
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in .env")

class Base(DeclarativeBase):
    pass

engine=create_engine(
    url=DATABASE_URL,
    #echo=True ## you'll see SQL statements in your terminal.
)

def enable_sqlite_foreign_keys(target_engine):
    """SQLite ignores foreign keys unless every connection asks for them; PostgreSQL always
    enforces them. Turning them on makes development behave like production. A no-op for
    other databases."""
    if target_engine.dialect.name != "sqlite":
        return target_engine

    @event.listens_for(target_engine, "connect")
    def _foreign_keys_on(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return target_engine

enable_sqlite_foreign_keys(engine)

SessionLocal=sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

def get_session_factory():
    """Dependency: how the post-response conversation save opens its OWN session (the request
    session is closed by then). Tests override it."""
    return SessionLocal

def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()


###Engine = Database communication infrastructure

###Session = Your database workspace