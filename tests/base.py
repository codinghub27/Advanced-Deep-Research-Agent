import unittest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from research_app.auth.auth import create_access_token
from research_app.db.crud import create_user, create_session, save_query
from research_app.db.database import Base, get_db, get_session_factory
from research_app.main import app


class ApiTestCase(unittest.TestCase):
    """Fresh in-memory SQLite database per test, wired in via get_db."""

    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False)

        def override_get_db():
            db = self.SessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_session_factory] = lambda: self.SessionLocal
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    # -- helpers -------------------------------------------------------------
    def make_user(self, username):
        with self.SessionLocal() as db:
            return create_user(username, "not-a-real-hash", db).id

    def make_session(self, user_id, title="s"):
        with self.SessionLocal() as db:
            return create_session(user_id, title, db).id

    def make_query(self, session_id, question="q", answer="a"):
        with self.SessionLocal() as db:
            save_query(session_id, question, answer, db)

    def count(self, model):
        with self.SessionLocal() as db:
            return db.query(model).count()

    @staticmethod
    def auth(username):
        return {"Authorization": f"Bearer {create_access_token({'sub': username})}"}
