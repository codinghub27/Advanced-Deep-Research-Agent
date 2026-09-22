"""P1.5 -- minimal durable evidence/claim persistence. SQLite in memory; no network/LLM."""
from __future__ import annotations

import unittest
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from research_app.db import conversations as repo
from research_app.db.database import Base, enable_sqlite_foreign_keys
from research_app.db.migrate import upgrade_to_head
from research_app.db.models import ResearchClaim, ResearchEvidence, Session as ChatSession, User


def memory_engine():
    return enable_sqlite_foreign_keys(
        create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}))


class MigrationTests(unittest.TestCase):
    def test_0003_runs_cleanly_on_an_empty_database(self):
        engine = memory_engine()
        upgrade_to_head(engine)  # 0001 -> 0002 -> 0003 in one go
        from sqlalchemy import inspect
        tables = set(inspect(engine).get_table_names())
        self.assertIn("research_evidence", tables)
        self.assertIn("research_claims", tables)

    def test_running_the_migration_twice_is_a_no_op(self):
        engine = memory_engine()
        upgrade_to_head(engine)
        upgrade_to_head(engine)  # must not raise (idempotent, like 0001/0002)

    def test_downgrade_removes_the_new_tables(self):
        from sqlalchemy import inspect
        engine = memory_engine()
        upgrade_to_head(engine)
        from research_app.db.migrate import downgrade
        downgrade(engine, "0002_conversations")
        tables = set(inspect(engine).get_table_names())
        self.assertNotIn("research_evidence", tables)
        self.assertNotIn("research_claims", tables)


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = memory_engine()
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False)
        with self.Session() as db:
            db.add(User(id=1, username="alice", hashed_password="x"))
            db.add(ChatSession(id=1, user_id=1, title="s"))
            db.commit()

    def _evidence_item(self, **overrides):
        item = {
            "task_id": "t1",
            "query": "how do I install fastapi",
            "source_id": "abc123",
            "source_type": "official_docs",
            "provider": "tavily",
            "url": "https://fastapi.tiangolo.com/tutorial/",
            "domain": "fastapi.tiangolo.com",
            "title": "Tutorial",
            "excerpt": "pip install fastapi",
            "published_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-06-01T00:00:00+00:00",
            "date_confidence": "exact",
            "date_source": "published_at",
            "freshness_status": "fresh",
            "content_classification": "official_documentation",
            "classification_confidence": 0.95,
            "authority_level": "official",
            "is_primary_source": True,
            "independently_verified": True,
            "included": True,
            "retrieved_at": "2026-09-22T00:00:00+00:00",
        }
        item.update(overrides)
        return item

    def _claim_item(self, **overrides):
        item = {
            "text": "FastAPI is installed with pip install fastapi.",
            "claim_type": "fact",
            "support_status": "directly_supported",
            "evidence_ids": ["ev-1"],
            "source_ids": ["abc123"],
            "excerpts": ["pip install fastapi"],
            "citation_id": "cit-1",
            "evidence_strength": 0.9,
            "freshness_status": "fresh",
            "conflict_status": False,
            "verification_status": "verified",
        }
        item.update(overrides)
        return item

    def test_save_turn_persists_evidence_and_claims(self):
        with self.Session() as db:
            repo.save_turn(
                db, session_id=1, turn_number=1, data={"query_text": "q"},
                evidence=[self._evidence_item()], claims=[self._claim_item()],
            )
        with self.Session() as db:
            self.assertEqual(db.query(ResearchEvidence).count(), 1)
            self.assertEqual(db.query(ResearchClaim).count(), 1)

    def test_evidence_row_carries_freshness_and_classification_fields(self):
        with self.Session() as db:
            conv = repo.save_turn(
                db, session_id=1, turn_number=1, data={"query_text": "q"},
                evidence=[self._evidence_item()],
            )
            conv_id = conv.id
        with self.Session() as db:
            [row] = repo.load_evidence(db, conv_id)
            self.assertEqual(row.source_type, "official_docs")
            self.assertEqual(row.date_confidence, "exact")
            self.assertEqual(row.freshness_status, "fresh")
            self.assertEqual(row.content_classification, "official_documentation")
            self.assertAlmostEqual(float(row.classification_confidence or 0), 0.95)
            self.assertEqual(row.authority_level, "official")
            self.assertTrue(row.is_primary_source)
            self.assertTrue(row.independently_verified)
            self.assertTrue(row.included)
            self.assertEqual(row.published_at, datetime(2026, 1, 1))
            self.assertEqual(row.content_updated_at, datetime(2026, 6, 1))

    def test_rejected_evidence_is_distinguished_from_included(self):
        with self.Session() as db:
            conv = repo.save_turn(
                db, session_id=1, turn_number=1, data={"query_text": "q"},
                evidence=[
                    self._evidence_item(url="https://a.example/1"),
                    self._evidence_item(
                        url="https://b.example/2", included=False,
                        rejection_reason="stale evidence outside the freshness window"),
                ],
            )
            conv_id = conv.id
        with self.Session() as db:
            rows = repo.load_evidence(db, conv_id)
        included = {r.url: r.included for r in rows}
        self.assertTrue(included["https://a.example/1"])
        self.assertFalse(included["https://b.example/2"])
        rejected = [r for r in rows if not r.included][0]
        self.assertEqual(rejected.rejection_reason, "stale evidence outside the freshness window")

    def test_claim_row_carries_claim_fields(self):
        with self.Session() as db:
            conv = repo.save_turn(
                db, session_id=1, turn_number=1, data={"query_text": "q"},
                claims=[self._claim_item()],
            )
            conv_id = conv.id
        with self.Session() as db:
            [claim] = repo.load_claims(db, conv_id)
            self.assertEqual(claim.claim_type, "fact")
            self.assertEqual(claim.support_status, "directly_supported")
            self.assertEqual(claim.evidence_ids, ["ev-1"])
            self.assertEqual(claim.source_ids, ["abc123"])
            self.assertEqual(claim.excerpts, ["pip install fastapi"])
            self.assertEqual(claim.citation_id, "cit-1")
            self.assertAlmostEqual(float(claim.evidence_strength or 0), 0.9)
            self.assertEqual(claim.verification_status, "verified")

    def test_save_turn_without_evidence_or_claims_is_unchanged(self):
        # Backward compatibility: every existing caller of save_turn (Phase 6) omits these
        # kwargs entirely and must keep working exactly as before.
        with self.Session() as db:
            conv = repo.save_turn(db, session_id=1, turn_number=1, data={"query_text": "q"},
                                   citations=[{"url": "https://example.com/a"}])
            conv_id = conv.id
        with self.Session() as db:
            self.assertEqual(repo.load_evidence(db, conv_id), [])
            self.assertEqual(repo.load_claims(db, conv_id), [])

    def test_deleting_a_session_cascades_to_evidence_and_claims(self):
        with self.Session() as db:
            repo.save_turn(
                db, session_id=1, turn_number=1, data={"query_text": "q"},
                evidence=[self._evidence_item()], claims=[self._claim_item()],
            )
        with self.Session() as db:
            session = db.get(ChatSession, 1)
            db.delete(session)
            db.commit()
        with self.Session() as db:
            self.assertEqual(db.query(ResearchEvidence).count(), 0)
            self.assertEqual(db.query(ResearchClaim).count(), 0)

    def test_unparseable_or_missing_dates_are_stored_as_null_not_invented(self):
        with self.Session() as db:
            conv = repo.save_turn(
                db, session_id=1, turn_number=1, data={"query_text": "q"},
                evidence=[self._evidence_item(published_at="not a date", updated_at=None)],
            )
            conv_id = conv.id
        with self.Session() as db:
            [row] = repo.load_evidence(db, conv_id)
            self.assertIsNone(row.published_at)
            self.assertIsNone(row.content_updated_at)


if __name__ == "__main__":
    unittest.main()
