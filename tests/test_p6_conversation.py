"""Phase 6, Part C: migrations, conversation persistence, history loading, the session API,
follow-ups through the real graph, and the cache interplay. SQLite in memory; no network."""
import json
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from research_app.agent import state as st
from research_app.agent import vectordb
from research_app.conversation.context import build_context, summarize_answer
from research_app.conversation.settings import ConversationSettings
from research_app.db import conversations as repo
from research_app.db.database import Base, enable_sqlite_foreign_keys
from research_app.db.migrate import downgrade, upgrade_to_head
from research_app.db.models import (
    Query, ResearchCitation, ResearchConversation, Session as ChatSession, User,
)
from research_app.domain import ConversationContext, RoutingDecision, SourceType
from tests.base import ApiTestCase
from tests.p6_helpers import ScriptedLLM, patches, understanding
from tests.test_p6_pipeline import relevant_fake


def memory_engine():
    return enable_sqlite_foreign_keys(
        create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}))


# --------------------------------------------------------------------------- migrations

class MigrationTests(unittest.TestCase):
    OLD_SCHEMA = (
        "create table users (id integer primary key, username varchar(100) not null, hashed_password varchar(255) not null, created_at datetime)",
        "create table sessions (id integer primary key, user_id integer not null references users(id), title varchar(255), created_at datetime)",
        "create table queries (id integer primary key, session_id integer not null references sessions(id), question varchar not null, answer varchar not null, sources json, created_at datetime)",
    )

    def test_runs_cleanly_on_an_empty_database(self):
        engine = memory_engine()
        upgrade_to_head(engine)
        names = set(inspect(engine).get_table_names())
        self.assertTrue({"users", "sessions", "queries", "research_conversations", "research_citations"} <= names)
        self.assertTrue({"updated_at", "is_active"} <= {c["name"] for c in inspect(engine).get_columns("sessions")})

    def test_runs_on_a_database_that_already_has_the_old_tables_and_keeps_its_rows(self):
        engine = memory_engine()
        with engine.begin() as c:
            for ddl in self.OLD_SCHEMA:
                c.execute(text(ddl))
            c.execute(text("insert into users values (1,'u','h','2026-01-01')"))
            c.execute(text("insert into sessions values (1,1,'old chat','2026-01-02 03:04:05')"))
            c.execute(text("insert into queries values (1,1,'q','a','{}','2026-01-02 03:04:06')"))
        upgrade_to_head(engine)
        with engine.connect() as c:
            row = c.execute(text("select title, updated_at, is_active from sessions")).one()
            self.assertEqual((row[0], str(row[1]), row[2]), ("old chat", "2026-01-02 03:04:05", 1))  # backfilled, active
            self.assertEqual(c.execute(text("select count(*) from queries")).scalar(), 1)

    def test_is_idempotent(self):
        engine = memory_engine()
        upgrade_to_head(engine)
        upgrade_to_head(engine)
        with engine.connect() as c:
            self.assertEqual(c.execute(text("select version_num from alembic_version")).scalar(), "0002_conversations")

    def test_downgrade_removes_only_the_phase6_objects(self):
        engine = memory_engine()
        upgrade_to_head(engine)
        downgrade(engine, "0001_baseline")
        insp = inspect(engine)
        self.assertNotIn("research_conversations", insp.get_table_names())
        self.assertIn("sessions", insp.get_table_names())
        self.assertNotIn("is_active", [c["name"] for c in insp.get_columns("sessions")])

    def test_foreign_keys_are_enforced(self):
        engine = memory_engine()
        upgrade_to_head(engine)
        with engine.begin() as c:
            with self.assertRaises(IntegrityError):
                c.execute(text("insert into research_conversations (id, session_id, turn_number, query_text, query_type,"
                               " is_follow_up, resolved_query, answer_text, confidence, critic_verdict, sources_consulted,"
                               " subquestions, research_metadata) values ('00000000000000000000000000000001', 999, 1, 'q', '',"
                               " 0, 'q', 'a', '', '', '[]', '[]', '{}')"))

    def test_a_turn_number_is_unique_within_a_session(self):
        engine = memory_engine()
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        with Session() as db:
            db.add(User(username="u", hashed_password="h"))
            db.commit()
            db.add(ChatSession(user_id=1, title="s"))
            db.commit()
            repo.save_turn(db, session_id=1, turn_number=1, data={"query_text": "q"})
            db.add(ResearchConversation(session_id=1, turn_number=1, query_text="dup"))
            with self.assertRaises(IntegrityError):
                db.commit()


# --------------------------------------------------------------------------- repository + context

class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = memory_engine()
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, autoflush=False)
        with self.Session() as db:
            db.add(User(username="u", hashed_password="h"))
            db.commit()
            db.add(ChatSession(user_id=1, title="New Research Session"))
            db.commit()

    def turn(self, n, **kw):
        data = {"query_text": f"question {n}", "resolved_query": f"resolved {n}", "answer_text": f"answer {n} " * 60,
                "sources_consulted": ["web"], **kw}
        with self.Session() as db:
            return repo.save_turn(db, session_id=1, turn_number=n, data=data, citations=kw.get("citations", ()))

    def test_title_is_the_first_80_characters_of_the_question(self):
        self.assertEqual(repo.make_session_title("short question"), "short question")
        long = repo.make_session_title("word " * 40, 80)
        self.assertLessEqual(len(long), 80)
        self.assertTrue(long.endswith("…"))
        self.assertEqual(repo.make_session_title("  a \n  b  "), "a b")
        self.assertEqual(repo.make_session_title("   "), "New Research Session")

    def test_a_turn_and_its_citations_are_saved(self):
        self.turn(1, citations=[{"index": 1, "source_type": "official_docs", "title": "T", "url": "https://d.example/x",
                                 "domain": "d.example", "snippet": "s", "retrieved_at": datetime(2026, 1, 1)}])
        with self.Session() as db:
            conv = db.query(ResearchConversation).one()
            self.assertEqual((conv.turn_number, conv.query_text, conv.resolved_query), (1, "question 1", "resolved 1"))
            (cite,) = conv.citations
            self.assertEqual((cite.citation_index, cite.source_type, cite.domain), (1, "official_docs", "d.example"))
            self.assertEqual(db.get(ChatSession, 1).title, "question 1")  # the default title is replaced on turn 1

    def test_turn_numbers_increment_and_a_taken_number_is_retried_once(self):
        self.turn(1)
        self.turn(2)
        self.turn(2)  # another request took 2 first: this one lands on 3
        with self.Session() as db:
            self.assertEqual(sorted(t.turn_number for t in db.query(ResearchConversation)), [1, 2, 3])
            self.assertEqual(repo.last_turn_number(db, 1), 3)

    def test_recent_turns_are_the_last_n_in_order(self):
        for n in range(1, 16):
            self.turn(n)
        with self.Session() as db:
            rows = repo.load_recent_turns(db, 1, 10)
            self.assertEqual([r.turn_number for r in rows], list(range(6, 16)))
            self.assertEqual(repo.count_turns(db, 1), 15)

    def test_deleting_a_session_removes_its_turns_and_citations(self):
        self.turn(1, citations=[{"index": 1, "source_type": "web", "url": "https://x.example"}])
        with self.Session() as db:
            db.delete(db.get(ChatSession, 1))
            db.commit()
            self.assertEqual((db.query(ResearchConversation).count(), db.query(ResearchCitation).count()), (0, 0))

    def test_soft_delete_hides_the_session_from_listings(self):
        with self.Session() as db:
            repo.soft_delete_session(db, db.get(ChatSession, 1))
            self.assertEqual(repo.list_user_sessions(db, 1, 20, 0), ([], 0))


def row(n, query, resolved=None, answer="", used=(), meta=None):
    return SimpleNamespace(turn_number=n, query_text=query, resolved_query=resolved or query, answer_text=answer,
                           sources_consulted=list(used), research_metadata=meta or {}, created_at=None)


class ContextTests(unittest.TestCase):
    def test_an_empty_session_gives_an_empty_context(self):
        ctx = build_context(7, [])
        self.assertEqual((ctx.session_id, ctx.turns, ctx.topics_discussed, ctx.technologies_mentioned, ctx.total_turns, ctx.is_empty),
                         (7, [], [], [], 0, True))

    def test_three_turns_load_in_order(self):
        ctx = build_context(1, [row(1, "How do I install FastAPI?", used=["official_docs"]), row(2, "add JWT to it", "add JWT to FastAPI"),
                                row(3, "what about OAuth2?")])
        self.assertEqual([t.turn_number for t in ctx.turns], [1, 2, 3])
        self.assertEqual(ctx.turns[1].resolved_query, "add JWT to FastAPI")
        self.assertEqual(ctx.turns[0].source_types_used, ["official_docs"])
        self.assertEqual(ctx.total_turns, 3)

    def test_the_answer_summary_is_truncated_never_the_full_answer(self):
        ctx = build_context(1, [row(1, "q", answer="word " * 400)], settings=ConversationSettings(answer_summary_length=300))
        self.assertLessEqual(len(ctx.turns[0].answer_summary), 300)
        self.assertEqual(summarize_answer("a  b\n c", 100), "a b c")

    def test_topics_and_technologies_are_extracted_without_an_llm(self):
        ctx = build_context(1, [row(1, "How do I install FastAPI?", answer="Use pip. Uvicorn runs it."),
                                row(2, "Set up Docker Compose", "Set up Docker Compose")])
        self.assertEqual(ctx.topics_discussed, ["How do I install FastAPI?", "Set up Docker Compose"])
        self.assertIn("FastAPI", ctx.technologies_mentioned)
        self.assertTrue(any("Docker" in t for t in ctx.technologies_mentioned))

    def test_total_turns_can_exceed_the_loaded_turns(self):
        ctx = build_context(1, [row(n, f"q{n}") for n in range(6, 16)], total_turns=15)
        self.assertEqual((len(ctx.turns), ctx.total_turns), (10, 15))

    def test_conversation_settings_parse(self):
        s = ConversationSettings.from_env({"CONVERSATION_HISTORY_LIMIT": "4", "ANSWER_SUMMARY_LENGTH": "100", "SESSION_TITLE_MAX_LENGTH": "40"})
        self.assertEqual((s.history_limit, s.answer_summary_length, s.title_max_length), (4, 100, 40))
        d = ConversationSettings.from_env({"CONVERSATION_HISTORY_LIMIT": "0", "ANSWER_SUMMARY_LENGTH": "x"})
        self.assertEqual((d.history_limit, d.answer_summary_length, d.title_max_length), (10, 300, 80))


# --------------------------------------------------------------------------- API

CITATION = {"index": 1, "marker": "[1]", "source_type": "official_docs", "title": "FastAPI docs",
            "url": "https://fastapi.tiangolo.com/tutorial/", "domain": "fastapi.tiangolo.com", "snippet": "Install it",
            "retrieved_at": "2026-01-01T00:00:00+00:00"}
DECISION = RoutingDecision(task_id="t1", sub_question="Install FastAPI", sources=[SourceType.OFFICIAL_DOCS, SourceType.WEB],
                           origins={"official_docs": "planner", "web": "policy"})


def fake_graph(answer="The answer [1].", **extra):
    state = {"sub_questions": ["Install FastAPI"], "final_answer": answer, "api_limit_reached": False,
             "citations": [CITATION], "routing_decisions": [DECISION], "confidence": "medium",
             "resolved_query": "resolved", "understanding": {"intent": "technical_howto"}, **extra}
    graph = mock.MagicMock()
    graph.ainvoke = mock.AsyncMock(return_value=state)

    async def astream(inputs, stream_mode=None):
        yield "updates", {"search_node": {"sources": [{"url": "https://x.example", "title": "X", "snippet": ""}],
                                          "routing_decisions": [DECISION]}}
        yield "updates", {"format_response": {"final_answer": answer, "citations": [CITATION]}}
    graph.astream = mock.MagicMock(side_effect=astream)
    return graph


class ConversationApiTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.alice = self.make_user("alice")
        self.bob = self.make_user("bob")
        self.headers = self.auth("alice")

    def post(self, body, graph=None, headers=None, path="/api/research"):
        with mock.patch("research_app.main.compiled_graph", graph or fake_graph()):
            return self.client.post(path, json=body, headers=headers or self.headers)

    def turns(self, session_id):
        with self.SessionLocal() as db:
            rows = repo.load_all_turns(db, session_id)
            for r in rows:
                list(r.citations)  # load them before the session closes
            db.expunge_all()
            return rows

    # ---- sessions
    def test_no_session_id_creates_a_session_and_returns_its_id(self):
        r = self.post({"question": "How do I install FastAPI?"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        with self.SessionLocal() as db:
            session = db.get(ChatSession, body["session_id"])
            self.assertEqual((session.user_id, session.title, session.is_active), (self.alice, "How do I install FastAPI?", True))
        self.assertEqual((body["turn_number"], body["is_follow_up"]), (1, False))

    def test_the_auto_title_is_at_most_80_characters(self):
        r = self.post({"question": "why " * 60})
        with self.SessionLocal() as db:
            self.assertLessEqual(len(db.get(ChatSession, r.json()["session_id"]).title), 80)

    def test_a_valid_session_id_loads_its_history_into_the_graph(self):
        sid = self.post({"question": "How do I install FastAPI?"}).json()["session_id"]
        graph = fake_graph()
        self.post({"question": "now add JWT to it", "session_id": sid}, graph)
        inputs = graph.ainvoke.await_args.args[0]
        ctx = inputs["conversation_context"]
        self.assertIsInstance(ctx, ConversationContext)
        self.assertEqual((inputs["session_id"], inputs["turn_number"], len(ctx.turns)), (sid, 2, 1))
        self.assertEqual(ctx.turns[0].query_text, "How do I install FastAPI?")
        self.assertLessEqual(len(ctx.turns[0].answer_summary), 300)

    def test_a_nonexistent_session_is_404_and_nothing_runs(self):
        graph = fake_graph()
        r = self.post({"question": "hi", "session_id": 9999}, graph)
        self.assertEqual(r.status_code, 404)
        graph.ainvoke.assert_not_awaited()

    def test_another_users_session_is_404(self):
        sid = self.make_session(self.bob)
        self.assertEqual(self.post({"question": "hi", "session_id": sid}).status_code, 404)

    def test_a_soft_deleted_session_is_treated_as_not_found(self):
        sid = self.post({"question": "first"}).json()["session_id"]
        self.assertEqual(self.client.delete(f"/sessions/{sid}", headers=self.headers).status_code, 200)
        self.assertEqual(self.post({"question": "again", "session_id": sid}).status_code, 404)

    def test_an_empty_session_has_an_empty_context(self):
        sid = self.make_session(self.alice)
        graph = fake_graph()
        self.post({"question": "hello", "session_id": sid}, graph)
        self.assertTrue(graph.ainvoke.await_args.args[0]["conversation_context"].is_empty)

    # ---- saving
    def test_the_conversation_and_its_citations_are_saved_after_the_response(self):
        r = self.post({"question": "How do I install FastAPI?"})
        (turn,) = self.turns(r.json()["session_id"])
        self.assertEqual((turn.turn_number, turn.query_text, turn.answer_text, turn.confidence, turn.query_type),
                         (1, "How do I install FastAPI?", "The answer [1].", "medium", "technical_howto"))
        self.assertEqual([(c.citation_index, c.source_type, c.domain) for c in turn.citations],
                         [(1, "official_docs", "fastapi.tiangolo.com")])
        self.assertEqual(turn.research_metadata["routing_decisions"][0]["sources"], ["official_docs", "web"])
        self.assertEqual(self.count(Query), 1)  # the legacy row is still written

    def test_turn_numbers_increment_within_a_session(self):
        sid = self.post({"question": "one"}).json()["session_id"]
        second = self.post({"question": "two", "session_id": sid}).json()
        third = self.post({"question": "three", "session_id": sid}).json()
        self.assertEqual((second["turn_number"], third["turn_number"]), (2, 3))
        self.assertEqual([t.turn_number for t in self.turns(sid)], [1, 2, 3])

    def test_a_failing_save_never_breaks_the_response(self):
        with mock.patch("research_app.research_service.repo.save_turn", side_effect=RuntimeError("db down")):
            with self.assertLogs("research_app.research_service", level="ERROR") as logs:
                r = self.post({"question": "hi"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["answer"], "The answer [1].")
        self.assertIn("Conversation save FAILED", logs.output[0])

    def test_a_failing_history_load_does_not_stop_research(self):
        sid = self.make_session(self.alice)
        with mock.patch("research_app.research_service.repo.load_recent_turns", side_effect=RuntimeError("x")):
            r = self.post({"question": "hi", "session_id": sid})
        self.assertEqual(r.status_code, 200)

    def test_the_legacy_queries_row_carries_citations_and_routing_for_reloading_a_chat(self):
        sid = self.post({"question": "hi"}).json()["session_id"]
        body = self.client.get(f"/api/sessions/{sid}/queries", headers=self.headers).json()
        q = body["queries"][0]
        self.assertEqual((q["question"], q["answer"]), ("hi", "The answer [1]."))
        self.assertEqual(q["citations"][0]["domain"], "fastapi.tiangolo.com")
        self.assertEqual(q["routing"][0]["sources"], ["official_docs", "web"])

    def test_legacy_rows_without_sources_still_load(self):
        sid = self.make_session(self.alice)
        self.make_query(sid, "old q", "old a")
        q = self.client.get(f"/api/sessions/{sid}/queries", headers=self.headers).json()["queries"][0]
        self.assertEqual((q["citations"], q["routing"]), ([], []))

    def test_the_response_keeps_the_legacy_keys_and_adds_the_new_ones(self):
        body = self.post({"question": "hi"}).json()
        self.assertTrue({"sub_questions", "answer", "session_id"} <= set(body))
        self.assertTrue({"turn_number", "is_follow_up", "resolved_query", "citations", "sources_consulted", "subquestions",
                         "confidence", "critic_verdict", "research_metadata"} <= set(body))
        meta = body["research_metadata"]
        self.assertTrue({"total_sources_found", "total_evidence_pieces", "gap_search_performed", "retry_performed",
                         "sources_failed", "routing_decisions"} <= set(meta))
        self.assertEqual(body["citations"][0]["marker"], "[1]")

    def test_a_cache_hit_is_saved_as_a_turn_too(self):
        graph = fake_graph("cached answer text", cache_hit=True, sub_questions=["(from cache)"], routing_decisions=[],
                           cached_routing=[DECISION.model_dump(mode="json")])
        body = self.post({"question": "What is FastAPI?"}, graph).json()
        self.assertEqual(body["critic_verdict"], "cached")
        self.assertTrue(body["research_metadata"]["cache_hit"])
        self.assertEqual(body["subquestions"], [])
        self.assertEqual(body["research_metadata"]["routing_decisions"][0]["sub_question"], "Install FastAPI")
        (turn,) = self.turns(body["session_id"])
        self.assertEqual(turn.query_type, "cache")

    # ---- stream
    def test_the_stream_reports_routing_and_citations_before_the_tokens_and_ends_with_the_session(self):
        r = self.post({"question": "How do I install FastAPI?"}, path="/api/research/stream")
        events = [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]
        types = [e["type"] for e in events]
        self.assertLess(types.index("routing"), types.index("token"))
        self.assertLess(types.index("citations"), types.index("token"))
        self.assertEqual(types[-2:], ["sources", "done"])
        routing = next(e for e in events if e["type"] == "routing")
        self.assertEqual(routing["decisions"][0]["sources"], ["official_docs", "web"])
        self.assertEqual(next(e for e in events if e["type"] == "citations")["citations"][0]["index"], 1)
        done = events[-1]
        self.assertEqual((done["type"], done["turn_number"], done["is_follow_up"]), ("done", 1, False))
        self.assertTrue(done["session_id"])
        self.assertEqual(len(self.turns(done["session_id"])), 1)

    def test_streaming_without_a_session_id_creates_one(self):
        r = self.post({"question": "hi there"}, path="/api/research/stream")
        done = json.loads([l for l in r.text.split("\n\n") if l.startswith("data: ")][-1][6:])
        with self.SessionLocal() as db:
            self.assertEqual(db.get(ChatSession, done["session_id"]).user_id, self.alice)

    # ---- session endpoints
    def test_list_sessions_is_paginated_and_only_lists_own_active_sessions(self):
        ids = [self.make_session(self.alice, f"s{i}") for i in range(5)]
        self.make_session(self.bob, "bobs")
        self.client.delete(f"/sessions/{ids[0]}", headers=self.headers)
        page = self.client.get("/sessions?limit=2&offset=0", headers=self.headers).json()
        self.assertEqual((len(page["sessions"]), page["total"], page["limit"], page["offset"]), (2, 4, 2, 0))
        rest = self.client.get("/sessions?limit=2&offset=2", headers=self.headers).json()
        seen = [s["id"] for s in page["sessions"] + rest["sessions"]]
        self.assertEqual(sorted(seen), sorted(ids[1:]))
        self.assertEqual(len(self.client.get("/sessions", headers=self.headers).json()["sessions"]), 4)  # default limit 20

    def test_list_sessions_validates_the_page_size(self):
        self.assertEqual(self.client.get("/sessions?limit=0", headers=self.headers).status_code, 422)
        self.assertEqual(self.client.get("/sessions?limit=101", headers=self.headers).status_code, 422)

    def test_get_session_returns_the_full_history_with_citations(self):
        sid = self.post({"question": "one"}).json()["session_id"]
        self.post({"question": "two", "session_id": sid})
        body = self.client.get(f"/sessions/{sid}", headers=self.headers).json()
        self.assertEqual((body["id"], body["turn_count"], [t["turn_number"] for t in body["turns"]]), (sid, 2, [1, 2]))
        self.assertEqual(body["turns"][0]["answer_text"], "The answer [1].")
        self.assertEqual(body["turns"][0]["citations"][0]["url"], "https://fastapi.tiangolo.com/tutorial/")

    def test_delete_is_a_soft_delete(self):
        sid = self.post({"question": "one"}).json()["session_id"]
        self.assertEqual(self.client.delete(f"/sessions/{sid}", headers=self.headers).json(), {"deleted": sid, "soft": True})
        with self.SessionLocal() as db:
            self.assertFalse(db.get(ChatSession, sid).is_active)
            self.assertEqual(repo.count_turns(db, sid), 1)  # nothing was removed
        self.assertEqual(self.client.get(f"/sessions/{sid}", headers=self.headers).status_code, 404)
        self.assertEqual(self.client.get("/sessions", headers=self.headers).json()["total"], 0)
        self.assertEqual(self.client.get("/api/sessions", headers=self.headers).json()["sessions"], [])

    def test_session_endpoints_enforce_auth_and_ownership(self):
        sid = self.make_session(self.alice)
        for method, path in (("get", "/sessions"), ("get", f"/sessions/{sid}"), ("delete", f"/sessions/{sid}")):
            self.assertEqual(getattr(self.client, method)(path).status_code, 401, path)
        bob = self.auth("bob")
        self.assertEqual(self.client.get(f"/sessions/{sid}", headers=bob).status_code, 404)
        self.assertEqual(self.client.delete(f"/sessions/{sid}", headers=bob).status_code, 404)
        self.assertEqual(self.client.get("/sessions", headers=bob).json()["total"], 0)

    def test_admin_clear_all_removes_conversations_without_a_foreign_key_error(self):
        with self.engine.connect() as c:
            c.exec_driver_sql("PRAGMA foreign_keys=ON")
        self.post({"question": "one"})
        with mock.patch.dict("os.environ", {"ADMIN_USERNAMES": "alice"}):
            r = self.client.delete("/admin/clear-all-data", headers=self.headers)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["conversations_deleted"], 1)
        self.assertEqual((self.count(ResearchConversation), self.count(ResearchCitation), self.count(ChatSession)), (0, 0, 0))


# --------------------------------------------------------------------------- follow-ups through the real graph

class FollowUpApiTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.uid = self.make_user("alice")
        self.headers = self.auth("alice")
        self.fake = relevant_fake()

    def ask(self, question, llm, session_id=None):
        p1, p2, p3, p4 = patches(self.fake, llm)
        body = {"question": question}
        if session_id:
            body["session_id"] = session_id
        with p1, p2, p3, p4:
            return self.client.post("/api/research", json=body, headers=self.headers)

    def test_a_follow_up_is_resolved_researched_and_stored_with_its_context(self):
        first = self.ask("How do I install FastAPI with uvicorn?", ScriptedLLM(
            understanding_=understanding(intent="technical_howto", technology="FastAPI"),
            answer="FastAPI installs with pip [1]. Uvicorn runs it [2]. Docs cover both [1][2]."))
        self.assertEqual(first.status_code, 200)
        sid = first.json()["session_id"]
        self.assertFalse(first.json()["is_follow_up"])

        llm = ScriptedLLM(understanding_=understanding(is_follow_up=True, resolved_query="deploy FastAPI", intent="technical_howto",
                                                       technology="FastAPI", context_topics=["FastAPI"]))
        second = self.ask("deploy it", llm, sid).json()
        self.assertEqual((second["is_follow_up"], second["resolved_query"], second["turn_number"]), (True, "deploy FastAPI", 2))
        self.assertIn("How do I install FastAPI with uvicorn?", llm.prompts_for("QueryUnderstanding")[0])
        planner_prompt = llm.prompts_for("SourceAwarePlan")[0]
        self.assertIn("Question: deploy FastAPI", planner_prompt)
        self.assertIn("Do not re-research", planner_prompt)
        self.assertIn("deploy FastAPI", llm.prompts[0])  # the writer answers the resolved question
        with self.SessionLocal() as db:
            turns = repo.load_all_turns(db, sid)
        self.assertEqual([(t.turn_number, t.is_follow_up, t.resolved_query) for t in turns],
                         [(1, False, "How do I install FastAPI with uvicorn?"), (2, True, "deploy FastAPI")])

    def test_a_topic_change_in_the_same_session_is_researched_independently(self):
        sid = self.ask("How do I install FastAPI?", ScriptedLLM()).json()["session_id"]
        llm = ScriptedLLM(understanding_=understanding(is_follow_up=False, resolved_query="How does Docker Compose work?"))
        out = self.ask("How does Docker Compose work?", llm, sid).json()
        self.assertEqual((out["is_follow_up"], out["resolved_query"]), (False, "How does Docker Compose work?"))


# --------------------------------------------------------------------------- cache interplay

class CacheTests(unittest.TestCase):
    def setUp(self):
        vectordb.clear_cache()
        self.addCleanup(vectordb.clear_cache)

    def context(self):
        return build_context(1, [row(1, "How do I install FastAPI?", answer="pip install fastapi")])

    def test_a_session_with_history_bypasses_the_cache(self):
        vectordb.store_cache("deploy it", "somebody else's answer about deploying something " * 3)
        out = st.semantic_cache_node({"question": "deploy it", "conversation_context": self.context()})
        self.assertFalse(out["cache_hit"])

    def test_an_empty_history_still_hits_the_cache_and_returns_the_citations(self):
        answer = "FastAPI installs with pip [1]. " * 3
        vectordb.store_cache("What is FastAPI?", answer)
        vectordb.store_cache_extras(answer, {"citations": [CITATION], "routing": [DECISION.model_dump(mode="json")],
                                             "confidence": "high"})
        out = st.semantic_cache_node({"question": "What is FastAPI?", "conversation_context": ConversationContext()})
        self.assertTrue(out["cache_hit"])
        self.assertEqual(out["citations"][0]["domain"], "fastapi.tiangolo.com")
        self.assertEqual((out["cached_routing"][0]["sub_question"], out["confidence"]), ("Install FastAPI", "high"))

    def test_a_legacy_cache_hit_has_exactly_the_old_shape(self):
        vectordb.store_cache("What is FastAPI?", "An answer that is long enough to be cached properly here.")
        out = st.semantic_cache_node({"question": "What is FastAPI?"})
        self.assertEqual(set(out), {"cache_hit", "final_answer", "messages", "search_results", "sources", "sub_questions"})

    def stored(self, **state):
        base = {"question": "What is FastAPI?", "final_answer": "FastAPI is a framework [1]. " * 4, "cache_hit": False,
                "synthesis": object(), "citations": [CITATION], "routing_decisions": [DECISION], "confidence": "high"}
        base.update(state)
        st.save_to_cache_node(base)
        return vectordb.lookup_cache("What is FastAPI?")[0]

    def test_a_follow_up_answer_is_not_cached(self):
        self.assertFalse(self.stored(is_follow_up=True))

    def test_an_answer_without_citations_is_not_cached(self):
        self.assertFalse(self.stored(citations=[]))

    def test_a_cited_standalone_answer_is_cached_with_its_sources(self):
        self.assertTrue(self.stored(is_follow_up=False))
        answer = vectordb.lookup_cache("What is FastAPI?")[1]
        self.assertEqual(vectordb.get_cache_extras(answer)["citations"][0]["index"], 1)


if __name__ == "__main__":
    unittest.main()
