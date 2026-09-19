import os
from unittest import mock

from research_app.db.models import Query, Session as ChatSession, User
from tests.base import ApiTestCase


def _fake_graph(final_answer="the answer"):
    """Stand-in for compiled_graph so no LLM / Tavily call is ever made."""
    graph = mock.MagicMock()
    graph.ainvoke = mock.AsyncMock(
        return_value={
            "sub_questions": [],
            "final_answer": final_answer,
            "api_limit_reached": False,
        }
    )

    async def astream(inputs, stream_mode=None):
        yield "updates", {
            "synthesize_node": {
                "final_answer": final_answer,
                "sources": [{"url": "https://example.com", "title": "Example"}],
            }
        }

    graph.astream = mock.MagicMock(side_effect=astream)
    return graph


class SessionOwnershipTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        alice_id = self.make_user("alice")
        bob_id = self.make_user("bob")
        self.alice_session = self.make_session(alice_id, "alice-session")
        self.bob_session = self.make_session(bob_id, "bob-session")
        self.make_query(self.alice_session, "alice-secret-question", "alice-secret-answer")

    # GET /api/sessions/{id}/queries
    def test_owner_can_read_own_session_queries(self):
        r = self.client.get(f"/api/sessions/{self.alice_session}/queries", headers=self.auth("alice"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["queries"][0]["question"], "alice-secret-question")

    def test_other_user_cannot_read_session_queries(self):
        r = self.client.get(f"/api/sessions/{self.alice_session}/queries", headers=self.auth("bob"))
        self.assertEqual(r.status_code, 404)
        self.assertNotIn("alice-secret", r.text)

    def test_missing_session_queries_is_404(self):
        r = self.client.get("/api/sessions/99999/queries", headers=self.auth("alice"))
        self.assertEqual(r.status_code, 404)

    def test_queries_requires_auth(self):
        r = self.client.get(f"/api/sessions/{self.alice_session}/queries")
        self.assertEqual(r.status_code, 401)

    # PATCH /api/sessions/{id}
    def test_owner_can_rename_own_session(self):
        r = self.client.patch(
            f"/api/sessions/{self.alice_session}", json={"title": "renamed"}, headers=self.auth("alice")
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["title"], "renamed")

    def test_other_user_cannot_rename_session(self):
        r = self.client.patch(
            f"/api/sessions/{self.alice_session}", json={"title": "hijacked"}, headers=self.auth("bob")
        )
        self.assertEqual(r.status_code, 404)
        with self.SessionLocal() as db:
            self.assertEqual(db.get(ChatSession, self.alice_session).title, "alice-session")

    # DELETE /api/sessions/{id} (already checked ownership; regression guard)
    def test_other_user_cannot_delete_session(self):
        r = self.client.delete(f"/api/sessions/{self.alice_session}", headers=self.auth("bob"))
        self.assertEqual(r.status_code, 404)
        self.assertEqual(self.count(ChatSession), 2)

    # POST /api/research
    def test_research_with_other_users_session_is_rejected(self):
        graph = _fake_graph()
        with mock.patch("research_app.main.compiled_graph", graph):
            r = self.client.post(
                "/api/research",
                json={"question": "hi", "session_id": self.alice_session},
                headers=self.auth("bob"),
            )
        self.assertEqual(r.status_code, 404)
        graph.ainvoke.assert_not_awaited()
        self.assertEqual(self.count(Query), 1)  # nothing written into alice's session

    def test_research_with_own_session_still_works(self):
        with mock.patch("research_app.main.compiled_graph", _fake_graph("fresh answer")):
            r = self.client.post(
                "/api/research",
                json={"question": "hi", "session_id": self.alice_session},
                headers=self.auth("alice"),
            )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["answer"], "fresh answer")
        self.assertEqual(r.json()["session_id"], self.alice_session)
        self.assertEqual(self.count(Query), 2)

    def test_research_without_session_id_uses_callers_own_session(self):
        with mock.patch("research_app.main.compiled_graph", _fake_graph()):
            r = self.client.post("/api/research", json={"question": "hi"}, headers=self.auth("bob"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["session_id"], self.bob_session)

    # POST /api/research/stream
    def test_stream_with_other_users_session_is_rejected(self):
        graph = _fake_graph()
        with mock.patch("research_app.main.compiled_graph", graph):
            r = self.client.post(
                "/api/research/stream",
                json={"question": "hi", "session_id": self.alice_session},
                headers=self.auth("bob"),
            )
        self.assertEqual(r.status_code, 404)
        graph.astream.assert_not_called()
        self.assertEqual(self.count(Query), 1)

    def test_stream_with_own_session_still_works(self):
        with mock.patch("research_app.main.compiled_graph", _fake_graph("streamed answer here")):
            r = self.client.post(
                "/api/research/stream",
                json={"question": "hi", "session_id": self.alice_session},
                headers=self.auth("alice"),
            )
        self.assertEqual(r.status_code, 200)
        self.assertIn('"type": "done"', r.text)
        self.assertIn('"type": "sources"', r.text)
        self.assertEqual(self.count(Query), 2)


class AdminAccessTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        alice_id = self.make_user("alice")
        self.make_user("bob")
        self.make_query(self.make_session(alice_id), "q", "a")

    def test_admin_endpoints_require_auth(self):
        self.assertEqual(self.client.get("/admin/db-stats").status_code, 401)
        self.assertEqual(self.client.delete("/admin/clear-all-data").status_code, 401)

    def test_fails_closed_when_admin_usernames_unset(self):
        with mock.patch.dict(os.environ, {"ADMIN_USERNAMES": ""}):
            self.assertEqual(self.client.get("/admin/db-stats", headers=self.auth("alice")).status_code, 403)
            r = self.client.delete("/admin/clear-all-data", headers=self.auth("alice"))
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.count(Query), 1)
        self.assertEqual(self.count(ChatSession), 1)

    def test_non_admin_is_forbidden_even_when_an_admin_is_configured(self):
        with mock.patch.dict(os.environ, {"ADMIN_USERNAMES": "alice"}):
            self.assertEqual(self.client.get("/admin/db-stats", headers=self.auth("bob")).status_code, 403)
            r = self.client.delete("/admin/clear-all-data", headers=self.auth("bob"))
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.count(Query), 1)

    def test_admin_can_read_stats(self):
        with mock.patch.dict(os.environ, {"ADMIN_USERNAMES": "alice"}):
            r = self.client.get("/admin/db-stats", headers=self.auth("alice"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"queries": 1, "sessions": 1, "users": 2})

    def test_admin_can_clear_data_and_list_tolerates_spaces(self):
        with mock.patch.dict(os.environ, {"ADMIN_USERNAMES": "carol, alice ,"}):
            r = self.client.delete("/admin/clear-all-data", headers=self.auth("alice"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.count(Query), 0)
        self.assertEqual(self.count(ChatSession), 0)
        self.assertEqual(self.count(User), 2)  # users are untouched by clear-all-data
