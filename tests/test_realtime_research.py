"""Real-time research: the agent knows today's date, does not search past years for "latest"
questions, asks the search provider for recent results, and never serves a stale cached answer for
them. Pure logic plus the search node with a fake Tavily; no network, no LLM."""
import os
import unittest
from datetime import datetime, timezone
from unittest import mock

from research_app.agent import state as st
from research_app.agent import temporal
from research_app.agent.pipeline.evidence import EvidencePool
from research_app.agent.pipeline.synthesis import NumberedSource, build_prompt
from research_app.domain import ResearchTask, SourceDocument, SourceType
from tests.p6_helpers import ScriptedLLM, plan, task, understanding
from tests.test_router_nodes import WEB_OK, FakeTavily, NodeTestBase

NOW = datetime(2026, 9, 19, 10, 30, tzinfo=timezone.utc)
YEAR = datetime.now(timezone.utc).year   # what the code under test sees as "now"


class TemporalTests(unittest.TestCase):
    def test_the_date_line_states_today_and_the_year(self):
        line = temporal.date_line(NOW)
        self.assertIn("2026-09-19", line)
        self.assertIn("current year is 2026", line)

    def test_time_sensitive_questions_are_recognised(self):
        for q in ("latest AI news", "What are the emerging trends in RAG?", "current price of Bitcoin",
                  "What's new in Python?", "recent research on scaling RAG", "breaking news today"):
            self.assertTrue(temporal.is_time_sensitive(q), q)

    def test_evergreen_questions_are_not(self):
        for q in ("What is the capital of France?", "How do I install FastAPI?", "history of the Roman empire",
                  "explain how a transformer works"):
            self.assertFalse(temporal.is_time_sensitive(q), q)

    def test_the_models_own_judgement_counts(self):
        self.assertTrue(temporal.is_time_sensitive("how is X doing", {"time_sensitivity": "current"}))
        self.assertTrue(temporal.is_time_sensitive("how is X doing", {"intent": "current_events"}))
        self.assertFalse(temporal.is_time_sensitive("how is X doing", {"time_sensitivity": "evergreen"}))

    def test_a_past_year_the_planner_added_becomes_the_current_year(self):
        self.assertEqual(temporal.refresh_stale_years("latest AI news 2024", now=NOW), "latest AI news 2026")
        self.assertEqual(temporal.refresh_stale_years("RAG trends 2023 and 2024", now=NOW), "RAG trends 2026")
        self.assertEqual(temporal.refresh_stale_years("AI 2023 vs 2024 benchmarks", now=NOW), "AI 2026 benchmarks")

    def test_a_year_the_user_wrote_is_kept(self):
        self.assertEqual(temporal.refresh_stale_years("AI news 2023", user_text="AI news from 2023", now=NOW),
                         "AI news 2023")

    def test_the_current_and_future_years_and_other_numbers_are_left_alone(self):
        for q in ("AI news 2026", "roadmap 2027", "GPT-4 release", "port 8080", "version 2024x", "model-2023-06"):
            self.assertEqual(temporal.refresh_stale_years(q, now=NOW), q)

    def test_the_date_window_matches_how_fresh_the_question_wants_things(self):
        self.assertEqual(temporal.recency_params("breaking news today"), {"time_range": "week", "topic": "news"})
        self.assertEqual(temporal.recency_params("latest AI news"), {"time_range": "month", "topic": "news"})
        self.assertEqual(temporal.recency_params("emerging trends in RAG"), {"time_range": "year"})
        self.assertEqual(temporal.recency_params("history of Rome"), {})

    def test_the_date_filter_can_be_switched_off(self):
        self.assertEqual(temporal.recency_params("latest AI news", env={"RECENCY_FILTER_ENABLED": "false"}), {})


class SearchNodeTests(NodeTestBase):
    async def search(self, query, question, fake=None):
        fake = fake or FakeTavily()
        with mock.patch.object(st, "tavily_tool", fake):
            update = await st._research_update(query, question=question, content_limit=400)
        return update, fake

    async def test_a_latest_question_gets_a_date_window_and_no_past_year(self):
        _update, fake = await self.search("latest AI news 2024", "latest AI news")
        (call,) = fake.of("web")
        self.assertEqual((call["query"], call["time_range"], call["topic"]), (f"latest AI news {YEAR}", "month", "news"))

    async def test_a_year_the_user_asked_for_is_searched_as_written(self):
        _update, fake = await self.search("AI news 2023", "what happened in AI news in 2023")
        self.assertEqual(fake.of("web")[0]["query"], "AI news 2023")

    async def test_an_evergreen_question_is_searched_exactly_as_before(self):
        _update, fake = await self.search("history of Rome 2020", "history of Rome")
        self.assertEqual(fake.of("web"), [{"query": "history of Rome 2020"}])

    async def test_nothing_in_the_window_retries_once_without_it(self):
        class Windowed(FakeTavily):
            async def ainvoke(self, payload):
                if self.kind(payload) == "web" and "time_range" in payload:
                    self.calls.append(payload)
                    return {"query": payload["query"], "results": []}
                return await super().ainvoke(payload)
        update, fake = await self.search("emerging trends in RAG", "emerging trends in RAG", Windowed())
        web = fake.of("web")
        self.assertEqual(len(web), 2)
        self.assertIn("time_range", web[0])
        self.assertNotIn("time_range", web[1])
        self.assertTrue(update["source_documents"])            # the retry's results are used

    async def test_results_inside_the_window_do_not_trigger_a_retry(self):
        _update, fake = await self.search("latest AI news", "latest AI news", FakeTavily(web=WEB_OK))
        self.assertEqual(len(fake.of("web")), 1)

    async def test_with_the_filter_off_only_the_year_guard_applies(self):
        os.environ["RECENCY_FILTER_ENABLED"] = "false"
        _update, fake = await self.search("latest AI news 2024", "latest AI news")
        self.assertEqual(fake.of("web"), [{"query": f"latest AI news {YEAR}"}])


class PromptAndPlanTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_planner_and_the_classifier_are_told_the_date(self):
        llm = ScriptedLLM(understanding_=understanding(time_sensitivity="current"))
        with mock.patch.object(st, "llm_groq", llm):
            await st.classify_node({"question": "latest AI news"})
            await st.planner_node({"question": "latest AI news", "resolved_query": "latest AI news",
                                   "understanding": {"time_sensitivity": "current"}})
        for prompt in (llm.prompts_for("QueryUnderstanding")[0], llm.prompts_for("SourceAwarePlan")[0]):
            self.assertIn("TODAY'S DATE:", prompt)
            self.assertIn(f"current year is {YEAR}", prompt)
        self.assertIn("Never put an earlier year in a query", llm.prompts_for("SourceAwarePlan")[0])

    async def plan_for(self, question, *texts):
        llm = ScriptedLLM(plan_=plan(*(task(t) for t in texts)))
        with mock.patch.object(st, "llm_groq", llm):
            return await st.planner_node({"question": question, "resolved_query": question, "understanding": {}})

    async def test_past_years_in_a_latest_plan_are_refreshed_even_if_the_model_ignores_the_prompt(self):
        out = await self.plan_for("latest research on RAG", "Emerging trends in RAG architectures 2024",
                                  "Research papers on RAG 2023 vs 2024", "Open-source RAG libraries 2024")
        self.assertEqual(out["sub_questions"], [f"Emerging trends in RAG architectures {YEAR}",
                                                f"Research papers on RAG {YEAR}", f"Open-source RAG libraries {YEAR}"])
        self.assertEqual([t.search_query for t in out["research_tasks"]], out["sub_questions"])

    async def test_a_year_named_by_the_user_or_an_evergreen_plan_is_left_alone(self):
        out = await self.plan_for("what were the latest RAG trends in 2023", "RAG trends 2023")
        self.assertEqual(out["sub_questions"], ["RAG trends 2023"])
        out = await self.plan_for("history of the transformer", "Transformer paper 2017")
        self.assertEqual(out["sub_questions"], ["Transformer paper 2017"])

    def test_the_synthesis_prompt_has_the_date_and_shows_when_a_source_was_published(self):
        doc = SourceDocument(url="https://a.example/x", published_at=datetime(2026, 3, 2, tzinfo=timezone.utc))
        source = NumberedSource(index=1, source_type=SourceType.WEB, title="t", url="https://a.example/x",
                                domain="a.example", source_id="s", retrieved_at=None, snippet="s", content="body",
                                label="[WEB]", documents=[doc])
        prompt = build_prompt("latest RAG news", "latest RAG news", [source], EvidencePool([ResearchTask(sub_question="q")]), None)
        self.assertIn("TODAY'S DATE:", prompt)
        self.assertIn("[1] [WEB] (published 2026-03-02)", prompt)
        self.assertIn("Freshness:", prompt)


class CacheTests(unittest.TestCase):
    def test_a_time_sensitive_question_is_never_answered_from_the_cache(self):
        lookup = mock.Mock(return_value=(True, "old answer from yesterday"))
        with mock.patch.object(st, "lookup_cache", lookup):
            out = st.semantic_cache_node({"question": "latest AI news"})
        self.assertFalse(out["cache_hit"])
        lookup.assert_not_called()

    def test_an_evergreen_question_still_uses_the_cache(self):
        lookup = mock.Mock(return_value=(True, "FastAPI is a web framework."))
        with mock.patch.object(st, "lookup_cache", lookup), mock.patch.object(st, "get_cache_extras", lambda a: None):
            out = st.semantic_cache_node({"question": "What is FastAPI?"})
        self.assertTrue(out["cache_hit"])

    def test_a_time_sensitive_answer_is_not_saved(self):
        store = mock.Mock()
        with mock.patch.object(st, "store_cache", store):
            st.save_to_cache_node({"question": "latest AI news", "final_answer": "x" * 200})
            store.assert_not_called()
            st.save_to_cache_node({"question": "What is FastAPI?", "final_answer": "x" * 200})
            store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
