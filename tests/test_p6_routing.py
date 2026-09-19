"""Phase 6, Part A: the planner proposes, the router decides. Per-task routing, planner output
parsing, query understanding and reference resolution. No network, no LLM."""
import unittest

from research_app.agent import state as st
from research_app.agent.pipeline.settings import PipelineSettings
from research_app.domain import (
    ORIGIN_FALLBACK, ORIGIN_PLANNER, ORIGIN_POLICY, ConversationContext, ConversationTurn, ResearchTask,
    SourceIntent, SourceType,
)
from research_app.sources.official_docs.settings import DocsSettings
from research_app.sources.routing import RouterSettings, route_sources, route_task, task_technology
from tests.p6_helpers import ScriptedLLM, plan, task, understanding

DOCS, GH, RD, WEB = SourceType.OFFICIAL_DOCS, SourceType.GITHUB, SourceType.REDDIT, SourceType.WEB


def route(text, intent=None, sources=(), technology=None, understanding_=None, cap=3, enabled=True, docs_enabled=True):
    t = ResearchTask(sub_question=text, source_intent=intent, suggested_sources=list(sources), technology=technology)
    return route_task(t, understanding_ or {}, RouterSettings(enabled=enabled, max_sources_per_task=cap),
                      DocsSettings(enabled=docs_enabled))


class PlannerSuggestionsTests(unittest.TestCase):
    def test_valid_suggestions_are_accepted_as_planner_origin(self):
        d = route("How do I install FastAPI with uvicorn?", "technical_howto", ["official_docs", "web"])
        self.assertEqual(d.sources, [DOCS, WEB])
        self.assertEqual(d.origins, {"official_docs": ORIGIN_PLANNER, "web": ORIGIN_PLANNER})
        self.assertEqual(d.dropped, [])

    def test_unknown_and_malformed_source_types_are_dropped_and_reported(self):
        d = route("Find GitHub examples of a Zorblax agent", "code_implementation",
                  ["bogus", "GitHub", "github", "", "twitter"])
        self.assertEqual(d.sources, [GH, WEB])  # GitHub / github de-duplicated, case-insensitive
        self.assertEqual([(x.source, x.reason) for x in d.dropped],
                         [("bogus", "unknown_source_type"), ("", "unknown_source_type"), ("twitter", "unknown_source_type")])

    def test_non_string_suggestions_are_reported_as_malformed(self):
        t = ResearchTask.model_construct(task_id="t", sub_question="q", search_query="q", source_types=[],
                                         preferred_domains=[], priority=0, status=None, source_intent=None,
                                         suggested_sources=[5, None, "web"], technology=None)
        d = route_task(t, {}, RouterSettings(), DocsSettings())
        self.assertEqual(d.sources, [WEB])
        self.assertEqual([x.reason for x in d.dropped], ["malformed_source", "malformed_source"])

    def test_too_many_suggestions_are_capped(self):
        d = route("Find GitHub examples", "code_implementation", ["github"] + ["bogus"] * 12)
        self.assertIn("too_many_suggestions", [x.reason for x in d.dropped])

    def test_an_unknown_intent_is_ignored_not_fatal(self):
        t = ResearchTask.model_construct(task_id="t", sub_question="Best laptops", search_query="Best laptops",
                                         source_intent="nonsense", suggested_sources=[], technology=None)
        self.assertEqual(route_task(t, {}, RouterSettings(), DocsSettings()).sources, [WEB])


class FallbackTests(unittest.TestCase):
    def test_no_hints_uses_the_phase5_route_for_the_question(self):
        q = "Find GitHub examples of LangGraph agents"
        d = route("sub one", understanding_={"resolved_query": q})
        self.assertEqual(d.sources, list(route_sources(q).sources))
        self.assertEqual(set(d.origins.values()), {ORIGIN_FALLBACK})

    def test_the_fallback_is_not_trimmed_by_the_cap(self):
        q = "Why am I getting this LangGraph error?"  # official_docs + github + reddit + web
        d = route("sub one", understanding_={"resolved_query": q}, cap=3)
        self.assertEqual(d.sources, [DOCS, GH, RD, WEB])

    def test_no_hints_and_no_question_text_routes_the_sub_question_itself(self):
        d = route("Real-world experiences using Qdrant")
        self.assertEqual(d.sources, [RD, WEB])

    def test_a_plain_question_is_web_only(self):
        self.assertEqual(route("Best laptops for students").sources, [WEB])

    def test_intent_without_suggestions_uses_the_intent_defaults_as_policy(self):
        d = route("Real-world experiences using Qdrant", "community_experience")
        self.assertEqual(d.sources, [RD, WEB])
        self.assertEqual(set(d.origins.values()), {ORIGIN_POLICY})


class EnforcementTests(unittest.TestCase):
    def test_official_docs_is_enforced_when_the_planner_omits_it_on_a_technical_task(self):
        d = route("How do I install FastAPI with uvicorn?", "technical_howto", ["web"])
        self.assertEqual(d.sources, [DOCS, WEB])
        self.assertEqual(d.origins["official_docs"], ORIGIN_POLICY)

    def test_reddit_never_substitutes_for_official_docs(self):
        d = route("How do I configure FastAPI middleware?", "technical_howto", ["reddit"])
        self.assertIn(DOCS, d.sources)
        self.assertEqual(d.origins["official_docs"], ORIGIN_POLICY)

    def test_a_technology_from_understanding_counts_for_a_technical_task(self):
        d = route("How do I add middleware?", "technical_howto", ["web"], understanding_={"technology": "FastAPI"})
        self.assertIn(DOCS, d.sources)

    def test_non_technical_stays_web_only_despite_over_suggestion(self):
        d = route("Best laptops for students", "general_research", ["github", "reddit", "official_docs", "web"])
        self.assertEqual(d.sources, [WEB])
        self.assertEqual(sorted(x.source for x in d.dropped), ["github", "official_docs", "reddit"])
        self.assertEqual({x.reason for x in d.dropped}, {"non_technical"})

    def test_a_general_sub_question_does_not_inherit_the_runs_technology(self):
        d = route("Best laptops for students", "general_research", ["github", "official_docs"],
                  understanding_={"technology": "LangGraph"})
        self.assertEqual(d.sources, [WEB])

    def test_docs_for_a_technology_that_is_not_in_the_registry_are_dropped_with_a_reason(self):
        d = route("How do I configure Zorblax?", "technical_howto", ["official_docs", "web"], technology="Zorblax")
        self.assertEqual(d.sources, [WEB])
        self.assertEqual([(x.source, x.reason) for x in d.dropped], [("official_docs", "technology_not_in_registry")])

    def test_a_comparison_with_a_technology_gets_official_docs(self):
        d = route("Compare FastAPI and Django setup", "comparison", ["web"])
        self.assertEqual(d.sources, [DOCS, WEB])

    def test_a_community_problem_question_does_not_force_official_docs(self):
        d = route("What problems are developers having with Qdrant in production?", "community_experience",
                  ["reddit", "github", "web"])
        self.assertEqual(d.sources, [GH, RD, WEB])

    def test_web_is_always_present_even_if_the_planner_left_it_out(self):
        d = route("How do I install FastAPI?", "technical_howto", ["official_docs"])
        self.assertIn(WEB, d.sources)
        self.assertEqual(d.origins["web"], ORIGIN_POLICY)


class CapTests(unittest.TestCase):
    Q = "Why does LangGraph raise this error?"

    def test_the_cap_trims_reddit_first_and_reports_it(self):
        d = route(self.Q, "troubleshooting", cap=3)
        self.assertEqual(d.sources, [DOCS, GH, WEB])
        self.assertEqual([(x.source, x.reason) for x in d.dropped], [("reddit", "over_cap")])

    def test_a_question_about_community_experience_keeps_reddit_and_trims_github(self):
        q = "What problems are developers having with Qdrant in production?"
        d = route("Qdrant production latency issues", "troubleshooting", ["reddit", "github", "web"],
                  understanding_={"resolved_query": q}, cap=3)
        self.assertEqual(d.sources, [DOCS, RD, WEB])
        self.assertEqual([(x.source, x.reason) for x in d.dropped], [("github", "over_cap")])

    def test_the_planner_writing_reddit_into_a_sub_question_does_not_unlock_reddit_for_a_non_technical_question(self):
        d = route("Reddit threads discussing real-world performance of student laptops", "community_experience",
                  ["reddit", "web"], understanding_={"resolved_query": "Best laptops for college students"})
        self.assertEqual(d.sources, [WEB])
        self.assertEqual([(x.source, x.reason) for x in d.dropped], [("reddit", "non_technical")])

    def test_a_non_technical_question_that_asks_for_reddit_keeps_it(self):
        d = route("student laptop opinions", "community_experience", ["reddit", "web"],
                  understanding_={"resolved_query": "What do students on reddit say about the best laptops?"})
        self.assertEqual(d.sources, [RD, WEB])

    def test_a_higher_cap_keeps_all_four(self):
        self.assertEqual(route(self.Q, "troubleshooting", cap=4).sources, [DOCS, GH, RD, WEB])

    def test_official_docs_and_web_are_never_dropped_by_the_cap(self):
        d = route(self.Q, "troubleshooting", cap=1)
        self.assertEqual(d.sources, [DOCS, WEB])

    def test_settings_parse_and_fall_back(self):
        self.assertEqual(RouterSettings.from_env({"MAX_SOURCES_PER_TASK": "2"}).max_sources_per_task, 2)
        for bad in ("0", "9", "x", ""):
            self.assertEqual(RouterSettings.from_env({"MAX_SOURCES_PER_TASK": bad}).max_sources_per_task, 3)


class KillSwitchTests(unittest.TestCase):
    def test_router_disabled_ignores_hints_and_keeps_the_phase4_behaviour(self):
        d = route("How do I install FastAPI?", "code_implementation", ["github", "reddit"], enabled=False)
        self.assertEqual(d.sources, [DOCS, WEB])  # the documentation detector + web
        self.assertEqual({x.reason for x in d.dropped}, {"router_disabled"})

    def test_docs_disabled_removes_official_docs(self):
        d = route("How do I install FastAPI?", "technical_howto", ["official_docs", "web"], docs_enabled=False)
        self.assertEqual(d.sources, [WEB])
        self.assertEqual([x.reason for x in d.dropped], ["docs_disabled"])


class DecisionTests(unittest.TestCase):
    def test_same_input_same_decision(self):
        def make():
            return route_task(ResearchTask(task_id="fixed", sub_question="Why does LangGraph raise this error?",
                                           source_intent="troubleshooting"), {}, RouterSettings(), DocsSettings())
        self.assertEqual(make().model_dump(), make().model_dump())

    def test_contents_of_a_decision(self):
        t = ResearchTask(task_id="abc", sub_question="Install FastAPI", source_intent="technical_howto",
                         suggested_sources=["web", "bogus"], technology="FastAPI")
        d = route_task(t, {}, RouterSettings(), DocsSettings())
        self.assertEqual((d.task_id, d.sub_question, d.source_intent), ("abc", "Install FastAPI", SourceIntent.TECHNICAL_HOWTO))
        self.assertEqual(d.sources, [DOCS, WEB])
        self.assertEqual(d.origins, {"official_docs": ORIGIN_POLICY, "web": ORIGIN_PLANNER})
        self.assertEqual(d.dropped[0].source, "bogus")
        self.assertTrue(d.includes(DOCS) and not d.includes(GH))
        self.assertEqual(d.outcomes, [])  # outcomes are attached later, by the executor

    def test_different_sub_questions_route_differently(self):
        setup = route("FastAPI setup steps", "technical_howto", ["official_docs", "web"], technology="FastAPI")
        opinions = route("What do developers think about Django?", "community_experience", ["reddit", "web"])
        self.assertEqual(setup.sources, [DOCS, WEB])
        self.assertEqual(opinions.sources, [RD, WEB])

    def test_task_technology_rules(self):
        own = ResearchTask(sub_question="q", technology="Own")
        self.assertEqual(task_technology(own, {"technology": "Run"}), "Own")
        legacy = ResearchTask(sub_question="q")
        self.assertEqual(task_technology(legacy, {"technology": "Run"}), "Run")
        technical = ResearchTask(sub_question="q", source_intent="technical_howto")
        self.assertEqual(task_technology(technical, {"technology": "Run"}), "Run")
        general = ResearchTask(sub_question="q", source_intent="general_research")
        self.assertIsNone(task_technology(general, {"technology": "Run"}))


class PlannerParsingTests(unittest.TestCase):
    def test_source_hints_become_tasks(self):
        result = plan(task("Install LangGraph", "technical_howto", ["official_docs", "web"], "LangGraph"),
                      task("Examples", "code_implementation", ["github"]))
        tasks = st.tasks_from_plan(result, "Q?")
        self.assertEqual([t.sub_question for t in tasks], ["Install LangGraph", "Examples"])
        self.assertEqual(tasks[0].source_intent, SourceIntent.TECHNICAL_HOWTO)
        self.assertEqual(tasks[0].suggested_sources, ["official_docs", "web"])
        self.assertEqual((tasks[0].technology, tasks[1].technology), ("LangGraph", None))

    def test_legacy_output_without_source_fields_still_works(self):
        tasks = st.tasks_from_plan(st.SearchPlan(sub_questions=["a b", "c d", "e f"]), "Q?")
        self.assertEqual([t.sub_question for t in tasks], ["a b", "c d", "e f"])
        self.assertTrue(all(t.suggested_sources == [] and t.source_intent is None for t in tasks))

    def test_blank_and_duplicate_sub_questions_are_dropped_and_the_count_is_capped(self):
        result = plan(*[task(t) for t in ["one", "  ", "One", "two", "three", "four"]])
        self.assertEqual([t.sub_question for t in st.tasks_from_plan(result, "Q?")], ["one", "two", "three"])

    def test_malformed_hints_are_cleaned(self):
        item = st.PlannedSubquestion.model_construct(text="x y", source_intent="weird",
                                                     suggested_sources=["web", 5, None, ""], technology=" FastAPI ")
        tasks = st.tasks_from_plan(st.SourceAwarePlan.model_construct(tasks=[item]), "Q?")
        self.assertEqual((tasks[0].source_intent, tasks[0].suggested_sources, tasks[0].technology),
                         (None, ["web"], "FastAPI"))

    def test_an_empty_plan_falls_back_to_the_question(self):
        tasks = st.tasks_from_plan(st.SourceAwarePlan(tasks=[]), "The question?", "FastAPI")
        self.assertEqual([(t.sub_question, t.technology) for t in tasks], [("The question?", "FastAPI")])


def context(*turns) -> ConversationContext:
    built = [ConversationTurn(turn_number=i, query_text=q, resolved_query=r, answer_summary=a)
             for i, (q, r, a) in enumerate(turns, start=1)]
    return ConversationContext(session_id=1, turns=built, technologies_mentioned=["FastAPI"], total_turns=len(built))


FASTAPI_TURN = ("How do I install FastAPI?", "How do I install FastAPI?", "Install FastAPI with pip install fastapi.")


class UnderstandingTests(unittest.IsolatedAsyncioTestCase):
    async def understand(self, question, result, ctx=None, fail=False):
        from unittest import mock
        llm = ScriptedLLM(understanding_=result, fail=["QueryUnderstanding"] if fail else ())
        state = {"question": question}
        if ctx is not None:
            state["conversation_context"] = ctx
        with mock.patch.object(st, "llm_groq", llm):
            return await st.classify_node(state), llm

    async def test_a_follow_up_is_detected_and_resolved_from_history(self):
        out, llm = await self.understand(
            "deploy it", understanding(is_follow_up=True, resolved_query="deploy FastAPI", intent="technical_howto",
                                       technology="FastAPI", context_topics=["FastAPI"]), context(FASTAPI_TURN))
        self.assertTrue(out["is_follow_up"])
        self.assertEqual(out["resolved_query"], "deploy FastAPI")
        self.assertEqual(out["understanding"]["context_topics"], ["FastAPI"])
        prompt = llm.prompts_for("QueryUnderstanding")[0]
        self.assertIn("CONVERSATION SO FAR", prompt)
        self.assertIn("How do I install FastAPI?", prompt)
        self.assertIn("Install FastAPI with pip", prompt)

    async def test_a_comparison_follow_up_resolves(self):
        out, _ = await self.understand(
            "compare it to Django", understanding(is_follow_up=True, resolved_query="compare FastAPI to Django"),
            context(FASTAPI_TURN))
        self.assertEqual(out["resolved_query"], "compare FastAPI to Django")

    async def test_a_standalone_question_is_unchanged(self):
        out, _ = await self.understand("What is Docker?", understanding(is_follow_up=False, resolved_query="What is Docker?"),
                                       context(FASTAPI_TURN))
        self.assertFalse(out["is_follow_up"])
        self.assertEqual(out["resolved_query"], "What is Docker?")

    async def test_a_rewrite_of_a_non_follow_up_is_ignored(self):
        out, _ = await self.understand("What is Docker?", understanding(is_follow_up=False, resolved_query="What is FastAPI?"),
                                       context(FASTAPI_TURN))
        self.assertEqual(out["resolved_query"], "What is Docker?")

    async def test_the_first_turn_is_never_a_follow_up_even_if_the_model_says_so(self):
        out, llm = await self.understand("deploy it", understanding(is_follow_up=True, resolved_query="deploy FastAPI"))
        self.assertFalse(out["is_follow_up"])
        self.assertEqual(out["resolved_query"], "deploy it")
        self.assertNotIn("CONVERSATION SO FAR", llm.prompts_for("QueryUnderstanding")[0])

    async def test_no_context_means_the_resolved_query_is_the_original(self):
        out, _ = await self.understand("Best laptops", understanding(resolved_query="something else"))
        self.assertEqual(out["resolved_query"], "Best laptops")

    async def test_an_empty_or_runaway_rewrite_falls_back_to_the_question(self):
        for bad in ("", "x" * 600):
            out, _ = await self.understand("deploy it", understanding(is_follow_up=True, resolved_query=bad),
                                           context(FASTAPI_TURN))
            self.assertEqual(out["resolved_query"], "deploy it")

    async def test_invalid_values_are_normalised(self):
        out, _ = await self.understand("q", understanding(intent="???", time_sensitivity="soonish"))
        self.assertEqual((out["understanding"]["intent"], out["understanding"]["time_sensitivity"]), ("unknown", "unknown"))

    async def test_a_failing_model_degrades_to_complex_and_standalone(self):
        out, _ = await self.understand("deploy it", None, context(FASTAPI_TURN), fail=True)
        self.assertEqual((out["is_simple"], out["is_follow_up"], out["resolved_query"]), (False, False, "deploy it"))


class ContextAwarePlannerTests(unittest.IsolatedAsyncioTestCase):
    async def plan_prompt(self, ctx, resolved="add JWT authentication to FastAPI", original="add JWT to it"):
        from unittest import mock
        llm = ScriptedLLM()
        state = {"question": original, "resolved_query": resolved, "conversation_context": ctx,
                 "understanding": {"technology": "FastAPI"}}
        with mock.patch.object(st, "llm_groq", llm):
            out = await st.planner_node(state)
        return llm.prompts_for("SourceAwarePlan")[0], out

    async def test_a_follow_up_plan_uses_the_resolved_query_and_asks_for_new_information_only(self):
        prompt, out = await self.plan_prompt(context(FASTAPI_TURN))
        self.assertIn("Question: add JWT authentication to FastAPI", prompt)
        self.assertIn("original wording: add JWT to it", prompt)
        self.assertIn("Do not re-research what earlier turns already covered", prompt)
        self.assertIn("NEW information", prompt)
        self.assertIn("only use facts stated in it", prompt)
        self.assertIn("changes topic", prompt)  # topic change: independent sub-questions
        self.assertIn("How do I install FastAPI?", prompt)
        self.assertEqual(len(out["research_tasks"]), 2)

    async def test_without_history_there_are_no_conversation_rules(self):
        prompt, _ = await self.plan_prompt(None, resolved="What is Docker?", original="What is Docker?")
        self.assertNotIn("RULES FOR THE CONVERSATION", prompt)
        self.assertNotIn("CONVERSATION SO FAR", prompt)

    async def test_the_prompt_explains_what_each_source_is_for(self):
        prompt, _ = await self.plan_prompt(None)
        for phrase in ("official_docs: authoritative API", "github: code, repos", "reddit: real-world experience",
                       "web: general information"):
            self.assertIn(phrase, prompt)

    async def test_a_failing_planner_researches_the_resolved_question(self):
        from unittest import mock
        llm = ScriptedLLM(fail=["SourceAwarePlan"])
        with mock.patch.object(st, "llm_groq", llm):
            out = await st.planner_node({"question": "it?", "resolved_query": "deploy FastAPI",
                                         "understanding": {"technology": "FastAPI"}})
        self.assertEqual(out["sub_questions"], ["deploy FastAPI"])
        self.assertEqual(out["research_tasks"][0].technology, "FastAPI")


class PipelineSettingsTests(unittest.TestCase):
    def test_defaults(self):
        s = PipelineSettings.from_env({})
        self.assertEqual((s.concurrency_limit, s.gap_search_enabled, s.max_targeted_queries, s.max_critic_retries,
                          s.relevance_threshold), (6, True, 2, 1, 0.3))

    def test_values_and_fallbacks(self):
        s = PipelineSettings.from_env({"CONCURRENCY_LIMIT": "3", "GAP_SEARCH_ENABLED": "false", "MAX_TARGETED_QUERIES": "1",
                                       "MAX_CRITIC_RETRIES": "0", "RELEVANCE_THRESHOLD": "0.5"})
        self.assertEqual((s.concurrency_limit, s.gap_search_enabled, s.max_targeted_queries, s.max_critic_retries,
                          s.relevance_threshold), (3, False, 1, 0, 0.5))
        bad = PipelineSettings.from_env({"CONCURRENCY_LIMIT": "x", "MAX_CRITIC_RETRIES": "5", "RELEVANCE_THRESHOLD": "2"})
        self.assertEqual((bad.concurrency_limit, bad.max_critic_retries, bad.relevance_threshold), (6, 1, 0.3))


if __name__ == "__main__":
    unittest.main()
