"""Phase 5: the deterministic source router. Pure rules, no network, no LLM."""
import os
import unittest
from unittest import mock

from research_app.domain import SourceType
from research_app.sources.official_docs import DocsSettings, get_registry, mentioned_technologies
from research_app.sources.routing import (
    SOURCE_ORDER,
    WEB_ONLY,
    RouterSettings,
    plan_technology_docs,
    route_sources,
)

OD, GH, RD, WEB = SourceType.OFFICIAL_DOCS, SourceType.GITHUB, SourceType.REDDIT, SourceType.WEB

# The examples from the Phase 5 brief.
BRIEF_EXAMPLES = {
    "How do I configure FastAPI?": (OD, WEB),
    "Find GitHub examples of LangGraph agents": (GH, WEB),
    "What problems are developers facing with LangGraph?": (OD, GH, RD, WEB),
    "Why am I getting this LangGraph error?": (OD, GH, RD, WEB),
    "Best laptops for students": (WEB,),
    "Real-world experiences using Qdrant": (RD, WEB),
}
MORE_EXAMPLES = {
    # documentation
    "How do I install FastAPI?": (OD, WEB),
    "How do I configure Docker Compose?": (OD, WEB),
    "What changed in Pydantic version 2?": (OD, WEB),
    # code
    "Show me an open source implementation of a RAG pipeline": (GH, WEB),
    "Find repos for Qdrant clients": (GH, WEB),
    "How do I implement auth in FastAPI?": (OD, GH, WEB),
    "Sample code for a LangGraph agent": (GH, WEB),
    "Pull requests adding streaming to LangChain": (GH, WEB),
    # community
    "What do people think about LangGraph?": (RD, WEB),
    "What do people think about FastAPI vs Flask?": (RD, WEB),
    "Has anyone used Qdrant in production?": (RD, WEB),
    "Is Docker worth it for solo devs? asking reddit": (RD, WEB),
    "Discussion about r/LangChain sentiment": (RD, WEB),
    "What problems do people face with electric cars?": (RD, WEB),
    # troubleshooting
    "How do I fix ImportError in FastAPI?": (OD, GH, RD, WEB),
    "Getting a TypeError with pydantic validators": (OD, GH, RD, WEB),
    "Why am I getting a python error?": (GH, RD, WEB),  # weak technology: no docs lookup
    "My script throws an exception and a traceback": (GH, RD, WEB),
    # nothing specialised
    "What is the capital of France?": (WEB,),
    "Compare offline test frameworks": (WEB,),
    "FastAPI vs Flask": (WEB,),
    "best FastAPI alternatives": (WEB,),
    "FastAPI popularity": (WEB,),
    "What is FastAPI?": (WEB,),
    "Why am I getting this error?": (WEB,),  # an error word with nothing technical
    "How do I fix my sleep schedule?": (WEB,),
    "Latest news about OpenAI": (WEB,),
    "best open source laptops for students": (GH, WEB),  # documented limitation: "open source" wording
}


class RouterTestBase(unittest.TestCase):
    """A clean environment and registry cache for every test."""

    def setUp(self):
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in [k for k in os.environ if k.startswith(("OFFICIAL_DOCS_", "SOURCE_ROUTER_"))]:
            del os.environ[key]
        get_registry.cache_clear()
        self.addCleanup(get_registry.cache_clear)


class RoutingTests(RouterTestBase):
    def assertRoutes(self, table):
        for question, expected in table.items():
            with self.subTest(question=question):
                self.assertEqual(route_sources(question).sources, expected)

    def test_the_examples_from_the_brief(self):
        self.assertRoutes(BRIEF_EXAMPLES)

    def test_more_questions(self):
        self.assertRoutes(MORE_EXAMPLES)

    def test_web_is_always_selected_and_order_is_fixed(self):
        for question in {**BRIEF_EXAMPLES, **MORE_EXAMPLES}:
            with self.subTest(question=question):
                plan = route_sources(question)
                self.assertIn(WEB, plan.sources)
                self.assertEqual(plan.sources, tuple(s for s in SOURCE_ORDER if s in plan.sources))
                self.assertEqual(set(plan.reasons), set(plan.sources))
                self.assertEqual(plan.reasons[WEB], "always")

    def test_not_every_source_for_every_query(self):
        plans = [route_sources(q) for q in {**BRIEF_EXAMPLES, **MORE_EXAMPLES}]
        self.assertTrue(any(len(p.sources) == 1 for p in plans))
        for question in ("How do I configure FastAPI?", "Find GitHub examples of LangGraph agents",
                         "Real-world experiences using Qdrant", "Best laptops for students"):
            self.assertLess(len(route_sources(question).sources), 4, question)

    def test_reasons_explain_each_choice(self):
        plan = route_sources("Why am I getting this LangGraph error?")
        self.assertEqual(plan.reasons[OD], "technology_problem:langgraph")
        self.assertEqual(plan.reasons[GH], "technical_problem")
        self.assertEqual(plan.reasons[RD], "problem_or_troubleshooting")
        self.assertEqual(route_sources("How do I install FastAPI?").reasons[OD], "documentation_query:fastapi")
        self.assertEqual(route_sources("find GitHub examples").reasons[GH], "explicit_code_or_github_wording")
        self.assertEqual(route_sources("Real-world experiences using Qdrant").reasons[RD], "community_experience")

    def test_docs_from_technology_is_only_set_when_the_phase_4_detector_said_no(self):
        self.assertTrue(route_sources("Why am I getting this LangGraph error?").docs_from_technology)
        self.assertTrue(route_sources("What problems are developers facing with LangGraph?").docs_from_technology)
        self.assertFalse(route_sources("How do I fix ImportError in FastAPI?").docs_from_technology)  # detector: how-to
        self.assertFalse(route_sources("How do I configure FastAPI?").docs_from_technology)
        self.assertFalse(route_sources("Real-world experiences using Qdrant").docs_from_technology)

    def test_case_and_punctuation_do_not_matter(self):
        self.assertEqual(route_sources("FIND GITHUB EXAMPLES OF LANGGRAPH AGENTS").sources, (GH, WEB))
        self.assertEqual(route_sources("real   world experiences,   using QDRANT!!").sources, (RD, WEB))
        self.assertEqual(route_sources("Why am I getting this LangGraph   error???").sources, (OD, GH, RD, WEB))

    def test_garbage_input_is_web_only_and_never_raises(self):
        for value in (None, "", "   ", "\n\t", 5, [], "😀" * 50, "x" * 100_000, "\x00\x01"):
            with self.subTest(value=repr(value)[:20]):
                self.assertEqual(route_sources(value).sources, (WEB,))

    def test_only_the_beginning_of_a_huge_question_is_examined(self):
        self.assertEqual(route_sources("filler " * 1000 + "find GitHub examples").sources, (WEB,))
        self.assertEqual(route_sources("find GitHub examples " + "filler " * 1000).sources, (GH, WEB))

    def test_web_only_constant(self):
        self.assertEqual((WEB_ONLY.sources, WEB_ONLY.docs_from_technology), ((WEB,), False))
        self.assertTrue(WEB_ONLY.includes(WEB) and not WEB_ONLY.includes(GH))


class KillSwitchTests(RouterTestBase):
    def test_router_disabled_keeps_exactly_the_phase_4_sources(self):
        os.environ["SOURCE_ROUTER_ENABLED"] = "false"
        expected = {
            "How do I configure FastAPI?": (OD, WEB),  # Phase 4 documentation detection stays on
            "Find GitHub examples of LangGraph agents": (WEB,),
            "Real-world experiences using Qdrant": (WEB,),
            "Why am I getting this LangGraph error?": (WEB,),
            "What problems are developers facing with LangGraph?": (WEB,),
            "Best laptops for students": (WEB,),
        }
        for question, sources in expected.items():
            with self.subTest(question=question):
                plan = route_sources(question)
                self.assertEqual(plan.sources, sources)
                self.assertFalse(plan.docs_from_technology)

    def test_official_docs_disabled_removes_documentation_only(self):
        os.environ["OFFICIAL_DOCS_ENABLED"] = "false"
        expected = {
            "How do I configure FastAPI?": (WEB,),
            "Why am I getting this LangGraph error?": (GH, RD, WEB),
            "Find GitHub examples of LangGraph agents": (GH, WEB),
            "Real-world experiences using Qdrant": (RD, WEB),
        }
        for question, sources in expected.items():
            with self.subTest(question=question):
                self.assertEqual(route_sources(question).sources, sources)

    def test_explicit_settings_override_the_environment(self):
        plan = route_sources("Find GitHub examples of LangGraph agents", RouterSettings(enabled=False), DocsSettings())
        self.assertEqual(plan.sources, (WEB,))

    def test_a_bad_registry_file_still_routes_with_the_builtin_registry(self):
        os.environ["OFFICIAL_DOCS_REGISTRY_PATH"] = os.path.join("no", "such", "file.json")
        with self.assertLogs("research_app.sources.official_docs.registry", level="ERROR"):
            plan = route_sources("Why am I getting this LangGraph error?")
        self.assertEqual(plan.sources, (OD, GH, RD, WEB))


class SettingsTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(RouterSettings.from_env({}), RouterSettings(enabled=True, timeout_s=15.0))

    def test_enabled_flag(self):
        for value in ("false", "FALSE", "0", "no", "off", " Off "):
            self.assertFalse(RouterSettings.from_env({"SOURCE_ROUTER_ENABLED": value}).enabled, value)
        for value in ("true", "1", "yes", "", "garbage"):
            self.assertTrue(RouterSettings.from_env({"SOURCE_ROUTER_ENABLED": value}).enabled, value)

    def test_timeout_parsing(self):
        self.assertEqual(RouterSettings.from_env({"SOURCE_ROUTER_TIMEOUT_S": "7.5"}).timeout_s, 7.5)
        for bad in ("0", "0.5", "61", "-3", "abc", "nan", "inf"):
            with self.subTest(value=bad), self.assertLogs("research_app.sources.routing.settings", level="WARNING"):
                self.assertEqual(RouterSettings.from_env({"SOURCE_ROUTER_TIMEOUT_S": bad}).timeout_s, 15.0)


class TechnologyDocsPlanTests(unittest.TestCase):
    def setUp(self):
        get_registry.cache_clear()
        self.addCleanup(get_registry.cache_clear)

    def test_plan_searches_the_registered_hosts_with_the_query_given(self):
        plan = plan_technology_docs("Why am I getting this LangGraph error?", "langgraph error handling",
                                    DocsSettings(timeout_s=7))
        self.assertEqual(plan.request.query, "langgraph error handling")
        self.assertEqual(plan.request.source_type, OD)
        self.assertEqual(plan.request.timeout_s, 7)
        self.assertIn("docs.langchain.com", plan.request.include_domains)

    def test_no_technology_or_disabled_means_no_plan(self):
        self.assertIsNone(plan_technology_docs("Why am I getting this error?", "q", DocsSettings()))
        self.assertIsNone(plan_technology_docs("LangGraph error", "q", DocsSettings(enabled=False)))
        self.assertIsNone(plan_technology_docs("Why is my python script slow?", "q", DocsSettings()))  # weak alias

    def test_mentioned_technologies_helper(self):
        registry = get_registry()
        ids = lambda text, **kw: tuple(t.id for t in mentioned_technologies(text, registry, **kw))  # noqa: E731
        self.assertEqual(ids("LangGraph error in FastAPI"), ("langgraph", "fastapi"))
        self.assertEqual(ids("a python error"), ())
        self.assertEqual(ids("a python error", include_weak=True), ("python",))
        self.assertEqual(ids("docker compose up fails"), ("docker-compose",))  # not also 'docker'
        for value in (None, "", "   ", 5):
            self.assertEqual(mentioned_technologies(value, registry), ())


if __name__ == "__main__":
    unittest.main()
