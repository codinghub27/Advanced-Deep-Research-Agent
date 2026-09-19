"""Documentation-intent detection: technology match + technical cue, conservative,
rule-based, deterministic."""
import unittest

from research_app.sources.official_docs import DocsEntry, detect_docs_intent, get_registry
from research_app.sources.official_docs.detector import MAX_TECHNOLOGIES
from research_app.sources.official_docs.registry import DocsRegistry


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_registry.cache_clear()
        cls.registry = get_registry()

    def detect(self, text):
        return detect_docs_intent(text, self.registry)

    def assertRoutes(self, text, ids):
        intent = self.detect(text)
        self.assertTrue(intent.is_documentation_query, f"{text!r}: {intent.reason}")
        self.assertEqual(list(intent.technology_ids), ids, text)
        self.assertTrue(intent.cues, text)

    def assertNotRouted(self, text, reason_prefix=None):
        intent = self.detect(text)
        self.assertFalse(intent.is_documentation_query, f"{text!r} was routed to {intent.technology_ids}")
        self.assertEqual(intent.technologies, ())
        if reason_prefix:
            self.assertTrue(intent.reason.startswith(reason_prefix), (text, intent.reason))

    # ---- the required examples --------------------------------------------------

    def test_required_examples_route_to_official_docs(self):
        self.assertRoutes("How do I install FastAPI?", ["fastapi"])
        self.assertRoutes("How do I use LangGraph?", ["langgraph"])
        self.assertRoutes("How do I configure Qdrant?", ["qdrant"])
        self.assertRoutes("How does PostgreSQL CREATE INDEX work?", ["postgresql"])
        self.assertRoutes("How do I use Docker Compose?", ["docker-compose"])  # not also "docker"

    def test_required_non_examples_do_not_route(self):
        self.assertNotRouted("FastAPI vs Flask", "vetoed:versus")
        self.assertNotRouted("best FastAPI alternatives", "vetoed:")
        self.assertNotRouted("FastAPI popularity", "vetoed:popularity")
        self.assertNotRouted("best laptops 2026", "no_technology")

    def test_manual_test_queries(self):
        self.assertRoutes("What changed in Pydantic 2?", ["pydantic"])

    # ---- positives ----------------------------------------------------------------

    def test_more_documentation_questions(self):
        self.assertRoutes("What is the official syntax for PostgreSQL partial indexes?", ["postgresql"])
        self.assertRoutes("docker-compose.yml syntax for healthcheck", ["docker-compose"])
        self.assertRoutes("SQLAlchemy 2.0 migration guide", ["sqlalchemy"])
        self.assertRoutes("Alembic autogenerate configuration", ["alembic"])
        self.assertRoutes("How do I use the OpenAI API?", ["openai-api"])
        self.assertRoutes("Tavily search API parameters", ["tavily"])
        self.assertRoutes("uvicorn command line options", ["uvicorn"])
        self.assertRoutes("best practices for configuring Qdrant", ["qdrant"])  # "best practices" is not opinion
        self.assertRoutes("HOW DO I INSTALL FASTAPI", ["fastapi"])  # case-insensitive
        self.assertRoutes("what's new in LangChain", ["langchain"])
        self.assertRoutes("Docker release notes", ["docker"])

    def test_several_technologies_in_text_order_and_capped(self):
        self.assertRoutes("Set up Uvicorn with FastAPI", ["uvicorn", "fastapi"])
        many = "How do I install fastapi uvicorn pydantic sqlalchemy alembic"
        intent = self.detect(many)
        self.assertEqual(len(intent.technologies), MAX_TECHNOLOGIES)
        self.assertEqual(list(intent.technology_ids), ["fastapi", "uvicorn", "pydantic"])

    def test_docker_inside_docker_compose_is_not_a_second_technology(self):
        self.assertEqual(self.detect("How do I install docker compose").technology_ids, ("docker-compose",))
        self.assertEqual(self.detect("how to install docker and docker compose").technology_ids,
                         ("docker", "docker-compose"))

    # ---- weak aliases ---------------------------------------------------------------

    def test_weak_alias_needs_a_specific_cue_not_just_how_do_i(self):
        self.assertNotRouted("How do I use Python for data science?", "weak_technology_needs_specific_cue")
        self.assertNotRouted("How does Claude work?", "weak_technology_needs_specific_cue")
        self.assertRoutes("How to install Python 3.12 on Windows", ["python"])
        self.assertRoutes("compose file syntax", ["docker-compose"])
        self.assertRoutes("anthropic api usage", ["claude-api"])  # strong alias
        self.assertRoutes("openai api key configuration", ["openai-api"])

    # ---- negatives ------------------------------------------------------------------

    def test_technology_without_a_cue_is_not_routed(self):
        self.assertNotRouted("What is FastAPI?", "no_cue")
        self.assertNotRouted("Tell me about Qdrant", "no_cue")

    def test_cue_without_a_technology_is_not_routed(self):
        self.assertNotRouted("How do I install a package?", "no_technology")
        self.assertNotRouted("How do I use a Python package?", "weak_technology")  # only the weak 'python'

    def test_opinion_and_comparison_phrasing_is_vetoed_even_with_a_cue(self):
        for text in (
            "How does FastAPI compare to Flask?",
            "Is Docker Compose better than Kubernetes?",
            "Should I choose Qdrant or Pinecone?",
            "pros and cons of PostgreSQL",
            "top 10 FastAPI extensions",
            "FastAPI reviews",
            "How do I choose between FastAPI and Django?",
            "Which is faster, FastAPI or Flask?",
            "Is PostgreSQL worth it?",
        ):
            with self.subTest(text=text):
                self.assertNotRouted(text, "vetoed:")

    def test_names_only_match_on_word_boundaries(self):
        for text in ("how to install myfastapi", "how to install fastapiish", "how to install postgresqlx",
                     "install the dockerization tool"):
            with self.subTest(text=text):
                self.assertNotRouted(text, "no_technology")

    # ---- robustness -------------------------------------------------------------------

    def test_empty_and_non_string_input(self):
        for value in ("", "   ", None, 42, b"how to install fastapi", ["fastapi"]):
            with self.subTest(value=value):
                self.assertNotRouted(value, "empty")

    def test_very_long_input_is_bounded(self):
        text = "how to install fastapi " + "x " * 50_000
        self.assertRoutes(text, ["fastapi"])
        # A technology beyond the analysed prefix is not seen.
        self.assertNotRouted("how to install " + "x " * 5000 + "fastapi", "no_technology")

    def test_detection_is_deterministic(self):
        first = self.detect("How do I set up Uvicorn with FastAPI and Docker?")
        for _ in range(5):
            self.assertEqual(self.detect("How do I set up Uvicorn with FastAPI and Docker?"), first)

    def test_uses_the_registry_it_is_given(self):
        custom = DocsRegistry([DocsEntry(
            id="acme", name="Acme", aliases=["acme cloud"], sites=[{"host": "docs.acme.example"}])])
        self.assertTrue(detect_docs_intent("How do I install Acme Cloud?", custom).is_documentation_query)
        self.assertFalse(detect_docs_intent("How do I install FastAPI?", custom).is_documentation_query)
        self.assertFalse(detect_docs_intent("How do I install FastAPI?", DocsRegistry([])).is_documentation_query)


if __name__ == "__main__":
    unittest.main()
