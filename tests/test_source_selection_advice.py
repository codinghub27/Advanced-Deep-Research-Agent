"""Source selection for advice/review questions, and the failure modes seen on one:
"review my LangGraph/Tavily/FastAPI agent, is it worth building, does it help in interviews?"
went to official docs + GitHub (the words were only DESCRIBING the project) and the summary was
reported as a model failure. Pure logic plus the synthesis step with a scripted LLM; no network."""
import unittest
from unittest import mock

from research_app.agent import state as st
from research_app.agent.pipeline.evidence import EvidencePool
from research_app.agent.pipeline.synthesis import run_synthesis
from research_app.domain import ResearchTask, SourceDocument, SourceIntent, SourceType
from research_app.sources.routing.router import route_sources
from research_app.sources.routing.task_router import route_task
from tests.p6_helpers import ScriptedLLM, understanding

DOCS, GITHUB, REDDIT, WEB = SourceType.OFFICIAL_DOCS, SourceType.GITHUB, SourceType.REDDIT, SourceType.WEB

REVIEW = ("im planning to build an advanced research agent: it's a LangGraph-based research agent that breaks a "
          "question into sub-questions, searches across the web, GitHub, and Reddit via Tavily, and streams over "
          "FastAPI/SSE. give honest review of it and is it worth to build this type of projects for a b.tech aids "
          "4th year student and is it help in interviews?")
ADVICE = {"intent": "advice", "technology": "", "resolved_query": REVIEW}


def task(text, intent, suggested, technology=None):
    return ResearchTask(sub_question=text, source_intent=intent, suggested_sources=suggested, technology=technology)


class DescriptiveMentionTests(unittest.TestCase):
    def test_describing_your_own_system_is_not_a_request_to_search_those_sites(self):
        self.assertEqual(route_sources(REVIEW).names, ("web",))
        self.assertEqual(route_sources("my agent searches Reddit and GitHub, how do I add caching").names, ("web",))
        self.assertEqual(route_sources("the tool scrapes reddit for leads and stores them").names, ("web",))

    def test_asking_to_search_them_still_counts(self):
        self.assertEqual(route_sources("search GitHub and Reddit for LangGraph memory issues").names,
                         ("github", "reddit", "web"))
        self.assertIn("reddit", route_sources("what do people on reddit say about FastAPI vs Django").names)
        self.assertIn("github", route_sources("find github repos for langgraph agents").names)


class AdviceRoutingTests(unittest.TestCase):
    def test_an_advice_sub_question_goes_to_reddit_and_web_only(self):
        d = route_task(task("Is building a LangGraph research agent a good final-year project for AI students",
                            "community_experience", ["official_docs", "github", "reddit", "web"], "LangGraph"), ADVICE)
        self.assertEqual(d.sources, [REDDIT, WEB])
        self.assertEqual(d.origins, {"reddit": "planner", "web": "planner"})
        self.assertEqual({(x.source, x.reason) for x in d.dropped},
                         {("official_docs", "advice_question"), ("github", "advice_question")})

    def test_reddit_and_web_are_added_even_if_the_planner_only_suggested_web(self):
        d = route_task(task("Do interviewers value RAG and agent projects on a fresher's resume", "general_research", ["web"]), ADVICE)
        self.assertEqual((d.sources, d.origins), ([REDDIT, WEB], {"reddit": "policy", "web": "planner"}))

    def test_a_technical_sub_question_of_an_advice_question_is_routed_normally(self):
        d = route_task(task("How do I install LangGraph and Tavily", "technical_howto", ["official_docs", "web"], "LangGraph"), ADVICE)
        self.assertIn(DOCS, d.sources)
        self.assertNotIn("advice_question", [x.reason for x in d.dropped])

    def test_the_same_sub_question_without_the_advice_intent_keeps_its_old_routing(self):
        sub = task("Is building a LangGraph research agent worth it", "comparison", ["official_docs", "github", "web"], "LangGraph")
        old = route_task(sub, {"intent": "research", "technology": "LangGraph"})
        self.assertIn(DOCS, old.sources)          # unchanged behaviour outside advice questions
        self.assertIn(GITHUB, old.sources)

    def test_the_router_kill_switch_still_wins(self):
        with mock.patch.dict("os.environ", {"SOURCE_ROUTER_ENABLED": "false"}):
            d = route_task(task("Is it worth it", "community_experience", ["reddit"]), ADVICE)
        self.assertNotEqual(d.sources, [REDDIT, WEB])


class AdviceUnderstandingTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_classifier_can_say_advice_and_is_told_what_it_means(self):
        llm = ScriptedLLM(understanding_=understanding(intent="advice", technology="", time_sensitivity="evergreen"))
        with mock.patch.object(st, "llm_groq", llm):
            out = await st.classify_node({"question": REVIEW})
        self.assertEqual(out["understanding"]["intent"], "advice")
        prompt = llm.prompts_for("QueryUnderstanding")[0]
        self.assertIn("advice", prompt)
        self.assertIn("THEIR OWN plan", prompt)

    async def test_the_planner_gets_the_advice_rules_only_for_advice_questions(self):
        async def planner_prompt(und):
            llm = ScriptedLLM()
            with mock.patch.object(st, "llm_groq", llm):
                await st.planner_node({"question": REVIEW, "resolved_query": REVIEW, "understanding": und})
            return llm.prompts_for("SourceAwarePlan")[0]
        advice = await planner_prompt(ADVICE)
        self.assertIn("RULES FOR THIS ADVICE QUESTION", advice)
        self.assertIn("Do not plan documentation or GitHub searches", advice)
        self.assertNotIn("RULES FOR THIS ADVICE QUESTION", await planner_prompt({"intent": "research"}))

    def test_a_simple_advice_question_becomes_a_community_task(self):
        self.assertEqual(st._SOURCE_INTENT_FOR["advice"], SourceIntent.COMMUNITY_EXPERIENCE)


class QueryLengthTests(unittest.TestCase):
    def test_a_long_question_is_cut_to_what_tavily_accepts_at_a_word_boundary(self):
        fitted = st._fit_query(REVIEW * 3)
        self.assertLessEqual(len(fitted), 400)
        self.assertTrue((REVIEW * 3).startswith(fitted.rsplit(" ", 1)[0]))
        self.assertFalse(fitted.endswith(" "))

    def test_a_short_query_is_unchanged(self):
        self.assertEqual(st._fit_query("latest AI news"), "latest AI news")


class SynthesisFailureTests(unittest.IsolatedAsyncioTestCase):
    def pool(self):
        t = ResearchTask(sub_question="is a research agent a good student project")
        docs = [SourceDocument(url=f"https://site{i}.example/p", title=f"Page {i}", task_id=t.task_id,
                               content="research agent student project interviews portfolio") for i in (1, 2)]
        pool = EvidencePool([t])
        pool.add_documents(docs)
        return pool

    async def synth(self, llm):
        with mock.patch.object(st, "llm_groq", llm):
            return await run_synthesis({"evidence_pool": self.pool(), "question": "is it worth it", "understanding": {}})

    async def test_a_short_uncited_reply_is_kept_and_the_sources_are_listed(self):
        result = await self.synth(ScriptedLLM(answer="The sources do not cover this."))
        self.assertTrue(result.answer_text.startswith("The sources do not cover this."))
        self.assertIn("Sources found:", result.answer_text)
        self.assertNotIn("did not respond", result.answer_text)
        self.assertEqual(result.confidence, "low")
        self.assertEqual(len(result.citations), 2)

    async def test_a_short_reply_that_cites_a_source_is_just_a_short_answer(self):
        result = await self.synth(ScriptedLLM(answer="Yes, it is common [1]."))
        self.assertNotIn("Sources found:", result.answer_text)
        self.assertEqual([c.index for c in result.citations], [1])

    async def test_an_empty_reply_is_a_failure_and_says_why_in_the_log(self):
        with self.assertLogs("research_app.agent.pipeline.synthesis", "WARNING") as logs:
            result = await self.synth(ScriptedLLM(answer="   "))
        self.assertIn("did not respond", result.answer_text)
        self.assertTrue(any("empty answer" in line for line in logs.output))

    async def test_a_model_error_is_logged_with_its_message(self):
        with self.assertLogs("research_app.agent.pipeline.synthesis", "WARNING") as logs:
            result = await self.synth(ScriptedLLM(fail=("synthesis",)))
        self.assertIn("did not respond", result.answer_text)
        self.assertTrue(any("RuntimeError: synthesis failed" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
