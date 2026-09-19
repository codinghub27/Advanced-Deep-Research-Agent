"""Phase 6, Part B: evidence pool, gap detection, synthesis, critic, retry and the whole pipeline
through the compiled graph. Tavily and the LLM are fakes."""
import json
import unittest
from unittest import mock

from research_app.agent import state as st
from research_app.agent.pipeline import nodes
from research_app.agent.pipeline.critic import run_critic, worth_retrying
from research_app.agent.pipeline.evidence import build_pool
from research_app.agent.pipeline.gaps import detect_gaps, keyword_query
from research_app.agent.pipeline.response import build_response
from research_app.agent.pipeline.synthesis import (
    build_sources, find_markers, normalize_markers, run_synthesis, strip_orphan_markers,
)
from research_app.domain import (
    Citation, CriticIssue, CriticIssueKind, CriticResult, CriticSeverity, CriticVerdict, ResearchTask,
    SourceDocument, SourceType, SynthesisResult,
)
from tests.p6_helpers import (
    GOOD_ANSWER, ScriptedLLM, critic_report, plan, run_graph, stream_graph, task, understanding,
)
from tests.test_router_nodes import DOCS_OK, GITHUB_OK, REDDIT_OK, FakeTavily

DOCS, GH, RD, WEB = SourceType.OFFICIAL_DOCS, SourceType.GITHUB, SourceType.REDDIT, SourceType.WEB
EMPTY = {"results": []}
PAD = " This sentence pads the answer so it is long enough to count as a real answer."
WEB_RELEVANT = {"results": [
    {"title": "Install LangGraph", "url": "https://blog.example/install-langgraph",
     "content": "Install LangGraph with pip and build your first agent.", "score": 0.9},
    {"title": "LangGraph agent examples", "url": "https://blog.example/agent-examples",
     "content": "Agent examples on GitHub for LangGraph.", "score": 0.8}]}


def doc(url, content, task_id=None, source_type=WEB, title="T", **kw):
    return SourceDocument(url=url, content=content, task_id=task_id, source_type=source_type, title=title, **kw)


def numbered_pool(*docs, tasks=()):
    return build_pool(list(tasks), docs)


# --------------------------------------------------------------------------- evidence

class EvidencePoolTests(unittest.TestCase):
    def setUp(self):
        self.a = ResearchTask(sub_question="install fastapi with uvicorn")
        self.b = ResearchTask(sub_question="fastapi community opinions")

    def test_collects_evidence_from_several_sub_questions(self):
        pool = build_pool([self.a, self.b], [
            doc("https://x.example/1", "Install FastAPI and run uvicorn", self.a.task_id),
            doc("https://y.example/2", "Developers share opinions about FastAPI community", self.b.task_id, RD)])
        self.assertEqual(len(pool), 2)
        self.assertEqual(sorted(pool.by_subquestion()), sorted([self.a.task_id, self.b.task_id]))
        self.assertEqual(pool.coverage(), ([self.a.task_id, self.b.task_id], []))

    def test_exact_duplicates_are_removed_but_the_link_to_the_second_task_is_kept(self):
        same = ("https://x.example/1", "Install FastAPI and fastapi opinions")
        pool = build_pool([self.a, self.b], [doc(*same, self.a.task_id), doc(*same, self.b.task_id)])
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool.coverage()[1], [])  # both sub-questions count it as evidence

    def test_same_url_with_different_content_is_kept(self):
        pool = build_pool([self.a], [doc("https://x.example/1", "Install FastAPI part one", self.a.task_id),
                                     doc("https://x.example/1", "Uvicorn part two", self.a.task_id)])
        self.assertEqual(len(pool), 2)

    def test_a_sub_question_with_no_evidence_is_a_gap(self):
        pool = build_pool([self.a, self.b], [doc("https://x.example/1", "Install FastAPI with uvicorn", self.a.task_id)])
        self.assertEqual(pool.coverage(), ([self.a.task_id], [self.b.task_id]))

    def test_irrelevant_text_is_not_evidence_for_a_sub_question(self):
        pool = build_pool([self.a], [doc("https://x.example/1", "Best pizza recipes in town", self.a.task_id)])
        self.assertEqual(pool.coverage(), ([], [self.a.task_id]))
        self.assertEqual(len(pool), 1)  # still available to synthesis

    def test_empty_results_are_handled(self):
        pool = build_pool([self.a], [])
        self.assertEqual((len(pool), pool.all_evidence(), pool.coverage()), (0, [], ([], [self.a.task_id])))
        self.assertEqual(len(build_pool([], [])), 0)

    def test_documents_without_text_are_not_evidence(self):
        pool = build_pool([self.a], [doc("https://x.example/1", "", self.a.task_id)])
        self.assertEqual(len(pool), 0)

    def test_evidence_is_ordered_official_first(self):
        pool = build_pool([self.a], [doc("https://w.example/", "install fastapi", self.a.task_id, WEB),
                                     doc("https://d.example/", "install fastapi", self.a.task_id, DOCS),
                                     doc("https://g.example/", "install fastapi", self.a.task_id, GH)])
        self.assertEqual([d.source_type for d in pool.all_evidence()], [DOCS, GH, WEB])


# --------------------------------------------------------------------------- gaps

class GapDetectionTests(unittest.TestCase):
    def setUp(self):
        self.a = ResearchTask(sub_question="how do I install fastapi with uvicorn?", technology="FastAPI")
        self.b = ResearchTask(sub_question="what are the best practices for structuring large projects")
        self.tasks = [self.a, self.b]

    def test_all_covered_means_no_gap_and_no_follow_up(self):
        pool = build_pool(self.tasks, [doc("https://x.example/1", "install fastapi uvicorn", self.a.task_id),
                                       doc("https://x.example/2", "best practices structuring large projects", self.b.task_id)])
        analysis = detect_gaps(pool, self.tasks)
        self.assertTrue(analysis.sufficient)
        self.assertEqual(analysis.follow_up_tasks, [])
        self.assertEqual(nodes.gap_router({"gap_analysis": analysis}), "synthesis_node")

    def test_a_missing_sub_question_produces_a_different_keyword_follow_up(self):
        pool = build_pool(self.tasks, [doc("https://x.example/1", "install fastapi uvicorn", self.a.task_id)])
        analysis = detect_gaps(pool, self.tasks)
        self.assertFalse(analysis.sufficient)
        (follow_up,) = analysis.follow_up_tasks
        self.assertEqual(follow_up.task_id, self.b.task_id)  # the evidence will land on the right sub-question
        self.assertNotEqual(follow_up.search_query, self.b.sub_question)
        self.assertIn("practices", follow_up.search_query)
        self.assertEqual(nodes.gap_router({"gap_analysis": analysis}), "targeted_search")

    def test_follow_ups_are_capped(self):
        many = [ResearchTask(sub_question=f"unrelated topic number {w}") for w in ("alpha", "bravo", "charlie", "delta")]
        analysis = detect_gaps(build_pool(many, []), many, max_queries=2)
        self.assertEqual(len(analysis.follow_up_tasks), 2)
        self.assertEqual(len([g for g in analysis.gaps if g.kind.value == "missing_subquestion"]), 4)

    def test_a_query_that_was_already_run_is_not_repeated(self):
        pool = build_pool(self.tasks, [])
        first = detect_gaps(pool, self.tasks)
        again = detect_gaps(pool, self.tasks, already_run=[t.search_query for t in first.follow_up_tasks])
        self.assertEqual(again.follow_up_tasks, [])

    def test_a_single_source_type_is_a_soft_gap_that_triggers_nothing(self):
        from research_app.domain import RoutingDecision
        pool = build_pool([self.a], [doc("https://x.example/1", "install fastapi uvicorn", self.a.task_id, WEB)])
        decision = RoutingDecision(task_id=self.a.task_id, sub_question=self.a.sub_question, sources=[DOCS, WEB])
        analysis = detect_gaps(pool, [self.a], decisions=[decision])
        self.assertTrue(analysis.sufficient)
        self.assertEqual([g.kind.value for g in analysis.gaps], ["weak_evidence"])
        self.assertEqual(analysis.follow_up_tasks, [])

    def test_keyword_query_keeps_key_terms_in_order(self):
        self.assertEqual(keyword_query("How do I install FastAPI with uvicorn?"), "install fastapi uvicorn")
        self.assertEqual(keyword_query("configure workers", "Gunicorn"), "Gunicorn configure workers")

    def test_the_gap_round_can_be_disabled(self):
        with mock.patch.dict("os.environ", {"GAP_SEARCH_ENABLED": "false"}):
            out = nodes.gap_detection_node({"evidence_pool": build_pool(self.tasks, []), "research_tasks": self.tasks})
        self.assertEqual(out["gap_analysis"].follow_up_tasks, [])


# --------------------------------------------------------------------------- synthesis

class MarkerTests(unittest.TestCase):
    def test_grouped_markers_are_normalised_and_code_is_left_alone(self):
        text = "See [1, 2] and [3].\n\n```py\nx = a[1, 2]\n```\n\nInline `b[4]` too."
        self.assertEqual(normalize_markers(text), "See [1][2] and [3].\n\n```py\nx = a[1, 2]\n```\n\nInline `b[4]` too.")
        self.assertEqual(find_markers(text), [1, 2, 3])

    def test_markdown_links_and_reference_definitions_are_not_citations(self):
        self.assertEqual(find_markers("[1](https://x.example) and\n[2]: https://y.example"), [])

    def test_orphan_markers_are_stripped_only_where_they_name_no_source(self):
        text, removed = strip_orphan_markers("Real claim [1]. Fake claim [9]. Code `a[9]`.", {1})
        self.assertEqual((text, removed), ("Real claim [1]. Fake claim. Code `a[9]`.", 1))


class SynthesisTests(unittest.IsolatedAsyncioTestCase):
    def tasks_and_pool(self, *docs):
        t = ResearchTask(sub_question="install langgraph", source_intent="technical_howto")
        docs = [d.model_copy(update={"task_id": t.task_id}) for d in docs]
        return t, build_pool([t], docs)

    async def synth(self, pool, llm, understanding_=None, **state):
        with mock.patch.object(st, "llm_groq", llm):
            return await run_synthesis({"evidence_pool": pool, "question": "How do I install LangGraph?",
                                        "understanding": understanding_ or {}, **state})

    def three_sources(self):
        return self.tasks_and_pool(
            doc("https://web.example/a", "install langgraph web body", source_type=WEB, title="Web A"),
            doc("https://docs.langchain.com/x", "install langgraph official body", source_type=DOCS, title="Docs", technology="LangGraph"),
            doc("https://github.com/o/r", "install langgraph github body", source_type=GH, title="Repo"))

    async def test_answer_has_markers_and_a_matching_citation_list(self):
        _, pool = self.three_sources()
        llm = ScriptedLLM(answer="Install it with pip [1]. The repo shows an example [2]. More [3]." + PAD)
        result = await self.synth(pool, llm)
        self.assertEqual(find_markers(result.answer_text), [1, 2, 3])
        self.assertEqual([c.index for c in result.citations], [1, 2, 3])
        self.assertEqual([c.marker for c in result.citations], ["[1]", "[2]", "[3]"])
        self.assertIn(result.confidence, ("high", "medium", "low"))
        for c in result.citations:
            self.assertTrue(c.url and c.domain and c.retrieved_at)

    async def test_official_documentation_is_source_one_and_the_prompt_says_to_trust_it(self):
        _, pool = self.three_sources()
        llm = ScriptedLLM(answer="Official [1]." + PAD)
        result = await self.synth(pool, llm, {"intent": "technical_howto"})
        prompt = llm.prompts[0]
        self.assertLess(prompt.index("[1] [OFFICIAL DOCUMENTATION: LangGraph]"), prompt.index("[GITHUB"))
        self.assertIn("OFFICIAL DOCUMENTATION GUIDANCE", prompt)
        self.assertIn("SOURCE LABEL GUIDANCE", prompt)
        self.assertIn("official documentation over web over GitHub/Reddit", prompt)  # conflicts
        self.assertEqual(result.citations[0].source_type, DOCS)

    async def test_markers_are_renumbered_by_first_use(self):
        _, pool = self.three_sources()
        result = await self.synth(pool, ScriptedLLM(answer="First the repo [2]. Then the docs [1][2]." + PAD))
        self.assertTrue(result.answer_text.startswith("First the repo [1]. Then the docs [2][1]."))
        self.assertEqual([(c.index, c.source_type) for c in result.citations], [(1, GH), (2, DOCS)])

    async def test_a_marker_for_a_source_that_does_not_exist_stays_for_the_critic(self):
        _, pool = self.three_sources()
        result = await self.synth(pool, ScriptedLLM(answer="Solid claim [1]. Invented claim [9]." + PAD))
        self.assertEqual(find_markers(result.answer_text), [1, 9])
        self.assertEqual(len(result.citations), 1)

    async def test_conflicting_sources_do_not_break_synthesis(self):
        _, pool = self.tasks_and_pool(
            doc("https://docs.langchain.com/x", "install langgraph requires python 3.10", source_type=DOCS, technology="LangGraph"),
            doc("https://reddit.example/r", "install langgraph works on python 3.8", source_type=RD))
        result = await self.synth(pool, ScriptedLLM(answer="The official docs require Python 3.10 [1]; a Reddit post disagrees [2]." + PAD))
        self.assertEqual(len(result.citations), 2)

    async def test_empty_evidence_gives_low_confidence_and_an_acknowledgment_without_calling_the_model(self):
        t = ResearchTask(sub_question="install langgraph")
        llm = ScriptedLLM()
        result = await self.synth(build_pool([t], []), llm)
        self.assertEqual((result.confidence, result.citations, llm.prompts), ("low", [], []))
        self.assertIn("could not find reliable sources", result.answer_text)
        self.assertEqual(result.subquestions_missed, ["install langgraph"])

    async def test_a_failing_model_returns_the_source_list_with_low_confidence(self):
        _, pool = self.three_sources()
        result = await self.synth(pool, ScriptedLLM(fail=["synthesis"]))
        self.assertEqual(result.confidence, "low")
        self.assertIn("could not write the summary", result.answer_text)
        self.assertTrue(result.citations)

    async def test_code_blocks_keep_their_indentation(self):
        _, pool = self.three_sources()
        answer = "Run this [1]:\n\n```python\ndef f():\n    return  1\n```\n" + PAD
        result = await self.synth(pool, ScriptedLLM(answer=answer))
        self.assertIn("    return  1", result.answer_text)

    async def test_covered_and_missed_sub_questions_are_reported(self):
        a = ResearchTask(sub_question="install langgraph")
        b = ResearchTask(sub_question="quantum error correction")
        pool = build_pool([a, b], [doc("https://x.example/1", "install langgraph guide", a.task_id)])
        result = await self.synth(pool, ScriptedLLM(answer="Guide [1]." + PAD))
        self.assertEqual((result.subquestions_covered, result.subquestions_missed),
                         (["install langgraph"], ["quantum error correction"]))

    async def test_only_the_source_window_is_sent_and_it_is_shared_across_types(self):
        t = ResearchTask(sub_question="install langgraph")
        docs = [doc(f"https://w{i}.example/", f"install langgraph {i}", t.task_id, WEB) for i in range(12)]
        docs.append(doc("https://github.com/o/r", "install langgraph repo", t.task_id, GH))
        sources = build_sources(build_pool([t], docs))
        self.assertEqual(len(sources), 12)
        self.assertIn(GH, [s.source_type for s in sources])

    def test_confidence_rules(self):
        from research_app.agent.pipeline.synthesis import assess_confidence
        c = lambda t, d: Citation(marker="[1]", source_id="s", url="https://x.example", source_type=t, domain=d)
        cites = [c(DOCS, "a.com"), c(WEB, "b.com"), c(WEB, "c.com")]
        self.assertEqual(assess_confidence(cites, [], technical=True), "high")
        self.assertEqual(assess_confidence(cites[1:] + [c(WEB, "d.com")], [], technical=True), "medium")  # no official docs
        self.assertEqual(assess_confidence(cites, ["missed"], technical=False), "medium")
        self.assertEqual(assess_confidence([], [], technical=False), "low")
        self.assertEqual(assess_confidence(cites, [], technical=False, fallback=True), "low")


# --------------------------------------------------------------------------- critic

def citation(i, t=DOCS, domain="d.example"):
    return Citation(marker=f"[{i}]", index=i, source_id=f"s{i}", url=f"https://{domain}/{i}", title=f"T{i}",
                    source_type=t, domain=domain)


def synthesis(text, cites, missed=(), covered=("q",), confidence="medium"):
    return SynthesisResult(answer_text=text, citations=cites, confidence=confidence,
                           subquestions_covered=list(covered), subquestions_missed=list(missed))


class CriticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.t = ResearchTask(sub_question="install langgraph")
        self.pool = build_pool([self.t], [
            doc("https://d.example/1", "install langgraph docs", self.t.task_id, DOCS, technology="LangGraph"),
            doc("https://d.example/2", "install langgraph more", self.t.task_id, DOCS),
            doc("https://d.example/3", "install langgraph three", self.t.task_id, DOCS)])

    async def critique(self, syn, report=None, fail=False, understanding_=None):
        llm = ScriptedLLM(critic=report or critic_report(), fail=["CriticReport"] if fail else ())
        with mock.patch.object(st, "llm_groq", llm):
            return await run_critic({"evidence_pool": self.pool, "question": "How do I install LangGraph?",
                                     "understanding": understanding_ or {}}, syn)

    async def test_a_good_answer_is_good(self):
        text = "Install it with pip [1]. Configure it in code [2]. Run it [3]."
        result = await self.critique(synthesis(text, [citation(1), citation(2), citation(3)]))
        self.assertEqual((result.verdict, result.passed, result.issues), (CriticVerdict.GOOD, True, []))

    async def test_an_orphan_citation_is_flagged(self):
        text = "Claim [1]. Another [2]. Third [3]. Invented [5]."
        result = await self.critique(synthesis(text, [citation(1), citation(2), citation(3)]))
        orphan = [i for i in result.issues if i.kind == CriticIssueKind.ORPHAN_CITATION]
        self.assertEqual(len(orphan), 1)
        self.assertIn("5", orphan[0].description)
        self.assertEqual(orphan[0].severity, CriticSeverity.HIGH)

    async def test_an_unsupported_claim_from_the_model_is_flagged(self):
        result = await self.critique(synthesis("Claim [1].", [citation(1)]),
                                     critic_report(unsupported=["It supports Python 2"]))
        issue = next(i for i in result.issues if i.kind == CriticIssueKind.UNSUPPORTED_CLAIM)
        self.assertIn("Python 2", issue.description)
        # one nit is noted, not a failure: a model reviewer finds something to say about any long answer
        self.assertEqual((issue.severity, result.verdict), (CriticSeverity.LOW, CriticVerdict.GOOD))

    async def test_several_unsupported_claims_make_the_answer_need_improvement(self):
        result = await self.critique(synthesis("Claim [1].", [citation(1)]),
                                     critic_report(unsupported=["one", "two", "three"]))
        self.assertEqual({i.severity for i in result.issues if i.kind == CriticIssueKind.UNSUPPORTED_CLAIM},
                         {CriticSeverity.MEDIUM})
        self.assertEqual(result.verdict, CriticVerdict.NEEDS_IMPROVEMENT)

    async def test_missing_information_is_noted_and_feeds_the_retry_queries(self):
        result = await self.critique(synthesis("Claim [1].", [citation(1)]), critic_report(missing=["Deployment options"]))
        self.assertEqual([i.severity for i in result.issues], [CriticSeverity.LOW])
        self.assertEqual(result.verdict, CriticVerdict.GOOD)
        self.assertIn("Deployment options".lower(), " ".join(result.suggestions).lower())

    async def test_uncited_paragraphs_are_flagged(self):
        long = "This paragraph makes plenty of claims but never cites anything at all, which is a problem for a research answer."
        text = f"{long}\n\n{long}\n\n{long}\n\nOne cited line [1]."
        result = await self.critique(synthesis(text, [citation(1)]))
        self.assertTrue(any(i.kind == CriticIssueKind.UNSUPPORTED_CLAIM and "paragraphs" in i.description for i in result.issues))

    async def test_no_citations_although_evidence_exists_is_bad(self):
        result = await self.critique(synthesis("An answer that cites nothing at all.", []))
        self.assertEqual(result.verdict, CriticVerdict.BAD)

    async def test_a_technical_claim_citing_only_reddit_is_a_weak_source(self):
        result = await self.critique(synthesis("Configure it like this [1].", [citation(1, RD, "reddit.com")]),
                                     understanding_={"intent": "technical_howto"})
        weak = [i for i in result.issues if i.kind == CriticIssueKind.WEAK_SOURCE]
        self.assertEqual([i.severity for i in weak], [CriticSeverity.HIGH])
        self.assertIn("Reddit", weak[0].description)

    async def test_official_docs_retrieved_but_not_cited_is_flagged_for_technical_answers(self):
        result = await self.critique(synthesis("From a blog [1].", [citation(1, WEB, "blog.example")]),
                                     understanding_={"intent": "technical_howto"})
        self.assertTrue(any(i.kind == CriticIssueKind.WEAK_SOURCE and "Official documentation" in i.description
                            for i in result.issues))

    async def test_reddit_for_a_non_technical_answer_is_fine(self):
        self.pool = build_pool([self.t], [doc("https://blog.example/1", "install langgraph opinions", self.t.task_id)])
        result = await self.critique(synthesis("People like it [1].", [citation(1, RD, "reddit.com")]))
        self.assertFalse(any(i.kind == CriticIssueKind.WEAK_SOURCE for i in result.issues))

    async def test_missing_coverage_and_incompleteness_are_flagged(self):
        result = await self.critique(
            synthesis("Partial [1].", [citation(1)], missed=["quantum error correction"], covered=[]),
            critic_report(covers=False, missing=["How to deploy"]))
        kinds = {i.kind for i in result.issues}
        self.assertTrue({CriticIssueKind.MISSING_COVERAGE, CriticIssueKind.INCOMPLETE} <= kinds)
        self.assertEqual(result.verdict, CriticVerdict.BAD)  # two high-severity issues

    async def test_suggestions_come_from_the_model_then_the_missing_pieces(self):
        result = await self.critique(
            synthesis("Partial [1].", [citation(1)], missed=["quantum error correction"]),
            critic_report(searches=["langgraph deploy guide"], missing=["Deployment options"]))
        self.assertEqual(result.suggestions[0], "langgraph deploy guide")
        self.assertIn("quantum error correction", result.suggestions)

    async def test_when_the_model_fails_the_deterministic_verdict_stands(self):
        result = await self.critique(synthesis("Claim [1]. Invented [7].", [citation(1)]), fail=True)
        self.assertEqual(result.verdict, CriticVerdict.NEEDS_IMPROVEMENT)
        self.assertTrue(any(i.kind == CriticIssueKind.ORPHAN_CITATION for i in result.issues))

    def test_when_a_retry_is_worth_it(self):
        issue = lambda s: CriticIssue(kind=CriticIssueKind.INCOMPLETE, description="d", severity=s)
        make = lambda v, *s: CriticResult(passed=False, verdict=v, issues=[issue(x) for x in s])
        self.assertTrue(worth_retrying(make(CriticVerdict.BAD)))
        self.assertTrue(worth_retrying(make(CriticVerdict.NEEDS_IMPROVEMENT, CriticSeverity.HIGH)))
        self.assertFalse(worth_retrying(make(CriticVerdict.NEEDS_IMPROVEMENT, CriticSeverity.MEDIUM)))
        self.assertFalse(worth_retrying(make(CriticVerdict.GOOD)))
        self.assertFalse(worth_retrying(None))
        self.assertFalse(worth_retrying(CriticResult(passed=True, verdict=None)))


# --------------------------------------------------------------------------- whole pipeline

def relevant_fake(**overrides):
    parts = dict(web=WEB_RELEVANT, github=GITHUB_OK, reddit=REDDIT_OK, docs=DOCS_OK)
    parts.update(overrides)
    return FakeTavily(**parts)


class PipelineGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_technical_question_runs_the_whole_pipeline_and_cites_official_docs(self):
        llm = ScriptedLLM()
        result, fake = await run_graph("How do I install and configure LangGraph?", llm, relevant_fake())
        self.assertEqual(result["critic_result"].verdict, CriticVerdict.GOOD)
        self.assertFalse(result.get("retry_performed"))
        self.assertFalse(result.get("gap_search_performed"))
        types = [c["source_type"] for c in result["citations"]]
        self.assertIn("official_docs", types)
        self.assertEqual(result["final_answer"], result["synthesis"].answer_text)
        self.assertEqual(len(llm.prompts), 1)  # one synthesis call, no retry
        self.assertEqual([d.stage for d in result["routing_decisions"]], ["search", "search"])

    async def test_each_sub_question_is_routed_on_its_own_with_outcomes(self):
        result, fake = await run_graph("How do I install and configure LangGraph?", ScriptedLLM(), relevant_fake())
        by_question = {d.sub_question: d for d in result["routing_decisions"]}
        self.assertEqual(by_question["Install LangGraph"].names, ("official_docs", "web"))
        self.assertEqual(by_question["LangGraph agent examples on GitHub"].names, ("official_docs", "github", "web"))
        for decision in by_question.values():
            self.assertEqual({o.source for o in decision.outcomes}, set(decision.sources))
            self.assertTrue(all(o.status == "ok" and o.result_count > 0 for o in decision.outcomes))
        self.assertTrue(all(d.task_id for d in result["source_documents"]))  # documents carry their task

    async def test_one_failing_source_does_not_affect_the_others(self):
        broken = relevant_fake(github={"error": RuntimeError("boom")})
        result, _ = await run_graph("How do I install and configure LangGraph?", ScriptedLLM(), broken)
        decision = next(d for d in result["routing_decisions"] if "GitHub" in d.sub_question)
        statuses = {o.source.value: o.status for o in decision.outcomes}
        self.assertEqual(statuses["github"], "failed")
        self.assertEqual((statuses["web"], statuses["official_docs"]), ("ok", "ok"))
        self.assertTrue(result["final_answer"])
        response = build_response(result, session_id=1, turn_number=1)
        self.assertIn("github:", response.research_metadata.sources_failed[0])

    async def test_simple_question_goes_through_the_same_pipeline(self):
        llm = ScriptedLLM(understanding_=understanding(is_simple=True, intent="technical_howto", technology="LangGraph"),
                          answer="LangGraph installs with pip [1].")
        result, fake = await run_graph("How do I install LangGraph?", llm, relevant_fake())
        self.assertEqual(result["sub_questions"], ["How do I install LangGraph?"])
        self.assertTrue(result["citations"])
        self.assertEqual(len(result["routing_decisions"]), 1)

    async def test_a_gap_triggers_one_targeted_round_that_adds_to_the_pool(self):
        # the second sub-question finds nothing on the first pass, then the keyword query finds something
        class Fake(FakeTavily):
            async def ainvoke(self, payload):
                self.calls.append(payload)
                if self.kind(payload) == "web" and payload["query"].startswith("How does quantum"):
                    return {"results": []}  # the planner wording finds nothing; the keyword query does
                if self.kind(payload) == "web" and "quantum" in payload["query"]:
                    return {"results": [{"title": "Quantum", "url": "https://q.example/x",
                                         "content": "Quantum error correction explained in depth.", "score": 1}]}
                return WEB_RELEVANT if self.kind(payload) == "web" else await super().ainvoke(payload)
        llm = ScriptedLLM(plan_=plan(task("Install LangGraph", "technical_howto", ["web"], "LangGraph"),
                                     task("How does quantum error correction work in practice?", "general_research", ["web"])),
                          answer="Install [1]. Quantum [2]." + PAD)
        result, fake = await run_graph("How do I install LangGraph and what is quantum error correction?", llm, Fake())
        self.assertTrue(result["gap_search_performed"])
        stages = [d.stage for d in result["routing_decisions"]]
        self.assertEqual((stages.count("search"), stages.count("gap_search")), (2, 1))
        self.assertIn("https://q.example/x", llm.prompts[0])  # the page found by the gap round reached the writer

    async def test_the_gap_round_happens_once_even_if_it_finds_nothing(self):
        llm = ScriptedLLM(plan_=plan(task("What is zorblax quux flimflam wibble?", "general_research", ["web"])),
                          answer="unused")
        result, fake = await run_graph("zorblax?", llm, FakeTavily(web=EMPTY, github=EMPTY, reddit=EMPTY, docs=EMPTY))
        self.assertEqual([d.stage for d in result["routing_decisions"]].count("gap_search"), 1)
        self.assertEqual(result["synthesis"].confidence, "low")
        self.assertEqual(llm.prompts, [])  # nothing to write from: the model is not asked to invent an answer
        self.assertIn("could not find reliable sources", result["final_answer"])

    async def test_a_bad_verdict_triggers_exactly_one_retry_and_the_better_attempt_wins(self):
        answers = ["Invented claim [9]. Another [8]. Third [7]." + PAD, GOOD_ANSWER]
        # The first answer cites nothing valid, so the critic bad-rates it without asking the model;
        # the model is asked once, about the re-written answer.
        llm = ScriptedLLM(answer=answers)
        result, fake = await run_graph("How do I install and configure LangGraph?", llm, relevant_fake())
        self.assertTrue(result["retry_performed"])
        self.assertEqual(len(llm.prompts), 2)  # first synthesis + one re-synthesis, never more
        self.assertEqual(result["critic_result"].verdict, CriticVerdict.GOOD)
        # nothing new to search for (the answer just cited nothing valid): it was re-written from the
        # same evidence, with the critic's findings in the prompt
        self.assertNotIn("retry", [d.stage for d in result["routing_decisions"]])
        self.assertIn("A REVIEWER FOUND THESE PROBLEMS", llm.prompts[1])
        self.assertNotIn("A REVIEWER FOUND", llm.prompts[0])
        self.assertNotIn("Note:", result["final_answer"])

    async def test_a_retry_searches_for_what_the_critic_says_is_missing(self):
        llm = ScriptedLLM(answer=GOOD_ANSWER, critic=critic_report(
            covers=False, missing=["Deployment options"], searches=["langgraph deployment guide"]))
        result, fake = await run_graph("How do I install and configure LangGraph?", llm, relevant_fake())
        self.assertTrue(result["retry_performed"])
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("langgraph deployment guide", [c["query"] for c in fake.calls])
        self.assertGreaterEqual([d.stage for d in result["routing_decisions"]].count("retry"), 1)

    async def test_a_good_verdict_skips_the_retry(self):
        llm = ScriptedLLM()
        result, _ = await run_graph("How do I install and configure LangGraph?", llm, relevant_fake())
        self.assertFalse(result.get("retry_performed"))
        self.assertEqual([n for n, _ in llm.structured_prompts].count("CriticReport"), 1)

    async def test_a_second_bad_verdict_returns_the_best_result_with_limitations(self):
        llm = ScriptedLLM(answer="Invented [9]. Again [8]. More [7]." + PAD, critic=critic_report(covers=False, searches=["langgraph reference"]))
        result, _ = await run_graph("How do I install and configure LangGraph?", llm, relevant_fake())
        self.assertTrue(result["retry_performed"])
        self.assertEqual(len(llm.prompts), 2)  # bounded: exactly one retry
        self.assertEqual(result["critic_result"].verdict, CriticVerdict.BAD)
        self.assertIn("Note:", result["final_answer"])
        self.assertEqual(result["confidence"], "low")
        self.assertTrue(result["limitations"])
        self.assertNotIn("[9]", result["final_answer"])  # dead markers are removed from the text

    async def test_retry_can_be_disabled(self):
        llm = ScriptedLLM(answer="Invented [9]. Again [8]. More [7]." + PAD, critic=critic_report(covers=False, searches=["x y z"]))
        with mock.patch.dict("os.environ", {"MAX_CRITIC_RETRIES": "0"}):
            result, _ = await run_graph("How do I install and configure LangGraph?", llm, relevant_fake())
        self.assertFalse(result.get("retry_performed"))
        self.assertEqual(len(llm.prompts), 1)

    async def test_no_retry_when_there_is_no_new_query_to_run(self):
        # every sub-question found nothing and the gap round already ran their keyword queries
        llm = ScriptedLLM(plan_=plan(task("What is zorblax quux?", "general_research", ["web"])))
        result, _ = await run_graph("zorblax?", llm, FakeTavily(web=EMPTY, github=EMPTY, reddit=EMPTY, docs=EMPTY))
        self.assertFalse(result.get("retry_performed"))

    async def test_only_format_response_emits_final_answer(self):
        updates, _ = await stream_graph("How do I install and configure LangGraph?", ScriptedLLM(), relevant_fake())
        emitters = [n for n, data in updates if data and data.get("final_answer")]
        self.assertEqual(emitters, ["format_response"])
        self.assertEqual(updates[-1][0], "save_to_cache_node")
        json.dumps([d.model_dump(mode="json") for n, data in updates if data for d in data.get("routing_decisions", [])])

    async def test_every_step_is_isolated_from_a_crash_in_the_critic(self):
        llm = ScriptedLLM(fail=["CriticReport"])
        result, _ = await run_graph("How do I install and configure LangGraph?", llm, relevant_fake())
        self.assertTrue(result["final_answer"])
        self.assertIn(result["critic_result"].verdict, (CriticVerdict.GOOD, CriticVerdict.NEEDS_IMPROVEMENT))


class OutcomeMappingTests(unittest.TestCase):
    def result(self, status, code, message="m"):
        from research_app.domain import ResearchError, ResearchResult, ResultStatus
        return ResearchResult(status=ResultStatus(status), source_type=DOCS,
                              error=ResearchError(code=code, message=message))

    def test_a_search_that_found_nothing_verified_is_empty_not_failed(self):
        for code in ("no_official_results", "no_domain_results"):
            self.assertEqual(st._outcome_from_result(DOCS, self.result("failed", code)).status, "empty")
        tavily_none = self.result("failed", "provider_error", "No search results found for 'q'.")
        self.assertEqual(st._outcome_from_result(WEB, tavily_none).status, "empty")

    def test_real_failures_stay_failures_and_keep_their_code(self):
        out = st._outcome_from_result(GH, self.result("failed", "provider_error", "boom"))
        self.assertEqual((out.status, out.error_code), ("failed", "provider_error"))
        self.assertEqual(st._outcome_from_result(GH, self.result("timeout", "timeout")).status, "timeout")


class ResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_response_describes_the_run(self):
        result, _ = await run_graph("How do I install and configure LangGraph?", ScriptedLLM(), relevant_fake(),
                                    session_id=4, turn_number=2)
        r = build_response(result, session_id=4, turn_number=2, latency_ms=12.34)
        self.assertEqual((r.session_id, r.turn_number, r.is_follow_up), (4, 2, False))
        self.assertEqual(r.resolved_query, "How do I install and configure LangGraph?")
        self.assertEqual(r.critic_verdict, "good")
        self.assertEqual(r.subquestions, ["Install LangGraph", "LangGraph agent examples on GitHub"])
        self.assertIn("official_docs", r.sources_consulted)
        meta = r.research_metadata
        self.assertGreaterEqual(meta.total_sources_found, meta.total_evidence_pieces - 1)
        self.assertEqual((meta.gap_search_performed, meta.retry_performed, meta.sources_failed), (False, False, []))
        self.assertEqual(len(meta.routing_decisions), 2)
        json.dumps(r.model_dump(mode="json"))

    def test_a_bare_state_still_builds(self):
        r = build_response({"question": "q", "final_answer": "a", "sub_questions": []}, session_id=1, turn_number=1)
        self.assertEqual((r.answer, r.critic_verdict, r.resolved_query), ("a", "unavailable", "q"))


if __name__ == "__main__":
    unittest.main()
