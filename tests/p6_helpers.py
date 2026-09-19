"""Shared fakes for the Phase 6 tests: a scripted LLM (understanding, planner, critic and the
free-text synthesis) and a graph runner. Nothing here touches the network or a real database."""
from __future__ import annotations

from typing import Callable, Optional, Union
from unittest import mock

from research_app.agent import state as st
from research_app.agent.graph import compiled_graph
from research_app.agent.pipeline.critic import CriticReport
from tests.test_router_nodes import FakeTavily  # answers by include_domains: web / github / reddit / docs

GOOD_ANSWER = ("LangGraph is installed with pip [1]. Agents are built from graphs and examples are in the "
               "repository [2]. The configuration is documented [1][2].")


def understanding(**kw) -> st.QueryUnderstanding:
    base = dict(is_simple=False, is_follow_up=False, resolved_query="", intent="unknown", technology="",
                time_sensitivity="unknown", context_topics=[])
    base.update(kw)
    return st.QueryUnderstanding(**base)


def plan(*tasks) -> st.SourceAwarePlan:
    return st.SourceAwarePlan(tasks=[st.PlannedSubquestion(**t) for t in tasks])


def task(text, intent="general_research", sources=("web",), technology=""):
    return dict(text=text, source_intent=intent, suggested_sources=list(sources), technology=technology)


def critic_report(covers=True, unsupported=(), missing=(), searches=()) -> CriticReport:
    return CriticReport(covers_question=covers, unsupported_claims=list(unsupported),
                        missing_information=list(missing), suggested_searches=list(searches))


class _Structured:
    def __init__(self, llm: "ScriptedLLM", schema):
        self.llm, self.schema = llm, schema

    async def ainvoke(self, prompt):
        self.llm.structured_prompts.append((self.schema.__name__, prompt))
        if self.schema.__name__ in self.llm.fail:
            raise RuntimeError(f"{self.schema.__name__} failed")
        value = {"QueryUnderstanding": self.llm.understanding, "SourceAwarePlan": self.llm.plan,
                 "CriticReport": self.llm.critic}[self.schema.__name__]
        return value() if callable(value) else value


class ScriptedLLM:
    """``answer`` is a string, a list (one per synthesis call, the last repeats) or a callable(n)."""

    def __init__(self, *, understanding_=None, plan_=None, answer: Union[str, list, Callable] = GOOD_ANSWER,
                 critic=None, fail=()):
        self.understanding = understanding_ or understanding(intent="technical_howto", technology="LangGraph")
        self.plan = plan_ or plan(
            task("Install LangGraph", "technical_howto", ["official_docs", "web"], "LangGraph"),
            task("LangGraph agent examples on GitHub", "code_implementation", ["github"], "LangGraph"))
        self.answer = answer
        self.critic = critic if critic is not None else critic_report()
        self.fail = set(fail)
        self.prompts: list[str] = []
        self.structured_prompts: list[tuple[str, str]] = []

    def with_structured_output(self, schema, **kwargs):
        return _Structured(self, schema)

    def prompts_for(self, schema_name: str) -> list[str]:
        return [p for name, p in self.structured_prompts if name == schema_name]

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        if "synthesis" in self.fail:
            raise RuntimeError("synthesis failed")
        n = len(self.prompts)
        if callable(self.answer):
            text = self.answer(n)
        elif isinstance(self.answer, list):
            text = self.answer[min(n, len(self.answer)) - 1]
        else:
            text = self.answer
        return mock.Mock(content=text)


def graph_inputs(question: str, **extra) -> dict:
    return {
        "question": question, "messages": [], "step_count": 0, "final_answer": "", "sub_questions": [],
        "search_results": [], "sources": [], "cache_hit": False, "api_limit_reached": False,
        "critic_score": 0, "critic_feedback": "", "is_simple": False, "history": [], **extra,
    }


def patches(fake: FakeTavily, llm: ScriptedLLM, *, lookup=lambda q: (False, ""), store: Optional[Callable] = None):
    return (
        mock.patch.object(st, "tavily_tool", fake),
        mock.patch.object(st, "llm_groq", llm),
        mock.patch.object(st, "lookup_cache", lookup),
        mock.patch.object(st, "store_cache", store or (lambda q, a: None)),
    )


async def run_graph(question: str, llm: ScriptedLLM, fake: Optional[FakeTavily] = None, **extra):
    fake = fake or FakeTavily()
    p1, p2, p3, p4 = patches(fake, llm)
    with p1, p2, p3, p4:
        result = await compiled_graph.ainvoke(graph_inputs(question, **extra))
    return result, fake


async def stream_graph(question: str, llm: ScriptedLLM, fake: Optional[FakeTavily] = None, **extra):
    fake = fake or FakeTavily()
    p1, p2, p3, p4 = patches(fake, llm)
    updates = []
    with p1, p2, p3, p4:
        async for mode, chunk in compiled_graph.astream(graph_inputs(question, **extra), stream_mode=["updates"]):
            node, data = next(iter(chunk.items()))
            updates.append((node, data))
    return updates, fake
