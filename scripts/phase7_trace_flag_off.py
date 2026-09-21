"""Phase 7 verification, pass 1: run the same scripted queries through whichever checkout is the cwd
and print a JSON trace (node order, answers, sources, search calls). Run it in a clean checkout of
880f4e0 (pre-Phase-7) and of the Phase 7 commit from that checkout's root, then diff the two outputs.
Nothing external is touched: ScriptedLLM + FakeTavily + a stubbed answer cache (tests/p6_helpers)."""
import asyncio
import json
import os
import sys
from unittest import mock

sys.path.insert(0, os.getcwd())
for k in ("GROQ_API_KEY", "TAVILY_API_KEY", "OPENROUTER_API_KEY", "SECRET_KEY", "OPENAI_API_KEY"):
    os.environ.setdefault(k, "dummy-not-a-real-key")
for key in [k for k in os.environ if k.startswith(("SOURCE_RAG", "RAG_", "RERANK", "QDRANT"))]:
    del os.environ[key]

from research_app.agent import state as st  # noqa: E402
from research_app.agent.graph import compiled_graph  # noqa: E402
from tests.p6_helpers import (  # noqa: E402
    GOOD_ANSWER, ScriptedLLM, critic_report, graph_inputs, understanding,
)
from tests.test_router_nodes import FakeTavily  # noqa: E402


def urls(items):
    out = []
    for s in items or []:
        out.append(getattr(s, "url", None) or (s.get("url") if isinstance(s, dict) else str(s)))
    return sorted(str(u) for u in out)


async def run(name, question, llm, fake=None, lookup=lambda q: (False, "")):
    fake = fake or FakeTavily()
    stored = []
    updates = []
    error = None
    with mock.patch.object(st, "tavily_tool", fake), mock.patch.object(st, "llm_groq", llm), \
            mock.patch.object(st, "lookup_cache", lookup), \
            mock.patch.object(st, "store_cache", lambda q, a: stored.append((q, a))):
        try:
            async for mode, chunk in compiled_graph.astream(graph_inputs(question), stream_mode=["updates"]):
                node, data = next(iter(chunk.items()))
                updates.append((node, data or {}))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    final = {}
    for node, data in updates:
        for k in ("final_answer", "critic_score", "sub_questions", "cache_hit"):
            if k in data:
                final[k] = data[k]
        if "sources" in data:
            final["sources"] = urls(data["sources"])
    return {
        "scenario": name,
        "question": question,
        "nodes": [n for n, _ in updates],
        "update_keys": {f"{i}:{n}": sorted(d) for i, (n, d) in enumerate(updates)},
        "final": final,
        "tavily_calls": [(c.get("query"), c.get("include_domains")) for c in fake.calls],
        "llm_free_text_calls": len(llm.prompts),
        "llm_structured_calls": [name for name, _ in llm.structured_prompts],
        "cache_stores": len(stored),
        "error": error,
    }


class SeqCritic:
    def __init__(self):
        self.n = 0

    def __call__(self):
        self.n += 1
        return critic_report(covers=False, unsupported=["x"]) if self.n == 1 else critic_report()


async def main():
    out = []
    out.append(await run("complex_howto", "How do I install LangGraph and build an agent?", ScriptedLLM()))
    out.append(await run("simple", "What is FastAPI?", ScriptedLLM(
        understanding_=understanding(is_simple=True, intent="general_research", technology=""))))
    out.append(await run("cache_hit", "What is FastAPI?", ScriptedLLM(),
                         lookup=lambda q: (True, "Cached answer about FastAPI. " * 3)))
    out.append(await run("critic_retry", "How do I install LangGraph and build an agent?",
                         ScriptedLLM(critic=SeqCritic())))
    out.append(await run("time_sensitive", "What is the latest version of FastAPI?", ScriptedLLM(
        understanding_=understanding(is_simple=True, intent="technical_howto", technology="FastAPI",
                                     time_sensitivity="current"))))
    out.append(await run("web_failure", "How do I install LangGraph and build an agent?", ScriptedLLM(),
                         fake=FakeTavily(web=RuntimeError("web down"))))
    print(json.dumps(out, indent=1, sort_keys=True, default=str))


asyncio.run(main())
