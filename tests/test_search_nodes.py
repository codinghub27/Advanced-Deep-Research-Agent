"""The two web search nodes after Phase 3: they normalize, then project back onto the
state keys the graph and the SSE `sources` event already used. Tavily and the LLM are
replaced by fakes, so nothing here touches the network."""
import json
import unittest
from unittest import mock

from research_app.agent import state as st
from research_app.agent.graph import compiled_graph
from research_app.domain import SourceDocument, SourceType


class FakeTavily:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def ainvoke(self, payload):
        self.calls.append(payload)
        return self.response


def legacy_web_search_update(query, results, limit):
    """The node body exactly as it was before Phase 3 (reference for equivalence)."""
    condensed, sources = [], []
    result_list = results.get("results", []) if isinstance(results, dict) else []
    for r in result_list[:3]:
        content = r.get("content", "")[:limit]
        url = r.get("url", "")
        title = r.get("title", "")
        if content:
            condensed.append(f"{url}\n{content}")
        if url:
            sources.append({"url": url, "title": title, "snippet": content[:150]})
    return {"search_results": [f"Query: {query}\n" + "\n---\n".join(condensed)], "sources": sources}


LONG = "Sentence about the topic. " * 60  # ~1500 chars, exercises both content limits

RESPONSES = {
    "normal": {"query": "q", "answer": "x", "results": [
        {"title": "One", "url": "https://one.example/a", "content": "First body text.", "score": 0.9},
        {"title": "Two", "url": "https://two.example/b?x=1", "content": LONG, "score": 0.8},
        {"title": "Three", "url": "https://three.example/c", "content": "Third body.", "score": 0.7},
    ]},
    "more_than_three": {"results": [
        {"title": str(n), "url": f"https://s{n}.example/", "content": f"body {n}"} for n in range(6)]},
    "content_missing": {"results": [
        {"title": "No body", "url": "https://nobody.example/", "content": ""},
        {"title": "Body", "url": "https://body.example/", "content": "has text"}]},
    "empty": {"results": []},
    "tavily_error": {"error": RuntimeError("boom")},
    "tavily_no_results": "No search results found for 'q'. Suggestions: broaden the query.",
}


class WebSearchNodeTests(unittest.IsolatedAsyncioTestCase):
    async def run_node(self, node, response, payload):
        fake = FakeTavily(response)
        with mock.patch.object(st, "tavily_tool", fake):
            out = await node(payload)
        return out, fake

    async def test_search_node_output_keys_and_shapes(self):
        out, fake = await self.run_node(st.search_node, RESPONSES["normal"], {"query": "my query"})
        self.assertEqual(fake.calls, [{"query": "my query"}])
        self.assertEqual(set(out), {"search_results", "sources", "source_documents"})
        self.assertTrue(all(set(s) == {"url", "title", "snippet"} for s in out["sources"]))
        self.assertTrue(all(isinstance(d, SourceDocument) for d in out["source_documents"]))
        self.assertEqual(len(out["search_results"]), 1)
        json.dumps(out["sources"])  # main.py serialises these into SSE events

    async def test_simple_search_node_also_sets_sub_questions(self):
        out, _ = await self.run_node(st.simple_search_node, RESPONSES["normal"], {"question": "capital of France?"})
        self.assertEqual(out["sub_questions"], ["capital of France?"])
        self.assertEqual(set(out), {"search_results", "sources", "source_documents", "sub_questions"})

    async def test_search_results_string_format(self):
        out, _ = await self.run_node(st.search_node, RESPONSES["normal"], {"query": "q1"})
        first, second, third = out["search_results"][0].split("\n---\n")
        self.assertEqual(first, "Query: q1\nhttps://one.example/a\nFirst body text.")
        self.assertEqual(second, "https://two.example/b?x=1\n" + LONG.strip()[:800])
        self.assertEqual(third, "https://three.example/c\nThird body.")

    async def test_content_limits_differ_per_node(self):
        out, _ = await self.run_node(st.search_node, RESPONSES["normal"], {"query": "q"})
        self.assertEqual(len(out["search_results"][0].split("\n---\n")[1].split("\n", 1)[1]), 800)
        out, _ = await self.run_node(st.simple_search_node, RESPONSES["normal"], {"question": "q"})
        self.assertEqual(len(out["search_results"][0].split("\n---\n")[1].split("\n", 1)[1]), 400)

    async def test_output_matches_pre_phase_3_behaviour_for_well_formed_results(self):
        for name in ("normal", "more_than_three", "content_missing", "empty", "tavily_error",
                     "tavily_no_results"):
            for node, key, limit in ((st.search_node, "query", 800), (st.simple_search_node, "question", 400)):
                with self.subTest(response=name, node=node.__name__):
                    out, _ = await self.run_node(node, RESPONSES[name], {key: "q"})
                    expected = legacy_web_search_update("q", RESPONSES[name], limit)
                    self.assertEqual(out["search_results"], expected["search_results"])
                    self.assertEqual(out["sources"], expected["sources"])

    async def test_documents_carry_query_provider_and_metadata(self):
        out, _ = await self.run_node(st.search_node, RESPONSES["normal"], {"query": "q7"})
        doc = out["source_documents"][0]
        self.assertEqual((doc.query, doc.provider, doc.source_type), ("q7", "tavily", SourceType.WEB))
        self.assertEqual(doc.metadata, {"score": 0.9})
        self.assertEqual(doc.domain, "one.example")

    async def test_at_most_three_results(self):
        out, _ = await self.run_node(st.search_node, RESPONSES["more_than_three"], {"query": "q"})
        self.assertEqual(len(out["source_documents"]), 3)
        self.assertEqual(len(out["sources"]), 3)

    async def test_tavily_error_degrades_to_empty_like_before(self):
        with self.assertLogs("research_app.agent.state", level="WARNING") as logs:
            out, _ = await self.run_node(st.search_node, RESPONSES["tavily_error"], {"query": "q"})
        self.assertEqual(out, {"search_results": ["Query: q\n"], "sources": [], "source_documents": []})
        self.assertIn("provider_error", logs.output[0])

    async def test_unusable_urls_no_longer_reach_sources_or_the_prompt(self):
        response = {"results": [
            {"title": "bad", "url": "javascript:alert(1)", "content": "evil"},
            {"title": "good", "url": "https://good.example/", "content": "fine"}]}
        with self.assertLogs("research_app.sources.normalizer", level="WARNING"):
            out, _ = await self.run_node(st.search_node, response, {"query": "q"})
        self.assertEqual([s["url"] for s in out["sources"]], ["https://good.example/"])
        self.assertNotIn("evil", out["search_results"][0])


class FakeStructured:
    def __init__(self, schema):
        self.schema = schema

    async def ainvoke(self, prompt):
        if self.schema is st.QuestionType:
            return st.QuestionType(is_simple=False)
        return st.SearchPlan(sub_questions=["sub one", "sub two", "sub three"])


class FakeLLM:
    def with_structured_output(self, schema, **kwargs):
        return FakeStructured(schema)

    async def ainvoke(self, prompt):
        return mock.Mock(content="A grounded report sentence with detail. " * 30)


# Same shape main.py builds; note there is no "source_documents" key in it.
GRAPH_INPUTS = {
    "question": "Compare offline test frameworks", "messages": [], "step_count": 0, "final_answer": "",
    "sub_questions": [], "search_results": [], "sources": [], "cache_hit": False,
    "api_limit_reached": False, "critic_score": 0, "critic_feedback": "", "is_simple": False, "history": [],
}


class CompiledGraphTests(unittest.IsolatedAsyncioTestCase):
    def patches(self):
        return (
            mock.patch.object(st, "tavily_tool", FakeTavily(RESPONSES["normal"])),
            mock.patch.object(st, "llm_groq", FakeLLM()),
            mock.patch.object(st, "lookup_cache", lambda q: (False, "")),
            mock.patch.object(st, "store_cache", lambda q, a: None),
        )

    async def test_ainvoke_with_main_py_inputs(self):
        p1, p2, p3, p4 = self.patches()
        with p1, p2, p3, p4:
            result = await compiled_graph.ainvoke(dict(GRAPH_INPUTS))
        self.assertEqual(result["sub_questions"], ["sub one", "sub two", "sub three"])
        self.assertEqual(len(result["search_results"]), 3)  # one per parallel search_node
        self.assertEqual(len(result["sources"]), 9)
        self.assertEqual(len(result["source_documents"]), 9)
        self.assertEqual({d.query for d in result["source_documents"]}, {"sub one", "sub two", "sub three"})
        self.assertGreater(len(result["final_answer"]), 500)

    async def test_astream_updates_are_serialisable_the_way_main_py_uses_them(self):
        p1, p2, p3, p4 = self.patches()
        seen = []
        with p1, p2, p3, p4:
            async for mode, chunk in compiled_graph.astream(dict(GRAPH_INPUTS), stream_mode=["updates"]):
                node, data = next(iter(chunk.items()))
                seen.append(node)
                if node == "search_node":
                    json.dumps(data["sources"])
                    self.assertEqual(len(data["source_documents"]), 3)
        self.assertEqual(seen.count("search_node"), 3)
        self.assertEqual(seen[-1], "save_to_cache_node")


if __name__ == "__main__":
    unittest.main()
