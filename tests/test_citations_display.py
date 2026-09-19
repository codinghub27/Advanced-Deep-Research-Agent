"""Source-citations display (Phase 6 add-on): what the API and the SSE stream hand to the chat UI.
SQLite in memory; no network. The page itself is checked by tests/ui/check_sources_ui.js."""
import json
import re
from unittest import mock

from research_app.agent.pipeline.nodes import format_response_node
from research_app.domain import Citation, SourceType
from research_app.domain.pipeline import SynthesisResult
from tests.base import ApiTestCase
from tests.test_p6_conversation import CITATION, DECISION, fake_graph

REQUIRED = {"index", "source_type", "title", "url", "domain", "snippet", "retrieved_at"}
SOURCE_TYPES = {"official_docs", "web", "github", "reddit"}
LEGACY = [{"url": "https://x.example", "title": "X", "snippet": ""}]
SECOND = {**CITATION, "index": 2, "marker": "[2]", "source_type": "reddit", "title": "Thread",
          "url": "https://www.reddit.com/r/python/comments/abc/", "domain": "reddit.com"}


def stream_graph(*updates):
    """A fake graph whose ``astream`` yields the given node updates, in order."""
    graph = mock.MagicMock()

    async def astream(inputs, stream_mode=None):
        for node, data in updates:
            yield "updates", {node: data}
    graph.astream = mock.MagicMock(side_effect=astream)
    return graph


class CitationsDisplayTests(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.make_user("alice")
        self.headers = self.auth("alice")

    def events(self, graph, path="/api/research/stream"):
        with mock.patch("research_app.main.compiled_graph", graph):
            r = self.client.post(path, json={"question": "How do I install FastAPI?"}, headers=self.headers)
        self.assertEqual(r.status_code, 200)
        return [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: ")]

    # ---- JSON API
    def test_the_api_response_carries_complete_citations(self):
        graph = fake_graph("Docs say so [1]. Users agree [2].", citations=[CITATION, SECOND])
        with mock.patch("research_app.main.compiled_graph", graph):
            body = self.client.post("/api/research", json={"question": "How do I install FastAPI?"},
                                    headers=self.headers).json()
        self.assertEqual([c["index"] for c in body["citations"]], [1, 2])
        for c in body["citations"]:
            self.assertTrue(REQUIRED <= set(c), f"missing {REQUIRED - set(c)}")
            self.assertIn(c["source_type"], SOURCE_TYPES)
        self.assertIn("sources_consulted", body)

    # ---- SSE
    def test_the_sources_event_comes_after_the_last_token_and_before_done_with_the_full_list(self):
        events = self.events(stream_graph(
            ("search_node", {"sources": LEGACY, "routing_decisions": [DECISION]}),
            ("format_response", {"final_answer": "Docs say so [1]. Users agree [2].",
                                 "citations": [CITATION, SECOND]})))
        types = [e["type"] for e in events]
        self.assertEqual(types[-2:], ["sources", "done"])
        self.assertGreater(types.index("sources"), max(i for i, t in enumerate(types) if t == "token"))
        sources = events[-2]
        self.assertEqual(sources["total"], 2)
        self.assertEqual([c["index"] for c in sources["citations"]], [1, 2])
        self.assertTrue(all(REQUIRED <= set(c) for c in sources["citations"]))
        self.assertEqual(sources["sources"], LEGACY)                # the legacy list is unchanged
        self.assertEqual(types.count("sources"), 1)                 # sent once, not per token

    def test_the_answers_markers_match_the_citations(self):
        events = self.events(stream_graph(
            ("format_response", {"final_answer": "A [1]. B [2].", "citations": [CITATION, SECOND]})))
        text = "".join(e["content"] for e in events if e["type"] == "token")
        cited = {c["index"] for c in next(e for e in events if e["type"] == "sources")["citations"]}
        self.assertEqual({int(n) for n in re.findall(r"\[(\d+)\]", text)}, cited)

    def test_no_citations_means_no_citations_in_the_event(self):
        events = self.events(stream_graph(("search_node", {"sources": LEGACY}),
                                          ("format_response", {"final_answer": "Plain answer.", "citations": []})))
        sources = next(e for e in events if e["type"] == "sources")   # the legacy event still exists
        self.assertNotIn("citations", sources)
        self.assertNotIn("total", sources)
        self.assertEqual(sources["sources"], LEGACY)

    def test_nothing_to_show_means_no_sources_event(self):
        events = self.events(stream_graph(("format_response", {"final_answer": "Plain answer.", "citations": []})))
        self.assertNotIn("sources", [e["type"] for e in events])
        self.assertEqual(events[-1]["type"], "done")

    def test_a_cache_hit_shows_its_stored_citations(self):
        events = self.events(stream_graph(
            ("semantic_cache_node", {"cache_hit": True, "final_answer": "Cached [1].", "citations": [CITATION]})))
        sources = next(e for e in events if e["type"] == "sources")
        self.assertEqual((sources["total"], sources["sources"]), (1, []))

    # ---- what the pipeline produces
    def test_format_response_only_lists_citations_the_text_uses_and_only_known_source_types(self):
        def cite(i, kind):
            return Citation(marker=f"[{i}]", source_id=f"s{i}", url=f"https://e{i}.example/", index=i,
                            domain=f"e{i}.example", source_type=kind)
        synthesis = SynthesisResult(
            answer_text="Claim [1]. Another [2]. Invented [7].",
            citations=[cite(1, SourceType.OFFICIAL_DOCS), cite(2, SourceType.GITHUB)], confidence="medium")
        out = format_response_node({"synthesis": synthesis})
        self.assertNotIn("[7]", out["final_answer"])
        self.assertEqual({int(n) for n in re.findall(r"\[(\d+)\]", out["final_answer"])},
                         {c["index"] for c in out["citations"]})
        self.assertTrue({c["source_type"] for c in out["citations"]} <= SOURCE_TYPES)
