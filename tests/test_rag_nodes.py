import os
import unittest
from unittest import mock

from research_app.agent import vectordb
from research_app.agent.graph import compiled_graph
from research_app.agent.pipeline import rag_nodes
from research_app.domain import SourceDocument
from research_app.rag.retriever import PROVIDER
from tests.rag_helpers import doc

RAG_ON = {"SOURCE_RAG_ENABLED": "true"}


def env(**values):
    return mock.patch.dict(os.environ, values, clear=False)


def state(simple=True, **extra):
    return {"question": "how do I install fastapi", "is_simple": simple, **extra}


def hit(text: str, score: float) -> SourceDocument:
    d = doc("https://fastapi.tiangolo.com/tutorial/", text, provider=PROVIDER)
    d.metadata["rag_score"] = score
    return d


class FakeRetriever:
    def __init__(self, docs=None, error=None):
        self.docs, self.error, self.queries = docs or [], error, []

    async def retrieve(self, query, filters=None, top_k=None):
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.docs


class GateRouterTests(unittest.TestCase):
    def test_flag_off_is_exactly_classify_router(self):
        with env(SOURCE_RAG_ENABLED="false"):
            self.assertEqual(rag_nodes.rag_gate_router(state(simple=True)), "simple_search_node")
            self.assertEqual(rag_nodes.rag_gate_router(state(simple=False)), "planner_node")

    def test_flag_unset_defaults_to_off(self):
        with mock.patch.dict(os.environ, clear=False) as e:
            e.pop("SOURCE_RAG_ENABLED", None)
            self.assertEqual(rag_nodes.rag_gate_router(state(simple=False)), "planner_node")

    def test_flag_on_goes_to_the_lookup(self):
        with env(**RAG_ON):
            self.assertEqual(rag_nodes.rag_gate_router(state()), rag_nodes.SOURCE_RAG_NODE)

    def test_time_sensitive_questions_skip_the_lookup(self):
        with env(**RAG_ON):
            question = {"question": "latest fastapi release", "is_simple": False}
            self.assertEqual(rag_nodes.rag_gate_router(question), "planner_node")
            modelled = state(simple=True, understanding={"time_sensitivity": "current"})
            self.assertEqual(rag_nodes.rag_gate_router(modelled), "simple_search_node")


class SourceRagNodeTests(unittest.IsolatedAsyncioTestCase):
    async def run_node(self, retriever, **extra):
        with env(**RAG_ON), mock.patch.object(rag_nodes, "get_retriever", return_value=retriever):
            return await rag_nodes.source_rag_node(state(**extra))

    async def test_enough_relevant_chunks_become_the_evidence(self):
        retriever = FakeRetriever([hit("install with pip", 0.9), hit("run uvicorn", 0.7), hit("unrelated", 0.1)])
        update = await self.run_node(retriever, resolved_query="install FastAPI")
        self.assertTrue(update["rag_hit"])
        self.assertEqual(retriever.queries, ["install FastAPI"])
        self.assertEqual(len(update["source_documents"]), 2)  # below-threshold chunks are dropped
        task = update["research_tasks"][0]
        self.assertEqual(update["sub_questions"], ["install FastAPI"])
        self.assertTrue(all(d.task_id == task.task_id for d in update["source_documents"]))
        self.assertEqual(update["rag_top_score"], 0.9)
        self.assertEqual(rag_nodes.rag_router({**state(), **update}), rag_nodes.EVIDENCE_NODE)

    async def test_too_few_relevant_chunks_fall_through_to_research(self):
        update = await self.run_node(FakeRetriever([hit("install with pip", 0.9), hit("other", 0.2)]))
        self.assertEqual(update, {"rag_hit": False, "rag_top_score": 0.9})
        self.assertEqual(rag_nodes.rag_router({**state(simple=False), **update}), "planner_node")
        self.assertEqual(rag_nodes.rag_router({**state(simple=True), **update}), "simple_search_node")

    async def test_nothing_retrieved_falls_through(self):
        update = await self.run_node(FakeRetriever([]))
        self.assertEqual(update, {"rag_hit": False, "rag_top_score": 0.0})

    async def test_threshold_and_minimum_come_from_settings(self):
        retriever = FakeRetriever([hit("a", 0.6)])
        with env(**RAG_ON, RAG_RELEVANCE_THRESHOLD="0.55", RAG_MIN_CHUNKS="1"), \
                mock.patch.object(rag_nodes, "get_retriever", return_value=retriever):
            update = await rag_nodes.source_rag_node(state())
        self.assertTrue(update["rag_hit"])


class IndexNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_off_by_default_and_never_calls_the_indexer(self):
        indexer = mock.AsyncMock()
        with env(SOURCE_RAG_INGEST_ENABLED="false"), \
                mock.patch.object(rag_nodes, "get_indexer", return_value=mock.Mock(index=indexer)):
            self.assertEqual(await rag_nodes.index_sources_node(state(source_documents=[doc("https://a.io/x", "t")])), {})
        indexer.assert_not_awaited()

    async def test_on_indexes_the_run_documents(self):
        indexer = mock.AsyncMock(return_value=3)
        docs = [doc("https://a.io/x", "text")]
        with env(SOURCE_RAG_INGEST_ENABLED="true"), \
                mock.patch.object(rag_nodes, "get_indexer", return_value=mock.Mock(index=indexer)):
            await rag_nodes.index_sources_node(state(source_documents=docs))
        indexer.assert_awaited_once_with(docs)

    async def test_cache_hits_are_not_indexed_and_failures_do_not_propagate(self):
        indexer = mock.AsyncMock(side_effect=RuntimeError("boom"))
        with env(SOURCE_RAG_INGEST_ENABLED="true"), \
                mock.patch.object(rag_nodes, "get_indexer", return_value=mock.Mock(index=indexer)):
            self.assertEqual(await rag_nodes.index_sources_node(state(cache_hit=True)), {})
            indexer.assert_not_awaited()
            self.assertEqual(await rag_nodes.index_sources_node(state(source_documents=[doc("https://a.io/x", "t")])), {})


class GraphContractTests(unittest.TestCase):
    def edges(self):
        graph = compiled_graph.get_graph()
        return {(e.source, e.target) for e in graph.edges}

    def test_phase_6_edges_are_unchanged(self):
        edges = self.edges()
        for edge in [("__start__", "semantic_cache_node"), ("classify_node", "simple_search_node"),
                     ("classify_node", "planner_node"), ("evidence_collection", "gap_detection"),
                     ("synthesis_node", "critic_node"), ("retry_node", "format_response"),
                     ("save_to_cache_node", "__end__"), ("semantic_cache_node", "classify_node")]:
            self.assertIn(edge, edges)

    def test_phase_7_edges(self):
        edges = self.edges()
        self.assertIn(("classify_node", "source_rag_node"), edges)
        self.assertIn(("source_rag_node", "evidence_collection"), edges)
        self.assertIn(("source_rag_node", "simple_search_node"), edges)
        self.assertIn(("source_rag_node", "planner_node"), edges)
        self.assertIn(("format_response", "index_sources_node"), edges)
        self.assertIn(("index_sources_node", "save_to_cache_node"), edges)
        self.assertNotIn(("format_response", "save_to_cache_node"), edges)
        # nothing leads back to classification or the lookup: still no loops
        self.assertFalse({s for s, t in edges if t in ("source_rag_node", "classify_node")} - {"classify_node", "semantic_cache_node"})


class AnswerCacheUntouchedTests(unittest.IsolatedAsyncioTestCase):
    async def test_rag_nodes_never_change_the_answer_cache(self):
        vectordb.clear_cache()
        vectordb.store_cache("what is fastapi", "FastAPI is a Python web framework. " * 3)
        before = vectordb.get_cache_stats()
        retriever = FakeRetriever([hit("a", 0.9), hit("b", 0.8)])
        indexer = mock.AsyncMock(return_value=1)
        with env(**RAG_ON, SOURCE_RAG_INGEST_ENABLED="true"), \
                mock.patch.object(rag_nodes, "get_retriever", return_value=retriever), \
                mock.patch.object(rag_nodes, "get_indexer", return_value=mock.Mock(index=indexer)):
            await rag_nodes.source_rag_node(state())
            await rag_nodes.index_sources_node(state(source_documents=[doc("https://a.io/x", "t")]))
        self.assertEqual(vectordb.get_cache_stats(), before)
        vectordb.clear_cache()


if __name__ == "__main__":
    unittest.main()
