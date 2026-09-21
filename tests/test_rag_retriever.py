import asyncio
import sys
import time
import unittest
from typing import Sequence
from unittest import mock

from research_app.domain import Reranker, RetrievalFilter, RetrievalMode, RetrievedChunk, SourceType
from research_app.rag import reranker as reranker_module
from research_app.rag.chunking import chunk_document
from research_app.rag.ingest import SourceIndexer, indexable
from research_app.rag.reranker import CrossEncoderReranker, NoopReranker, build_reranker
from research_app.rag.retriever import PROVIDER, HybridRetriever
from research_app.rag.settings import RagSettings
from tests.rag_helpers import CORPUS, doc, make_store, settings


class ReverseReranker:
    """Puts the worst fused hit first, so a changed order proves the reranker ran."""

    def rerank(self, _query: str, chunks: Sequence[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        ordered = list(reversed(chunks))[:top_k]
        return [RetrievedChunk(chunk=c.chunk, score=0.9 - 0.1 * i, rank=i, fused_score=c.score, reranked=True)
                for i, c in enumerate(ordered)]


def seeded_retriever(reranker=None, **overrides) -> HybridRetriever:
    store = make_store(**overrides)
    store.upsert([c for d in CORPUS for c in chunk_document(d, 1200, 100)])
    return HybridRetriever(settings(**overrides), store, reranker)


class RetrieveTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_source_documents_with_scores(self):
        docs = await seeded_retriever().retrieve("how do I install fastapi", top_k=2)
        self.assertEqual(len(docs), 2)
        first = docs[0]
        self.assertEqual(first.domain, "fastapi.tiangolo.com")
        self.assertIs(first.source_type, SourceType.OFFICIAL_DOCS)
        self.assertTrue(first.credibility.is_official)
        self.assertEqual(first.provider, PROVIDER)
        self.assertEqual(first.query, "how do I install fastapi")
        self.assertIn("pip install fastapi", first.content or "")
        self.assertEqual(first.metadata["rag_rank"], 0)
        self.assertFalse(first.metadata["rag_reranked"])
        self.assertTrue(0.0 <= first.metadata["rag_score"] <= 1.0)

    async def test_filters_are_applied(self):
        docs = await seeded_retriever().retrieve(
            "server", RetrievalFilter(source_types=[SourceType.REDDIT]))
        self.assertTrue(docs and all(d.source_type is SourceType.REDDIT for d in docs))

    async def test_top_k_defaults_come_from_settings(self):
        self.assertEqual(len(await seeded_retriever(top_k=3).retrieve("fastapi index tomato server")), 3)

    async def test_dense_mode_is_the_rollback_path(self):
        docs = await seeded_retriever(mode=RetrievalMode.DENSE).retrieve("install fastapi", top_k=2)
        self.assertEqual(docs[0].domain, "fastapi.tiangolo.com")

    async def test_rerank_changes_order_and_marks_documents(self):
        plain = await seeded_retriever().retrieve("install fastapi server", top_k=len(CORPUS))
        reranked = await seeded_retriever(ReverseReranker(), rerank_enabled=True, rerank_top_k=3).retrieve(
            "install fastapi server")
        self.assertEqual([d.url for d in reranked][0], [d.url for d in plain][-1])
        self.assertTrue(all(d.metadata["rag_reranked"] for d in reranked))
        self.assertLessEqual(len(reranked), 3)

    async def test_rerank_provider_none_skips_the_reranker(self):
        exploding = mock.Mock(spec=Reranker)
        exploding.rerank.side_effect = AssertionError("must not be called")
        retriever = seeded_retriever(exploding, rerank_enabled=True, rerank_provider="none")
        self.assertTrue(await retriever.retrieve("install fastapi"))

    async def test_empty_query_and_empty_store_return_nothing(self):
        self.assertEqual(await seeded_retriever().retrieve("   "), [])
        empty = HybridRetriever(settings(), make_store())
        self.assertEqual(await empty.retrieve("anything"), [])  # no collection yet: failure -> []

    async def test_store_failure_returns_nothing(self):
        retriever = seeded_retriever()
        with mock.patch.object(retriever.store, "search", side_effect=ConnectionError("qdrant down")):
            self.assertEqual(await retriever.retrieve("install fastapi"), [])

    async def test_timeout_returns_nothing(self):
        retriever = seeded_retriever(timeout_s=1)
        with mock.patch.object(retriever.store, "search", side_effect=lambda *_a, **_k: time.sleep(1.5)):
            started = time.perf_counter()
            self.assertEqual(await retriever.retrieve("install fastapi"), [])
        self.assertLess(time.perf_counter() - started, 1.4)

    async def test_cancellation_is_not_swallowed(self):
        retriever = seeded_retriever()
        with mock.patch.object(retriever.store, "search", side_effect=lambda *_a, **_k: time.sleep(0.5)):
            task = asyncio.create_task(retriever.retrieve("install fastapi"))
            await asyncio.sleep(0.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task


class RerankerTests(unittest.TestCase):
    def chunks(self):
        store = make_store()
        store.upsert([c for d in CORPUS for c in chunk_document(d, 1200, 100)])
        return store.search("install fastapi index", limit=4)

    def test_factory_follows_settings(self):
        self.assertIsInstance(build_reranker(RagSettings(rerank_enabled=False)), NoopReranker)
        self.assertIsInstance(build_reranker(RagSettings(rerank_provider="none")), NoopReranker)
        self.assertIsInstance(build_reranker(RagSettings(rerank_provider="cohere")), NoopReranker)
        self.assertIsInstance(build_reranker(RagSettings()), CrossEncoderReranker)  # lazy: nothing loads

    def test_new_provider_is_one_registry_entry(self):
        with mock.patch.dict(reranker_module.RERANKERS, {"fake": lambda s: ReverseReranker()}):
            built = build_reranker(RagSettings(rerank_provider="fake"))
        self.assertIsInstance(built, ReverseReranker)
        self.assertIsInstance(built, Reranker)

    def test_cross_encoder_reorders_by_model_score(self):
        chunks = self.chunks()
        model = mock.Mock()
        model.predict.return_value = [0.1, 0.9, 0.5, 0.2][:len(chunks)]
        reranker = CrossEncoderReranker("m")
        reranker._model = model
        out = reranker.rerank("q", chunks, 2)
        self.assertEqual([r.chunk.chunk_id for r in out], [chunks[1].chunk.chunk_id, chunks[2].chunk.chunk_id])
        self.assertEqual([round(r.score, 1) for r in out], [0.9, 0.5])
        self.assertTrue(all(r.reranked for r in out))
        self.assertEqual(out[0].fused_score, chunks[1].fused_score)  # the raw pre-rerank score is kept

    def test_unavailable_model_keeps_the_fused_order(self):
        chunks = self.chunks()
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):  # import raises ImportError
            reranker = CrossEncoderReranker("m")
            out = reranker.rerank("q", chunks, 3)
            again = reranker.rerank("q", chunks, 3)
        self.assertEqual([r.chunk.chunk_id for r in out], [c.chunk.chunk_id for c in chunks[:3]])
        self.assertFalse(any(r.reranked for r in out))
        self.assertEqual(len(again), 3)

    def test_scoring_failure_keeps_the_fused_order(self):
        chunks = self.chunks()
        reranker = CrossEncoderReranker("m")
        reranker._model = mock.Mock(predict=mock.Mock(side_effect=RuntimeError("oom")))
        self.assertEqual(len(reranker.rerank("q", chunks, 2)), 2)
        self.assertEqual(reranker.rerank("q", [], 2), [])


class IndexerTests(unittest.IsolatedAsyncioTestCase):
    def test_indexable_skips_empty_and_already_stored_documents(self):
        stored = doc("https://a.example.com/1", "text", provider=PROVIDER)
        empty = doc("https://a.example.com/2", "   ")
        fresh = doc("https://a.example.com/3", "text")
        self.assertEqual(indexable([stored, empty, fresh]), [fresh])

    async def test_index_writes_chunks_and_never_raises(self):
        store = make_store(chunk_size=200, chunk_overlap=20)
        indexer = SourceIndexer(settings(chunk_size=200, chunk_overlap=20), store)
        written = await indexer.index([doc("https://a.example.com/long", "lorem ipsum " * 100)])
        self.assertGreater(written, 1)
        self.assertEqual(store.count(), written)
        with mock.patch.object(store, "upsert", side_effect=ConnectionError("down")):
            self.assertEqual(await indexer.index([doc("https://a.example.com/n", "new text")]), 0)


class SettingsTests(unittest.TestCase):
    def test_defaults_are_off_and_hybrid(self):
        s = RagSettings.from_env({})
        self.assertFalse(s.source_rag_enabled)
        self.assertFalse(s.ingest_enabled)
        self.assertIs(s.mode, RetrievalMode.HYBRID)
        self.assertEqual((s.collection, s.rrf_k, s.top_k, s.rerank_top_k), ("source_chunks", 60, 8, 5))
        self.assertEqual(s.rerank_model, "BAAI/bge-reranker-v2-m3")

    def test_env_overrides_and_malformed_values(self):
        s = RagSettings.from_env({
            "SOURCE_RAG_ENABLED": "true", "RAG_MODE": "dense", "RAG_RRF_K": "20", "RAG_TOP_K": "abc",
            "SOURCE_CHUNKS_COLLECTION": "chunks_x", "RERANK_MODEL": "org/model", "RAG_DENSE_TOP_K": "12",
            "RAG_RELEVANCE_THRESHOLD": "2", "QDRANT_API_KEY": "secret"})
        self.assertTrue(s.source_rag_enabled)
        self.assertIs(s.mode, RetrievalMode.DENSE)
        self.assertEqual((s.rrf_k, s.top_k, s.dense_top_k, s.collection, s.rerank_model),
                         (20, 8, 12, "chunks_x", "org/model"))
        self.assertEqual(s.relevance_threshold, 0.5)
        self.assertNotIn("secret", repr(s))

    def test_bad_mode_and_overlap_fall_back(self):
        s = RagSettings.from_env({"RAG_MODE": "nope", "RAG_CHUNK_SIZE": "400", "RAG_CHUNK_OVERLAP": "400"})
        self.assertIs(s.mode, RetrievalMode.HYBRID)
        self.assertLess(s.chunk_overlap, s.chunk_size)


if __name__ == "__main__":
    unittest.main()
