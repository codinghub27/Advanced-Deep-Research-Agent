import unittest
from datetime import timedelta
from unittest import mock

from qdrant_client import models

from research_app.domain import RetrievalFilter, RetrievalMode, SourceType, utc_now
from research_app.rag.chunking import chunk_document
from tests.rag_helpers import CORPUS, doc, make_store


def seeded_store(**overrides):
    store = make_store(**overrides)
    chunks = [c for d in CORPUS for c in chunk_document(d, 1200, 100)]
    store.upsert(chunks)
    return store


def urls(hits):
    return [h.chunk.url for h in hits]


class UpsertTests(unittest.TestCase):
    def test_reindexing_overwrites_instead_of_duplicating(self):
        store = seeded_store()
        before = store.count()
        store.upsert([c for d in CORPUS for c in chunk_document(d, 1200, 100)])
        self.assertEqual(store.count(), before)
        self.assertEqual(before, len(CORPUS))

    def test_empty_upsert_creates_nothing(self):
        store = make_store()
        self.assertEqual(store.upsert([]), 0)


class SearchModeTests(unittest.TestCase):
    def setUp(self):
        self.store = seeded_store()

    def test_dense_finds_the_related_document(self):
        hits = self.store.search("how to install fastapi", mode=RetrievalMode.DENSE, limit=3)
        self.assertEqual(hits[0].chunk.domain, "fastapi.tiangolo.com")

    def test_sparse_matches_the_exact_rare_token(self):
        hits = self.store.search("concurrently", mode=RetrievalMode.SPARSE, limit=3)
        self.assertEqual(hits[0].chunk.domain, "www.postgresql.org")
        self.assertEqual(hits[0].score, 1.0)  # sparse scores are relative to the best hit

    def test_hybrid_combines_both_legs_and_scores_stay_in_range(self):
        hits = self.store.search("install fastapi concurrently", mode=RetrievalMode.HYBRID, limit=4)
        domains = [h.chunk.domain for h in hits[:2]]
        self.assertIn("fastapi.tiangolo.com", domains)
        self.assertIn("www.postgresql.org", domains)
        self.assertTrue(all(0.0 <= h.score <= 1.0 for h in hits))
        self.assertEqual([h.rank for h in hits], list(range(len(hits))))

    def test_rrf_constant_changes_scores(self):
        low = seeded_store(rrf_k=1).search("install fastapi", mode=RetrievalMode.HYBRID, limit=2)
        high = seeded_store(rrf_k=500).search("install fastapi", mode=RetrievalMode.HYBRID, limit=2)
        self.assertNotEqual(low[0].fused_score, high[0].fused_score)

    def test_client_side_rrf_is_used_when_the_server_rejects_rrf_with_k(self):
        store = seeded_store()
        real = store.client.query_points

        def reject_rrf(*args, **kwargs):
            if isinstance(kwargs.get("query"), models.RrfQuery):
                raise RuntimeError("unknown variant rrf")
            return real(*args, **kwargs)

        with mock.patch.object(store.client, "query_points", side_effect=reject_rrf):
            hits = store.search("install fastapi", mode=RetrievalMode.HYBRID, limit=3)
        self.assertEqual(hits[0].chunk.domain, "fastapi.tiangolo.com")

    def test_no_match_in_an_empty_collection_returns_nothing(self):
        store = make_store()
        store.ensure_collection()
        self.assertEqual(store.search("anything", limit=3), [])


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.store = seeded_store()

    def search(self, flt, mode=RetrievalMode.HYBRID):
        return self.store.search("fastapi uvicorn server index", mode=mode, flt=flt, limit=10)

    def test_source_type(self):
        for mode in RetrievalMode:
            hits = self.search(RetrievalFilter(source_types=[SourceType.REDDIT]), mode)
            self.assertTrue(hits and all(h.chunk.source_type is SourceType.REDDIT for h in hits), mode)

    def test_domain_and_technology(self):
        hits = self.search(RetrievalFilter(domains=["FastAPI.tiangolo.com"]))
        self.assertEqual({h.chunk.domain for h in hits}, {"fastapi.tiangolo.com"})
        hits = self.search(RetrievalFilter(technology="fastapi"))
        self.assertEqual({h.chunk.technology for h in hits}, {"fastapi"})
        self.assertEqual(len(hits), 2)

    def test_retrieved_at_range(self):
        old = utc_now() - timedelta(days=400)
        store = make_store()
        store.upsert([c for d in [doc("https://old.example.com/a", "fastapi old page", retrieved_at=old),
                                   doc("https://new.example.com/a", "fastapi new page")]
                      for c in chunk_document(d, 1200, 100)])
        cutoff = utc_now() - timedelta(days=30)
        recent = store.search("fastapi page", flt=RetrievalFilter(retrieved_after=cutoff), limit=5)
        self.assertEqual(urls(recent), ["https://new.example.com/a"])
        stale = store.search("fastapi page", flt=RetrievalFilter(retrieved_before=cutoff), limit=5)
        self.assertEqual(urls(stale), ["https://old.example.com/a"])

    def test_filter_with_no_match_returns_nothing(self):
        self.assertEqual(self.search(RetrievalFilter(technology="cobol")), [])

    def test_empty_filter_does_not_filter(self):
        self.assertEqual(len(self.search(RetrievalFilter())), len(CORPUS))


class IsolationTests(unittest.TestCase):
    def test_answer_cache_collection_is_never_touched(self):
        store = make_store()
        client = store.client
        client.create_collection("answer_cache", vectors_config=models.VectorParams(
            size=4, distance=models.Distance.COSINE))
        client.upsert("answer_cache", points=[models.PointStruct(id=1, vector=[1, 0, 0, 0], payload={"q": "x"})])
        store.upsert([c for d in CORPUS for c in chunk_document(d, 1200, 100)])
        store.search("install fastapi", limit=3)
        self.assertEqual(client.count("answer_cache", exact=True).count, 1)
        self.assertEqual(store.count(), len(CORPUS))
        self.assertEqual(sorted(c.name for c in client.get_collections().collections),
                         ["answer_cache", "source_chunks"])


if __name__ == "__main__":
    unittest.main()
