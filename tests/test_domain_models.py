import asyncio
import json
import unittest
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from research_app.domain import (
    AnswerCache,
    Citation,
    CriticResult,
    Evidence,
    Gap,
    GapAnalysis,
    GapKind,
    ResearchError,
    ResearchQuery,
    ResearchResult,
    ResearchRun,
    ResearchTask,
    ResultStatus,
    SearchRequest,
    SourceAdapter,
    SourceDocument,
    SourceType,
    canonical_url,
    source_id_for,
)


class SourceDocumentTests(unittest.TestCase):
    def test_domain_and_id_are_derived(self):
        doc = SourceDocument(url="https://Docs.Python.org/3/library/asyncio.html")
        self.assertEqual(doc.domain, "docs.python.org")
        self.assertEqual(doc.source_id, source_id_for(doc.url))
        self.assertEqual(doc.source_type, SourceType.WEB)

    def test_source_id_ignores_case_slash_fragment_and_default_port(self):
        variants = [
            "https://example.com/a",
            "HTTPS://Example.COM/a/",
            "https://example.com:443/a#section",
        ]
        self.assertEqual(len({source_id_for(u) for u in variants}), 1)
        self.assertNotEqual(source_id_for("https://example.com/a"), source_id_for("https://example.com/b"))
        self.assertNotEqual(source_id_for("https://example.com/a?x=1"), source_id_for("https://example.com/a?x=2"))
        self.assertEqual(canonical_url("http://example.com:8080/x/"), "http://example.com:8080/x")

    def test_rejects_non_http_urls(self):
        for bad in ["javascript:alert(1)", "ftp://example.com/x", "file:///etc/passwd", "example.com", "", "https://"]:
            with self.subTest(url=bad):
                with self.assertRaises(ValidationError):
                    SourceDocument(url=bad)

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            SourceDocument(url="https://example.com", snipet="typo")

    def test_datetimes_are_utc_aware(self):
        naive = datetime(2026, 1, 2, 3, 4, 5)
        ist = timezone(timedelta(hours=5, minutes=30))
        doc = SourceDocument(
            url="https://example.com",
            published_at=naive,
            retrieved_at=datetime(2026, 1, 2, 8, 30, tzinfo=ist),
        )
        self.assertEqual(doc.published_at, naive.replace(tzinfo=timezone.utc))
        self.assertEqual(doc.retrieved_at, datetime(2026, 1, 2, 3, 0, tzinfo=timezone.utc))
        self.assertEqual(SourceDocument(url="https://example.com").retrieved_at.utcoffset(), timedelta(0))

    def test_official_docs_metadata_fields(self):
        doc = SourceDocument(
            url="https://fastapi.tiangolo.com/tutorial/",
            source_type=SourceType.OFFICIAL_DOCS,
            technology="fastapi",
            version="0.141",
        )
        self.assertEqual((doc.technology, doc.version), ("fastapi", "0.141"))
        self.assertEqual(doc.source_type.value, "official_docs")

    def test_json_round_trip(self):
        doc = SourceDocument(url="https://example.com/a", title="T", metadata={"stars": 5})
        again = SourceDocument.model_validate_json(doc.model_dump_json())
        self.assertEqual(doc, again)
        self.assertEqual(json.loads(doc.model_dump_json())["source_type"], "web")


class QueryAndTaskTests(unittest.TestCase):
    def test_query_defaults_and_validation(self):
        q = ResearchQuery(text="what is x")
        self.assertIsNone(q.complexity)
        self.assertEqual(q.intent.value, "unknown")
        self.assertEqual(q.history, [])
        with self.assertRaises(ValidationError):
            ResearchQuery(text="")

    def test_task_search_query_defaults_to_sub_question(self):
        t = ResearchTask(sub_question="best laptops 2026")
        self.assertEqual(t.search_query, "best laptops 2026")
        self.assertEqual(t.source_types, [SourceType.WEB])
        self.assertTrue(t.task_id)
        self.assertNotEqual(t.task_id, ResearchTask(sub_question="x").task_id)
        self.assertEqual(ResearchTask(sub_question="a", search_query="b").search_query, "b")

    def test_search_request_matches_current_tavily_settings(self):
        r = SearchRequest(query="q")
        self.assertEqual((r.max_results, r.search_depth, r.source_type), (3, "advanced", SourceType.WEB))
        with self.assertRaises(ValidationError):
            SearchRequest(query="q", max_results=0)
        with self.assertRaises(ValidationError):
            SearchRequest(query="q", timeout_s=0)
        with self.assertRaises(ValidationError):
            SearchRequest(query="q", search_depth="deep")


class ResultEvidenceTests(unittest.TestCase):
    def test_failed_and_timeout_results_require_an_error(self):
        for status in (ResultStatus.FAILED, ResultStatus.TIMEOUT):
            with self.subTest(status=status):
                with self.assertRaises(ValidationError):
                    ResearchResult(status=status)
        err = ResearchError(code="timeout", message="tavily timed out", retryable=True)
        result = ResearchResult(status=ResultStatus.TIMEOUT, error=err)
        self.assertFalse(result.succeeded)
        self.assertTrue(ResearchResult().succeeded)
        self.assertTrue(ResearchResult(status=ResultStatus.PARTIAL).succeeded)

    def test_evidence_hash_ignores_case_and_whitespace(self):
        kw = dict(source_id="s", url="https://example.com")
        a = Evidence(text="Hello   World\n", **kw)
        b = Evidence(text="hello world", **kw)
        c = Evidence(text="different", **kw)
        self.assertEqual(a.content_hash, b.content_hash)
        self.assertNotEqual(a.content_hash, c.content_hash)

    def test_scores_are_bounded(self):
        with self.assertRaises(ValidationError):
            Evidence(source_id="s", url="https://example.com", text="t", quality=1.5)
        with self.assertRaises(ValidationError):
            CriticResult(passed=True, score=-0.1)

    def test_citation_requires_http_url(self):
        Citation(marker="[1]", source_id="s", url="https://example.com")
        with self.assertRaises(ValidationError):
            Citation(marker="[1]", source_id="s", url="javascript:alert(1)")

    def test_gap_analysis_and_critic_hold_follow_up_tasks(self):
        task = ResearchTask(sub_question="what changed in v2?")
        analysis = GapAnalysis(
            iteration=1,
            sufficient=False,
            gaps=[Gap(kind=GapKind.MISSING_SUBQUESTION, description="no v2 info", task_id=task.task_id)],
            follow_up_tasks=[task],
        )
        self.assertEqual(GapAnalysis.model_validate_json(analysis.model_dump_json()), analysis)
        self.assertEqual(CriticResult(passed=False, retry_tasks=[task]).retry_tasks[0].sub_question, "what changed in v2?")
        with self.assertRaises(ValidationError):
            GapAnalysis(iteration=-1, sufficient=True)


class RunTests(unittest.TestCase):
    def test_run_defaults_and_round_trip(self):
        run = ResearchRun(query=ResearchQuery(text="q"))
        self.assertEqual(run.status.value, "pending")
        self.assertFalse(run.cache_hit)
        self.assertTrue(run.run_id)
        run.sources.append(SourceDocument(url="https://example.com"))
        run.evidence.append(Evidence(source_id=run.sources[0].source_id, url="https://example.com", text="fact"))
        self.assertEqual(ResearchRun.model_validate_json(run.model_dump_json()), run)


class InterfaceTests(unittest.TestCase):
    def test_fake_adapter_and_cache_satisfy_protocols(self):
        class FakeAdapter:
            source_type = SourceType.WEB

            async def search(self, request: SearchRequest) -> ResearchResult:
                return ResearchResult(
                    task_id=request.task_id,
                    documents=[SourceDocument(url="https://example.com", query=request.query)],
                )

        class FakeCache:
            def lookup(self, question):
                return False, ""

            def store(self, question, answer):
                return None

        self.assertIsInstance(FakeAdapter(), SourceAdapter)
        self.assertIsInstance(FakeCache(), AnswerCache)
        self.assertNotIsInstance(object(), SourceAdapter)
        result = asyncio.run(FakeAdapter().search(SearchRequest(query="q", task_id="t1")))
        self.assertEqual((result.task_id, result.documents[0].query), ("t1", "q"))


if __name__ == "__main__":
    unittest.main()
