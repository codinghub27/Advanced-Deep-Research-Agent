"""Tavily response -> ResearchResult. Canned payloads only; Tavily is never called."""
import unittest
from datetime import datetime, timezone

from research_app.domain import ResearchResult, ResultStatus, SourceType
from research_app.sources import normalize_tavily_response

T0 = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def item(n, **overrides):
    base = {
        "title": f"Result {n}",
        "url": f"https://site{n}.example/page",
        "content": f"Body of result {n}.",
        "score": 0.9 - n / 100,
    }
    base.update(overrides)
    return base


def response(*items, **extra):
    return {"query": "q", "answer": "LLM answer", "results": list(items), "images": [],
            "response_time": 1.2, **extra}


class SuccessTests(unittest.TestCase):
    def test_ok_result(self):
        res = normalize_tavily_response(response(item(1), item(2), item(3)), query="q", retrieved_at=T0)
        self.assertIsInstance(res, ResearchResult)
        self.assertEqual((res.status, res.provider, res.source_type), (ResultStatus.OK, "tavily", SourceType.WEB))
        self.assertIsNone(res.error)
        self.assertTrue(res.succeeded)
        self.assertEqual([d.title for d in res.documents], ["Result 1", "Result 2", "Result 3"])

    def test_documents_carry_provenance_and_provider_metadata(self):
        res = normalize_tavily_response(response(item(1)), query="the query", task_id="t9", retrieved_at=T0)
        (doc,) = res.documents
        self.assertEqual((doc.provider, doc.query, doc.task_id, doc.retrieved_at),
                         ("tavily", "the query", "t9", T0))
        self.assertEqual(doc.source_type, SourceType.WEB)
        self.assertEqual(doc.original_url, "https://site1.example/page")
        self.assertEqual(doc.metadata, {"score": 0.89})
        self.assertEqual(res.task_id, "t9")

    def test_response_level_fields_are_not_turned_into_documents(self):
        res = normalize_tavily_response(response(item(1)), query="q")
        self.assertEqual(len(res.documents), 1)  # `answer` is not a source

    def test_empty_results_is_a_valid_empty_ok(self):
        res = normalize_tavily_response(response(), query="q")
        self.assertEqual((res.status, res.documents, res.error), (ResultStatus.OK, [], None))

    def test_max_results_caps_raw_results_before_normalizing(self):
        res = normalize_tavily_response(response(*[item(n) for n in range(1, 6)]), query="q", max_results=3)
        self.assertEqual([d.title for d in res.documents], ["Result 1", "Result 2", "Result 3"])

    def test_results_without_content_are_kept(self):
        res = normalize_tavily_response(response(item(1, content=None), item(2, content="")), query="q")
        self.assertEqual(len(res.documents), 2)
        self.assertEqual([d.content for d in res.documents], [None, None])

    def test_results_share_one_retrieval_time(self):
        res = normalize_tavily_response(response(item(1), item(2)), query="q")
        self.assertEqual(len({d.retrieved_at for d in res.documents}), 1)


class PartialAndFailureTests(unittest.TestCase):
    def test_some_unusable_results_gives_partial(self):
        with self.assertLogs("research_app.sources.normalizer", level="WARNING"):
            res = normalize_tavily_response(response(item(1), item(2, url=""), item(3)), query="q")
        self.assertEqual(res.status, ResultStatus.PARTIAL)
        self.assertTrue(res.succeeded)
        self.assertEqual([d.title for d in res.documents], ["Result 1", "Result 3"])

    def test_cap_is_applied_before_filtering(self):
        # Same as the old `result_list[:3]`: an unusable result inside the cap still uses a slot.
        with self.assertLogs("research_app.sources.normalizer", level="WARNING"):
            res = normalize_tavily_response(
                response(item(1), item(2, url="javascript:x"), item(3), item(4)), query="q", max_results=3)
        self.assertEqual([d.title for d in res.documents], ["Result 1", "Result 3"])
        self.assertEqual(res.status, ResultStatus.PARTIAL)

    def test_all_results_unusable_is_failed(self):
        with self.assertLogs("research_app.sources.normalizer", level="WARNING"):
            res = normalize_tavily_response(response(item(1, url=None), "junk", None), query="q")
        self.assertEqual(res.status, ResultStatus.FAILED)
        self.assertEqual(res.error.code, "no_valid_results")
        self.assertEqual(res.documents, [])
        self.assertFalse(res.succeeded)

    def test_swallowed_exception_from_the_tavily_wrapper(self):
        res = normalize_tavily_response({"error": RuntimeError("HTTP 401 Unauthorized")}, query="q")
        self.assertEqual(res.status, ResultStatus.FAILED)
        self.assertEqual(res.error.code, "provider_error")
        self.assertEqual(res.error.message, "RuntimeError: HTTP 401 Unauthorized")
        self.assertEqual(res.error.source_type, SourceType.WEB)
        self.assertEqual(res.documents, [])

    def test_tool_exception_message_string_such_as_no_results(self):
        # handle_tool_error=True turns a ToolException into a plain string return value.
        res = normalize_tavily_response("No search results found for 'q'. Suggestions: ...", query="q")
        self.assertEqual(res.status, ResultStatus.FAILED)
        self.assertEqual(res.error.code, "provider_error")
        self.assertTrue(res.error.message.startswith("No search results found"))
        self.assertEqual(res.documents, [])

    def test_error_message_is_bounded(self):
        res = normalize_tavily_response({"error": "x" * 5000}, query="q")
        self.assertEqual(len(res.error.message), 200)

    def test_results_present_wins_over_error_key(self):
        res = normalize_tavily_response({"error": "ignored", "results": [item(1)]}, query="q")
        self.assertEqual(res.status, ResultStatus.OK)

    def test_unexpected_shapes_fail_cleanly_and_never_raise(self):
        for bad in (None, 5, [item(1)], {}, {"results": None},
                    {"results": "text"}, {"results": {"0": item(1)}}, {"answer": "only"}):
            with self.subTest(bad=bad):
                res = normalize_tavily_response(bad, query="q", task_id="t")
                self.assertEqual(res.status, ResultStatus.FAILED)
                self.assertEqual(res.error.code, "invalid_response")
                self.assertEqual((res.documents, res.provider, res.task_id), ([], "tavily", "t"))


if __name__ == "__main__":
    unittest.main()
