"""Unit tests for the source-agnostic normalizer. Pure functions: no network, no LLM,
no database, no clock dependence (retrieved_at is passed in where it is asserted)."""
import os
import subprocess
import sys
import unittest
from datetime import datetime, timedelta, timezone

from research_app.domain import SourceDocument, SourceType, source_id_for
from research_app.sources import (
    SNIPPET_MAX_CHARS,
    NormalizationError,
    clean_text,
    normalize_source,
    normalize_sources,
    normalize_url,
    parse_datetime,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

RAW = {
    "title": "Quickstart - FastAPI",
    "url": "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "content": "Create a file main.py.\nRun the server with uvicorn.",
    "score": 0.91,
}


class ValidResultTests(unittest.TestCase):
    def test_normal_result(self):
        doc = normalize_source(RAW, provider="tavily", query="how to run fastapi", retrieved_at=T0)
        self.assertIsInstance(doc, SourceDocument)
        self.assertEqual(doc.title, "Quickstart - FastAPI")
        self.assertEqual(doc.url, "https://fastapi.tiangolo.com/tutorial/first-steps/")
        self.assertEqual(doc.domain, "fastapi.tiangolo.com")
        self.assertEqual(doc.content, "Create a file main.py.\nRun the server with uvicorn.")
        self.assertEqual(doc.snippet, "Create a file main.py. Run the server with uvicorn.")
        self.assertEqual(doc.source_type, SourceType.WEB)
        self.assertEqual(doc.retrieved_at, T0)

    def test_dump_and_reload_round_trip(self):
        doc = normalize_source(RAW, provider="tavily", query="q", retrieved_at=T0)
        self.assertEqual(SourceDocument.model_validate(doc.model_dump(mode="json")), doc)


class MissingFieldTests(unittest.TestCase):
    def test_missing_title(self):
        doc = normalize_source({"url": "https://example.com/a", "content": "body"})
        self.assertIsNone(doc.title)
        self.assertEqual(doc.content, "body")

    def test_blank_title_is_none(self):
        self.assertIsNone(normalize_source({"url": "https://example.com", "title": "  \n "}).title)

    def test_missing_url_is_rejected(self):
        for raw in (
            {"title": "t", "content": "c"},
            {"url": None},
            {"url": ""},
            {"url": "   "},
            {"url": 123},
            {"url": ["https://example.com"]},
        ):
            with self.subTest(raw=raw), self.assertRaises(NormalizationError):
                normalize_source(raw)

    def test_unusable_urls_are_rejected(self):
        for url in (
            "javascript:alert(1)",
            "file:///etc/passwd",
            "ftp://example.com/x",
            "example.com/no-scheme",
            "https://",
            "http://[::1",                       # unterminated IPv6
            "https://example.com:notaport/",
            "https://user:secret@example.com/",  # credentials never enter a document
        ):
            with self.subTest(url=url), self.assertRaises(NormalizationError):
                normalize_source({"url": url})

    def test_missing_content_and_snippet(self):
        doc = normalize_source({"url": "https://example.com/a", "title": "T"})
        self.assertIsNone(doc.content)
        self.assertIsNone(doc.snippet)

    def test_snippet_without_content(self):
        doc = normalize_source({"url": "https://example.com/a", "snippet": "  only a   snippet "})
        self.assertIsNone(doc.content)
        self.assertEqual(doc.snippet, "only a snippet")

    def test_missing_optional_fields_default_cleanly(self):
        doc = normalize_source({"url": "https://example.com/a"})
        self.assertEqual(doc.metadata, {})
        self.assertIsNone(doc.author)
        self.assertIsNone(doc.published_at)
        self.assertIsNone(doc.technology)
        self.assertIsNone(doc.version)


class UrlAndDomainTests(unittest.TestCase):
    def test_host_and_scheme_lowercased_domain_extracted(self):
        doc = normalize_source({"url": "HTTPS://Docs.Python.ORG/3/Library/Asyncio.html"})
        self.assertEqual(doc.url, "https://docs.python.org/3/Library/Asyncio.html")  # path case kept
        self.assertEqual(doc.domain, "docs.python.org")

    def test_default_ports_dropped_others_kept_domain_has_no_port(self):
        self.assertEqual(normalize_url("http://example.com:80/x"), "http://example.com/x")
        self.assertEqual(normalize_url("https://example.com:443/x"), "https://example.com/x")
        self.assertEqual(normalize_url("https://example.com:8443/x"), "https://example.com:8443/x")
        self.assertEqual(normalize_source({"url": "https://example.com:8443/x"}).domain, "example.com")

    def test_query_and_fragment_kept(self):
        url = "https://example.com/docs?version=3&q=a#install"
        self.assertEqual(normalize_url(url), url)

    def test_whitespace_and_control_characters_removed(self):
        self.assertEqual(normalize_url("  https://example.com/a\n\t "), "https://example.com/a")
        self.assertEqual(normalize_url("https://exam\tple.com/a\x00"), "https://example.com/a")

    def test_www_is_not_stripped(self):
        self.assertEqual(normalize_source({"url": "https://www.example.com/a"}).domain, "www.example.com")

    def test_ipv6_literal(self):
        self.assertEqual(normalize_url("http://[::1]:8000/x"), "http://[::1]:8000/x")

    def test_original_url_is_kept_as_returned(self):
        doc = normalize_source({"url": "  HTTPS://Example.COM:443/A#frag "})
        self.assertEqual(doc.original_url, "HTTPS://Example.COM:443/A#frag")
        self.assertEqual(doc.url, "https://example.com/A#frag")


class SourceTypeTests(unittest.TestCase):
    def test_default_is_web(self):
        self.assertEqual(normalize_source({"url": "https://example.com"}).source_type, SourceType.WEB)

    def test_planned_source_types_can_be_assigned(self):
        for st in ("web", "official_docs", "github", "reddit"):
            with self.subTest(source_type=st):
                doc = normalize_source({"url": "https://example.com"}, source_type=st)
                self.assertEqual(doc.source_type, SourceType(st))

    def test_unknown_source_type_is_a_caller_error(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_source({"url": "https://example.com"}, source_type="blog")
        self.assertNotIsInstance(ctx.exception, NormalizationError)


class StableIdTests(unittest.TestCase):
    def test_id_is_pinned(self):
        # Ids will be persisted in later phases; changing the derivation is a breaking change.
        doc = normalize_source({"url": "https://Docs.Python.org/3/"})
        self.assertEqual(doc.source_id, "b4365d45965be1c1c1af00c627b043aa")

    def test_same_url_same_id_regardless_of_other_fields(self):
        a = normalize_source({"url": "https://example.com/a", "title": "A", "content": "one"}, query="q1")
        b = normalize_source({"url": "https://example.com/a", "title": "B", "content": "two"},
                             query="q2", source_type=SourceType.OFFICIAL_DOCS)
        self.assertEqual(a.source_id, b.source_id)

    def test_cosmetic_url_variants_share_an_id(self):
        ids = {
            normalize_source({"url": u}).source_id
            for u in (
                "https://example.com/a",
                "HTTPS://EXAMPLE.COM/a",
                "https://example.com:443/a",
                "https://example.com/a/",
                "https://example.com/a#section",
            )
        }
        self.assertEqual(len(ids), 1)

    def test_different_urls_get_different_ids(self):
        ids = {
            normalize_source({"url": u}).source_id
            for u in ("https://example.com/a", "https://example.com/b", "https://example.com/a?x=1",
                      "http://example.com/a")
        }
        self.assertEqual(len(ids), 4)

    def test_id_matches_domain_helper(self):
        doc = normalize_source({"url": "https://example.com/a"})
        self.assertEqual(doc.source_id, source_id_for("https://example.com/a"))


class MetadataTests(unittest.TestCase):
    def test_unmapped_fields_are_kept_verbatim(self):
        doc = normalize_source({**RAW, "favicon": "https://x/y.ico", "nested": {"a": [1, 2]}, "flag": True})
        self.assertEqual(doc.metadata, {"score": 0.91, "favicon": "https://x/y.ico",
                                        "nested": {"a": [1, 2]}, "flag": True})

    def test_consumed_fields_are_not_duplicated_into_metadata(self):
        doc = normalize_source({"url": "https://example.com", "title": "T", "content": "C",
                                "snippet": "S", "author": "A", "published_at": "2025-01-01"})
        self.assertEqual(doc.metadata, {})

    def test_wrongly_typed_field_is_dropped_but_not_lost(self):
        doc = normalize_source({"url": "https://example.com", "title": 42, "content": ["a", "b"]})
        self.assertIsNone(doc.title)
        self.assertIsNone(doc.content)
        self.assertEqual(doc.metadata, {"title": 42, "content": ["a", "b"]})

    def test_non_string_keys_do_not_break_validation(self):
        doc = normalize_source({"url": "https://example.com", 7: "seven"})
        self.assertEqual(doc.metadata, {"7": "seven"})

    def test_input_mapping_is_not_mutated(self):
        raw = dict(RAW)
        normalize_source(raw)
        self.assertEqual(raw, RAW)


class ProvenanceTests(unittest.TestCase):
    def test_provenance_fields(self):
        doc = normalize_source(RAW, source_type="official_docs", provider="tavily",
                               query="how to run fastapi", task_id="task-1", retrieved_at=T0)
        self.assertEqual(doc.provider, "tavily")
        self.assertEqual(doc.source_type, SourceType.OFFICIAL_DOCS)
        self.assertEqual(doc.original_url, RAW["url"])
        self.assertEqual(doc.query, "how to run fastapi")
        self.assertEqual(doc.task_id, "task-1")
        self.assertEqual(doc.retrieved_at, T0)

    def test_provenance_is_not_invented(self):
        doc = normalize_source({"url": "https://example.com"})
        self.assertIsNone(doc.provider)
        self.assertIsNone(doc.query)
        self.assertIsNone(doc.task_id)

    def test_retrieval_time_is_set_and_utc_when_not_given(self):
        doc = normalize_source({"url": "https://example.com"})
        self.assertEqual(doc.retrieved_at.utcoffset(), timedelta(0))


class TextCleaningTests(unittest.TestCase):
    def test_content_keeps_lines_and_indentation(self):
        raw = "Example:\r\n\r\n\r\n\r\n    def f():\r\n        return 1   \r\n"
        self.assertEqual(clean_text(raw), "Example:\n\n    def f():\n        return 1")

    def test_invisible_and_control_characters_removed(self):
        self.assertEqual(clean_text("a b​c﻿\x00d\x07e"), "a bcde")

    def test_unicode_is_composed(self):
        self.assertEqual(clean_text("café"), "café")

    def test_single_line_collapses_all_whitespace(self):
        self.assertEqual(clean_text("  A \n\n  title\twith   gaps ", single_line=True), "A title with gaps")

    def test_empty_or_wrong_type_is_none(self):
        for value in ("", "   \n\t", None, 5, b"bytes", ["x"]):
            with self.subTest(value=value):
                self.assertIsNone(clean_text(value))

    def test_original_wording_is_untouched(self):
        # Not the lossy sanitize_text: punctuation, symbols and non-ASCII survive.
        text = "Price: ₹500 — “quoted” °C ^ é"
        self.assertEqual(clean_text(text), text)

    def test_derived_snippet_is_first_150_chars_of_collapsed_content(self):
        content = ("word " * 100).strip() + "\n" + "tail"
        doc = normalize_source({"url": "https://example.com", "content": content})
        self.assertEqual(len(doc.snippet), SNIPPET_MAX_CHARS)
        self.assertEqual(doc.snippet, " ".join(content.split())[:150])
        self.assertEqual(doc.content, content)  # content itself is never truncated

    def test_provided_snippet_wins(self):
        doc = normalize_source({"url": "https://example.com", "content": "long body", "snippet": "given"})
        self.assertEqual(doc.snippet, "given")


class DateTests(unittest.TestCase):
    def test_accepted_formats(self):
        expected = datetime(2025, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
        for value in ("2025-03-04T05:06:07Z", "2025-03-04T05:06:07+00:00", "Tue, 04 Mar 2025 05:06:07 GMT",
                      expected, datetime(2025, 3, 4, 5, 6, 7)):
            with self.subTest(value=value):
                doc = normalize_source({"url": "https://example.com", "published_at": value})
                self.assertEqual(doc.published_at, expected)

    def test_offsets_are_converted_to_utc(self):
        doc = normalize_source({"url": "https://example.com", "published_at": "2025-03-04T05:06:07+02:00"})
        self.assertEqual(doc.published_at, datetime(2025, 3, 4, 3, 6, 7, tzinfo=timezone.utc))

    def test_published_date_alias(self):
        doc = normalize_source({"url": "https://example.com", "published_date": "Tue, 04 Mar 2025 05:06:07 GMT"})
        self.assertEqual(doc.published_at, datetime(2025, 3, 4, 5, 6, 7, tzinfo=timezone.utc))
        self.assertEqual(doc.metadata, {})

    def test_unparseable_date_is_none_and_kept_in_metadata(self):
        doc = normalize_source({"url": "https://example.com", "published_at": "last Tuesday"})
        self.assertIsNone(doc.published_at)
        self.assertEqual(doc.metadata, {"published_at": "last Tuesday"})

    def test_parse_datetime_rejects_non_dates(self):
        for value in (None, "", "  ", 5, 1.5, ["2025-01-01"], True):
            with self.subTest(value=value):
                self.assertIsNone(parse_datetime(value))


class BatchTests(unittest.TestCase):
    def test_results_normalize_independently_in_order(self):
        raws = [
            {"url": "https://a.example/1", "title": "one", "content": "c1"},
            {"title": "no url"},
            {"url": "https://b.example/2", "title": "two"},
        ]
        with self.assertLogs("research_app.sources.normalizer", level="WARNING") as logs:
            docs = normalize_sources(raws, provider="tavily", query="q")
        self.assertEqual([d.title for d in docs], ["one", "two"])
        self.assertEqual([d.content for d in docs], ["c1", None])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("Skipping result 1 from tavily", logs.output[0])

    def test_no_deduplication(self):
        docs = normalize_sources([{"url": "https://example.com/a"}, {"url": "https://EXAMPLE.com/a/"}])
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].source_id, docs[1].source_id)

    def test_batch_shares_one_retrieval_time_and_options(self):
        docs = normalize_sources([{"url": "https://a.example"}, {"url": "https://b.example"}],
                                 retrieved_at=T0, source_type="github", query="q", task_id="t")
        self.assertEqual({d.retrieved_at for d in docs}, {T0})
        self.assertEqual({(d.source_type, d.query, d.task_id) for d in docs}, {(SourceType.GITHUB, "q", "t")})

    def test_empty_input(self):
        self.assertEqual(normalize_sources([]), [])


class MalformedInputTests(unittest.TestCase):
    MALFORMED = [None, "https://example.com", 42, ["url"], ("url", "https://example.com"),
                 {}, {"url": {"href": "https://example.com"}}, object()]

    def test_single_malformed_results_raise_normalization_error(self):
        for raw in self.MALFORMED:
            with self.subTest(raw=raw), self.assertRaises(NormalizationError):
                normalize_source(raw)

    def test_batch_survives_malformed_results_and_returns_only_valid_documents(self):
        good = {"url": "https://example.com/ok", "title": "ok"}
        with self.assertLogs("research_app.sources.normalizer", level="WARNING"):
            docs = normalize_sources([*self.MALFORMED, good])
        self.assertEqual([d.title for d in docs], ["ok"])

    def test_error_and_log_never_contain_raw_content(self):
        secret = "SECRET-BODY-TEXT"
        with self.assertLogs("research_app.sources.normalizer", level="WARNING") as logs:
            normalize_sources([{"title": secret, "content": secret}])
        self.assertNotIn(secret, "\n".join(logs.output))
        with self.assertRaises(NormalizationError) as ctx:
            normalize_source({"url": "javascript:alert(1)", "content": secret})
        self.assertNotIn(secret, str(ctx.exception))
        self.assertNotIn("alert", str(ctx.exception))


class IsolationTests(unittest.TestCase):
    def test_sources_package_pulls_in_no_app_or_framework_modules(self):
        code = (
            "import sys, research_app.sources\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "{'langgraph','langchain_core','langchain','langchain_groq','langchain_tavily',"
            "'qdrant_client','sqlalchemy','fastapi'} "
            "or m.startswith(('research_app.agent','research_app.db','research_app.main','research_app.auth')))\n"
            "print(','.join(bad))\n"
        )
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                             capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "", f"sources imported: {out.stdout.strip()}")


if __name__ == "__main__":
    unittest.main()
