"""P1.4 -- source-type (content) classification. Pure-function tests, no LLM/network/DB."""
from __future__ import annotations

import unittest

from research_app.domain import AuthorityLevel, ContentClassification, SourceDocument, SourceType
from research_app.sources.classification import classify_content


def _doc(**kwargs) -> SourceDocument:
    kwargs.setdefault("url", "https://example.com/a")
    return SourceDocument(**kwargs)


class OfficialDocsTests(unittest.TestCase):
    def test_plain_docs_page_is_official_documentation(self):
        doc = classify_content(_doc(
            url="https://fastapi.tiangolo.com/tutorial/", source_type=SourceType.OFFICIAL_DOCS))
        self.assertEqual(doc.content_classification, ContentClassification.OFFICIAL_DOCUMENTATION)
        self.assertEqual(doc.authority_level, AuthorityLevel.OFFICIAL)
        self.assertTrue(doc.is_primary_source)
        self.assertTrue(doc.independently_verified)
        self.assertGreaterEqual(float(doc.classification_confidence or 0), 0.9)

    def test_changelog_path_is_official_announcement(self):
        doc = classify_content(_doc(
            url="https://docs.example.com/changelog/2026", source_type=SourceType.OFFICIAL_DOCS))
        self.assertEqual(doc.content_classification, ContentClassification.OFFICIAL_ANNOUNCEMENT)
        self.assertEqual(doc.authority_level, AuthorityLevel.OFFICIAL)

    def test_announcing_title_is_official_announcement(self):
        doc = classify_content(_doc(
            url="https://docs.example.com/x", title="Announcing FastAPI 1.0",
            source_type=SourceType.OFFICIAL_DOCS))
        self.assertEqual(doc.content_classification, ContentClassification.OFFICIAL_ANNOUNCEMENT)


class GitHubTests(unittest.TestCase):
    def test_repository_kind(self):
        doc = classify_content(_doc(
            url="https://github.com/langchain-ai/langgraph", source_type=SourceType.GITHUB,
            metadata={"github": {"kind": "repository", "owner": "langchain-ai", "repo": "langgraph"}}))
        self.assertEqual(doc.content_classification, ContentClassification.GITHUB_REPOSITORY)
        self.assertEqual(doc.authority_level, AuthorityLevel.COMMUNITY)
        self.assertTrue(doc.is_primary_source)

    def test_issue_kind_is_discussion_not_authority(self):
        doc = classify_content(_doc(
            url="https://github.com/o/r/issues/1", source_type=SourceType.GITHUB,
            metadata={"github": {"kind": "issue", "number": 1}}))
        self.assertEqual(doc.content_classification, ContentClassification.GITHUB_ISSUE_OR_DISCUSSION)
        self.assertEqual(doc.authority_level, AuthorityLevel.COMMUNITY)
        self.assertFalse(doc.is_primary_source)

    def test_missing_github_metadata_is_unknown_not_upgraded(self):
        # A GitHub result whose kind we can't determine must not be silently treated as a
        # repository -- "repository lists as proof of importance" caution (P1.4).
        doc = classify_content(_doc(url="https://github.com/orgs/x", source_type=SourceType.GITHUB))
        self.assertEqual(doc.content_classification, ContentClassification.UNKNOWN)
        self.assertLess(float(doc.classification_confidence or 0), 0.5)


class RedditTests(unittest.TestCase):
    def test_post_kind(self):
        doc = classify_content(_doc(
            url="https://www.reddit.com/r/LangChain/comments/abc123/x/", source_type=SourceType.REDDIT,
            metadata={"reddit": {"kind": "post", "subreddit": "LangChain"}}))
        self.assertEqual(doc.content_classification, ContentClassification.REDDIT_POST_OR_DISCUSSION)
        self.assertEqual(doc.authority_level, AuthorityLevel.COMMUNITY)
        self.assertFalse(doc.is_primary_source)


class WebDomainTests(unittest.TestCase):
    def test_arxiv_is_preprint_not_peer_reviewed(self):
        doc = classify_content(_doc(url="https://arxiv.org/abs/2401.00001", source_type=SourceType.WEB))
        self.assertEqual(doc.content_classification, ContentClassification.PREPRINT)
        self.assertFalse(doc.independently_verified)  # never claimed peer-reviewed

    def test_recognised_venue_is_research_paper(self):
        doc = classify_content(_doc(url="https://www.nature.com/articles/x", source_type=SourceType.WEB))
        self.assertEqual(doc.content_classification, ContentClassification.RESEARCH_PAPER)
        self.assertEqual(doc.authority_level, AuthorityLevel.ACADEMIC)

    def test_recognised_news_domain(self):
        doc = classify_content(_doc(url="https://www.reuters.com/technology/x", source_type=SourceType.WEB))
        self.assertEqual(doc.content_classification, ContentClassification.NEWS_REPORT)
        self.assertFalse(doc.is_primary_source)

    def test_medium_is_expert_blog_not_a_paper(self):
        doc = classify_content(_doc(url="https://medium.com/@someone/post", source_type=SourceType.WEB))
        self.assertEqual(doc.content_classification, ContentClassification.EXPERT_BLOG)
        self.assertFalse(doc.independently_verified)

    def test_blog_path_hint_without_a_listed_domain(self):
        doc = classify_content(_doc(url="https://company.example.com/blog/announcing-x",
                                     source_type=SourceType.WEB))
        self.assertEqual(doc.content_classification, ContentClassification.EXPERT_BLOG)

    def test_gov_domain_is_technical_report(self):
        doc = classify_content(_doc(url="https://www.nist.gov/publications/x", source_type=SourceType.WEB))
        self.assertEqual(doc.content_classification, ContentClassification.TECHNICAL_REPORT)

    def test_snippet_only_result_is_search_snippet(self):
        doc = classify_content(_doc(
            url="https://unknown-blogless-domain.example/x", source_type=SourceType.WEB,
            snippet="short snippet", content=None))
        self.assertEqual(doc.content_classification, ContentClassification.SEARCH_SNIPPET)

    def test_short_content_still_counts_as_snippet_only(self):
        doc = classify_content(_doc(
            url="https://unknown-blogless-domain.example/x", source_type=SourceType.WEB,
            content="too short to be a full source"))
        self.assertEqual(doc.content_classification, ContentClassification.SEARCH_SNIPPET)

    def test_full_content_unknown_domain_is_unknown_not_upgraded(self):
        doc = classify_content(_doc(
            url="https://unknown-blogless-domain.example/x", source_type=SourceType.WEB,
            content="x" * 500))
        self.assertEqual(doc.content_classification, ContentClassification.UNKNOWN)
        self.assertEqual(doc.authority_level, AuthorityLevel.UNKNOWN)


class PurityTests(unittest.TestCase):
    def test_classify_content_does_not_mutate_other_fields(self):
        original = _doc(url="https://fastapi.tiangolo.com/x", title="T", content="c",
                         source_type=SourceType.OFFICIAL_DOCS)
        classified = classify_content(original)
        self.assertEqual(classified.url, original.url)
        self.assertEqual(classified.title, original.title)
        self.assertEqual(classified.content, original.content)
        self.assertEqual(classified.source_type, original.source_type)
        # the input is untouched (classify_content returns a copy)
        self.assertIsNone(original.content_classification)

    def test_rag_document_and_cache_are_out_of_scope_not_misclassified(self):
        for source_type in (SourceType.RAG_DOCUMENT, SourceType.CACHE):
            with self.subTest(source_type=source_type):
                doc = classify_content(_doc(source_type=source_type))
                self.assertEqual(doc.content_classification, ContentClassification.UNKNOWN)
                self.assertEqual(doc.classification_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
