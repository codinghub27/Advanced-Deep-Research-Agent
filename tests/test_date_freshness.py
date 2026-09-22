"""P1.1 (freshness classification) + P1.2 (date extraction/normalization).

Pure-function tests: no LLM, no Tavily, no database, no Qdrant.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from research_app.agent.temporal import (
    FreshnessCategory,
    FreshnessPolicy,
    SourceFreshnessStatus,
    classify_freshness,
    evaluate_source_freshness,
)
from research_app.domain import DateConfidence, SourceDocument
from research_app.sources.normalizer import normalize_source, parse_relative_date


def _dt(days_ago: int, *, now: datetime) -> datetime:
    return now - timedelta(days=days_ago)


NOW = datetime(2026, 9, 22, tzinfo=timezone.utc)


class ClassifyFreshnessTests(unittest.TestCase):
    def test_current_information_for_short_window_wording(self):
        policy = classify_freshness("What is happening in the news today?", now=NOW)
        self.assertEqual(policy.category, FreshnessCategory.CURRENT_INFORMATION)
        self.assertFalse(policy.cache_allowed)
        self.assertFalse(policy.rag_allowed)
        self.assertTrue(policy.live_search_required)
        self.assertEqual(policy.preferred_window_days, 7)

    def test_recent_for_latest_wording(self):
        # "release notes" is itself a _MEDIUM (current-window) cue in agent.temporal, so this
        # uses "this year" instead to isolate the RECENT (30-day) branch from CURRENT_INFORMATION.
        policy = classify_freshness("What is trending in AI research this year?", now=NOW)
        self.assertEqual(policy.category, FreshnessCategory.RECENT)
        self.assertFalse(policy.cache_allowed)
        self.assertEqual(policy.preferred_window_days, 30)

    def test_current_information_for_release_notes_wording(self):
        # "release notes"/"changelog" are treated as CURRENT_INFORMATION (7-day window,
        # not 30): a changelog question wants the newest entry, not anything from the
        # last month, so the tighter window is deliberate, not a misclassification.
        policy = classify_freshness("What are the latest LangGraph release notes?", now=NOW)
        self.assertEqual(policy.category, FreshnessCategory.CURRENT_INFORMATION)
        self.assertFalse(policy.cache_allowed)

    def test_model_judged_current_intent_without_wording_cues(self):
        policy = classify_freshness(
            "How is it going", understanding={"intent": "current_events"}, now=NOW)
        self.assertEqual(policy.category, FreshnessCategory.CURRENT_INFORMATION)

    def test_historical_wording_allows_cache_and_rag(self):
        policy = classify_freshness("What is the history of Python?", now=NOW)
        self.assertEqual(policy.category, FreshnessCategory.HISTORICAL)
        self.assertTrue(policy.cache_allowed)
        self.assertTrue(policy.rag_allowed)
        self.assertFalse(policy.live_search_required)

    def test_version_dependent_requires_a_documentation_question(self):
        policy = classify_freshness(
            "What changed in Pydantic 2?", is_documentation_query=True, now=NOW)
        self.assertEqual(policy.category, FreshnessCategory.VERSION_DEPENDENT)
        self.assertFalse(policy.cache_allowed)
        self.assertTrue(policy.official_verification_required)

    def test_version_wording_without_docs_intent_is_not_version_dependent(self):
        # "version" alone (e.g. "which version of this song") should not trigger
        # VERSION_DEPENDENT unless the caller's docs detector actually fired.
        policy = classify_freshness("which version of this song is better", now=NOW)
        self.assertNotEqual(policy.category, FreshnessCategory.VERSION_DEPENDENT)

    def test_as_of_wording_extracts_a_past_year_cutoff(self):
        policy = classify_freshness("What was the population as of 2020?", now=NOW)
        self.assertEqual(policy.category, FreshnessCategory.AS_OF_DATE)
        self.assertEqual(policy.cutoff_date.year, 2020)
        self.assertTrue(policy.official_verification_required)

    def test_as_of_wording_ignores_a_future_year(self):
        policy = classify_freshness("population as of 2099", now=NOW)
        self.assertIsNone(policy.cutoff_date)

    def test_stable_default_allows_cache_and_rag(self):
        policy = classify_freshness("How does a hash table work?", now=NOW)
        self.assertEqual(policy.category, FreshnessCategory.STABLE)
        self.assertTrue(policy.cache_allowed)
        self.assertTrue(policy.rag_allowed)

    def test_empty_question_is_unknown_and_discloses_the_assumption(self):
        policy = classify_freshness("   ", now=NOW)
        self.assertEqual(policy.category, FreshnessCategory.UNKNOWN)
        self.assertFalse(policy.cache_allowed)
        self.assertIsNotNone(policy.assumption)

    def test_historical_wins_over_a_stray_recent_word_only_when_historical_cue_present(self):
        # Sanity: historical cue still checked before time-sensitivity.
        policy = classify_freshness("What is the history of the latest smartphone designs?", now=NOW)
        self.assertEqual(policy.category, FreshnessCategory.HISTORICAL)


class EvaluateSourceFreshnessTests(unittest.TestCase):
    def _doc(self, **kwargs) -> SourceDocument:
        return SourceDocument(url="https://example.com/a", **kwargs)

    def test_no_date_is_unknown(self):
        policy = FreshnessPolicy(category=FreshnessCategory.RECENT, reason="t", preferred_window_days=30)
        status = evaluate_source_freshness(self._doc(), policy, now=NOW)
        self.assertEqual(status, SourceFreshnessStatus.UNKNOWN)

    def test_within_window_is_fresh(self):
        policy = FreshnessPolicy(category=FreshnessCategory.RECENT, reason="t", preferred_window_days=30)
        doc = self._doc(published_at=_dt(5, now=NOW))
        self.assertEqual(evaluate_source_freshness(doc, policy, now=NOW), SourceFreshnessStatus.FRESH)

    def test_outside_window_is_stale(self):
        policy = FreshnessPolicy(category=FreshnessCategory.CURRENT_INFORMATION, reason="t",
                                  preferred_window_days=7)
        doc = self._doc(published_at=_dt(30, now=NOW))
        self.assertEqual(evaluate_source_freshness(doc, policy, now=NOW), SourceFreshnessStatus.STALE)

    def test_no_window_treats_a_dated_source_as_fresh(self):
        # STABLE/HISTORICAL policies have no preferred_window_days: age alone doesn't
        # disqualify an authoritative old source (CLAUDE.md P1.3).
        policy = FreshnessPolicy(category=FreshnessCategory.STABLE, reason="t")
        doc = self._doc(published_at=_dt(2000, now=NOW))
        self.assertEqual(evaluate_source_freshness(doc, policy, now=NOW), SourceFreshnessStatus.FRESH)

    def test_future_date_is_future_invalid(self):
        policy = FreshnessPolicy(category=FreshnessCategory.STABLE, reason="t")
        doc = self._doc(published_at=NOW + timedelta(days=30))
        self.assertEqual(evaluate_source_freshness(doc, policy, now=NOW), SourceFreshnessStatus.FUTURE_INVALID)

    def test_updated_before_published_is_a_date_conflict(self):
        policy = FreshnessPolicy(category=FreshnessCategory.STABLE, reason="t")
        doc = self._doc(published_at=_dt(5, now=NOW), updated_at=_dt(200, now=NOW))
        self.assertEqual(evaluate_source_freshness(doc, policy, now=NOW), SourceFreshnessStatus.DATE_CONFLICT)

    def test_updated_at_is_preferred_over_published_at_as_the_reference(self):
        policy = FreshnessPolicy(category=FreshnessCategory.RECENT, reason="t", preferred_window_days=30)
        # published long ago, but updated recently -> fresh
        doc = self._doc(published_at=_dt(400, now=NOW), updated_at=_dt(1, now=NOW))
        self.assertEqual(evaluate_source_freshness(doc, policy, now=NOW), SourceFreshnessStatus.FRESH)


class ParseRelativeDateTests(unittest.TestCase):
    def test_days_ago(self):
        result = parse_relative_date("3 days ago", now=NOW)
        self.assertEqual(result, NOW - timedelta(days=3))

    def test_weeks_months_years_ago(self):
        self.assertEqual(parse_relative_date("2 weeks ago", now=NOW), NOW - timedelta(days=14))
        self.assertEqual(parse_relative_date("1 month ago", now=NOW), NOW - timedelta(days=30))
        self.assertEqual(parse_relative_date("1 year ago", now=NOW), NOW - timedelta(days=365))

    def test_yesterday_and_today(self):
        self.assertEqual(parse_relative_date("yesterday", now=NOW), NOW - timedelta(days=1))
        self.assertEqual(parse_relative_date("today", now=NOW), NOW)
        self.assertEqual(parse_relative_date("just now", now=NOW), NOW)

    def test_not_a_relative_expression(self):
        self.assertIsNone(parse_relative_date("sometime in the past", now=NOW))
        self.assertIsNone(parse_relative_date("2024-01-01", now=NOW))
        self.assertIsNone(parse_relative_date(None, now=NOW))
        self.assertIsNone(parse_relative_date(42, now=NOW))


class NormalizerDateExtractionTests(unittest.TestCase):
    def test_absolute_published_date_is_exact(self):
        doc = normalize_source(
            {"url": "https://example.com/a", "published_at": "2026-01-01T00:00:00+00:00"},
            retrieved_at=NOW,
        )
        self.assertIsNotNone(doc.published_at)
        self.assertEqual(doc.date_confidence, DateConfidence.EXACT)
        self.assertEqual(doc.date_source, "published_at")
        self.assertIsNone(doc.updated_at)

    def test_relative_published_date_is_approximate(self):
        doc = normalize_source(
            {"url": "https://example.com/a", "published_at": "3 days ago"}, retrieved_at=NOW,
        )
        self.assertEqual(doc.published_at, NOW - timedelta(days=3))
        self.assertEqual(doc.date_confidence, DateConfidence.APPROXIMATE)
        self.assertEqual(doc.date_source, "relative:published_at")

    def test_unparseable_date_is_unknown_confidence_and_kept_in_metadata(self):
        doc = normalize_source(
            {"url": "https://example.com/a", "published_at": "sometime last spring"}, retrieved_at=NOW,
        )
        self.assertIsNone(doc.published_at)
        self.assertEqual(doc.date_confidence, DateConfidence.UNKNOWN)
        self.assertEqual(doc.date_source, "published_at")
        self.assertEqual(doc.metadata.get("published_at"), "sometime last spring")

    def test_missing_date_has_no_confidence(self):
        doc = normalize_source({"url": "https://example.com/a"}, retrieved_at=NOW)
        self.assertIsNone(doc.published_at)
        self.assertIsNone(doc.date_confidence)
        self.assertIsNone(doc.date_source)

    def test_updated_at_is_extracted_and_preferred_as_the_reference(self):
        doc = normalize_source(
            {
                "url": "https://example.com/a",
                "published_at": "2020-01-01T00:00:00+00:00",
                "updated_at": "2026-09-01T00:00:00+00:00",
            },
            retrieved_at=NOW,
        )
        self.assertEqual(doc.published_at.year, 2020)
        self.assertEqual(doc.updated_at.year, 2026)
        self.assertEqual(doc.date_source, "updated_at")  # reference field, not published_at

    def test_alternate_updated_field_names_are_recognized(self):
        for key in ("updated_date", "last_updated", "modified_at"):
            with self.subTest(key=key):
                doc = normalize_source(
                    {"url": "https://example.com/a", key: "2026-06-01T00:00:00+00:00"}, retrieved_at=NOW,
                )
                self.assertEqual(doc.updated_at.year, 2026)
                self.assertNotIn(key, doc.metadata)

    def test_existing_behaviour_unaffected_when_no_date_fields_present(self):
        # Equivalence guard: a plain result with no date fields normalizes exactly as
        # before this change (Phase 3 contract).
        doc = normalize_source(
            {"url": "https://example.com/a", "title": "T", "content": "c"}, retrieved_at=NOW,
        )
        self.assertEqual(doc.title, "T")
        self.assertEqual(doc.content, "c")
        self.assertIsNone(doc.published_at)
        self.assertIsNone(doc.updated_at)


if __name__ == "__main__":
    unittest.main()
