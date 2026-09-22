"""Current-date awareness for real-time research.

A language model does not know what day it is: left alone, it writes "latest ... 2024" queries
in 2026. Everything here is pure and deterministic (no LLM, no network):

``date_line``           the sentence that goes into every prompt that reasons about time
``is_time_sensitive``   does the question ask about the latest / current state of something?
``refresh_stale_years`` a past year the planner added on its own becomes the current year
``recency_params``      the Tavily ``time_range`` (and ``topic``) for a time-sensitive question

``RECENCY_FILTER_ENABLED`` (default true) turns the Tavily date filter off; the date in the
prompts and the stale-year guard stay on. Read at call time.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Mapping, Optional

from research_app.conversation.settings import read_bool
from research_app.domain.enums import FreshnessCategory, SourceFreshnessStatus
from research_app.domain.models import SourceDocument, utc_now


def today(now: Optional[datetime] = None) -> date:
    return (now or datetime.now(timezone.utc)).date()


def date_line(now: Optional[datetime] = None) -> str:
    """One sentence for a prompt. ``now`` is injectable for tests."""
    day = today(now)
    return (f"TODAY'S DATE: {day.isoformat()} (the current year is {day.year}). Your own knowledge ends "
            f"before this date, so never treat an earlier year as \"now\".")


# --------------------------------------------------------------------------- time sensitivity

# Words that ask about the state of things NOW. Bare "now"/"new" are left out on purpose: too common.
_NEWS = re.compile(r"\b(news|breaking|headlines?)\b", re.I)
_SHORT = re.compile(r"\b(today|tonight|right now|this week|past week|last week|breaking|just (?:released|announced|launched))\b", re.I)
_MEDIUM = re.compile(r"\b(this month|past month|last month|current(?:ly)?|news|headlines?|what'?s new|what changed|release notes?|changelog)\b", re.I)
_RECENT = re.compile(
    r"\b(latest|newest|most recent|recent(?:ly)?|this year|trending|upcoming|emerging|state[- ]of[- ]the[- ]art|sota|"
    r"up[- ]to[- ]date|nowadays|these days|roadmap)\b", re.I)


def _text(*parts: object) -> str:
    return " ".join(p for p in parts if isinstance(p, str) and p)


def is_time_sensitive(question: str, understanding: Optional[Mapping] = None) -> bool:
    """True when the answer depends on today's date. The model's judgement (``time_sensitivity``
    recent/current, intent ``current_events``) or the wording of the question is enough."""
    u = understanding or {}
    if u.get("time_sensitivity") in ("recent", "current") or u.get("intent") == "current_events":
        return True
    text = _text(question, u.get("resolved_query"))
    return bool(_SHORT.search(text) or _MEDIUM.search(text) or _RECENT.search(text))


# --------------------------------------------------------------------------- stale years

_YEAR = re.compile(r"(?<![\w.\-/])(?:19|20)\d{2}(?![\w\-/])")
# "2026 vs 2026", "2026-2026", "2026, 2026" after two stale years became the same year
_DOUBLE = re.compile(r"\b((?:19|20)\d{2})(?:\s*(?:vs\.?|versus|and|to|through|[-–,])\s*)\1\b", re.I)


def refresh_stale_years(query: str, *, user_text: str = "", now: Optional[datetime] = None) -> str:
    """Replace a past year in ``query`` with the current year unless the user wrote that year
    themselves. Future years and the current year are left alone. Only call this for a
    time-sensitive question: for "history of X in 1998" the year is the point."""
    current = today(now).year
    kept = {m.group(0) for m in _YEAR.finditer(user_text or "")}

    def swap(match: "re.Match[str]") -> str:
        year = match.group(0)
        return year if int(year) >= current or year in kept else str(current)

    refreshed = _YEAR.sub(swap, query)
    return _DOUBLE.sub(r"\1", refreshed) if refreshed != query else query


# --------------------------------------------------------------------------- Tavily date filter

def recency_params(question: str, understanding: Optional[Mapping] = None, *,
                   env: Optional[Mapping[str, str]] = None) -> dict:
    """Extra Tavily arguments for a time-sensitive question, else ``{}``. Narrow for "today" and
    "this week", a month for news and "current", a year for "latest"/"recent"/"emerging". Tavily
    advises broad ranges for niche topics, so the default is the year; the caller retries once
    without the filter if it returns nothing."""
    if not read_bool(env if env is not None else os.environ, "RECENCY_FILTER_ENABLED", True):
        return {}
    if not is_time_sensitive(question, understanding):
        return {}
    u = understanding or {}
    text = _text(question, u.get("resolved_query"))
    if _SHORT.search(text):
        params = {"time_range": "week"}
    elif _MEDIUM.search(text) or u.get("time_sensitivity") == "current" or u.get("intent") == "current_events":
        params = {"time_range": "month"}
    else:
        params = {"time_range": "year"}
    if _NEWS.search(text):
        params["topic"] = "news"
    return params


# --------------------------------------------------------------------------- freshness policy (P1.1)

_VERSION_CUE = re.compile(
    r"\b(version|changelog|release notes?|migrate|migration|upgrade|deprecated|"
    r"what changed|breaking changes?)\b", re.I)
_HISTORICAL_CUE = re.compile(
    r"\b(history of|historical(?:ly)?|originally|used to be|in the (?:19|20)\d0s)\b", re.I)
_AS_OF_CUE = re.compile(r"\bas of\b", re.I)


@dataclass(frozen=True)
class FreshnessPolicy:
    """A request's freshness requirement (P1.1): category plus the usage decisions that
    follow from it. Pure data -- nothing here reads the cache, Qdrant or a search tool;
    callers (semantic_cache_node, the RAG gate, synthesis/critic) decide what to do with it.
    ``assumption`` is set only when the category was ambiguous and a safer default was chosen,
    so callers can disclose it (CLAUDE.md P1: "For ambiguous cases ... disclose the assumption")."""

    category: FreshnessCategory
    reason: str
    cutoff_date: Optional[date] = None
    preferred_window_days: Optional[int] = None
    cache_allowed: bool = True
    rag_allowed: bool = True
    live_search_required: bool = False
    official_verification_required: bool = False
    assumption: Optional[str] = None


def _extract_cutoff_year(text: str, *, now: Optional[datetime] = None) -> Optional[date]:
    match = _YEAR.search(text)
    if not match:
        return None
    year = int(match.group(0))
    if year > today(now).year:
        return None  # never accept a future cutoff from a misparse; no invented date
    return date(year, 1, 1)


def classify_freshness(
    question: str,
    understanding: Optional[Mapping] = None,
    *,
    is_documentation_query: bool = False,
    now: Optional[datetime] = None,
) -> FreshnessPolicy:
    """Explicit freshness category and usage policy for one request. Extends
    ``is_time_sensitive`` (still called, unchanged, below) rather than replacing it -- the
    recent/current split reuses the exact same wording regexes and model judgement.

    ``is_documentation_query`` should be the existing docs-intent detector's result
    (``sources.official_docs.detector.detect_docs_intent``) when the caller already has it,
    so a bare mention of the word "version" only becomes VERSION_DEPENDENT alongside a real
    registered technology, not any stray use of the word.
    """
    text = _text(question, (understanding or {}).get("resolved_query"))

    if _VERSION_CUE.search(text) and is_documentation_query:
        return FreshnessPolicy(
            category=FreshnessCategory.VERSION_DEPENDENT,
            reason="version/changelog/migration wording on a documentation question",
            cache_allowed=False,
            rag_allowed=True,
            live_search_required=True,
            official_verification_required=True,
        )
    if _HISTORICAL_CUE.search(text):
        return FreshnessPolicy(
            category=FreshnessCategory.HISTORICAL,
            reason="explicit historical wording",
            cache_allowed=True,
            rag_allowed=True,
            live_search_required=False,
            official_verification_required=False,
        )
    if is_time_sensitive(question, understanding):
        is_current = bool(
            _SHORT.search(text) or _MEDIUM.search(text)
            or (understanding or {}).get("time_sensitivity") == "current"
            or (understanding or {}).get("intent") == "current_events"
        )
        if is_current:
            return FreshnessPolicy(
                category=FreshnessCategory.CURRENT_INFORMATION,
                reason="today/this-week/current wording or model-judged current intent",
                preferred_window_days=7,
                cache_allowed=False,
                rag_allowed=False,
                live_search_required=True,
                official_verification_required=False,
            )
        return FreshnessPolicy(
            category=FreshnessCategory.RECENT,
            reason="latest/recent/trending wording or model-judged recent intent",
            preferred_window_days=30,
            cache_allowed=False,
            rag_allowed=False,
            live_search_required=True,
            official_verification_required=False,
        )
    if _AS_OF_CUE.search(text):
        cutoff = _extract_cutoff_year(text, now=now)
        return FreshnessPolicy(
            category=FreshnessCategory.AS_OF_DATE,
            reason="explicit as-of wording",
            cutoff_date=cutoff,
            cache_allowed=False,
            rag_allowed=True,
            live_search_required=True,
            official_verification_required=True,
        )
    if not text.strip():
        return FreshnessPolicy(
            category=FreshnessCategory.UNKNOWN,
            reason="no question text to classify",
            cache_allowed=False,
            rag_allowed=True,
            live_search_required=False,
            official_verification_required=False,
            assumption="empty/unclassifiable question; defaulting to the safer no-cache policy",
        )
    return FreshnessPolicy(
        category=FreshnessCategory.STABLE,
        reason="no time-sensitive, historical, version or as-of wording detected",
        cache_allowed=True,
        rag_allowed=True,
        live_search_required=False,
        official_verification_required=False,
    )


def evaluate_source_freshness(
    source: SourceDocument,
    policy: FreshnessPolicy,
    *,
    now: Optional[datetime] = None,
) -> SourceFreshnessStatus:
    """One source's freshness, evaluated against ``policy`` -- the request's own freshness
    requirement, never one global threshold (CLAUDE.md P1.3). Never invents a date: a source
    with neither ``published_at`` nor ``updated_at`` is ``UNKNOWN``, not assumed fresh or stale.
    """
    now = now or utc_now()
    published = source.published_at
    updated = source.updated_at
    if published is not None and updated is not None and updated < published - timedelta(days=1):
        return SourceFreshnessStatus.DATE_CONFLICT
    reference = updated or published
    if reference is None:
        return SourceFreshnessStatus.UNKNOWN
    if reference > now + timedelta(days=1):
        return SourceFreshnessStatus.FUTURE_INVALID
    if policy.preferred_window_days is None:
        return SourceFreshnessStatus.FRESH
    age_days = (now - reference).days
    return (
        SourceFreshnessStatus.FRESH if age_days <= policy.preferred_window_days
        else SourceFreshnessStatus.STALE
    )
