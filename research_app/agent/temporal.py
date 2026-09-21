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
from datetime import date, datetime, timezone
from typing import Mapping, Optional

from research_app.conversation.settings import read_bool


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
