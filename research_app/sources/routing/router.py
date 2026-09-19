"""The deterministic source router: which sources should this question be sent to?

Rule-based, no LLM, no scoring. Web is always selected (it is the baseline every question
has always had); official documentation, GitHub and Reddit are added when the question
carries a matching signal. A question with no signal is web-only, exactly as before.

Signals (regexes over the lower-cased question; see the tables below):

  DOCS        Phase 4 ``detect_docs_intent``: technology + technical cue, not opinion phrasing
  CODE        GitHub/repo/pull-request/source-code/open-source wording; or example/sample/
              implementation/boilerplate wording next to a registry technology
  EXPERIENCE  Reddit wording, real-world experience, "what do people/users think", community
  PROBLEMS    problem/challenge/pain-point wording about developers/users/people
  TROUBLE     error/exception/"not working"/fix wording with a technical anchor (a registry
              technology, a traceback/exception token, or a ``FooError`` class name)

  official_docs  DOCS; or a registry technology plus PROBLEMS/TROUBLE
  github         CODE; or (PROBLEMS/TROUBLE with a technical anchor)
  reddit         EXPERIENCE; or PROBLEMS; or TROUBLE
  web            always

``SOURCE_ROUTER_ENABLED=false`` keeps only the Phase 4 behaviour (web + DOCS).

The router only decides. It does not fetch anything; ``agent/state.py`` executes the plan.
Must stay importable without ``research_app.agent`` / LangChain.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional

from research_app.domain import SourceType
from research_app.sources.official_docs.adapter import build_docs_request
from research_app.sources.official_docs.detector import (
    DocsIntent,
    detect_docs_intent,
    mentioned_technologies,
)
from research_app.sources.official_docs.plan import DocsPlan
from research_app.sources.official_docs.registry import get_registry, normalize_text
from research_app.sources.official_docs.settings import DocsSettings
from research_app.sources.routing.settings import RouterSettings

logger = logging.getLogger(__name__)

_MAX_TEXT_CHARS = 2000  # bounds regex work on very long input

# Fixed execution/presentation order: authoritative first.
SOURCE_ORDER = (SourceType.OFFICIAL_DOCS, SourceType.GITHUB, SourceType.REDDIT, SourceType.WEB)


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


_GITHUB_EXPLICIT = _rx(
    r"\bgit ?hub\b|\brepos?\b|\brepositor(?:y|ies)\b|\bpull requests?\b|\bopen source\b"
    r"|\bsource code\b|\bcode (?:examples?|samples?|snippets?)\b"
    r"|\b(?:sample|example|starter|reference) (?:code|projects?|repos?|implementations?|apps?)\b"
)
# Only meaningful next to a registry technology ("examples of LangGraph agents").
_CODE_SOFT = _rx(r"\b(?:examples?|samples?|implementations?|implement(?:s|ing)?|boilerplate|templates?|starter|demos?)\b")

_REDDIT_EXPLICIT = _rx(r"\breddit\b|\bsubreddits?\b|\br/[a-z0-9_]{2,21}\b|\bredditors?\b")
_EXPERIENCE = _rx(
    r"\breal world (?:experiences?|usage|feedback|reports?|stories)\b"
    r"|\b(?:experiences?|feedback) (?:with|using|of|from)\b"
    r"|\bwhat do (?:people|developers|users|engineers|folks|students|others) (?:think|say|use|recommend)\b"
    r"|\b(?:people|developers|users|engineers) (?:say|report|think|complain|recommend)\b"
    r"|\bcommunity (?:opinions?|feedback|discussions?|experiences?|consensus|sentiment)\b"
    r"|\b(?:opinions?|discussions?) (?:on|about)\b"
    r"|\banyone (?:here |else )?(?:using|used|tried|switched|migrated)\b"
    r"|\b(?:has|have) (?:anyone|anybody)\b"
)

_PROBLEM_WORD = _rx(
    r"\b(?:problems?|issues?|challenges?|pain points?|complaints?|struggl(?:e|es|ing)|frustrations?"
    r"|limitations?|drawbacks?|gotchas?|pitfalls?)\b"
)
_ACTOR = _rx(r"\b(?:developers?|users?|people|engineers?|teams?|community|folks|programmers?|practitioners)\b")

_TROUBLE_CUE = _rx(
    r"\b(?:errors?|exceptions?|tracebacks?|stack ?traces?|crash(?:es|ed|ing)?|bugs?|broken|failing"
    r"|fails?|failed|failures?|not working|doesn't work|does not work|isn't working"
    r"|won't (?:work|start|run|install|build|compile|connect)|can't|cannot|unable to"
    r"|troubleshoot(?:ing)?|fix(?:es|ed|ing)?|timeouts?|timed out|hangs?|hanging)\b"
)
# Technical anchors that are not registry technologies.
_ANCHOR_TEXT = _rx(
    r"\b(?:tracebacks?|stack ?traces?|segfault|segmentation fault|syntax error|import error"
    r"|module not found|compile error|build error|runtime error|null pointer|exceptions?)\b"
)
_ANCHOR_CLASS = _rx(r"\b[A-Z][A-Za-z]+(?:Error|Exception)\b")  # TypeError, HTTPError (case-sensitive)


@dataclass(frozen=True)
class RouteSignals:
    docs_intent: DocsIntent
    strong_technologies: tuple[str, ...] = ()  # registry ids; weak aliases (python, compose) excluded
    any_technology: bool = False  # including weak aliases
    github_explicit: bool = False
    code_soft: bool = False
    reddit_explicit: bool = False
    experience: bool = False
    problems: bool = False
    trouble: bool = False
    code_anchor: bool = False

    @property
    def technical(self) -> bool:
        return self.any_technology or self.code_anchor


@dataclass(frozen=True)
class RoutePlan:
    """Sources to search, in ``SOURCE_ORDER``, and why (``reasons`` is for logs and tests)."""

    sources: tuple[SourceType, ...]
    reasons: Mapping[SourceType, str] = field(default_factory=dict)
    # True when official docs were added because of technology + problem/trouble wording,
    # i.e. the Phase 4 documentation detector did NOT select them.
    docs_from_technology: bool = False

    def includes(self, source_type: SourceType) -> bool:
        return source_type in self.sources

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.value for s in self.sources)


WEB_ONLY = RoutePlan(sources=(SourceType.WEB,), reasons={SourceType.WEB: "default"})


def detect_signals(text: str, docs_settings: DocsSettings) -> RouteSignals:
    registry = get_registry(docs_settings.registry_path)
    normalized = normalize_text(text[:_MAX_TEXT_CHARS]).replace("’", "'")
    raw = text[:_MAX_TEXT_CHARS]

    strong = mentioned_technologies(raw, registry)
    weak_or_strong = mentioned_technologies(raw, registry, include_weak=True)
    return RouteSignals(
        docs_intent=detect_docs_intent(raw, registry),
        strong_technologies=tuple(t.id for t in strong),
        any_technology=bool(weak_or_strong),
        github_explicit=bool(_GITHUB_EXPLICIT.search(normalized)),
        code_soft=bool(_CODE_SOFT.search(normalized)),
        reddit_explicit=bool(_REDDIT_EXPLICIT.search(normalized)),
        experience=bool(_EXPERIENCE.search(normalized)),
        problems=bool(_PROBLEM_WORD.search(normalized) and _ACTOR.search(normalized)),
        # An error class name (TypeError) is both the cue and the technical anchor.
        trouble=bool(_TROUBLE_CUE.search(normalized) or _ANCHOR_CLASS.search(raw)),
        code_anchor=bool(_ANCHOR_TEXT.search(normalized) or _ANCHOR_CLASS.search(raw)),
    )


def route_sources(
    question: Optional[str],
    settings: Optional[RouterSettings] = None,
    docs_settings: Optional[DocsSettings] = None,
) -> RoutePlan:
    """Choose the sources for ``question``. May raise (registry problems); callers isolate
    that and fall back to web-only."""
    settings = settings or RouterSettings.from_env()
    docs_settings = docs_settings or DocsSettings.from_env()
    if not isinstance(question, str) or not question.strip():
        return WEB_ONLY

    sig = detect_signals(question, docs_settings)
    trouble = sig.trouble and sig.technical  # an error word alone is not enough
    problem_or_trouble = sig.problems or trouble

    reasons: dict[SourceType, str] = {}
    docs_from_technology = False

    if docs_settings.enabled and sig.docs_intent.is_documentation_query:
        reasons[SourceType.OFFICIAL_DOCS] = "documentation_query:" + ",".join(sig.docs_intent.technology_ids)
    elif settings.enabled and docs_settings.enabled and sig.strong_technologies and problem_or_trouble:
        reasons[SourceType.OFFICIAL_DOCS] = "technology_problem:" + ",".join(sig.strong_technologies)
        docs_from_technology = True

    if settings.enabled:
        if sig.github_explicit:
            reasons[SourceType.GITHUB] = "explicit_code_or_github_wording"
        elif sig.code_soft and sig.strong_technologies:
            reasons[SourceType.GITHUB] = "code_examples_for:" + ",".join(sig.strong_technologies)
        elif problem_or_trouble and sig.technical:
            reasons[SourceType.GITHUB] = "technical_problem"

        if sig.reddit_explicit:
            reasons[SourceType.REDDIT] = "explicit_reddit_wording"
        elif sig.experience:
            reasons[SourceType.REDDIT] = "community_experience"
        elif problem_or_trouble:
            reasons[SourceType.REDDIT] = "problem_or_troubleshooting"

    reasons[SourceType.WEB] = "always"
    sources = tuple(s for s in SOURCE_ORDER if s in reasons)
    plan = RoutePlan(sources=sources, reasons={s: reasons[s] for s in sources},
                     docs_from_technology=docs_from_technology)
    logger.info("Source route: %s (%s)", ",".join(plan.names),
                "; ".join(f"{s.value}={r}" for s, r in plan.reasons.items() if s != SourceType.WEB) or "web only")
    return plan


def plan_technology_docs(
    question: str, query: str, docs_settings: Optional[DocsSettings] = None
) -> Optional[DocsPlan]:
    """An official-docs search plan for the technologies named in ``question``, used when
    the router selected documentation because of a problem/troubleshooting question (the
    Phase 4 detector, which needs a documentation cue, said no). ``None`` = nothing to
    search. May raise; callers isolate failures."""
    docs_settings = docs_settings or DocsSettings.from_env()
    if not docs_settings.enabled:
        return None
    registry = get_registry(docs_settings.registry_path)
    technologies = mentioned_technologies(question, registry)
    if not technologies:
        return None
    intent = DocsIntent(True, technologies=technologies, cues=("problem_or_troubleshooting",),
                        reason="router")
    request = build_docs_request(intent, query, timeout_s=docs_settings.timeout_s)
    if request is None:
        return None
    return DocsPlan(request=request, registry=registry)
