"""Per-task source routing (Phase 6): the planner PROPOSES, this module DECIDES.

``route_task(task, understanding) -> RoutingDecision``. A pure, deterministic function
(no LLM, no I/O, same input -> same decision). It reuses the Phase 5 rules (``detect_signals``
and ``route_sources``) instead of duplicating them.

Where the sources come from
  - the planner's ``suggested_sources`` that are valid (origin ``planner``);
  - otherwise the defaults of the planner's ``source_intent`` (origin ``policy``);
  - otherwise the Phase 5 route (origin ``fallback``): the same question-level rules Phase 5 ran
    on the whole (resolved) question, so planner output without hints behaves exactly as before.
  Unknown values are dropped and reported, never trusted.

Enforcement (the router overrides the planner)
  1. A technology from the docs registry + a technical task -> ``official_docs`` is included
     even if the planner left it out. Reddit never substitutes for it.
  2. A task with no technical signal stays web-only; planner suggestions of documentation/
     GitHub are dropped unless the sub-question itself calls for them. Reddit is kept only if
     the USER's question asks for it (reddit wording, real-world experience, ...): a planner
     writing "Reddit" into a sub-question of its own does not count.
  3. At most ``MAX_SOURCES_PER_TASK`` sources: Reddit is trimmed first, then GitHub -- GitHub
     first when the user's question asks for community experience. ``official_docs`` and
     ``web`` are never trimmed by the cap.
  4. ``SOURCE_ROUTER_ENABLED=false`` or ``OFFICIAL_DOCS_ENABLED=false`` are honoured.
  Rules 1-3 apply to planner-led decisions. A task with no usable hint gets the Phase 5 route
  unchanged (including a four-source route, which the cap does not trim).

Must stay importable without ``research_app.agent`` / LangChain.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from research_app.domain import (
    ORIGIN_FALLBACK,
    ORIGIN_PLANNER,
    ORIGIN_POLICY,
    DroppedSource,
    ResearchTask,
    RoutingDecision,
    SourceIntent,
    SourceType,
)
from research_app.sources.official_docs.detector import mentioned_technologies
from research_app.sources.official_docs.registry import get_registry
from research_app.sources.official_docs.settings import DocsSettings
from research_app.sources.routing.router import SOURCE_ORDER, detect_signals, route_sources
from research_app.sources.routing.settings import RouterSettings

logger = logging.getLogger(__name__)

ALLOWED_SOURCES = frozenset(SOURCE_ORDER)  # official_docs, github, reddit, web

TECHNICAL_INTENTS = frozenset({
    SourceIntent.TECHNICAL_HOWTO,
    SourceIntent.LIBRARY_USAGE,
    SourceIntent.CODE_IMPLEMENTATION,
    SourceIntent.TROUBLESHOOTING,
})

DOCS, GITHUB, REDDIT, WEB = (SourceType.OFFICIAL_DOCS, SourceType.GITHUB, SourceType.REDDIT, SourceType.WEB)

# Intent -> default sources. Web is always added on top (it is the baseline every task has).
INTENT_DEFAULTS: Mapping[SourceIntent, tuple[SourceType, ...]] = {
    SourceIntent.TECHNICAL_HOWTO: (DOCS, WEB),
    SourceIntent.LIBRARY_USAGE: (DOCS, GITHUB),
    SourceIntent.CODE_IMPLEMENTATION: (DOCS, GITHUB, WEB),
    SourceIntent.TROUBLESHOOTING: (DOCS, GITHUB, REDDIT, WEB),
    SourceIntent.COMMUNITY_EXPERIENCE: (REDDIT, WEB),
    SourceIntent.GENERAL_RESEARCH: (WEB,),
    SourceIntent.COMPARISON: (WEB,),  # + official_docs when a technology is named
    SourceIntent.CURRENT_EVENTS: (WEB,),
}

_TRIM_ORDER = (REDDIT, GITHUB)  # dropped first when over the cap; docs and web are protected
_MAX_HINTS = 8


def _parse_intent(value: Any) -> Optional[SourceIntent]:
    if isinstance(value, SourceIntent):
        return value
    if isinstance(value, str):
        try:
            return SourceIntent(value.strip().lower())
        except ValueError:
            logger.warning("Planner gave an unknown source_intent %r; ignored", value[:40])
    return None


def _validate_suggestions(raw: Any) -> tuple[list[SourceType], list[DroppedSource]]:
    """Valid, de-duplicated source types in the planner's order, and the rest as drops."""
    valid: list[SourceType] = []
    dropped: list[DroppedSource] = []
    if not isinstance(raw, (list, tuple)):
        return valid, dropped
    for item in list(raw)[:_MAX_HINTS]:
        if not isinstance(item, str):
            dropped.append(DroppedSource(source=repr(item)[:40], reason="malformed_source"))
            continue
        try:
            source = SourceType(item.strip().lower())
        except ValueError:
            dropped.append(DroppedSource(source=item[:40], reason="unknown_source_type"))
            continue
        if source not in ALLOWED_SOURCES:
            dropped.append(DroppedSource(source=item[:40], reason="unsupported_source_type"))
        elif source not in valid:
            valid.append(source)
    if len(raw) > _MAX_HINTS:
        dropped.append(DroppedSource(source="...", reason="too_many_suggestions"))
    return valid, dropped


def _derived_intent(sources: set[SourceType]) -> SourceIntent:
    """Only for reporting a decision made without a planner intent."""
    if {DOCS, GITHUB, REDDIT} <= sources:
        return SourceIntent.TROUBLESHOOTING
    if GITHUB in sources:
        return SourceIntent.CODE_IMPLEMENTATION
    if DOCS in sources:
        return SourceIntent.TECHNICAL_HOWTO
    if REDDIT in sources:
        return SourceIntent.COMMUNITY_EXPERIENCE
    return SourceIntent.GENERAL_RESEARCH


def _get(understanding: Any, key: str) -> Any:
    if understanding is None:
        return None
    if isinstance(understanding, Mapping):
        return understanding.get(key)
    return getattr(understanding, key, None)


def task_technology(task: ResearchTask, understanding: Any = None) -> Optional[str]:
    """The technology a sub-question is about: the planner's own answer when it gave one. Only a
    sub-question with no technology of its own that is technical (or has no intent, i.e. legacy
    planner output: "how do I add middleware?" in a FastAPI conversation) inherits the one the
    query-understanding step found. A sub-question the planner called general research or current
    events does not: it is not about that technology."""
    own = (task.technology or "").strip()
    if own:
        return own
    intent = _parse_intent(task.source_intent)
    if intent is None or intent in TECHNICAL_INTENTS:
        inherited = (_get(understanding, "technology") or "").strip()
        return inherited or None
    return None


def _advice_decision(task: ResearchTask, intent: Optional[SourceIntent], suggestions: list[SourceType],
                     dropped: list[DroppedSource]) -> RoutingDecision:
    """Reddit (what people who have been there say) and web (articles, guides). Official
    documentation and code repositories do not answer "is it worth it"; a suggestion of either is
    reported as dropped."""
    origins = {s: ORIGIN_PLANNER if s in suggestions else ORIGIN_POLICY for s in (REDDIT, WEB)}
    for source in (DOCS, GITHUB):
        if source in suggestions:
            dropped.append(DroppedSource(source=source.value, reason="advice_question"))
    ordered = [s for s in SOURCE_ORDER if s in origins]
    decision = RoutingDecision(
        task_id=task.task_id,
        sub_question=task.sub_question,
        source_intent=intent or SourceIntent.COMMUNITY_EXPERIENCE,
        sources=ordered,
        origins={s.value: origins[s] for s in ordered},
        dropped=dropped,
    )
    logger.info("Task route: %s (advice question)%s", ",".join(decision.names),
                (" dropped=" + ",".join(f"{d.source}:{d.reason}" for d in dropped)) if dropped else "")
    return decision


def route_task(
    task: ResearchTask,
    understanding: Any = None,
    settings: Optional[RouterSettings] = None,
    docs_settings: Optional[DocsSettings] = None,
) -> RoutingDecision:
    """Decide the sources for one sub-question. ``understanding`` (a mapping or object with
    ``technology``) supplies the technology the query-understanding step identified. May
    raise (registry problems); callers isolate that and fall back to Phase 5 / web-only."""
    settings = settings or RouterSettings.from_env()
    docs_settings = docs_settings or DocsSettings.from_env()

    query = (task.search_query or task.sub_question).strip()
    technology = task_technology(task, understanding) or ""
    # The technology named by the planner / understanding step joins the text the rules read,
    # so "How do I add middleware?" under a FastAPI conversation still has a technology.
    signal_text = query if not technology or technology.lower() in query.lower() else f"{query} {technology}"

    registry = get_registry(docs_settings.registry_path)
    sig = detect_signals(signal_text, docs_settings)
    # Phase 5 rules over the sub-question: what the question independently supports (used to
    # gate planner suggestions). The fallback route below is over the whole question instead.
    baseline = route_sources(signal_text, settings, docs_settings)
    question_text = (_get(understanding, "resolved_query") or "").strip()
    fallback = route_sources(question_text, settings, docs_settings) if question_text else baseline
    # Does the USER's question ask for community experience (reddit wording, "what do developers
    # think", problems people have)? An error message alone does not: GitHub issues serve it better.
    question_signals = detect_signals(question_text or signal_text, docs_settings)
    wants_community = question_signals.reddit_explicit or question_signals.experience or question_signals.problems
    tech_entries = mentioned_technologies(signal_text, registry)
    technology_intent = bool(tech_entries)

    intent = _parse_intent(task.source_intent)
    suggestions, dropped = _validate_suggestions(task.suggested_sources)
    # Planner-led: the planner supplied a usable hint. Only then do the enforcement rules and
    # the source cap apply; a task with no hint gets exactly the Phase 5 route.
    planner_led = settings.enabled and (bool(suggestions) or intent is not None)

    # An advice/review question ("is it worth building X? does it help in interviews?"): the
    # technologies and sites the user names describe THEIR project, they are not the subject.
    # A sub-question that is itself technical or a documentation lookup is routed as usual.
    if (settings.enabled and _get(understanding, "intent") == "advice"
            and intent not in TECHNICAL_INTENTS and not sig.docs_intent.is_documentation_query):
        return _advice_decision(task, intent, suggestions, dropped)

    origins: dict[SourceType, str] = {}
    if not settings.enabled:
        # Kill switch: exactly the Phase 4 behaviour (web + the documentation detector).
        for source in fallback.sources:
            origins[source] = ORIGIN_FALLBACK
        dropped = [DroppedSource(source=str(raw)[:40], reason="router_disabled")
                   for raw in (task.suggested_sources or [])]
    elif suggestions:
        for source in suggestions:
            origins[source] = ORIGIN_PLANNER
    elif intent is not None:
        for source in INTENT_DEFAULTS[intent]:
            origins[source] = ORIGIN_POLICY
    else:
        for source in fallback.sources:
            origins[source] = ORIGIN_FALLBACK

    if planner_led:
        # "Technical" here is the brief's list (how-to / config / API / install / version): a
        # technical intent, or the Phase 4 documentation detector. A question about community
        # problems with a technology is not one; on the fallback path the Phase 5 rules
        # already decided.
        technical_task = intent in TECHNICAL_INTENTS or sig.docs_intent.is_documentation_query
        non_technical = not (technology_intent or sig.technical or sig.github_explicit
                             or intent in TECHNICAL_INTENTS)

        # Rule 2: no technical signal -> web only. Documentation/GitHub stay only if the sub-question
        # itself calls for them (Phase 5 rules); Reddit only if the USER's question asks for community
        # views ("what do students on reddit say ...") -- not because the planner wrote "Reddit" into a
        # sub-question of its own.
        if non_technical:
            for source in (DOCS, GITHUB, REDDIT):
                allowed = fallback.includes(source) if source == REDDIT else baseline.includes(source)
                if source in origins and not allowed:
                    if origins[source] == ORIGIN_PLANNER:
                        dropped.append(DroppedSource(source=source.value, reason="non_technical"))
                    del origins[source]

        # Rule 1: technology + technical task -> official docs, whatever the planner said.
        wants_docs = technology_intent and (technical_task or intent == SourceIntent.COMPARISON)
        if wants_docs and DOCS not in origins:
            origins[DOCS] = ORIGIN_POLICY

    if planner_led:
        # Documentation needs a registry technology to search; without one the adapter has nothing
        # to do, so say so instead of pretending it ran.
        if DOCS in origins:
            if not docs_settings.enabled:
                reason = "docs_disabled"
            elif not technology_intent:
                reason = "technology_not_in_registry"
            else:
                reason = ""
            if reason:
                if origins[DOCS] == ORIGIN_PLANNER:
                    dropped.append(DroppedSource(source=DOCS.value, reason=reason))
                del origins[DOCS]

        # Web is the baseline every task has (Phase 5); the planner cannot remove it.
        origins.setdefault(WEB, ORIGIN_POLICY)

        # Rule 3: the cap. Reddit, then GitHub -- unless the user's question asks for community
        # experience, then GitHub goes first. Documentation and web are never trimmed.
        for source in ((GITHUB, REDDIT) if wants_community else _TRIM_ORDER):
            if len(origins) <= settings.max_sources_per_task:
                break
            if source in origins:
                dropped.append(DroppedSource(source=source.value, reason="over_cap"))
                del origins[source]

    ordered = [s for s in SOURCE_ORDER if s in origins]
    decision = RoutingDecision(
        task_id=task.task_id,
        sub_question=task.sub_question,
        source_intent=intent or _derived_intent(set(ordered)),
        sources=ordered,
        origins={s.value: origins[s] for s in ordered},
        dropped=dropped,
        docs_from_technology=DOCS in origins and (
            fallback.docs_from_technology if not (suggestions or intent) or not settings.enabled
            else not sig.docs_intent.is_documentation_query),
    )
    logger.info("Task route: %s (%s)%s", ",".join(decision.names),
                ",".join(f"{s.value}={origins[s]}" for s in ordered),
                (" dropped=" + ",".join(f"{d.source}:{d.reason}" for d in dropped)) if dropped else "")
    return decision
