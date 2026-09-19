"""Documentation-intent detection: rule-based, deterministic, no LLM call.

A question is a documentation query only when ALL of these hold:

1. it names a technology from the registry (an alias, on word boundaries);
2. it carries a technical/documentation cue (install, configure, "how do I", API,
   version, ...). An alias listed as ``weak`` (``python``, ``compose``, ``openai``)
   additionally needs a *specific* cue; a bare "how do I ..." is not enough for it;
3. it is not opinion/comparison phrasing (vs, alternatives, best, popularity, ...).

Anything else is not routed to official docs and research proceeds exactly as before.
This is intentionally conservative: a miss costs nothing (web search still runs), a
false positive adds a pointless docs search.

Must stay importable without ``research_app.agent`` / LangChain.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from research_app.sources.official_docs.registry import DocsEntry, DocsRegistry, normalize_text

MAX_TECHNOLOGIES = 3
_MAX_TEXT_CHARS = 2000  # bounds regex work on very long input

# Cues that point at concrete technical facts.
_SPECIFIC_CUES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("install", r"\b(?:install(?:s|ed|ing|ation)?|set ?up|getting started|quick ?start)\b"),
        ("configure", r"\b(?:configur(?:e|es|ed|ing|ation|ations)|config|settings?|environment variables?|env vars?)\b"),
        ("syntax_api", r"\b(?:syntax|parameters?|arguments?|options?|flags?|commands?|usage|api|sdk|cli|reference|docs|documentation|official)\b"),
        ("version", r"\b(?:versions?|release notes?|changelog|what changed|what'?s new|breaking changes?|migrat(?:e|es|ed|ing|ion)|upgrad(?:e|es|ed|ing)|deprecat(?:ed|es|ion))\b"),
    )
)
# "how do I ..." style: technical in spirit but says nothing concrete.
_GENERIC_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("how_to", re.compile(r"\bhow (?:do|can|could|should|would|does|to)\b")),
)
# Opinion / comparison phrasing: not an authoritative-documentation question.
_VETOES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("versus", r"\b(?:vs\.?|versus)\b"),
        ("compare", r"\bcompar(?:e|es|ed|ing|ison|isons)\b"),
        ("alternatives", r"\balternatives?\b"),
        ("better_than", r"\bbetter than\b"),
        ("pros_cons", r"\bpros and cons\b"),
        ("popularity", r"\bpopular(?:ity)?\b"),
        ("best", r"\bbest\b(?! practices?)"),
        ("top_n", r"\btop \d+\b"),
        ("reviews", r"\breviews?\b"),
        ("worth_it", r"\bworth it\b"),
        ("choose", r"\b(?:choose|choosing|which (?:is|one|should))\b"),
        ("between", r"\bbetween .{1,60} and\b"),
    )
)


@dataclass(frozen=True)
class DocsIntent:
    """Outcome of detection. ``reason`` explains a negative result (for logs/tests)."""

    is_documentation_query: bool
    technologies: tuple[DocsEntry, ...] = ()
    cues: tuple[str, ...] = ()
    reason: str = ""

    @property
    def technology_ids(self) -> tuple[str, ...]:
        return tuple(t.id for t in self.technologies)


@dataclass
class _Hit:
    entry: DocsEntry
    weak: bool
    start: int
    end: int


def _find_hits(text: str, registry: DocsRegistry) -> list[_Hit]:
    """Alias occurrences, longest alias first, never overlapping (so the ``docker`` inside
    ``docker compose`` is not a second technology), returned in text order."""
    taken: list[tuple[int, int]] = []
    hits: list[_Hit] = []
    for item in registry.alias_patterns():
        for found in item.pattern.finditer(text):
            span = (found.start(), found.end())
            if any(span[0] < end and start < span[1] for start, end in taken):
                continue
            taken.append(span)
            hits.append(_Hit(item.entry, item.weak, *span))
    hits.sort(key=lambda h: (h.start, h.end))
    return hits


def mentioned_technologies(
    text: Optional[str], registry: DocsRegistry, *, include_weak: bool = False,
    max_technologies: int = MAX_TECHNOLOGIES,
) -> tuple[DocsEntry, ...]:
    """Registry technologies named in ``text``, with no cue or veto rules (Phase 5: the
    source router decides separately whether documentation is useful). Weak aliases
    (``python``, ``compose``, ``openai``) are left out unless ``include_weak``."""
    if not isinstance(text, str) or not text.strip():
        return ()
    normalized = normalize_text(text[:_MAX_TEXT_CHARS]).replace("’", "'")
    found: dict[str, DocsEntry] = {}
    for hit in _find_hits(normalized, registry):
        if hit.weak and not include_weak:
            continue
        found.setdefault(hit.entry.id, hit.entry)
    return tuple(found.values())[:max_technologies]


def detect_docs_intent(
    text: Optional[str], registry: DocsRegistry, *, max_technologies: int = MAX_TECHNOLOGIES
) -> DocsIntent:
    if not isinstance(text, str) or not text.strip():
        return DocsIntent(False, reason="empty")
    normalized = normalize_text(text[:_MAX_TEXT_CHARS]).replace("’", "'")

    hits = _find_hits(normalized, registry)
    if not hits:
        return DocsIntent(False, reason="no_technology")

    specific = tuple(name for name, rx in _SPECIFIC_CUES if rx.search(normalized))
    generic = tuple(name for name, rx in _GENERIC_CUES if rx.search(normalized))
    cues = specific + generic

    for name, rx in _VETOES:
        if rx.search(normalized):
            return DocsIntent(False, cues=cues, reason=f"vetoed:{name}")
    if not cues:
        return DocsIntent(False, reason="no_cue")

    accepted: dict[str, DocsEntry] = {}
    for hit in hits:
        if hit.weak and not specific:
            continue
        accepted.setdefault(hit.entry.id, hit.entry)
    if not accepted:
        return DocsIntent(False, cues=cues, reason="weak_technology_needs_specific_cue")

    technologies = tuple(accepted.values())[:max_technologies]
    return DocsIntent(True, technologies=technologies, cues=cues, reason="matched")
