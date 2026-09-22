"""Raw source result -> ``SourceDocument`` (the common internal format).

Source-agnostic. A raw result is a mapping using this small vocabulary; a
source-specific adapter (see ``sources/tavily.py``) is responsible for getting
its payload into that shape:

    url            required, absolute http(s)
    title          optional
    content        optional, the body text
    snippet        optional; derived from ``content`` when absent
    author         optional
    published_at   optional (alias: ``published_date``); datetime, ISO-8601 or RFC 2822
    <anything else>  kept verbatim in ``SourceDocument.metadata``

What it does NOT do (later phases): rank, de-duplicate, score quality, extract
evidence, or fetch anything. Two raw results with the same URL become two
documents with the same ``source_id``.

Must stay importable without ``research_app.agent`` / LangChain (see
``tests/test_source_normalizer.py``).
"""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from research_app.domain import DateConfidence, SourceDocument, SourceType, utc_now

logger = logging.getLogger(__name__)

# Derived snippet length. Matches the 150 characters the SSE ``sources`` event
# has always carried.
SNIPPET_MAX_CHARS = 150

_URL_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
# C0 controls except \t \n \r (line handling is done separately).
_TEXT_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH = {ord("​"): None, ord("﻿"): None}
_BLANK_LINES = re.compile(r"\n{3,}")


class NormalizationError(ValueError):
    """A raw result cannot become a ``SourceDocument``. The message describes the
    problem only; it never contains the raw content."""


# --------------------------------------------------------------------------- field helpers

def normalize_url(raw: str) -> str:
    """Tidy a URL for storage and display; raises ``NormalizationError`` if it is
    not an absolute http(s) URL.

    Removes control characters and surrounding whitespace, lower-cases scheme and
    host, and drops default ports. Path, query and fragment are kept as given (a
    fragment can identify the documented section, so it stays; identity ignores it
    via ``source_id``). URLs carrying credentials (``user:pass@host``) are rejected
    so they can never reach citations or logs.
    """
    cleaned = _URL_CONTROL_CHARS.sub("", raw).strip()
    try:
        parts = urlsplit(cleaned)
        host = parts.hostname
        port = parts.port
    except ValueError:
        raise NormalizationError("invalid url") from None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https") or not host:
        raise NormalizationError("url must be an absolute http(s) URL")
    if "@" in parts.netloc:
        raise NormalizationError("url must not contain credentials")
    if ":" in host:  # IPv6 literal; hostname strips the brackets
        host = f"[{host}]"
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"
    return urlunsplit((scheme, host, parts.path, parts.query, parts.fragment))


def clean_text(value: Any, *, single_line: bool = False) -> Optional[str]:
    """Normalise text; ``None`` for non-strings and for text that ends up empty.

    Always: Unicode NFC, non-breaking and zero-width spaces, control characters.
    ``single_line=True`` (titles, snippets) collapses all whitespace. Otherwise
    line breaks are kept and only trailing spaces, ``\\r\\n`` and runs of blank
    lines are tidied. Spaces *inside* a line are left alone so indented code
    survives.
    """
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFC", value).translate(_ZERO_WIDTH).replace(" ", " ")
    text = _TEXT_CONTROL_CHARS.sub("", text)
    if single_line:
        text = " ".join(text.split())
    else:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        text = _BLANK_LINES.sub("\n\n", text).strip("\n")
    return text or None


def parse_datetime(value: Any) -> Optional[datetime]:
    """``datetime`` / ISO-8601 / RFC 2822 -> ``datetime``; ``None`` if unparseable.
    Naive results are tagged UTC by ``SourceDocument``."""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None


_RELATIVE_DATE = re.compile(
    r"^(?:(?P<amount>\d+)\s+(?P<unit>day|week|month|year)s?\s+ago|(?P<yesterday>yesterday)"
    r"|(?P<today>today|just now))$",
    re.I,
)
_RELATIVE_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}


def parse_relative_date(value: Any, *, now: Optional[datetime] = None) -> Optional[datetime]:
    """"N days/weeks/months/years ago", "yesterday", "today"/"just now" -> ``datetime``,
    anchored to ``now`` (the batch's ``retrieved_at`` when called from the normalizer, since
    that is the only clock a provider's relative wording can be read against). ``None`` for
    anything else -- this never guesses at free text, only an exact relative-date phrase.
    """
    if not isinstance(value, str):
        return None
    match = _RELATIVE_DATE.match(value.strip())
    if not match:
        return None
    anchor = now or utc_now()
    if match.group("yesterday"):
        return anchor - timedelta(days=1)
    if match.group("today"):
        return anchor
    amount = int(match.group("amount"))
    unit = match.group("unit").lower()
    return anchor - timedelta(days=amount * _RELATIVE_DAYS[unit])


def _extract_date_field(
    raw: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    anchor: datetime,
    consumed: set,
) -> tuple[Optional[datetime], Optional[DateConfidence], Optional[str]]:
    """Try each key in ``keys`` in order, mirroring the original single-field loop exactly:
    the first key with a *parseable* value wins and is consumed; a key whose value cannot be
    parsed (absolute or relative) is left unconsumed so it survives in ``metadata`` verbatim
    (never invented, never dropped), and the loop moves on to the next key. Adds relative-date
    support and confidence/source bookkeeping on top of the original behaviour.
    """
    confidence: Optional[DateConfidence] = None
    source: Optional[str] = None
    for key in keys:
        value = raw.get(key)
        if value is None:
            consumed.add(key)
            continue
        parsed = parse_datetime(value)
        if parsed is not None:
            consumed.add(key)
            return parsed, DateConfidence.EXACT, key
        relative = parse_relative_date(value, now=anchor)
        if relative is not None:
            consumed.add(key)
            return relative, DateConfidence.APPROXIMATE, f"relative:{key}"
        if confidence is None:
            confidence = DateConfidence.UNKNOWN
            source = key
    return None, confidence, source


# --------------------------------------------------------------------------- normalizers

def normalize_source(
    raw: Any,
    *,
    source_type: SourceType = SourceType.WEB,
    provider: Optional[str] = None,
    query: Optional[str] = None,
    task_id: Optional[str] = None,
    retrieved_at: Optional[datetime] = None,
) -> SourceDocument:
    """Convert one raw result. Raises ``NormalizationError`` if it has no usable
    URL or is not a mapping. An unknown ``source_type`` raises ``ValueError``
    (a caller bug, not bad data)."""
    source_type = SourceType(source_type)
    if not isinstance(raw, Mapping):
        raise NormalizationError(f"raw result must be a mapping, got {type(raw).__name__}")
    raw_url = raw.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise NormalizationError("result has no url")
    url = normalize_url(raw_url)

    # A key is "consumed" once its value has been used; everything else is kept in
    # metadata so nothing the provider sent is silently lost.
    consumed = {"url"}

    def text(key: str, *, single_line: bool) -> Optional[str]:
        value = raw.get(key)
        if value is None or isinstance(value, str):
            consumed.add(key)
        return clean_text(value, single_line=single_line)

    title = text("title", single_line=True)
    author = text("author", single_line=True)
    snippet = text("snippet", single_line=True)
    content = text("content", single_line=False)
    if snippet is None and content:
        snippet = " ".join(content.split())[:SNIPPET_MAX_CHARS]

    date_anchor = retrieved_at or utc_now()
    published_at, published_confidence, published_source = _extract_date_field(
        raw, ("published_at", "published_date"), anchor=date_anchor, consumed=consumed,
    )
    updated_at, updated_confidence, updated_source = _extract_date_field(
        raw, ("updated_at", "updated_date", "last_updated", "modified_at"),
        anchor=date_anchor, consumed=consumed,
    )
    # The freshness reference prefers "last updated" over "published" (P1.2); its
    # confidence/source describe whichever field that reference came from.
    if updated_at is not None:
        date_confidence, date_source = updated_confidence, updated_source
    elif published_at is not None:
        date_confidence, date_source = published_confidence, published_source
    else:
        date_confidence = updated_confidence or published_confidence
        date_source = updated_source or published_source

    metadata = {str(k): v for k, v in raw.items() if k not in consumed}
    extra: dict[str, Any] = {"retrieved_at": retrieved_at} if retrieved_at else {}
    try:
        return SourceDocument(
            source_type=source_type,
            title=title,
            url=url,
            author=author,
            published_at=published_at,
            updated_at=updated_at,
            date_confidence=date_confidence,
            date_source=date_source,
            snippet=snippet,
            content=content,
            provider=provider,
            original_url=raw_url.strip(),
            query=query,
            task_id=task_id,
            metadata=metadata,
            **extra,
        )
    except ValidationError as exc:
        raise NormalizationError("result failed validation") from exc


def normalize_sources(
    raws: Iterable[Any],
    *,
    retrieved_at: Optional[datetime] = None,
    **kwargs: Any,
) -> list[SourceDocument]:
    """Normalise each raw result independently, in order. Unusable results are
    skipped and logged (reason and position only); one bad result never affects
    the others. No de-duplication. The batch shares one ``retrieved_at``."""
    retrieved_at = retrieved_at or utc_now()
    docs: list[SourceDocument] = []
    for index, raw in enumerate(raws):
        try:
            docs.append(normalize_source(raw, retrieved_at=retrieved_at, **kwargs))
        except NormalizationError as exc:
            logger.warning(
                "Skipping result %d from %s: %s", index, kwargs.get("provider") or "unknown", exc
            )
    return docs
