"""Official-documentation registry: technology -> official sites (host + path prefixes).

Data, not code: the built-in registry is ``data/official_docs.json``; a user file named
by ``OFFICIAL_DOCS_REGISTRY_PATH`` is merged over it (same ``id`` replaces the built-in
entry, new ids are appended). Nothing here knows about a specific technology.

``DocsRegistry.match_url`` is the **security boundary** that decides whether a URL may
be labelled ``official_docs``. It is deliberately strict:

- https only; no credentials; no port other than 443;
- the hostname must EQUAL a registered host (a leading ``www.`` is ignored on both
  sides). There is no suffix or subdomain matching, so ``fastapi.tiangolo.com.evil.com``,
  ``evilfastapi.tiangolo.com`` and ``fastapi.tiangolo.com@evil.com`` all fail. To accept
  a subdomain, list it explicitly;
- if the site has path prefixes, the URL path (percent-decoded once, dot-segments
  resolved, as a client would resolve it) must be at or below one of them on a path
  *segment* boundary, so ``/compose/../engine`` is not under ``/compose`` and
  ``/langgraphevil`` is not under ``/langgraph``;
- when several sites match, the longest matching path prefix wins, then registry order,
  so the result never depends on dict/set ordering.

Must stay importable without ``research_app.agent`` / LangChain.
"""
from __future__ import annotations

import functools
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Collection, NamedTuple, Optional, Sequence
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

BUILTIN_REGISTRY_PATH = Path(__file__).parent / "data" / "official_docs.json"

_HOST_RE = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_URL_PATH = 2048  # bounds the work done by version regexes on hostile input
_MAX_REGISTRY_BYTES = 1_000_000


def normalize_text(value: str) -> str:
    """Lower-case, NFKC, hyphens/underscores as spaces, whitespace collapsed.

    The one text derivation shared by alias registration and query matching, so
    ``docker-compose`` and ``Docker Compose`` are the same thing."""
    text = unicodedata.normalize("NFKC", value).lower()
    text = re.sub(r"[-_]+", " ", text)
    return " ".join(text.split())


def host_key(host: str) -> str:
    """Comparison key for a hostname: lower-case, no trailing dot, no leading ``www.``."""
    host = host.lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def normalize_path(raw_path: str) -> Optional[str]:
    """Resolve a URL path the way a client would; ``None`` if it looks hostile."""
    if len(raw_path) > _MAX_URL_PATH:
        return None
    path = unquote(raw_path)
    if "\\" in path or any(ord(ch) < 32 or ord(ch) == 127 for ch in path):
        return None
    out: list[str] = []
    for segment in path.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if out:
                out.pop()
            continue
        out.append(segment)
    return "/" + "/".join(out)


def _under_prefix(path: str, prefix: str) -> bool:
    """``prefix`` is normalised (no trailing slash; ``""`` = whole host)."""
    return not prefix or path == prefix or path.startswith(prefix + "/")


# --------------------------------------------------------------------------- models

class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DocsSite(_Model):
    """One official documentation location for a technology."""

    host: str
    path_prefixes: tuple[str, ...] = ()
    version_patterns: tuple[str, ...] = ()

    @field_validator("host")
    @classmethod
    def _valid_host(cls, value: str) -> str:
        value = value.strip().lower()
        if not _HOST_RE.match(value):
            raise ValueError("host must be a bare hostname (no scheme, port, path or wildcard)")
        return value

    @field_validator("path_prefixes")
    @classmethod
    def _valid_prefixes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for value in values:
            value = value.strip()
            if not value.startswith("/") or any(c in value for c in "?#\\ \t\n") or ".." in value.split("/"):
                raise ValueError("path prefix must start with '/' and be a plain path")
            value = value.rstrip("/")
            if not value:  # "/" means the whole host
                return ()
            cleaned.append(value)
        return tuple(dict.fromkeys(cleaned))

    @field_validator("version_patterns")
    @classmethod
    def _valid_patterns(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            if len(value) > 200:
                raise ValueError("version pattern too long")
            try:
                compiled = re.compile(value)
            except re.error:
                raise ValueError("version pattern is not a valid regular expression") from None
            if "version" not in compiled.groupindex:
                raise ValueError("version pattern needs a (?P<version>...) group")
        return values


class DocsEntry(_Model):
    """A technology and where its official documentation lives.

    ``aliases`` are unambiguous names. ``weak_aliases`` are names that are also
    ordinary words or too broad on their own (``python``, ``compose``): the detector
    needs a *specific* technical cue (install, configure, API, version, ...) before it
    accepts one, not just "how do I".
    """

    id: str
    name: str = Field(min_length=1, max_length=100)
    aliases: tuple[str, ...] = Field(min_length=1)
    weak_aliases: tuple[str, ...] = ()
    sites: tuple[DocsSite, ...] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError("id must be lower-case letters, digits, '-' or '_'")
        return value

    @field_validator("aliases", "weak_aliases")
    @classmethod
    def _valid_aliases(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = [normalize_text(v) for v in values]
        if any(len(a) < 2 or len(a) > 60 for a in cleaned):
            raise ValueError("an alias must be 2-60 characters")
        return tuple(dict.fromkeys(cleaned))

    def search_hosts(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(site.host for site in self.sites))


class DocsMatch(NamedTuple):
    entry: DocsEntry
    site: DocsSite
    matched_prefix: Optional[str]  # the site path prefix the URL fell under, if any
    version: Optional[str]  # from the URL, via the site's version patterns


class AliasPattern(NamedTuple):
    alias: str
    entry: DocsEntry
    weak: bool
    pattern: "re.Pattern[str]"


# --------------------------------------------------------------------------- registry

class DocsRegistry:
    """Immutable, deterministic lookup over a sequence of entries (order = precedence)."""

    def __init__(self, entries: Sequence[DocsEntry]):
        self._entries = tuple(entries)
        self._by_id = {e.id: e for e in self._entries}

        # host key -> [(order, entry, site)]
        self._sites: dict[str, list[tuple[int, DocsEntry, DocsSite]]] = {}
        for order, entry in enumerate(self._entries):
            for site in entry.sites:
                self._sites.setdefault(host_key(site.host), []).append((order, entry, site))

        # Longest alias first (so "docker compose" is tried before "docker"), then
        # alphabetical: a fixed order independent of insertion.
        seen: dict[str, str] = {}
        patterns: list[AliasPattern] = []
        for entry in self._entries:
            for weak, names in ((False, entry.aliases), (True, entry.weak_aliases)):
                for alias in names:
                    if alias in seen:
                        logger.warning("Docs registry: alias %r of %r already belongs to %r; ignored",
                                       alias, entry.id, seen[alias])
                        continue
                    seen[alias] = entry.id
                    regex = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")
                    patterns.append(AliasPattern(alias, entry, weak, regex))
        patterns.sort(key=lambda p: (-len(p.alias), p.alias))
        self._aliases = tuple(patterns)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[DocsEntry, ...]:
        return self._entries

    def get(self, entry_id: str) -> Optional[DocsEntry]:
        return self._by_id.get(entry_id)

    def alias_patterns(self) -> tuple[AliasPattern, ...]:
        return self._aliases

    def match_url(self, url: str, *, allowed_hosts: Optional[Collection[str]] = None) -> Optional[DocsMatch]:
        """The registry site ``url`` belongs to, or ``None`` if it is not official.

        ``allowed_hosts`` (the hosts a search was restricted to) narrows the candidates
        further: a host that was not asked for is never accepted."""
        try:
            parts = urlsplit(url.strip())
            hostname = parts.hostname
            port = parts.port
        except (ValueError, AttributeError):
            return None
        if parts.scheme.lower() != "https" or not hostname or "@" in parts.netloc:
            return None
        if port not in (None, 443):
            return None
        path = normalize_path(parts.path)
        if path is None:
            return None

        key = host_key(hostname)
        allowed = None if allowed_hosts is None else {host_key(h) for h in allowed_hosts}
        if allowed is not None and key not in allowed:
            return None

        best: Optional[tuple[int, int]] = None  # (prefix length, -order): larger wins
        best_match: Optional[tuple[DocsEntry, DocsSite, Optional[str]]] = None
        for order, entry, site in self._sites.get(key, ()):
            if not site.path_prefixes:
                prefix, rank = None, (0, -order)
            else:
                matching = [p for p in site.path_prefixes if _under_prefix(path, p)]
                if not matching:
                    continue
                prefix = max(matching, key=len)
                rank = (len(prefix), -order)
            if best is None or rank > best:
                best, best_match = rank, (entry, site, prefix)
        if best_match is None:
            return None
        entry, site, prefix = best_match
        return DocsMatch(entry, site, prefix, _extract_version(site, path))


def _extract_version(site: DocsSite, path: str) -> Optional[str]:
    for pattern in site.version_patterns:
        found = re.search(pattern, path)
        if found and found.group("version"):
            return found.group("version")
    return None


# --------------------------------------------------------------------------- loading

def parse_entries(data: Any, *, source: str) -> list[DocsEntry]:
    """Validate registry JSON. An invalid entry is skipped and logged (its id and the
    problem locations only); it never prevents the others from loading."""
    if not isinstance(data, dict) or not isinstance(data.get("technologies"), list):
        raise ValueError(f"{source}: expected an object with a 'technologies' list")
    entries: list[DocsEntry] = []
    seen: set[str] = set()
    for index, raw in enumerate(data["technologies"]):
        label = raw.get("id") if isinstance(raw, dict) and isinstance(raw.get("id"), str) else f"#{index}"
        try:
            entry = DocsEntry.model_validate(raw)
        except ValidationError as exc:
            where = ", ".join(".".join(str(p) for p in err["loc"]) for err in exc.errors())
            logger.warning("Docs registry (%s): skipping entry %s, invalid fields: %s", source, label, where)
            continue
        if entry.id in seen:
            logger.warning("Docs registry (%s): duplicate id %s; keeping the first", source, entry.id)
            continue
        seen.add(entry.id)
        entries.append(entry)
    return entries


def load_entries(path: Path | str) -> list[DocsEntry]:
    """Read and validate one registry file. Raises ``OSError`` / ``ValueError`` if the
    file itself is unusable."""
    file = Path(path)
    if file.stat().st_size > _MAX_REGISTRY_BYTES:
        raise ValueError(f"{file.name}: registry file is too large")
    with file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return parse_entries(data, source=file.name)


def merge_entries(base: Sequence[DocsEntry], extra: Sequence[DocsEntry]) -> list[DocsEntry]:
    """``extra`` replaces a ``base`` entry with the same id (keeping its position);
    other ``extra`` entries are appended."""
    replacements = {e.id: e for e in extra}
    merged = [replacements.pop(e.id, e) for e in base]
    merged.extend(e for e in extra if e.id in replacements)
    return merged


@functools.lru_cache(maxsize=8)
def get_registry(extra_path: Optional[str] = None) -> DocsRegistry:
    """The built-in registry, plus ``extra_path`` merged over it. Cached per path (a
    changed file needs a restart). Never raises: an unreadable built-in file yields an
    empty registry and an unreadable extra file is ignored, both logged, so official
    docs turn themselves off instead of breaking research."""
    try:
        entries = load_entries(BUILTIN_REGISTRY_PATH)
    except (OSError, ValueError) as exc:
        logger.error("Built-in official-docs registry unusable (%s); official docs disabled", type(exc).__name__)
        entries = []
    if extra_path:
        try:
            entries = merge_entries(entries, load_entries(extra_path))
        except (OSError, ValueError) as exc:
            logger.error("OFFICIAL_DOCS_REGISTRY_PATH unusable (%s); using the built-in registry only",
                         type(exc).__name__)
    return DocsRegistry(entries)
