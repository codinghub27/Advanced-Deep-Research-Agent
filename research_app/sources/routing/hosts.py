"""Strict hostname validation for community/code sources.

A result is GitHub or Reddit ONLY if the hostname of its URL is one of an exact allowlist.
Titles, snippets and text are never consulted. Suffix matching is deliberately not used, so
``github.com.evil.com``, ``evilgithub.com``, ``github.io`` and ``githubusercontent.com`` are
all rejected; so are userinfo tricks (``github.com@evil.com``), backslash tricks, non-default
ports, trailing-dot hosts and non-ASCII look-alikes. A miss costs nothing (the result is
dropped); a false accept would let an arbitrary site carry a trusted-looking label.

Pure functions, no I/O. Must stay importable without ``research_app.agent`` / LangChain.
"""
from __future__ import annotations

from typing import Optional
from urllib.parse import urlsplit

GITHUB_HOSTS = frozenset({"github.com", "www.github.com"})
REDDIT_HOSTS = frozenset({"reddit.com", "www.reddit.com", "old.reddit.com"})

_DEFAULT_PORTS = (None, 80, 443)


def hostname_of(url: object) -> Optional[str]:
    """The lower-cased hostname of an absolute http(s) URL, or ``None`` when the URL is
    not a plain ``scheme://host[:default-port]/...`` URL."""
    if not isinstance(url, str) or any(ord(ch) < 32 or ord(ch) == 127 for ch in url):
        return None
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not host:
        return None
    if "@" in parts.netloc or "\\" in parts.netloc or port not in _DEFAULT_PORTS:
        return None
    return host


def is_github_url(url: object) -> bool:
    return hostname_of(url) in GITHUB_HOSTS


def is_reddit_url(url: object) -> bool:
    return hostname_of(url) in REDDIT_HOSTS
