"""Source normalisation (Phase 3): raw provider results -> ``SourceDocument``.

Official documentation (Phase 4) lives in the ``research_app.sources.official_docs``
subpackage, and the source router with the GitHub/Reddit adapters (Phase 5) in
``research_app.sources.routing``; import them explicitly (they are not re-exported here)."""
from research_app.sources.normalizer import (
    SNIPPET_MAX_CHARS,
    NormalizationError,
    clean_text,
    normalize_source,
    normalize_sources,
    normalize_url,
    parse_datetime,
)
from research_app.sources.tavily import normalize_tavily_response

__all__ = [
    "NormalizationError", "SNIPPET_MAX_CHARS", "clean_text", "normalize_source",
    "normalize_sources", "normalize_tavily_response", "normalize_url", "parse_datetime",
]
