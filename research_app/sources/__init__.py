"""Source normalisation (Phase 3): raw provider results -> ``SourceDocument``.

Official documentation (Phase 4) lives in the ``research_app.sources.official_docs``
subpackage; import it explicitly (it is not re-exported here)."""
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
