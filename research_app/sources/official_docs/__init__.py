"""Official documentation as a first-class source (Phase 4).

registry.py   technology -> official hosts/paths (JSON data) and the URL verification
detector.py   rule-based "is this a documentation question, about which technology?"
adapter.py    SourceAdapter: domain-restricted search -> verified ``official_docs`` documents
priority.py   official documentation sorts first for documentation questions
plan.py       the one call the agent makes (Phase 6 can replace it)
settings.py   OFFICIAL_DOCS_* environment settings
"""
from research_app.sources.official_docs.adapter import (
    DEFAULT_MAX_RESULTS,
    OfficialDocsAdapter,
    build_docs_request,
)
from research_app.sources.official_docs.detector import DocsIntent, detect_docs_intent, mentioned_technologies
from research_app.sources.official_docs.plan import DocsPlan, plan_official_docs
from research_app.sources.official_docs.priority import authority_rank, is_official, prioritize
from research_app.sources.official_docs.registry import (
    DocsEntry,
    DocsMatch,
    DocsRegistry,
    DocsSite,
    get_registry,
    load_entries,
    merge_entries,
)
from research_app.sources.official_docs.settings import DocsSettings

__all__ = [
    "DEFAULT_MAX_RESULTS", "DocsEntry", "DocsIntent", "DocsMatch", "DocsPlan", "DocsRegistry",
    "DocsSettings", "DocsSite", "OfficialDocsAdapter", "authority_rank", "build_docs_request",
    "detect_docs_intent", "get_registry", "is_official", "load_entries", "mentioned_technologies",
    "merge_entries",
    "plan_official_docs", "prioritize",
]
