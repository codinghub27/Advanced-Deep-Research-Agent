"""Source router (Phase 5): official docs, web, GitHub and Reddit as first-class sources.

hosts.py     strict hostname allowlists (GitHub, Reddit); look-alikes are rejected
adapters.py  SourceAdapter for GitHub and Reddit: Tavily restricted with ``include_domains``,
             every result URL verified, then re-typed and given URL-derived metadata
router.py    rule-based ``route_sources(question) -> RoutePlan`` (no LLM, no scoring)
task_router.py  Phase 6: ``route_task(task, understanding) -> RoutingDecision``; validates the
             planner's proposals with the router rules (planner proposes, router decides)
executor.py  concurrent, failure-isolated execution of adapters
settings.py  SOURCE_ROUTER_* environment settings

Official documentation stays in ``research_app.sources.official_docs`` (Phase 4). Web search
stays the direct Tavily path in ``agent/state.py``. The provider is always the one Tavily
tool; there is no GitHub/Reddit API.
"""
from research_app.sources.routing.adapters import (
    DEFAULT_MAX_RESULTS,
    GITHUB_POLICY,
    REDDIT_POLICY,
    DomainPolicy,
    DomainSourceAdapter,
    GitHubAdapter,
    RedditAdapter,
    adapter_for,
    build_request,
    github_metadata,
    reddit_metadata,
)
from research_app.sources.routing.executor import run_adapters
from research_app.sources.routing.hosts import (
    GITHUB_HOSTS,
    REDDIT_HOSTS,
    hostname_of,
    is_github_url,
    is_reddit_url,
)
from research_app.sources.routing.router import (
    SOURCE_ORDER,
    WEB_ONLY,
    RoutePlan,
    RouteSignals,
    detect_signals,
    plan_technology_docs,
    route_sources,
)
from research_app.sources.routing.settings import RouterSettings
from research_app.sources.routing.task_router import (
    INTENT_DEFAULTS,
    TECHNICAL_INTENTS,
    route_task,
    task_technology,
)

__all__ = [
    "DEFAULT_MAX_RESULTS", "INTENT_DEFAULTS", "TECHNICAL_INTENTS", "route_task", "task_technology", "DomainPolicy", "DomainSourceAdapter", "GITHUB_HOSTS", "GITHUB_POLICY",
    "GitHubAdapter", "REDDIT_HOSTS", "REDDIT_POLICY", "RedditAdapter", "RoutePlan", "RouteSignals",
    "RouterSettings", "SOURCE_ORDER", "WEB_ONLY", "adapter_for", "build_request", "detect_signals",
    "github_metadata", "hostname_of", "is_github_url", "is_reddit_url", "plan_technology_docs",
    "reddit_metadata", "route_sources", "run_adapters",
]
