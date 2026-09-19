"""Phase 2 must not change runtime behaviour: the domain package stays
self-contained and the app's public surface matches the Phase 1 baseline."""
import os
import subprocess
import sys
import unittest

from research_app.agent.graph import compiled_graph
from research_app.main import app

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_ROUTES = {
    ("DELETE", "/admin/clear-all-data"),
    ("DELETE", "/api/sessions/{session_id}"),
    ("GET", "/"),
    ("GET", "/admin/db-stats"),
    ("GET", "/api/research/history"),
    ("GET", "/api/sessions"),
    ("GET", "/api/sessions/{session_id}/queries"),
    ("GET", "/health"),
    ("GET", "/research/ui"),
    ("GET", "/test-db"),
    ("PATCH", "/api/sessions/{session_id}"),
    ("POST", "/api/research"),
    ("POST", "/api/research/stream"),
    ("POST", "/api/sessions"),
    ("POST", "/auth/login"),
    ("POST", "/auth/register"),
}
# Phase 6 added these (the conversation endpoints); the 16 routes above are unchanged.
PHASE6_ROUTES = {
    ("DELETE", "/sessions/{session_id}"),
    ("GET", "/sessions"),
    ("GET", "/sessions/{session_id}"),
}
BASELINE_NODES = {
    "semantic_cache_node", "classify_node", "simple_search_node", "planner_node",
    "search_node", "synthesize_node", "save_to_cache_node",
}
# Phase 6 replaced the single synthesis step (synthesize_node) with the evidence pipeline; every
# other baseline node is still there, unchanged in name.
PHASE6_NODES = (BASELINE_NODES - {"synthesize_node"}) | {
    "evidence_collection", "gap_detection", "targeted_search", "synthesis_node",
    "critic_node", "retry_node", "format_response",
}
FRAMEWORK_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


class DomainIsolationTests(unittest.TestCase):
    def test_domain_import_pulls_in_no_app_or_framework_modules(self):
        code = (
            "import sys, research_app.domain, research_app.domain.legacy\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "{'langgraph','langchain_core','langchain','langchain_groq','langchain_tavily',"
            "'qdrant_client','sqlalchemy','fastapi'} "
            "or m.startswith(('research_app.agent','research_app.db','research_app.main','research_app.auth')))\n"
            "print(','.join(bad))\n"
        )
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(out.stdout.strip(), "", f"domain imported: {out.stdout.strip()}")

    def test_sources_packages_pull_in_no_app_or_framework_modules(self):
        # Phase 4/5: the official-docs and routing packages (registry, detector, adapters,
        # router) must stay importable without LangChain/LangGraph/agent code, like the
        # rest of sources/.
        code = (
            "import sys, research_app.sources, research_app.sources.official_docs, "
            "research_app.sources.routing\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "{'langgraph','langchain_core','langchain','langchain_groq','langchain_tavily',"
            "'qdrant_client','sqlalchemy','fastapi'} "
            "or m.startswith(('research_app.agent','research_app.db','research_app.main','research_app.auth')))\n"
            "print(','.join(bad))\n"
        )
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(out.stdout.strip(), "", f"sources imported: {out.stdout.strip()}")


class BaselineSnapshotTests(unittest.TestCase):
    def test_application_routes_are_the_baseline_plus_the_session_endpoints(self):
        routes = {
            (method, r.path)
            for r in app.routes
            if r.path not in FRAMEWORK_PATHS
            for method in (getattr(r, "methods", None) or ())
            if method != "HEAD"
        }
        self.assertEqual(routes, BASELINE_ROUTES | PHASE6_ROUTES)
        self.assertTrue(BASELINE_ROUTES <= routes)

    def test_graph_nodes_are_the_baseline_plus_the_phase6_pipeline(self):
        nodes = set(compiled_graph.get_graph().nodes) - {"__start__", "__end__"}
        self.assertEqual(nodes, PHASE6_NODES)
        self.assertTrue((BASELINE_NODES - {"synthesize_node"}) <= nodes)


if __name__ == "__main__":
    unittest.main()
