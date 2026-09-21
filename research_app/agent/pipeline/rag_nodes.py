"""Graph nodes for source RAG (Phase 7). Both are off unless their flag is set.

Flow (see ``agent/graph.py``):
  classify_node -> rag_gate_router --(SOURCE_RAG_ENABLED, not time-sensitive)--> source_rag_node
      source_rag_node -> rag_router --(relevant stored chunks)--> evidence_collection
                                    --(otherwise)--> the Phase 6 route (simple_search / planner)
  format_response -> index_sources_node -> save_to_cache      (SOURCE_RAG_INGEST_ENABLED)

With both flags off ``rag_gate_router`` is exactly ``classify_router`` and ``index_sources_node``
does nothing, so the graph behaves as in Phase 6. There is still no edge back to an earlier node.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from research_app.agent import temporal
from research_app.agent.state import _SOURCE_INTENT_FOR, classify_router
from research_app.domain import ResearchTask
from research_app.rag.ingest import SourceIndexer
from research_app.rag.retriever import HybridRetriever
from research_app.rag.settings import RagSettings

logger = logging.getLogger(__name__)

SOURCE_RAG_NODE = "source_rag_node"
EVIDENCE_NODE = "evidence_collection"


@lru_cache(maxsize=2)
def get_retriever(settings: RagSettings) -> HybridRetriever:
    return HybridRetriever(settings)


@lru_cache(maxsize=2)
def get_indexer(settings: RagSettings) -> SourceIndexer:
    return SourceIndexer(settings)


def rag_gate_router(state) -> str:
    """Where ``classify_node`` goes next: the stored-source lookup when it is on and the question
    does not depend on today's date, otherwise where it always went."""
    settings = RagSettings.from_env()
    if settings.source_rag_enabled and not temporal.is_time_sensitive(
            state.get("question", ""), state.get("understanding")):
        return SOURCE_RAG_NODE
    return classify_router(state)


async def source_rag_node(state):
    """Look the question up in ``source_chunks``. Enough relevant chunks become the run's evidence;
    anything else (too few, a failure) leaves the state untouched so normal research follows."""
    settings = RagSettings.from_env()
    query = state.get("resolved_query") or state["question"]
    docs = await get_retriever(settings).retrieve(query)
    top = max((d.metadata.get("rag_score", 0.0) for d in docs), default=0.0)
    relevant = [d for d in docs if d.metadata.get("rag_score", 0.0) >= settings.relevance_threshold]
    hit = len(relevant) >= settings.min_chunks
    logger.info("Source RAG: %d chunks, %d relevant (threshold %.2f, need %d); top %.2f -> %s",
                len(docs), len(relevant), settings.relevance_threshold, settings.min_chunks, top,
                "answer from stored sources" if hit else "live research")
    if not hit:
        return {"rag_hit": False, "rag_top_score": top}
    understanding = state.get("understanding") or {}
    task = ResearchTask(
        sub_question=query,
        source_intent=_SOURCE_INTENT_FOR.get(str(understanding.get("intent") or "")),
        technology=understanding.get("technology") or None,
    )
    return {
        "source_documents": [d.model_copy(update={"task_id": task.task_id}) for d in relevant],
        "sub_questions": [query],
        "research_tasks": [task],
        "rag_hit": True,
        "rag_top_score": top,
    }


def rag_router(state) -> str:
    return EVIDENCE_NODE if state.get("rag_hit") else classify_router(state)


async def index_sources_node(state):
    """Add the run's fresh source documents to ``source_chunks``. Never fails the run."""
    try:
        settings = RagSettings.from_env()
        if settings.ingest_enabled and not state.get("cache_hit"):
            await get_indexer(settings).index(state.get("source_documents") or [])
    except Exception as exc:
        logger.warning("Source indexing skipped (%s: %s)", type(exc).__name__, exc)
    return {}
