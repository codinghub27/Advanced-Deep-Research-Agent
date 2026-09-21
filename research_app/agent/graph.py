from langgraph.graph import StateGraph, START, END
from research_app.agent.state import (
    ResearchState,
    semantic_cache_node,
    classify_node,
    classify_router,
    cache_router,
    simple_search_node,
    planner_node,
    planner_router,
    search_node,
    save_to_cache_node,
)
from research_app.agent.pipeline.critic import critic_node
from research_app.agent.pipeline.nodes import (
    critic_router,
    evidence_collection_node,
    format_response_node,
    gap_detection_node,
    gap_router,
    retry_node,
    targeted_search_node,
)
from research_app.agent.pipeline.rag_nodes import (
    EVIDENCE_NODE,
    SOURCE_RAG_NODE,
    index_sources_node,
    rag_gate_router,
    rag_router,
    source_rag_node,
)
from research_app.agent.pipeline.synthesis import synthesis_node

graph = StateGraph(ResearchState)

graph.add_node("semantic_cache_node", semantic_cache_node)
graph.add_node("classify_node", classify_node)  # query understanding (Phase 6: context-aware)
graph.add_node("simple_search_node", simple_search_node)
graph.add_node("planner_node", planner_node)
graph.add_node("search_node", search_node)
graph.add_node("evidence_collection", evidence_collection_node)
graph.add_node("gap_detection", gap_detection_node)
graph.add_node("targeted_search", targeted_search_node)
graph.add_node("synthesis_node", synthesis_node)
graph.add_node("critic_node", critic_node)
graph.add_node("retry_node", retry_node)
graph.add_node("format_response", format_response_node)
graph.add_node("save_to_cache_node", save_to_cache_node)
graph.add_node(SOURCE_RAG_NODE, source_rag_node)  # Phase 7: only reachable with SOURCE_RAG_ENABLED
graph.add_node("index_sources_node", index_sources_node)  # Phase 7: a no-op unless SOURCE_RAG_INGEST_ENABLED

graph.add_edge(START, "semantic_cache_node")

# FIX: provide the mapping for cache_router
graph.add_conditional_edges(
    "semantic_cache_node",
    cache_router,
    {
        "classify_node": "classify_node",
        END: END,
    }
)

# Phase 7: rag_gate_router is classify_router unless SOURCE_RAG_ENABLED sends the question to the
# stored-source lookup first; rag_router then continues to evidence or to the same two routes.
graph.add_conditional_edges(
    "classify_node",
    rag_gate_router,
    {
        SOURCE_RAG_NODE: SOURCE_RAG_NODE,
        "simple_search_node": "simple_search_node",
        "planner_node": "planner_node",
        END: END,
    }
)
graph.add_conditional_edges(
    SOURCE_RAG_NODE,
    rag_router,
    {
        EVIDENCE_NODE: EVIDENCE_NODE,
        "simple_search_node": "simple_search_node",
        "planner_node": "planner_node",
    }
)

# Research: one search for a simple question, one per sub-question otherwise. Both paths join here.
graph.add_edge("simple_search_node", "evidence_collection")
graph.add_conditional_edges("planner_node", planner_router)
graph.add_edge("search_node", "evidence_collection")

# Evidence -> one optional gap-filling round -> synthesis.
graph.add_edge("evidence_collection", "gap_detection")
graph.add_conditional_edges(
    "gap_detection",
    gap_router,
    {"targeted_search": "targeted_search", "synthesis_node": "synthesis_node"},
)
graph.add_edge("targeted_search", "synthesis_node")

# Synthesis -> critic -> one optional retry -> final answer. No edge leads back to an earlier node.
graph.add_edge("synthesis_node", "critic_node")
graph.add_conditional_edges(
    "critic_node",
    critic_router,
    {"retry_node": "retry_node", "format_response": "format_response"},
)
graph.add_edge("retry_node", "format_response")

graph.add_edge("format_response", "index_sources_node")
graph.add_edge("index_sources_node", "save_to_cache_node")
graph.add_edge("save_to_cache_node", END)

compiled_graph = graph.compile()
