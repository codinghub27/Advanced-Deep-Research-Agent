from langgraph.graph import StateGraph, START, END
from research_app.agent.state import (
    ResearchState,
    semantic_cache_node,
    classify_node,
    classify_router,
    cache_router,
    simple_search_node,
    synthesize_node,
    planner_node,
    planner_router,
    search_node,
    save_to_cache_node,
)

graph = StateGraph(ResearchState)

graph.add_node("semantic_cache_node", semantic_cache_node)
graph.add_node("classify_node", classify_node)
graph.add_node("simple_search_node", simple_search_node)
graph.add_node("planner_node", planner_node)
graph.add_node("search_node", search_node)
graph.add_node("synthesize_node", synthesize_node)
graph.add_node("save_to_cache_node", save_to_cache_node)

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

graph.add_conditional_edges(
    "classify_node",
    classify_router,
    {
        "simple_search_node": "simple_search_node",
        "planner_node": "planner_node",
        END: END,
    }
)

graph.add_edge("simple_search_node", "synthesize_node")
graph.add_conditional_edges("planner_node", planner_router)
graph.add_edge("search_node", "synthesize_node")
graph.add_edge("synthesize_node", "save_to_cache_node")
graph.add_edge("save_to_cache_node", END)

compiled_graph = graph.compile()