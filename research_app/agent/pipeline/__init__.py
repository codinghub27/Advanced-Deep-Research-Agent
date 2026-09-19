"""Phase 6 research pipeline nodes: evidence, gaps, synthesis, critic, retry, response.

The nodes are plain functions over ``ResearchState`` (see ``agent/graph.py``). They read
``state.llm_groq`` and ``state.tavily_tool`` at call time, so the existing test seam (patching
those two module attributes) keeps working.
"""
