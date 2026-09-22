from pydantic import BaseModel
from typing import Optional


class RequestResearch(BaseModel):
    question:str
    session_id:Optional[int] = None  # Auto-use latest session if not provided

class RequestRegister(BaseModel):
    username:str
    password:str

NODE_LABELS = {
    "semantic_cache_node": "🔍 Checking cache...",
    "classify_node": "🧠 Classifying question...",
    "planner_node": "📋 Planning research...",
    "simple_search_node": "🌐 Searching web...",
    "search_node": "🌐 Searching web...",
    "synthesize_node": "✍️ Writing answer...",
    "evidence_collection": "🗂️ Collecting evidence...",
    "gap_detection": "🔎 Checking coverage...",
    "targeted_search": "🌐 Searching for missing information...",
    "synthesis_node": "✍️ Writing answer...",
    "critic_node": "🧐 Reviewing the answer...",
    "retry_node": "🔁 Improving the answer...",
    "format_response": "📎 Preparing citations...",
    "save_to_cache_node": "💾 Saving to cache...",
    # Phase 7: run unconditionally (index_sources_node is a no-op unless SOURCE_RAG_INGEST_ENABLED;
    # source_rag_node only runs with SOURCE_RAG_ENABLED), so both need a label even with the flags off.
    "source_rag_node": "📚 Checking stored sources...",
    "index_sources_node": "💾 Indexing sources...",
}
