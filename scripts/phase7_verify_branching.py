"""Phase 7 verification, pass 2: flag-on branching, run through the real compiled graph of the cwd checkout.
Run from the repo root:  .venv/Scripts/python -W ignore scripts/phase7_verify_branching.py  (prints JSON).
Uses an isolated embedded Qdrant folder (P2_STORE_DIR, default in the temp dir; it is deleted first).
Real: fastembed dense+BM25, embedded Qdrant (isolated folder, collection `source_chunks`), reranker.
Faked (no network): the LLM (ScriptedLLM) and Tavily (FakeTavily)."""
import asyncio
import json
import logging
import os
import shutil
import sys
import urllib.request
from unittest import mock

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
for k in ("GROQ_API_KEY", "TAVILY_API_KEY", "OPENROUTER_API_KEY", "SECRET_KEY", "OPENAI_API_KEY"):
    os.environ.setdefault(k, "dummy-not-a-real-key")
import tempfile  # noqa: E402
STORE_DIR = os.environ.get("P2_STORE_DIR") or os.path.join(tempfile.gettempdir(), "phase7_verify_qdrant")
shutil.rmtree(STORE_DIR, ignore_errors=True)
os.environ.update({
    "QDRANT_URL": STORE_DIR, "SOURCE_CHUNKS_COLLECTION": "source_chunks", "RAG_TIMEOUT_S": "300",
    "SOURCE_RAG_ENABLED": "true", "SOURCE_RAG_INGEST_ENABLED": "false",
})

from qdrant_client import models as qm  # noqa: E402
from research_app.agent import state as st, vectordb  # noqa: E402
from research_app.agent.graph import compiled_graph  # noqa: E402
from research_app.agent.pipeline import rag_nodes  # noqa: E402
from research_app.rag import store as store_mod  # noqa: E402
from research_app.rag.ingest import SourceIndexer  # noqa: E402
from research_app.rag.settings import RagSettings  # noqa: E402
import rag_manual  # noqa: E402
from tests.p6_helpers import ScriptedLLM, graph_inputs, plan, task, understanding  # noqa: E402
from tests.test_router_nodes import FakeTavily  # noqa: E402

# one shared embedded client (the retriever and the indexer each build a store; local mode allows one client)
SHARED = store_mod.make_client(RagSettings.from_env())
store_mod.make_client = lambda settings: SHARED

log_lines = []


class Capture(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if record.name.endswith("rag_nodes") or "Source indexing" in msg:
            log_lines.append(msg)


logging.getLogger().addHandler(Capture())
logging.getLogger().setLevel(logging.INFO)


def counts():
    out = {}
    for c in SHARED.get_collections().collections:
        out[c.name] = SHARED.count(c.name, exact=True).count
    return out


def server_research_cache():
    try:
        with urllib.request.urlopen("http://localhost:6333/collections/research_cache", timeout=3) as r:
            return json.load(r)["result"]["points_count"]
    except Exception as exc:
        return f"unreachable ({type(exc).__name__})"


async def run(name, question, llm, env=None, fake=None):
    fake = fake or FakeTavily()
    log_lines.clear()
    updates, error = [], None
    with mock.patch.dict(os.environ, env or {}), mock.patch.object(st, "tavily_tool", fake), \
            mock.patch.object(st, "llm_groq", llm), mock.patch.object(st, "lookup_cache", lambda q: (False, "")), \
            mock.patch.object(st, "store_cache", lambda q, a: None):
        try:
            async for mode, chunk in compiled_graph.astream(graph_inputs(question), stream_mode=["updates"]):
                node, data = next(iter(chunk.items()))
                updates.append((node, data or {}))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    rag = next((d for n, d in updates if n == "source_rag_node"), {})
    fmt = next((d for n, d in updates if n == "format_response"), {})
    return {
        "scenario": name, "question": question, "nodes": [n for n, _ in updates],
        "rag_hit": rag.get("rag_hit"), "rag_top_score": rag.get("rag_top_score"),
        "tavily_calls": len(fake.calls),
        "structured_llm_calls": [n for n, _ in llm.structured_prompts],
        "free_text_llm_calls": len(llm.prompts),
        "final_answer": (fmt.get("final_answer") or "")[:200],
        "log": list(log_lines), "error": error,
    }


async def top_chunks(query):
    docs = await rag_nodes.get_retriever(RagSettings.from_env()).retrieve(query)
    return [(round(d.metadata.get("rag_score", 0), 3), d.title) for d in docs]


async def main():
    rerank = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    result = {"rerank_model": rerank, "rerank_enabled": os.environ.get("RERANK_ENABLED", "true (default)")}
    # answer_cache stand-in inside the same isolated Qdrant store, to prove nothing else is written
    SHARED.create_collection("answer_cache", vectors_config=qm.VectorParams(size=4, distance=qm.Distance.COSINE))
    SHARED.upsert("answer_cache", [qm.PointStruct(id=i, vector=[1, 0, 0, i / 10], payload={"q": f"cached {i}"})
                                   for i in range(1, 4)])
    result["memory_answer_cache_before"] = vectordb.get_cache_stats()
    result["server_research_cache_before"] = server_research_cache()

    S, T = rag_manual.SourceType, rag_manual._doc
    extra = [  # synthetic paraphrases (like the rest of the seed), so FastAPI has >= RAG_MIN_CHUNKS relevant chunks
        T("https://fastapi.tiangolo.com/virtual-environments/", "FastAPI: installation in a virtual environment",
          "Create and activate a virtual environment, then install FastAPI with pip install \"fastapi[standard]\". "
          "The standard extras include Uvicorn and the fastapi command line tool.", S.OFFICIAL_DOCS, "fastapi"),
        T("https://fastapi.tiangolo.com/fastapi-cli/", "FastAPI CLI: run the development server",
          "Start the development server with fastapi dev main.py. It reloads automatically when the code changes. "
          "Use fastapi run for production instead of the development server.", S.OFFICIAL_DOCS, "fastapi"),
    ]
    seeded = await SourceIndexer(RagSettings.from_env()).index([*rag_manual.CORPUS, *extra])
    result["seeded_chunks"] = seeded
    result["counts_after_seed"] = counts()

    q_rel = "How do I install FastAPI and start the development server?"
    q_rel2 = "What does CREATE INDEX CONCURRENTLY do in PostgreSQL?"
    q_irr = "What are the health benefits of intermittent fasting?"
    result["retrieval_relevant"] = await top_chunks(q_rel)
    result["retrieval_relevant2"] = await top_chunks(q_rel2)
    result["retrieval_irrelevant"] = await top_chunks(q_irr)

    def llm_rel(tech):
        return ScriptedLLM(understanding_=understanding(is_simple=False, intent="technical_howto", technology=tech))

    def llm_irr():
        return ScriptedLLM(understanding_=understanding(is_simple=False, intent="general_research"),
                           plan_=plan(task("Health benefits of intermittent fasting", "general_research", ["web"])))

    runs = []
    runs.append(await run("a_relevant_fastapi", q_rel, llm_rel("FastAPI")))
    q_low = "what do I type to set up the python web framework with automatic docs"
    result["retrieval_low_overlap"] = await top_chunks(q_low)
    runs.append(await run("a_low_word_overlap", q_low, llm_rel("FastAPI")))
    runs.append(await run("a_single_chunk_topic_default_min2", q_rel2, llm_rel("PostgreSQL")))
    runs.append(await run("a_single_chunk_topic_min1", q_rel2, llm_rel("PostgreSQL"), env={"RAG_MIN_CHUNKS": "1"}))
    counts_before_irr = counts()
    runs.append(await run("b_irrelevant_flag_on_ingest_on", q_irr, llm_irr(), env={"SOURCE_RAG_INGEST_ENABLED": "true"}))
    counts_after_irr = counts()
    runs.append(await run("b_irrelevant_flag_off", q_irr, llm_irr(), env={"SOURCE_RAG_ENABLED": "false"}))
    result["runs"] = runs
    result["counts_before_irrelevant_run"] = counts_before_irr
    result["counts_after_irrelevant_run_with_ingest"] = counts_after_irr
    result["counts_final"] = counts()
    result["memory_answer_cache_after"] = vectordb.get_cache_stats()
    result["server_research_cache_after"] = server_research_cache()
    print(json.dumps(result, indent=1, default=str))


asyncio.run(main())
