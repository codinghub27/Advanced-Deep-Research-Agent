# Advanced Deep Research Agent

A multi-source research agent built with **FastAPI** and **LangGraph**. Ask a question and it plans the research, searches the right sources (official documentation, the web, GitHub, Reddit), collects evidence, writes an answer with numbered citations, and reviews that answer with a critic before returning it.

The project is being upgraded from a working prototype into a production-grade system in small, numbered phases. Phases 1–6 are complete; Phase 7 (hybrid RAG) is built but switched off by default. See [Project status](#project-status).

---

## Features

- **Source-aware planning.** The planner splits a question into sub-questions and suggests sources for each one. A deterministic router (no LLM) validates and enforces the choice.
- **Official documentation first.** For install, configuration, API and version questions, the router detects the technology and searches its official docs with domain restrictions. Docs come from a configurable registry, not from the planner prompt.
- **Four source types.** `official_docs`, `web`, `github`, `reddit`. GitHub and Reddit results are checked against strict hostname allowlists, and community content is never treated as equal authority to official docs.
- **Parallel, failure-isolated search.** Searches run concurrently under a process-wide limit and per-source timeouts. One failed or slow source does not fail the run.
- **Evidence pipeline.** Results are normalized into one `SourceDocument` model, pooled per sub-question, de-duplicated and scored for coverage. One bounded gap-filling search round runs when a sub-question has no evidence.
- **Grounded synthesis.** The answer is written from collected evidence only, with `[N]` citations that map to a source list. Code, not the model, builds the citation list and confidence.
- **Critic with one retry.** Deterministic checks (orphan citations, uncited paragraphs, Reddit-only technical answers, unused official docs) plus one LLM review. A bad answer gets one targeted retry; there are no unbounded loops.
- **Conversations.** Sessions and turns are stored in the database. Follow-up questions are resolved against earlier turns ("What about OAuth2 instead?").
- **Date-aware research.** Questions about the latest or current state get a recency window on web search.
- **Streaming chat UI.** A single-page UI at `/research/ui` streams progress and tokens over SSE, shows a live router panel, clickable citation markers, a collapsible Sources panel and a Search-plan drawer.
- **Hybrid source RAG (Phase 7, off by default).** Qdrant dense + BM25 sparse retrieval, reciprocal-rank fusion and cross-encoder reranking over stored source documents.

---

## How it works

```
START → semantic_cache ──hit──────────────────────────────────────────→ END
          └─miss→ classify (query understanding: follow-up, intent, technology)
                    ├─ simple  → simple_search ───────────────────┐
                    └─ complex → planner ─Send→ search × N ───────┤   per-sub-question routing
                                                                   ▼
        evidence_collection → gap_detection ─gap→ targeted_search ─┐   (one round)
                                   └────────── sufficient ─────────┤
                                                                   ▼
        synthesis → critic ─bad→ retry ─┐                               (at most one)
                       └──── good ──────┤
                                        ▼
             format_response → index_sources → save_to_cache → END
```

Every loop is bounded by the graph topology: there is no edge back to an earlier node.

With `SOURCE_RAG_ENABLED=true`, a cache miss first checks stored source chunks in Qdrant. If enough relevant chunks are found, they go straight to the evidence pipeline; otherwise the run falls back to live research. Time-sensitive questions always skip stored sources.

### Source routing examples

| Question | Sources |
|---|---|
| How do I install FastAPI? | official_docs + web |
| Find GitHub examples of LangGraph agents | github + web |
| Real-world experiences using Qdrant | reddit + web |
| Why am I getting this LangGraph error? | official_docs + github + reddit + web |
| Best laptops for college students | web |

The official-docs registry ships with 16 technologies (Python, FastAPI, Uvicorn, Pydantic, SQLAlchemy, Alembic, LangGraph, LangChain, Qdrant, PostgreSQL, Docker, Docker Compose, Tavily, OpenAI API, Claude API, Groq). Add more with a JSON file pointed to by `OFFICIAL_DOCS_REGISTRY_PATH`, using the format in `research_app/sources/official_docs/data/official_docs.json`.

---

## Tech stack

| Area | Tools |
|---|---|
| API | FastAPI, Uvicorn, SSE streaming |
| Orchestration | LangGraph, LangChain |
| LLM | OpenRouter or Groq via LangChain (`research_app/agent/llms.py`) |
| Search | Tavily (web, plus domain-restricted searches for docs, GitHub, Reddit) |
| Database | PostgreSQL via SQLAlchemy + Alembic migrations (SQLite works for tests) |
| Vector store | Qdrant (`source_chunks` collection, Phase 7) |
| Embeddings / rerank | fastembed (`BAAI/bge-small-en-v1.5`, `Qdrant/bm25`), sentence-transformers (`BAAI/bge-reranker-v2-m3`) |
| Auth | bcrypt + JWT (python-jose) |

---

## Repository layout

```
research_app/
  main.py                FastAPI app, routes, SSE streaming
  research_service.py    research request orchestration
  agent/                 LangGraph graph, state, nodes, LLM client, answer cache
    pipeline/            evidence pool, gap detection, synthesis, critic, response
  sources/               Tavily client, result normalizer
    official_docs/       technology detector, docs registry, docs adapter
    routing/             source router, per-task router, GitHub/Reddit adapters, executor
  rag/                   chunking, embeddings, Qdrant store, hybrid retriever, reranker
  domain/                Pydantic domain models and interfaces
  conversation/          conversation context for follow-ups
  db/                    SQLAlchemy models, CRUD, Alembic migrations
  auth/                  JWT auth
  schemas/               request schemas and SSE node labels
  templates/chat.html    chat UI
  docker-compose.yml     local Qdrant
scripts/                 manual check tools (RAG seeding/querying, graph traces)
tests/                   offline unittest suite (no network, no real DB)
docs/phases/             one design + results document per phase
CLAUDE.md                architecture and engineering rules for the upgrade
```

---

## Getting started

### Prerequisites

- Python 3.11+ (developed on 3.14)
- PostgreSQL
- A Tavily API key and an LLM key (OpenRouter or Groq)
- Docker (only if you want Qdrant for Phase 7 hybrid RAG)

### 1. Install

```bash
git clone https://github.com/codinghub27/Advanced-Deep-Research-Agent.git
cd Advanced-Deep-Research-Agent
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Fill in at least:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | e.g. `postgresql+psycopg://USER:PASSWORD@localhost:5432/DBNAME` |
| `SECRET_KEY` | JWT signing key: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `TAVILY_API_KEY` | search |
| `OPENROUTER_API_KEY` | LLM. The client in `research_app/agent/llms.py` currently uses OpenRouter; a Groq client (`GROQ_API_KEY`) is there commented out. |
| `ADMIN_USERNAMES` | optional, comma-separated users allowed to call `/admin/*` |

Every other setting is optional and has a safe default. `.env.example` documents them all. Never commit `.env`.

### 3. Run

From the **repository root**:

```bash
uvicorn research_app.main:app --reload --port 8000
```

Database migrations run automatically on startup (`alembic upgrade head`).

- Chat UI: http://localhost:8000/research/ui
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

Register a user with `POST /auth/register`, log in with `POST /auth/login`, and use the returned token as `Authorization: Bearer <token>`.

### 4. (Optional) Qdrant for hybrid RAG

```bash
docker compose -f research_app/docker-compose.yml up -d
```

Then set `SOURCE_RAG_INGEST_ENABLED=true` (index each run's sources) and `SOURCE_RAG_ENABLED=true` (answer from stored sources when relevant). Phase 7 is not production-ready yet, so keep both off unless you are testing it. To try the retriever without the app:

```bash
python scripts/rag_manual.py seed
python scripts/rag_manual.py query "how do I install fastapi" --mode hybrid --rerank
```

Embedding and reranker models download on first use (the reranker is about 2.3 GB).

---

## API

| Endpoint | Auth | Description |
|---|---|---|
| `POST /auth/register`, `POST /auth/login` | – | create a user, get a JWT |
| `POST /api/research` | JWT | run research, return the full answer as JSON |
| `POST /api/research/stream` | JWT | run research as Server-Sent Events |
| `GET /api/research/history` | JWT | past queries |
| `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}` | JWT | conversations with turns and citations |
| `/api/sessions/*` | JWT | legacy session endpoints (kept for compatibility) |
| `GET /admin/db-stats`, `DELETE /admin/clear-all-data` | admin | requires `ADMIN_USERNAMES` |
| `GET /health`, `GET /research/ui` | – | health check, chat UI |

Request:

```json
{ "question": "How do I install and configure FastAPI with uvicorn?", "session_id": 12 }
```

Leave out `session_id` to start a new conversation; the new id is returned. The response keeps the original keys (`sub_questions`, `answer`, `session_id`) and adds `citations`, `sources_consulted`, `confidence`, `critic_verdict`, `turn_number`, `is_follow_up`, `resolved_query` and `research_metadata` (sources found, gap search, retry, failed sources, routing decisions, latency).

The stream emits `progress`, `routing`, `citations`, `token`, `sources`, `cache_hit` and `done` events.

---

## Configuration

Main switches (see `.env.example` and the phase docs for the full list):

| Variable | Default | Effect |
|---|---|---|
| `OFFICIAL_DOCS_ENABLED` | true | official documentation source |
| `SOURCE_ROUTER_ENABLED` | true | GitHub and Reddit sources |
| `MAX_SOURCES_PER_TASK` | 3 | sources per sub-question (docs and web are never trimmed) |
| `CONCURRENCY_LIMIT` | 6 | searches in flight across the process |
| `GAP_SEARCH_ENABLED` | true | one gap-filling search round |
| `MAX_CRITIC_RETRIES` | 1 | 0 disables the critic retry |
| `RECENCY_FILTER_ENABLED` | true | date window for "latest / current" questions |
| `CONVERSATION_HISTORY_LIMIT` | 10 | earlier turns used for follow-ups |
| `SOURCE_RAG_ENABLED` / `SOURCE_RAG_INGEST_ENABLED` | false | Phase 7 hybrid RAG |
| `RAG_MODE` | hybrid | `hybrid`, `dense` or `sparse` |
| `RERANK_ENABLED` | true | cross-encoder reranking |
| `LOG_LEVEL` / `LOG_NODE_TRACE` | INFO / false | logging detail |

The Phase 7 RAG settings (`QDRANT_URL`, `RAG_*`, `RERANK_*`) are listed in [PHASE-07-HYBRID-RAG.md](docs/phases/PHASE-07-HYBRID-RAG.md).

---

## Tests

The suite uses the standard library `unittest`. It forces dummy keys, uses in-memory SQLite and fakes the LLM and search calls, so it needs no network or credentials.

```bash
python -m unittest discover -s tests -t .
```

Live manual test cases for each phase are recorded in its phase document.

---

## Project status

| Phase | Topic | Status |
|---|---|---|
| 01 | [Repository audit and baseline](docs/phases/PHASE-01-AUDIT.md) | ✅ Done |
| 02 | [Domain contracts](docs/phases/PHASE-02-CONTRACTS.md) | ✅ Done |
| 03 | [Source normalization](docs/phases/PHASE-03-SOURCE-NORMALIZATION.md) | ✅ Done |
| 04 | [Official documentation source](docs/phases/PHASE-04-OFFICIAL-DOCS.md) | ✅ Done |
| 05 | [GitHub, Reddit and source router](docs/phases/PHASE-05-SOURCE-ROUTER.md) | ✅ Done |
| 06 | [Planner, evidence pipeline, critic, conversations](docs/phases/PHASE-06-COMPLETE-RESEARCH-SYSTEM.md) | ✅ Done |
| 07 | [Hybrid source RAG](docs/phases/PHASE-07-HYBRID-RAG.md) | 🚧 In progress, flag off |
| 08 | [Evidence store](docs/phases/PHASE-08-EVIDENCE.md) | Planned |
| 09 | [Gap detection and iterative research](docs/phases/PHASE-09-GAP-DETECTION.md) | Planned |
| 10 | [Synthesis and grounding](docs/phases/PHASE-10-SYNTHESIS.md) | Planned |
| 11 | [Critic and targeted retry](docs/phases/PHASE-11-CRITIC-RETRY.md) | Planned |
| 12 | [Semantic cache and freshness](docs/phases/PHASE-12-CACHE.md) | Planned |
| 13 | [Reliability](docs/phases/PHASE-13-RELIABILITY.md) | Planned |
| 14 | [Security and API hardening](docs/phases/PHASE-14-SECURITY.md) | Planned |
| 15 | [Performance and cost](docs/phases/PHASE-15-PERFORMANCE.md) | Planned |
| 16 | [API, streaming, long-running research](docs/phases/PHASE-16-API-LONG-RUNNING.md) | Planned |
| 17 | [Docker and production config](docs/phases/PHASE-17-DOCKER.md) | Planned |
| 18 | [Observability](docs/phases/PHASE-18-OBSERVABILITY.md) | Planned |
| 19 | [Final manual validation](docs/phases/PHASE-19-FINAL-VALIDATION.md) | Planned |

### Known limitations

- The answer cache is still an in-process dictionary with word-overlap matching. It is lost on restart and has no freshness rules (Phase 12).
- Evidence lives in memory per request; there is no durable evidence store yet (Phase 8).
- GitHub and Reddit are searched through Tavily with domain restrictions, not through their own APIs.
- Phase 7 has open review items (relevance gate without the reranker, inline indexing latency) and should stay disabled.
- `research_app/start_all.ps1` has hard-coded local paths; use the commands above instead.

LangSmith and automated evaluation are deliberately out of scope for now.

---

## Contributing / development workflow

Work happens one phase at a time, following [CLAUDE.md](CLAUDE.md):

1. Read `CLAUDE.md` and the current phase file in `docs/phases/`.
2. Inspect only the relevant code and write a plan before changing anything.
3. Implement only that phase, keeping existing API contracts and behavior.
4. Run focused tests, then the full suite.
5. Update the phase document with changes, test results, decisions and known issues.
6. Commit as `phase-N: <short description>`.

Prefer small, reviewable changes (a new module, adapter or migration) over rewrites, and never commit secrets.
