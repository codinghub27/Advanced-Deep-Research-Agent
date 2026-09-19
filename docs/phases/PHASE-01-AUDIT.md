# PHASE-01 — Repository Audit and Baseline

## Status
Completed

(Audit delivered and smoke-tested at import level. The live manual baseline cases below are defined but were **not executed**; they need a database, LLM and Tavily credentials. See Test Results.)

## Goal
Map the existing system, graph, APIs, storage, tools, tests, and technical debt without changing behavior.

## Scope
Inspected every tracked source file under `research_app/` (~3,000 lines): FastAPI app, LangGraph graph/state/nodes, Tavily tool, cache, DB models/CRUD, auth, schemas, `chat.html`, `docker-compose.yml`, scripts, `requirements.txt`, and git history. Environment variables were read by name only; no secret values were recorded.

## Out of Scope
- Future phases
- LangSmith/evaluation/datasets
- Any code change (none was made in this phase)

## Current Implementation

### Repository layout
```
research_app/
  main.py                 FastAPI app, all routes, SSE streaming
  agent/graph.py          StateGraph wiring
  agent/state.py          ResearchState, all nodes, Tavily tool, sanitize_text
  agent/llms.py           ChatGroq client (openai/gpt-oss-120b)
  agent/vectordb.py       "semantic cache" (in-process dict)
  agent/errors.py         Groq rate-limit helpers (imported nowhere)
  agent/stream.py         empty file
  db/{database,models,crud}.py   SQLAlchemy engine, 3 tables, CRUD
  auth/auth.py            JWT create/verify
  schemas/schemas.py      RequestResearch, RequestRegister, NODE_LABELS
  templates/chat.html     single-page UI (marked + Prism, SSE consumer)
  stream.py               UNRELATED Streamlit client (name clashes with agent/stream.py)
  register_user.py, start_all.ps1, docker-compose.yml
```

### Current LangGraph flow
```
START → semantic_cache_node ──hit───────────────────────────────→ END
              └─miss→ classify_node ──simple──→ simple_search_node ─┐
                            └─complex→ planner_node ─Send×3→ search_node ─┤
                                                              synthesize_node
                                                              → save_to_cache_node → END
```
- Compiled at import as `compiled_graph` (`graph.py:54`); 7 nodes. Invoked via `ainvoke` (`/api/research`) and `astream(stream_mode=["updates"])` (`/api/research/stream`).
- `ResearchState` (`state.py:45`) is a `TypedDict`. `sources`, `search_results`, `messages` use `operator.add` reducers.
- Declared but never written by any node: `step_count`, `critic_score`, `critic_feedback`, `api_limit_reached`. `history` is loaded on every request and never read.
- `classify_router` lists `END` in the path map but never returns it.

### CLAUDE.md "existing stack" vs. reality
| Item | Reality |
|---|---|
| Qdrant | **Not used.** `qdrant-client`/`langchain-qdrant` installed; compose file runs a container; no Python code calls Qdrant. |
| Semantic cache | In-process dict, word-set Jaccard ≥ 0.85 (`vectordb.py`). No embeddings, no Qdrant, no TTL. |
| RAG | Does not exist. |
| Critic | Does not exist (`Evaluate` model and state fields are unused). |
| PostgreSQL | Active `DATABASE_URL` is SQLite (`sqlite:///./test.db`); a `postgresql+psycopg` URL is commented out in `.env`. |
| Planner | Exists: LLM structured output → 3 web queries; falls back to the raw question. No intent/source awareness. |
| Parallel research | Exists: fixed `Send` fan-out of 3 `search_node`s. |
| Tavily | Exists: `TavilySearch(max_results=3, search_depth="advanced", include_answer=True)`; the answer is discarded. |
| Synthesis | Exists: one Groq call, "minimum 1000 words", `[Source, Year]` citations that cannot be traced to sources; fallback call on failure. |
| Auth/sessions | Exist: bcrypt + JWT (HS256, 30 min); `users`/`sessions`/`queries`. |
| URL/document extraction | Does not exist. |

### API endpoints
| Endpoint | Auth | Notes |
|---|---|---|
| `GET /`, `/health` | none | |
| `GET /test-db` | none | `SELECT version()`; returns raw exception text on error |
| `GET /research/ui` | none | serves `chat.html` |
| `POST /auth/register`, `/auth/login` | none | login returns 404 (unknown user) vs 401 (bad password) |
| `POST /api/research` | JWT | returns `{sub_questions, answer, session_id}` |
| `POST /api/research/stream` | JWT | SSE events: `progress`, `cache_hit`, `token`, `sources`, `done` (no `error` event) |
| `GET /api/research/history` | JWT | |
| `POST`/`GET /api/sessions` | JWT | |
| `GET /api/sessions/{id}/queries` | JWT | **no ownership check** |
| `PATCH /api/sessions/{id}` | JWT | **no ownership check** |
| `DELETE /api/sessions/{id}` | JWT | ownership checked |
| `DELETE /admin/clear-all-data`, `GET /admin/db-stats` | JWT (any user) | **no admin role**; clear-all deletes every user's data |

These 16 application routes are the contract to preserve. Also present: FastAPI's `/openapi.json`, `/docs`, `/docs/oauth2-redirect`, `/redoc`.

### Storage
- **PostgreSQL/SQLAlchemy (currently SQLite):** tables `users`, `sessions`, `queries` via `create_all` at startup; alembic is listed but there are no migrations. `queries.sources` (JSON) is never populated (always `{}`).
- **Qdrant:** no collections exist in code.
- **Cache:** `_exact_cache` module dict in `vectordb.py`: lost on restart, unbounded, shared across users, stores answer only (cache hits return no sources), no freshness. Word-order-insensitive matching means "Is Python faster than Rust?" and "Is Rust faster than Python?" match at 1.0 (from code reading; not executed).

### Configuration (variable names only)
Read by code: `DATABASE_URL`, `SECRET_KEY`, `ALGORITHM` (default HS256), `TAVILY_API_KEY`, `GROQ_API_KEY`. Present in `.env` but unused by Python: `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `AGENTROUTER_API_KEY`, `QDRANT_API_KEY`, `QDRANT_URL`, `QDRANT_TIMEOUT`, `GROQ_API_KEY1`. No dev/test/prod separation. `.env` is loaded by `load_dotenv()` in `vectordb.py`, `llms.py`, `database.py`, `auth.py`; `state.py` reads the Tavily key at import and only works because `vectordb` is imported first.

### Tests
None. No pytest, no test files, no CI.

## Planned Changes
Phase 1 made no code changes. Only this document was edited.

Recommended before Phase 2, as separate small commits (**not started**; awaiting the go-ahead):
1. Admin-role check on `/admin/*`; ownership checks on `GET`/`PATCH /api/sessions/{id}` and on `session_id` in `/api/research*`.
2. Fix `default=datetime.utcnow()` → `default=datetime.utcnow` (`models.py:26,49,84`).
3. Add minimal `logging.basicConfig` (app `logger.info` output is currently invisible).
4. Clean `requirements.txt`; add `.env.example` (names + placeholders only).
5. `git rm --cached` the 23 tracked `.pyc`/`.idea` files.

## Files Expected
- Create: none (this phase)
- Modify: `docs/phases/PHASE-01-AUDIT.md` only
- Delete: none

Likely to change in later phases:
| Phase | Files |
|---|---|
| 2 | new `research_app/domain/` (models, interfaces); adapters around `state.py`, `schemas.py` |
| 3 | new `sources/` adapters + normalizer; `state.py` (`search_node`, `simple_search_node`) |
| 4 | new official-docs registry (config), search adapter, extractor; `state.py`, `graph.py` |
| 5 | new GitHub/Reddit adapters; `requirements.txt`, `.env` |
| 6 | `state.py` (planner, `planner_router`), `graph.py`, `NODE_LABELS` |
| 7 | `vectordb.py` (separate cache from source chunks), new retrieval module, `requirements.txt` (BM25, embeddings) |
| 8 | `db/models.py`, `db/crud.py`, new migrations, `state.py` |
| 9–11 | `graph.py`, `state.py`, `main.py` (`event_generator`) |
| 12 | `vectordb.py`, `state.py` (cache nodes) |
| 13, 15 | `llms.py`, tool wrappers, `main.py` |
| 14 | `main.py`, `auth/auth.py`, `db/crud.py`, `chat.html`, `schemas.py` |
| 16 | `main.py`, `schemas.py`, `chat.html` |
| 17 | new Dockerfile, `docker-compose.yml`, env handling |
| 18 | `main.py`, `state.py`, `database.py`, logging config |

## Architecture Decisions
- **This repo is the authoritative codebase** (confirmed by the user). Phases 7 (hybrid RAG), 8 (evidence store), 11 (critic) and 12 (semantic cache) are greenfield builds, not upgrades.
- **PostgreSQL migration is deferred** (user decision). Development stays on SQLite for now; revisit no later than Phase 8.
- **Secrets:** the user has removed `.env` from git history and is rotating keys. `.env` stays untracked and gitignored; an `.env.example` is the intended pattern.
- **Pre-Phase-2 hardening:** the user delegated the sequencing. Recommendation is to do the items under Planned Changes before Phase 2, as separate commits.
- The SSE event protocol (`progress`/`cache_hit`/`token`/`sources`/`done`) and the 16 routes above are the stable contract that `chat.html` depends on.

## Dependencies
Installed in `.venv` (Python 3.14.0): fastapi 0.141.1, starlette 1.6.0, langgraph 1.2.11, langchain 1.4.2, langchain-core 1.6.3, langchain-tavily 0.2.18, langchain-groq 1.1.3, qdrant-client 1.19.1, langchain-qdrant 1.1.0, SQLAlchemy 2.0.54, psycopg 3.3.6, pydantic 2.12.5, python-jose 3.5.0, bcrypt 5.0.0, numpy 2.5.3.

Not installed: pytest, rank-bm25, streamlit, any embedding library.

`requirements.txt` problems: unpinned; typo `python-jose[cryptogaphy]`; `dotenv` should be `python-dotenv`; duplicate `langchain-openai`; stdlib `typing` listed; `bcrypt` is imported but not listed (comes via `passlib[bcrypt]`); `streamlit`/`requests` used by `research_app/stream.py` but not listed; unused: `passlib`, `wikipedia`, `langchain-ollama`, `langchain-google-genai`, `langchain-openrouter`, `alembic`.

## Implementation Notes
- Run the app from the **repo root**: `uvicorn research_app.main:app`. `start_all.ps1` is stale (hard-coded paths to another folder; `uvicorn main:app` from inside `research_app/` fails on `research_app.*` imports).
- `research_app/stream.py` is a Streamlit client, not backend streaming. `research_app/agent/stream.py` is empty.
- Run Python with `PYTHONDONTWRITEBYTECODE=1` until the tracked `.pyc` files are untracked, or every run dirties `git status`.
- `sanitize_text` (`state.py:16`) collapses runs of spaces (destroys code indentation) and strips characters such as `’ “ ” — € £ ₹ ° ^`. It must be fixed before technical/docs answers (Phases 4, 10).
- The "streaming" answer is generated in full by `synthesize_node` and then split into 3-word chunks (`main.py:315`); there is no token streaming.
- On synthesis failure the fallback text "Unable to generate comprehensive report. Please try again." (58 chars) passes the 50-char cache floor, so it is cached and saved to the DB as a real answer (from code reading).
- Simple-question path also goes through the "minimum 1000 words" synthesis prompt.
- Live manual tests write to `./test.db` relative to the working directory (gitignored).

## Manual Test Cases
Run against a running server (`uvicorn research_app.main:app` from repo root) with valid keys. All **Actual** fields are "Not run" for this phase.

### Test 1 — Health and import
Input: `GET /health`
Expected: `{"status": "ok"}`
Actual: Not run live. Import smoke test passed (see Test Results).

### Test 2 — Register and login
Input: `POST /auth/register`, then `POST /auth/login` with the same credentials
Expected: register → `{"success": ...}`; login → 200 with `access_token`. Duplicate register → 400. Wrong password → 401. Unknown user → 404.
Actual: Not run

### Test 3 — Unauthenticated / bad token
Input: `POST /api/research` with no token, then with a garbage token
Expected: 401 both times
Actual: Not run

### Test 4 — Simple question (streaming)
Input: `POST /api/research/stream` `{"question": "What is the capital of France?"}`
Expected: `progress` events for `semantic_cache_node`, `classify_node`, `simple_search_node`, `synthesize_node`, `save_to_cache_node`; `token` events carrying the report; then `sources`, then `done`.
Actual: Not run

### Test 5 — Complex question (streaming)
Input: `{"question": "Compare the best laptops for programming in 2026"}`
Expected: `planner_node`, then three `search_node` progress events, `synthesize_node`; up to 9 sources, de-duplicated by URL; a long report.
Actual: Not run

### Test 6 — Cache hit
Input: repeat Test 5's exact question
Expected: `cache_hit` event, tokens with `cache_hit: true`, **no** `sources` event; non-streaming variant returns `sub_questions: ["(from cache)"]`.
Actual: Not run

### Test 7 — Session CRUD
Input: create, list, rename, fetch queries, delete a session; then `POST /api/research` with that `session_id`
Expected: each call succeeds; deleted session → 404 on research.
Actual: Not run

### Test 8 — Baseline defect confirmation (current, defective behavior)
Input: (a) ask "Is Python faster than Rust?" then "Is Rust faster than Python?"; (b) as user B, `GET /api/sessions/{user A's id}/queries`; (c) as any user, `DELETE /admin/clear-all-data` on a throwaway DB.
Expected (today): (a) second question is a cache hit; (b) 200 with user A's data; (c) all users' data deleted. These are defects to be fixed, recorded so later phases can show the change.
Actual: Not run

## Commands Run
```text
git ls-files; wc -l on tracked sources; cat requirements.txt README.md .gitignore
sed on .env to print variable NAMES only (values redacted)
Read/Grep on research_app/**/*.py, templates/chat.html
.venv python -m pip list (filtered)
.venv python: import research_app.agent.graph / research_app.main; list graph nodes/edges and routes
git log --all -- .env; git for-each-ref refs/original; git status; git diff
Test-NetConnection localhost:5432 (a listener was present; no connection attempted)
```
All Python runs used `PYTHONDONTWRITEBYTECODE=1`.

## Test Results
- Import smoke test (run twice, before and after the git history rewrite): PASS. Graph compiles with 7 nodes; `research_app.main` imports on Python 3.14.0; 16 application routes registered.
- No live endpoint, LLM, Tavily or database calls were made (would consume credits and write to the DB).
- The cache, DB-timestamp, fallback-caching and ownership findings are from code reading, not execution.
- After the history rewrite: `git log --all -- .env` returns nothing and `refs/original` is absent.

## Security / Reliability Notes
Ordered by severity:
1. **Secrets.** `.env` was committed in early history. The user reports GitHub blocked the push, then rewrote history and is rotating keys. Rotate every key that was in the file, including `SECRET_KEY`, `TAVILY_API_KEY` and `QDRANT_API_KEY`, not just the ones the scanner flagged.
2. **Any authenticated user can delete all data** (`/admin/clear-all-data`); `/admin/db-stats` is likewise ungated.
3. **Missing ownership checks:** `GET`/`PATCH /api/sessions/{id}`, and `session_id` accepted in `/api/research*` without verifying it belongs to the caller (`main.py:184`, `:419`, `:439`).
4. **Stored XSS / token theft:** `marked.parse()` output goes to `innerHTML` unsanitized (`chat.html:805,821`), the JWT is in `localStorage` (`:362`), and `marked` loads from a CDN unpinned without SRI. Source hrefs are HTML-escaped but not scheme-checked. The code-block renderer interpolates `lang` unescaped.
5. **Prompt injection:** raw search content is interpolated into the synthesis prompt.
6. **Auth:** no rate limiting or password policy; login enumerates users (404 vs 401); a JWT without `sub` raises an uncaught `KeyError` (500); `SECRET_KEY` is not validated at startup; CORS is `*` with credentials.
7. **`/test-db`** is unauthenticated and leaks exception text.
8. **No input limits** on `question`; no request timeout or cancellation.
9. **No failure isolation:** `search_node`/`simple_search_node` have no error handling, so one Tavily failure kills the run (500, or a dead SSE stream with no `error`/`done` event). Bare `except:` in `classify_node` and the planner. `errors.py` rate-limit handling is dead code.
10. **SSRF:** not applicable today (the app fetches no URLs); becomes relevant with Phase 4 extraction.
11. The streaming generator reuses the request-scoped DB session; this depends on FastAPI's yield-dependency lifetime, which has changed across versions. Verify at manual-test time and pin FastAPI.

## Known Issues
- **Correctness:** `datetime.utcnow()` evaluated at import gives every row the server-start timestamp (ordering ambiguous); `username` not unique; `hashed_password` annotated `Mapped[int]`; `queries.sources` never saved; failure text cached and persisted; word-order-insensitive cache matching.
- **Quality:** citations untraceable; "1000 words" forced even for simple questions from 400-char snippets; per-result evidence capped at 400/800 chars; `include_answer=True` wasted; `sanitize_text` corrupts code and punctuation; streaming is simulated.
- **Performance:** sync SQLAlchemy inside `async def` routes blocks the event loop; the 20-query history is loaded every request and unused; `ChatGroq(timeout=None)`; up to 3–4 serial LLM calls per request; O(n) cache scan; no global concurrency limit across requests.
- **Ops/hygiene:** logging never configured; no tests; unpinned/incorrect `requirements.txt`; 23 `.pyc`/`.idea` files still tracked; venv is Python 3.14 while tracked bytecode is 3.11; `start_all.ps1` and `register_user.py` use stale paths/imports; `docker-compose.yml` runs only Qdrant with an unpinned `latest` tag and a `curl` healthcheck that the image may not support (unverified).
- **Not verified:** live end-to-end behavior of any endpoint.

## Git Commit
`phase-1: repository audit and baseline` (hash omitted on purpose; history was rewritten, see `git log`).

## Next Phase
Optional pre-Phase-2 hardening (see Planned Changes), then **Phase 02 — Architecture and Domain Contracts** (`docs/phases/PHASE-02-CONTRACTS.md`): introduce the Pydantic domain models and service interfaces without changing behavior. Start it in a fresh session.
