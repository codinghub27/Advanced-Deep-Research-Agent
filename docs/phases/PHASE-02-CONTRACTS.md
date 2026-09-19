# PHASE-02 — Architecture and Domain Contracts

## Status
Completed

(Contracts, interfaces, legacy adapters and tests are implemented and pass. No live endpoint, LLM, Tavily or PostgreSQL run was made; see Test Results.)

## Goal
Introduce stable domain models and boundaries while preserving existing behavior.

## Scope
- New `research_app/domain/` package: Pydantic contracts, two `Protocol` interfaces, and pure shape adapters for structures that exist today.
- Tests for the above, plus a snapshot test that pins the Phase 1 app surface.

## Out of Scope
- Future phases. In particular: no Tavily payload adapter and no parsing of the `search_results` strings (Phase 3), no runtime wiring, no DB tables or migrations (Phase 8), no cache changes (Phase 12).
- LangSmith/evaluation/datasets

## Current Implementation
Before this phase there were no domain objects: a `TypedDict` (`ResearchState`), `{url,title,snippet}` source dicts, `sub_questions: list[str]`, a string-concatenated `search_results`, and an unused `Evaluate` model. `main.py` builds a 13-key `inputs` dict and reads `final_answer`, `sub_questions`, and per-node `sources` / `cache_hit`. That contract, the 16 routes, the SSE events and the 7-node graph are unchanged.

## Planned Changes (as implemented)
1. `research_app/domain/enums.py`: `SourceType` (`web`, `official_docs`, `github`, `reddit`, `rag_document`, `cache`), `Complexity`, `QueryIntent`, `TimeSensitivity`, `TaskStatus`, `ResultStatus`, `RunStatus`, `GapKind`, `CriticIssueKind`.
2. `research_app/domain/models.py`: the contracts below.
3. `research_app/domain/interfaces.py`: `SourceAdapter`, `AnswerCache`.
4. `research_app/domain/legacy.py`: adapters between today's structures and the contracts.
5. `research_app/domain/__init__.py`: public re-exports (`from research_app.domain import ...`).
6. Tests (see Test Results).

### Contracts
All models: `extra="forbid"`; datetimes are timezone-aware UTC (naive input is tagged as UTC, not shifted).

| Model | Fields | Legacy counterpart |
|---|---|---|
| `ResearchQuery` | `text`, `normalized_text`, `user_id`, `session_id`, `complexity`, `intent`, `time_sensitivity`, `technology`, `is_documentation_query`, `history[HistoryTurn]` | `question`, `is_simple`, `history` |
| `SearchRequest` | `query`, `source_type`, `include_domains`, `max_results=3`, `search_depth="advanced"`, `timeout_s`, `run_id`, `task_id` | Tavily config (3 / advanced) |
| `ResearchTask` | `task_id`, `sub_question`, `search_query` (defaults to `sub_question`), `source_types=[web]`, `preferred_domains`, `priority`, `status` | item of `sub_questions` |
| `SourceDocument` | `source_id` (hash of canonical URL), `source_type`, `title`, `url` (http/https only), `domain` (derived), `author`, `published_at`, `retrieved_at`, `snippet`, `content`, `technology`, `version`, `credibility`, `query`, `task_id`, `metadata` | `{url,title,snippet}` |
| `ResearchResult` | `task_id`, `source_type`, `provider`, `status`, `documents`, `error`, `latency_ms`, `started_at`, `completed_at`. FAILED/TIMEOUT require an `error`. | return value of a search node |
| `Evidence` | `evidence_id`, `run_id`, `task_id`, `source_id`, `text`, `url`, `source_type`, `retrieved_at`, `query`, `rank`, `score`, `quality`, `confidence`, `content_hash` (derived, dedup key), `metadata` | none |
| `Citation` | `marker`, `source_id`, `evidence_id`, `url`, `title`, `source_type`, `excerpt` | none |
| `GapAnalysis` | `iteration`, `sufficient`, `covered_task_ids`, `gaps[Gap]`, `follow_up_tasks`, `rationale` | none |
| `CriticResult` | `passed`, `score` (0–1), `issues[CriticIssue]`, `feedback`, `retry_tasks` | `Evaluate`, `critic_*` (unused; not mapped) |
| `ResearchRun` | `run_id`, `user_id`, `session_id`, `query`, `status`, `started_at`, `completed_at`, `tasks`, `results`, `sources`, `evidence`, `gap_analyses`, `critic_results`, `citations`, `final_answer`, `cache_hit`, `retry_count`, `errors` | a `queries` row (plus new fields) |

Also: `ResearchError`, `SourceCredibility`, `Gap`, `CriticIssue`, `HistoryTurn`; helpers `canonical_url`, `source_id_for`, `new_id`, `utc_now`.

### Legacy adapters (`domain/legacy.py`, unused by the runtime)
`source_from_legacy` / `source_to_legacy` / `sources_from_legacy`, `query_from_state`, `tasks_from_sub_questions` / `tasks_to_sub_questions`, `run_from_state`.

## Files Expected
- Create: `research_app/domain/{__init__,enums,models,interfaces,legacy}.py`, `tests/test_domain_{models,legacy,isolation}.py`
- Modify: this document only. **No runtime file changed** (`state.py`, `graph.py`, `main.py`, `schemas.py`, `db/*`, `auth/*`, `requirements.txt`, `chat.html` untouched).
- Delete: None

## Architecture Decisions
- **Additive, not wired.** Nothing in the running app imports `research_app.domain`. Phase 3 does the first wiring, converting at node boundaries via `legacy.py`. `ResearchState` stays the graph contract.
  - *Update (Phase 3):* now wired. The two web search nodes in `agent/state.py` import `domain.legacy`; `SourceDocument` gained `provider` and `original_url`. See `PHASE-03-SOURCE-NORMALIZATION.md`.
- **Domain package is self-contained.** It may not import `research_app.agent/db/main/auth` or LangChain/LangGraph/SQLAlchemy/FastAPI (`state.py` builds a Tavily client at import). Enforced by a subprocess test.
- **Interfaces: only two.** `SourceAdapter` (adapters return a failed `ResearchResult` instead of raising) and `AnswerCache` (mirrors today's `lookup_cache`/`store_cache`; Phase 12 widens it). Critic, gap-detector and evidence-store interfaces wait for Phases 8, 9, 11.
- **`GapAnalysis` folds unsupported claims and contradictions into `gaps`** (by `GapKind`) rather than separate lists.
- **`ResearchRun.sources` added** (not in the original plan): legacy sources are not linked to tasks, and the final de-duplicated source list is part of the target answer, so it needs a home.
- **`SearchRequest` included** (CLAUDE.md Rule 5) although the Phase 2 list omitted it.
- **`is_simple` is read only when `classified=True`.** `main.py` initialises it to `False` before classification, which would otherwise be misread as "complex".
- **`"(from cache)"` sub-question sentinel is not turned into a task.** A test checks the sentinel still appears in `state.py`.
- **Scores are 0–1.** The unused legacy `critic_score: int` has no defined scale, so it is not mapped.
- **Contract stability rule:** adding an optional field is compatible; renaming/removing needs a note here first. Enum values will be persisted later; treat renames as breaking.
- **Timezones:** domain is aware-UTC; DB columns are naive UTC (`_utc_now`). Convert at the Phase 8 persistence boundary.
- **Hardening was committed separately first** (`0f7f8a1`), so this phase's diff contains only new files.

## Dependencies
- None added. `pydantic` was already in `requirements.txt` (2.12.5 installed).

## Implementation Notes
- Run tests from the repo root: `python -m unittest discover -s tests -t .`
- Import contracts from `research_app.domain`, not from submodules.
- `SourceDocument.url`, `Evidence.url`, `Citation.url` reject non-http(s) schemes (e.g. `javascript:`).
- `canonical_url` lower-cases scheme/host, drops default ports, fragment and trailing slashes; the query string is kept as given. It is for identity, not for fetching.

## Manual Test Cases
### Test 1 — App still starts and serves the same surface
Input: `uvicorn research_app.main:app` from the repo root; `GET /health`; open `/docs`.
Expected: `{"status": "ok"}`; the same 16 application routes as Phase 1.
Actual: Not run live (import/route/graph snapshot covered by `test_domain_isolation.py`).

### Test 2 — Existing research flow unchanged
Input: run Phase 1 manual tests 4–6 (simple question, complex question, cache hit) against a live server with valid keys.
Expected: identical SSE event sequence and answers as before this phase.
Actual: Not run (needs LLM/Tavily credentials).

### Test 3 — Contracts usable from a shell
Input: `python -c "from research_app.domain import SourceDocument as S; d=S(url='https://Docs.Python.org/3/'); print(d.domain, d.source_id)"`
Expected: `docs.python.org` and a 32-character hex id; `S(url='javascript:alert(1)')` raises `ValidationError`.
Actual: Run by hand: printed `docs.python.org b4365d45965be1c1c1af00c627b043aa 32`; `javascript:` URL raised `ValidationError`.

## Commands Run
```text
git diff --stat; git status --short; git add -n (hardening files); git commit (hardening, 0f7f8a1)
.venv python: import research_app.main + compiled_graph -> record 16 routes / 7 nodes baseline
python -m unittest tests.test_domain_models tests.test_domain_legacy tests.test_domain_isolation -v
python -m unittest discover -s tests -t .
mutation checks (see below); git diff --stat (tracked files: empty)
```
All Python runs used `PYTHONDONTWRITEBYTECODE=1`.

## Test Results
- Focused: 32/32 pass (`test_domain_models` 17, `test_domain_legacy` 12, `test_domain_isolation` 3 including the route and graph snapshots).
- Full suite: 53/53 pass (21 existing + 32 new).
- Mutation checks: (1) the isolation detector fires when `sqlalchemy` is imported; (2) truncating the snippet in `source_to_legacy` makes the round-trip test fail. Both reverted.
- Snapshot: 16 application routes and 7 graph nodes, identical to the Phase 1 baseline.
- Not run: any live endpoint, LLM, Tavily or PostgreSQL call.

## Security / Reliability Notes
- URL fields accept only absolute http(s) URLs with a host, so `javascript:`, `file:`, `ftp:` and scheme-less values cannot enter `SourceDocument`/`Evidence`/`Citation`. This is a shape check only; SSRF and prompt-injection defences are Phase 14.
- `sources_from_legacy` skips and logs invalid entries instead of raising (failure isolation); it logs the exception type only, not the content.
- `extra="forbid"` turns misspelled fields into errors rather than silent data loss.

## Known Issues
- **Contracts are unused at runtime until Phase 3.** Only tests exercise them, so real-world fit is unproven; expect additive revisions in Phases 3–11.
- **Two representations coexist** (`ResearchState` dicts and domain models). Adapters and round-trip tests are the drift guard.
- **Legacy data is lossy.** Legacy sources have no content (only a 150-character snippet), so `SourceDocument.content` is `None` when converted. `Evidence`, `Citation`, `GapAnalysis`, `CriticResult` have no legacy counterpart.
- **De-duplication differs slightly:** `main.py` de-duplicates sources by exact URL; the domain uses canonical URL (case, trailing slash, fragment, default port). `www.` and query-parameter differences are not normalised.
- **`AnswerCache` is not implemented by `vectordb.py`** (module functions, not an object). A wrapper belongs to Phase 12.
- **`test_placeholder_matches_what_the_graph_emits` reads `research_app/agent/state.py` by relative path**, so tests must run from the repo root (the documented command).
- Naive datetimes are assumed to be UTC; a naive local time would be mislabelled.
- Carried over from Phase 1, not addressed here: `.env` has `LANGSMITH_TRACING` enabled (conflicts with CLAUDE.md; turn it off), PostgreSQL path untested, `/test-db` unauthenticated. The Python 3.14 "Pydantic V1" warning comes from `langchain_core` and is pre-existing.
- Git prints LF→CRLF warnings on add/commit for new files; harmless, no `.gitattributes` exists.

## Git Commit
`phase-2: domain contracts` (hash omitted on purpose; see `git log`). Preceded by `0f7f8a1 pre-phase-2: ownership/admin checks, utc timestamps, logging, requirements cleanup`.

## Next Phase
**Phase 03 — Research Result and Source Normalization** (`docs/phases/PHASE-03-SOURCE-NORMALIZATION.md`): write the Tavily adapter that returns `ResearchResult`/`SourceDocument`, convert at the `search_node`/`simple_search_node` boundary through `legacy.py`, and keep `search_results`/`sources` in state byte-compatible. Start in a fresh session.
