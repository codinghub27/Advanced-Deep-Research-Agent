# PHASE-07 — Hybrid Source RAG

## Status
In Progress — **not production-ready; keep `SOURCE_RAG_ENABLED=false`.** The retrieval layer is built and verified
(Pass 1 / Pass 2 below, with the real `bge-reranker-v2-m3`). Open items before the flag should ever be turned on:
the flag-off node-trace deviation (Verification, Pass 1), the review findings R1–R3 (relevance gate fails open without
the reranker; a RAG hit can still trigger a live gap search; inline indexing latency), and `.env.example` not updated.
Phases 8–19 are **not started** (see "Known Issues / Out of Scope").

## Goal
Implement Qdrant dense + BM25 + rank fusion + reranking over stored source documents, kept
separate from the answer cache.

## Scope
- New `source_chunks` Qdrant collection (named vectors `dense` + `sparse`) and its payload indexes.
- `HybridRetriever.retrieve(query, filters, top_k) -> list[SourceDocument]`: dense + BM25 → RRF → optional rerank.
- Metadata filters: `source_type`, `domain`, `technology`, `retrieved_at` range.
- Chunking + indexing of a run's source documents (`SOURCE_RAG_INGEST_ENABLED`).
- A flag-gated graph detour (`SOURCE_RAG_ENABLED`): stored sources first, live research if they are not relevant.
- Every tunable in environment settings; reranker behind a swappable interface.

## Out of Scope
- Future phases (evidence store in PostgreSQL = Phase 8; semantic answer cache in Qdrant + freshness = Phase 12).
- LangSmith / evaluation / datasets.
- Any change to `agent/vectordb.py` (the answer cache) or to API contracts.

## Current Implementation
Verified by inspection before coding (the repo is thinner than CLAUDE.md describes):
- Answer cache = `research_app/agent/vectordb.py`: an in-process dict with Jaccard word overlap
  (`CACHE_SCORE_THRESHOLD=0.85`). No Qdrant, no embeddings. Only caller: `agent/state.py`
  (`semantic_cache_node`, `save_to_cache_node`). **Untouched by this phase.**
- Qdrant was only a docker-compose service plus unused `qdrant-client` / `langchain-qdrant`. There was
  **no retriever and no dense-only call** to replace.
- `EvidencePool` (`agent/pipeline/evidence.py`) is per-request and in-memory (word-overlap relevance).
- No `sources`/`evidence` PostgreSQL tables (only `research_conversations`, `research_citations`), so chunk
  metadata lives in the Qdrant payload. Durable evidence tables are Phase 8.

## Implemented Changes
With both flags **off**, answers, sources, search calls and LLM calls are identical to Phase 6 (Pass 1), but the node
trace is **not** byte-for-byte: a no-op `index_sources_node` runs between `format_response` and `save_to_cache_node`
and reaches API clients as one extra SSE `progress` event. With `SOURCE_RAG_ENABLED=true`:

```
semantic_cache -> classify_node -> rag_gate_router
   time-sensitive / flag off -> classify_router (simple_search | planner)     [unchanged]
   otherwise -> source_rag_node -> rag_router
        >= RAG_MIN_CHUNKS chunks with score >= RAG_RELEVANCE_THRESHOLD -> evidence_collection (existing path)
        otherwise                                                        -> classify_router (live research)
format_response -> index_sources_node -> save_to_cache_node       (index_sources_node is a no-op unless SOURCE_RAG_INGEST_ENABLED)
```
No edge goes back to an earlier node.

- **Retrieval** (`rag/store.py`): hybrid = two Qdrant prefetch legs (dense cosine, sparse BM25 with IDF applied by Qdrant),
  fused server-side with `RrfQuery(Rrf(k=RAG_RRF_K))`. If a server rejects that query, the same fusion is done
  client-side with `qdrant_client.hybrid.fusion.reciprocal_rank_fusion` (tested). Filters are applied to both legs.
- **Scores are 0–1** so one threshold works everywhere: reranked → sigmoid probability; hybrid → RRF score / best possible (2/k);
  dense → cosine; sparse → relative to the best hit (debug mode only). Raw pre-rerank score is kept in `fused_score`.
- **Reranker** (`rag/reranker.py`): `Reranker` protocol (`domain/interfaces.py`) + `RERANKERS` registry keyed by
  `RERANK_PROVIDER`. A Cohere/Jina swap = one class + one registry line. Default `cross_encoder` loads lazily on the first call;
  if it cannot load or score, the fused order is returned (logged once). A sigmoid is forced on `predict()` because some
  checkpoints (found with ms-marco MiniLM) return raw logits.
- **Rollback**: `SOURCE_RAG_ENABLED=false` (Phase 6 graph exactly) and `RAG_MODE=dense` (dense-only baseline through the same store).
- **Reliability**: every retrieval/indexing call runs in a worker thread under `RAG_TIMEOUT_S`; Qdrant requests use
  `QDRANT_TIMEOUT_S`. Any error/timeout returns `[]` / `0` and is logged, so a run falls back to live research. Cancellation still propagates.
  `retrieve` never creates the collection (only indexing does).

### Settings (all environment variables, read at call time; malformed values fall back to the default)
| Variable | Default | Meaning |
|---|---|---|
| `SOURCE_RAG_ENABLED` | false | stored-source lookup after a cache miss |
| `SOURCE_RAG_INGEST_ENABLED` | false | index each run's documents into `source_chunks` |
| `QDRANT_URL` | `http://localhost:6333` | server URL; `:memory:` or a folder path = embedded engine (tests, seed script) |
| `QDRANT_API_KEY` | – | secret, never logged (`repr=False`) |
| `QDRANT_TIMEOUT_S` | 5 (1–120) | per Qdrant request |
| `SOURCE_CHUNKS_COLLECTION` | `source_chunks` | collection name |
| `RAG_MODE` | `hybrid` | `hybrid` / `dense` / `sparse` |
| `RAG_DENSE_MODEL` / `RAG_SPARSE_MODEL` | `BAAI/bge-small-en-v1.5` / `Qdrant/bm25` | fastembed models |
| `RAG_DENSE_TOP_K` / `RAG_SPARSE_TOP_K` | 30 / 30 | candidates per leg |
| `RAG_TOP_K` | 8 | chunks after fusion (rerank candidate pool) |
| `RAG_RRF_K` | 60 | RRF constant |
| `RAG_RELEVANCE_THRESHOLD` / `RAG_MIN_CHUNKS` | 0.5 / 2 | gate to answer from stored sources |
| `RAG_TIMEOUT_S` | 30 | whole retrieval / indexing call |
| `RAG_EMBED_BATCH_SIZE` | 32 | embedding batch |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | 1200 / 150 | characters |
| `RERANK_ENABLED` | true | reranking on/off |
| `RERANK_PROVIDER` | `cross_encoder` | `cross_encoder` / `none` / anything added to `RERANKERS` |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | CrossEncoder model |
| `RERANK_TOP_K` / `RERANK_MAX_LENGTH` / `RERANK_BATCH_SIZE` | 5 / 512 / 16 | rerank output / tokens / batch |

## Files Expected (actual)
- Create:
  - `research_app/rag/{__init__,settings,chunking,embeddings,reranker,store,retriever,ingest}.py`
  - `research_app/agent/pipeline/rag_nodes.py`
  - `scripts/rag_manual.py` (seed + manual query tool; own collection, refuses `source_chunks`)
  - `tests/rag_helpers.py`, `tests/test_rag_{chunking,store,retriever,nodes}.py`
- Modify:
  - `research_app/domain/{models,enums,interfaces,__init__}.py`: `SourceChunk`, `RetrievalFilter`, `RetrievedChunk`, `RetrievalMode`, `Retriever`, `Reranker`
  - `research_app/agent/graph.py`: two nodes, gate/router edges
  - `research_app/agent/state.py`: two `ResearchState` fields (`rag_hit`, `rag_top_score`), nothing else
  - `tests/test_domain_isolation.py`: node snapshot now `PHASE7_NODES` (baseline + Phase 6 + the two new nodes)
  - `requirements.txt`, `.gitignore` (`.rag_manual/`)
- Delete: none.

## Architecture Decisions
- **BM25 = Qdrant sparse vectors (fastembed `Qdrant/bm25`), not `rank_bm25`**: persists with the dense vectors, one filter applies to both legs and fusion happens in one call; `rank_bm25` is in-memory, rebuilt on start, and cannot filter natively.
- **Chunks keep their original `source_type`** (official_docs stays official_docs) so source priority and citations keep working; `provider="source_chunks"` and `metadata.rag_*` mark the origin.
- **Chunk ids are deterministic** (uuid5 of source id + index): re-indexing is an idempotent upsert; documents that came from `source_chunks` are not re-indexed.
- **Deviation from the plan**: `retrieve()` returns `list[SourceDocument]` with scores in `metadata` (as requested) instead of a `RetrievalResult` model, so `RetrievalRequest`/`RetrievalResult` were not created; `RetrievedChunk` is the internal boundary.
- **Deviation**: the plan said "call the existing dense-only path"; none existed, so the rollback is the two flags + `RAG_MODE=dense`.
- `technology` is stored lower-cased so the keyword filter is case-insensitive.
- The time-sensitivity gate reuses `agent/temporal.is_time_sensitive` (same rule the cache uses): live questions never come from stored sources.

## Dependencies
- Added `fastembed` (dense + BM25 on ONNX; no torch) and `sentence-transformers` (CrossEncoder; pulls in torch, imported lazily).
- `torch>=2.9` pinned explicitly in `requirements.txt` (Python 3.14 wheels; the venv has torch 2.14.0, sentence-transformers 6.1.0, fastembed 0.8.0).
- Not added: `rank_bm25`; `langchain-qdrant` stays unused (no prefetch/RRF-k control).
- First use downloads models (≈130 MB fastembed; ≈2.3 GB for `bge-reranker-v2-m3`).

## Implementation Notes
- `research_app/agent/temporal.py` (Phase 6 real-time work) is imported by `rag_nodes.py` and is committed with this phase because HEAD would not import without it.
- The embedded engine (`:memory:` / folder) is used by tests and the seed script; a folder can be opened by one process at a time.
- Pyright (CLI, project venv) on all touched files: 0 errors. Baseline errors left alone: `graph.py:40` (`search_node` signature), `domain/legacy.py` (2).

## Manual Test Cases
Corpus: `scripts/rag_manual.py seed` (9 docs: FastAPI/PostgreSQL/Qdrant official docs, 2 Reddit, 1 old web page dated 400 days ago, 1 unrelated). Real fastembed models; embedded engine in `.rag_manual/`, then repeated on the real Qdrant server (`localhost:6333`) in a throw-away collection `source_chunks_manual_test`, dropped afterwards.

### Test 1 — dense-only baseline
Input: `query "what do I type to set up the python web framework with automatic docs" --mode dense`
Expected: the FastAPI install page ranks first without sharing keywords.
Actual: PASS — #0 FastAPI tutorial (0.598), #1 old FastAPI notes (0.571), #2 FastAPI dependencies (0.566). Note the tight spread: dense-only scores are weak for gating.

### Test 2 — BM25 only
Input: `query "CONCURRENTLY" --mode sparse`
Expected: only the chunk containing the literal token.
Actual: PASS — single hit, PostgreSQL CREATE INDEX (score 1.000).

### Test 3 — hybrid with fusion
Input: `query "reciprocal rank fusion of dense and sparse vectors" --mode hybrid`, `--rrf-k 60` vs `--rrf-k 2`
Expected: Qdrant hybrid-queries page first; scores change with k.
Actual: PASS — same order; fused score 0.0333 (k=60) vs 1.0000 (k=2), rank-2 0.0164 vs 0.3333.

### Test 4 — hybrid + rerank
Input: `query "how do I install fastapi" --mode hybrid --rerank` with `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`; then `RERANK_MODEL=bogus/not-a-model`.
Expected: reranked flag set, scores 0–1 and graded, order can differ from fused; a bad model falls back to the fused order.
Actual: PASS with the MiniLM stand-in — FastAPI tutorial 1.000, dependencies 0.403, old notes 0.088, Reddit 0.059 (after forcing the sigmoid; before, raw logits were clamped to 1.000/0.000). The bogus model logged `Reranker bogus/not-a-model unavailable (RepositoryNotFoundError); keeping the fused order` and returned the fused order.
**NOT RUN with `BAAI/bge-reranker-v2-m3`**: the download stalled (≈67 MB of ≈2.3 GB in 4 min) and was aborted. See Known Issues #1.

### Test 5 — metadata filters
Input: `--source-type reddit --technology fastapi`; `--after-days 30`; `--domain qdrant.tech`
Expected: only matching payloads.
Actual: PASS on the embedded engine and on the real server (server-side payload indexes created) — 1 Reddit FastAPI hit; 8 of 9 chunks (the 400-day-old page excluded); 1 qdrant.tech hit.

### Test 6 — answer cache isolation
Input: run `source_rag_node` + `index_sources_node` (both flags on) with one item in the in-memory answer cache.
Expected: cache unchanged; nothing written outside `source_chunks*`.
Actual: PASS — answer cache items 1 → 1 and `get_cache_stats()` identical; `git diff research_app/agent/vectordb.py` empty; embedded store lists only `source_chunks_manual_test` (9 points, re-index idempotent). On the real server the pre-existing `research_cache` collection stayed at **6 points before and after**; only the test collection was created and then dropped. (There is no `answer_cache` collection in this repo — the cache is in-memory until Phase 12.)

### Test 7 — Qdrant unavailable
Input: `--url http://localhost:6399` (nothing listening) and a server without the collection.
Expected: warning, no exception, empty result.
Actual: PASS — `(no results)` after ≈12 s (includes cold model import); missing-collection 404 also returns `[]`.

## Commands Run
```text
.venv/Scripts/python -m pip install fastembed sentence-transformers
.venv/Scripts/python -W ignore -m unittest tests.test_rag_chunking tests.test_rag_store tests.test_rag_retriever tests.test_rag_nodes tests.test_domain_isolation
.venv/Scripts/python -W ignore -m unittest discover -s tests -t .
npx -y pyright --pythonpath .venv/Scripts/python.exe research_app/rag research_app/agent/pipeline/rag_nodes.py scripts/rag_manual.py tests/test_rag_*.py tests/rag_helpers.py research_app/domain/*.py
.venv/Scripts/python -W ignore scripts/rag_manual.py seed | stats | drop | query "..." --mode dense|sparse|hybrid [--rerank|--no-rerank] [--rrf-k N] [--source-type T] [--technology X] [--domain D] [--after-days N]
.venv/Scripts/python -W ignore scripts/rag_manual.py --url http://localhost:6333 --collection source_chunks_manual_test seed | query | stats | drop
```
(The repo has no pytest; tests are `unittest`.)

## Test Results
- New: 59 tests across chunking, store (dense/sparse/hybrid, RRF fallback, filters, `answer_cache` isolation), retriever/reranker/indexer/settings, nodes/graph contract — all pass (`test_domain_isolation` included).
- Full suite on the clean committed trees (git worktrees, dummy API keys, `unittest discover -s tests -t .`): **515 tests at `880f4e0` (pre-Phase-7), 570 at `986b243` (Phase 7) — 0 failures in both.** (An earlier 617 figure came from the dirty working tree with uncommitted Phase 6 real-time tests; `tests/test_source_router.py` had a stray ` hy` typed on line 1 there, uncommitted and unrelated to this phase.)
- Pyright (CLI) on touched files: 0 Phase 7 errors; 3 baseline errors remain (`graph.py:40`, `domain/legacy.py` x2).
- Pass 1 / Pass 2 below.

## Security / Reliability Notes
- Every external call has a timeout and returns empty on failure; one failed store never fails the run. Indexing never raises, but it is awaited inline (`format_response -> index_sources_node -> save_to_cache_node`), so with ingest on it delays the answer-cache save and the end of the stream by up to `RAG_TIMEOUT_S` (review finding R3); it is not fire-and-forget.
- `QDRANT_API_KEY` is not logged or repr'd. `scripts/rag_manual.py` refuses the production collection name.
- Stored web content is untrusted and re-enters synthesis through the existing evidence path; prompt-injection hardening is Phase 14. Stale content is a risk: chunks keep `retrieved_at`, but freshness policy is Phase 12 (time-sensitive questions bypass stored sources).

## Known Issues
1. ~~Default reranker not verified with real weights~~ — **closed 2026-09-21**: the ≈2.3 GB download completed and Pass 2 ran on the real `BAAI/bge-reranker-v2-m3` (scores in Pass 2). The standalone Test 4 command was not re-run with it.
2. **`.env.example` not updated** — editing it was blocked by the session's permission settings; the variables are documented in the table above and in `rag/settings.py`.
3. **Cold start**: the first retrieval with the reranker enabled loads/downloads the model inside `RAG_TIMEOUT_S`; if it exceeds it, that run falls back to live research while the load finishes in the background. Warm the model before enabling in production (e.g. one `scripts/rag_manual.py query`) or raise `RAG_TIMEOUT_S`.
4. **Gate thresholds are uncalibrated**: without the reranker, dense/hybrid scores are weakly discriminative (Test 1 shows a 0.57–0.60 spread); `RAG_RELEVANCE_THRESHOLD=0.5` is a starting point. That is why both flags default off.
5. Ingest embeds inside the request's post-answer path (in a worker thread under `RAG_TIMEOUT_S`); throughput/backgrounding is a Phase 15/16 topic.
6. Docker was not running; server-side behaviour was verified against the existing local Qdrant server on `localhost:6333` instead.
7. Local-engine shutdown prints a harmless `QdrantClient.__del__` traceback.

## Verification (graph wiring, 2026-09-21)
Reproduce from the repo root (`.venv/Scripts/python -W ignore <script>`): `scripts/phase7_trace_flag_off.py`
(pass 1, run in each checkout) and `scripts/phase7_verify_branching.py` (pass 2). Both use `ScriptedLLM` +
`FakeTavily` from the existing tests, so no LLM/Tavily credentials or network are needed; retrieval, embeddings, Qdrant
(embedded, isolated folder, collection `source_chunks`) and the reranker are real. Note: the pre-Phase-7 live baseline
queries were never executed in Phase 1 (they need a DB, LLM and Tavily), so there is no recorded pre-Phase-7 output;
pass 1 instead diffs two clean git worktrees (`880f4e0` vs `986b243`) running identical scripted queries.

### Pass 1 — regression, `SOURCE_RAG_ENABLED=false` (default)
6 scenarios: complex how-to, simple question, cache hit, critic retry, time-sensitive, web tool raising.
- **Identical** in both trees: final answer, sources, Tavily calls (query + domains), free-text/structured LLM call counts, cache writes, errors.
- **Deviation (reported, not fixed):** the node trace differs by one node in 4 of 6 scenarios (all except cache hit and the crashing web-failure run, which never reaches it): `format_response -> index_sources_node -> save_to_cache_node`. The node returns `{}`. `main.py` (`~L299-303`) emits `{type: 'progress', node, label}` for every node and `NODE_LABELS` has no entry for `index_sources_node` (or `source_rag_node`), so clients get one extra event whose label is the raw node name. `chat.html` ignores unknown nodes (`NODE_STEP[data.node] ?? lastStep`); a strict or label-displaying client would not. This violates "graph behavior identical with the flag off" (CLAUDE.md Rule 3, additive only). Smallest fixes: a conditional edge that bypasses the node unless `SOURCE_RAG_INGEST_ENABLED`, and `NODE_LABELS` entries for the new nodes.
- Pre-existing (both trees): a raw exception from the web search tool propagates out of `search_node` and ends the run (Phase 13 territory).
- Suites: 515 pass (pre) / 570 pass (post), see Test Results.

### Pass 2 — branching, `SOURCE_RAG_ENABLED=true`, real `BAAI/bge-reranker-v2-m3`
Seed: the 9-doc `rag_manual` corpus + 2 extra synthetic FastAPI pages = 11 chunks in `source_chunks` (embedded Qdrant, temp folder). Also an `answer_cache` collection (3 points) in the same store.

| Scenario | Top score / chunks >= 0.5 | Path | Tavily / planner LLM |
|---|---|---|---|
| a. relevant: "How do I install FastAPI and start the development server?" | 0.999 / 3 | `source_rag -> evidence_collection -> gap_detection -> synthesis -> critic -> format_response -> index_sources -> save_to_cache` | 0 / none |
| a'. same topic, one on-topic chunk (PostgreSQL CREATE INDEX), default `RAG_MIN_CHUNKS=2` | 0.999 / 1 | **falls through** to `planner -> search x2 -> ... -> targeted_search -> ...` | 8 / yes |
| a''. same, `RAG_MIN_CHUNKS=1` | 0.999 / 1 | hits (as in a) | 0 / none |
| a'''. Test-1 query ("what do I type to set up the python web framework with automatic docs") | 0.002 / 0 | **falls through** (the reranker scores it ~0 against the FastAPI pages) | 8 / yes |
| b. irrelevant: "health benefits of intermittent fasting" | 0.000 / 0 | falls through to `planner -> search -> evidence -> gap -> targeted_search -> synthesis -> critic -> retry -> format` | 2 / yes |
| b'. same question, flag off | – | node list identical to b minus `source_rag_node` | 2 / yes |

- **a:** the RAG-hit branch never calls the planner, `search_node` or Tavily; the LLM calls were understanding + critic + one synthesis. It is *not* literally `source_rag -> synthesis`: it goes through `evidence_collection` and `gap_detection` (pure Python, no I/O) first, which is what reuses the existing synthesis/critic unchanged.
- **b:** the fall-through is the unchanged full-research path; b vs b' differ only by the extra `source_rag_node` step. (Path equality with the pre-Phase-7 tree for the same code was shown in Pass 1.)
- **c (answer cache untouched):** the `answer_cache` collection stayed at 3 points while `source_chunks` went 11 -> 15 (ingest on, after the live run of scenario b). In-memory answer cache (`vectordb`) 0 -> 0; the local Qdrant server's existing `research_cache` (read-only check) 6 -> 6. `agent/vectordb.py` has no diff. Caveat: the graph's own `store_cache` was stubbed in the harness, so this shows that source_chunks writes do not touch the answer cache, not that `save_to_cache_node` is unchanged.
- **Finding:** with the real reranker scores are near-binary (0.999 vs ~0.00). Default `RAG_MIN_CHUNKS=2` therefore needs two on-topic chunks; a corpus with a single on-topic chunk per subject always falls through (cost: an extra live search, not a wrong answer). The Test-1 style semantic query with no shared keywords also misses (a''').
- Harness caveats: retriever and indexer share one Qdrant client via a patch of `store.make_client` (embedded mode allows one client per folder); `RAG_TIMEOUT_S=300` so the first model load is not cut off; the corpus is synthetic paraphrases, not fetched pages.

### Independent review (CLAUDE.md Phase 7 + Rules 3, 8, 9, 10)
No agent named `code-reviewer` exists in this session (custom agents: `.claude/agents/researcher.yaml` only; built-ins: claude, Explore, general-purpose, Plan, ...). A read-only `general-purpose` agent reviewed the clean `986b243` tree instead. Its verdict: the library layer meets the Phase 7 goals; the graph integration does **not yet** meet Rules 8/9 safely; Rule 3 holds for the answer cache and is additive-only for the SSE stream; Rule 10 holds (no eval/LangSmith code); "do not enable `SOURCE_RAG_ENABLED` until R1–R3 are fixed". Evidence levels are as the reviewer stated them:
- **R1 (major, run by the reviewer)** — the gate is rank-derived when the reranker is off or failed to load (`RERANK_ENABLED=false`, or the sticky `_failed` after a load error): hybrid `rag_score` = RRF rank / (2/k), dense has no similarity floor, so an off-topic question returned three chunks scoring 1.0 / 0.976 / 0.976 and would be answered from stored sources. Fix: fail closed unless results are `reranked`, or add a dense-cosine floor; retry a failed reranker load after a cooldown. (Consistent with Known Issue 4 below, but worse than that entry says.)
- **R2 (major, code reading; not reproduced in Pass 2)** — a RAG hit can still trigger a live search: `evidence_for()` filters by word-overlap (`relevance()`, default 0.3), a semantic hit with little word overlap yields "no evidence" for the sub-question, `gap_detection` then emits follow-ups and `targeted_search` calls Tavily (`GAP_SEARCH_ENABLED` default true); a critic retry can also search. No test covers `source_rag -> evidence_collection -> gap_detection`. In Pass 2 the reranker gate itself rejected the low-overlap query, so the hit + gap combination did not occur. Fix: treat `provider == "source_chunks"` documents as relevant in the evidence pool; add a flow test.
- **R3 (major, code reading)** — inline indexing delays `done`/answer-cache save by up to `RAG_TIMEOUT_S` (first use includes a model download). Fix: order `save_to_cache -> index_sources -> END` or a tracked background task.
- **R4 (major, run by the reviewer)** — `sentence-transformers` + `torch>=2.9` exist only to allow `bge-reranker-v2-m3`; fastembed ships ONNX cross-encoders (bge-reranker-base, jina-reranker-v2-base-multilingual, MiniLM), so torch is avoidable. The `torch>=2.9` pin requested for this phase was added regardless. Decision needed: keep torch + bge-v2-m3, or move the `cross_encoder` provider to fastembed.
- **R5 (quality)** — `synthesis.build_sources` truncates a page's joined chunks to 1000 chars (web/Reddit) or 1500 (official docs) while `RAG_CHUNK_SIZE` defaults to 1200, so the gating chunk of a non-official page can be cut before synthesis sees it.
- **R6 (minor, mostly read)** — Rule 8 gaps: `wait_for(to_thread)` cancels the await, not the thread (cold-start requests can pile up on the reranker lock); lazily-built store/reranker without a lock; `_hybrid_points` catches every exception and retries as separate queries (a blackholed Qdrant can exceed `QDRANT_TIMEOUT_S`); embedding/rerank have only the outer timeout; retriever and indexer build separate stores.
- **R7 (minor)** — stale/poisoned content: no default max age (`RetrievalFilter.retrieved_after` exists but the node never passes it); a shrunken re-indexed page leaves old tail chunks (no delete-by-source); no quality filter on ingest. TTL/invalidation is Phase 12 and injection hardening Phase 14, but Phase 7 makes one poisoned page re-enter synthesis for every related later query.
- **R8 (minor)** — `NODE_LABELS` / SSE deviation: same as Pass 1.
- **R9 (minor)** — `RAG_MIN_CHUNKS` counts chunks, not sources (two chunks of one page satisfy it), and a lone strong chunk is discarded on a miss instead of being kept as evidence. Not a defect by itself.
- **R10 (nit)** — `agent/temporal.py` (104 lines) is Phase 6 real-time work carried into this commit; only `is_time_sensitive` is used (scope mixing, Rule 4). Graph never uses the metadata filters, so official docs cannot be preferred on a RAG hit; the legacy SSE `sources` list is empty on a RAG hit (`citations` are populated).

### Decisions
- Reported the flag-off node-trace deviation and did **not** change graph code during verification; the fix is small but changes wiring, so it needs an explicit go-ahead.
- Kept `RAG_MIN_CHUNKS=2` / `RAG_RELEVANCE_THRESHOLD=0.5` defaults; the hit scenario was demonstrated by seeding a corpus with >= 2 on-topic chunks (and separately with `RAG_MIN_CHUNKS=1`), not by tuning the defaults.
- Seed corpus is synthetic; the verification scripts live in `scripts/` and never write to the server's collections (embedded store only).

## Known Issues / Out of Scope — NOT production-ready
Phase 7 delivers only the retrieval layer and a flag-gated graph detour. **Phases 8–19 are not started.** In particular:

| Phase | Status | What is still missing |
|---|---|---|
| 8 Evidence pipeline and store | Not started | Evidence is still per-request and in-memory (word-overlap relevance); no PostgreSQL `sources`/`evidence` tables; chunk metadata lives only in the Qdrant payload; no evidence enrichment or quality filter on ingest |
| 9 Gap detection and iterative research | Not started | Gap detection does not use RAG evidence sensibly yet (R2); one bounded gap round only |
| 10 Synthesis, citations, grounding | Not started | Same Phase 6 synthesis; truncation of chunks (R5); no claim-level grounding |
| 11 Critic and targeted retry | Not started | Phase 6 critic reused unchanged |
| 12 Semantic cache and freshness | Not started | The answer cache is an **in-process dict with Jaccard word overlap** (`agent/vectordb.py`), not Qdrant and not semantic; no `answer_cache` Qdrant collection exists; no TTL/freshness/invalidation; stored chunks have no max age (R7) |
| 13 Reliability and fault tolerance | Not started | Retries/backoff/circuit breaking; raw tool exceptions still end a run; R6 |
| 14 Security and API hardening | Not started | Prompt-injection defenses for stored/retrieved content, SSRF, rate limits, auth review |
| 15 Performance and cost | Not started | Ingest runs inline (R3); no embedding batching/reuse work beyond a batch size setting |
| 16 API, streaming, long-running research | Not started | No run IDs, status tracking, cancellation; SSE labels for the new nodes missing (Pass 1) |
| 17 Docker and production config | Not started | Docker not run in this phase; no image/health checks for the RAG stack or model warm-up |
| 18 Observability | Not started | Log lines only; no metrics/tracing for retrieval latency or scores |
| 19 Final manual validation | Not started | The 20-case matrix has not been run; this phase's manual tests are Pass 1/2 + Tests 1-7 |

Other open items: `.env.example` is not updated (edits are blocked by the session's permission settings; variables are in the settings table above); the standalone Test 4 was not re-run with `bge-reranker-v2-m3`; the gate thresholds are uncalibrated on real data.

## Git Commit
`986b243` (`phase-7: pin torch>=2.9 ...`) and the final checkpoint `phase-7: hybrid source RAG, flag-gated node wired into graph` (docs + verification scripts).
`2ced340` `phase-7: hybrid source RAG (dense + BM25 + RRF + rerank)` on branch `phase-7-hybrid-rag`, pushed to `origin` (not merged into `main`).

## Next Phase
Phase 8 — Evidence Pipeline and Evidence Store (durable evidence metadata in PostgreSQL, searchable content in Qdrant).
