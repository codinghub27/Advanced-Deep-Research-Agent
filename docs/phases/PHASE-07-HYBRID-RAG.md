# PHASE-07 — Hybrid Source RAG

## Status
In Progress — implementation done and verified; two items open (see Known Issues #1 and #2):
the default reranker `BAAI/bge-reranker-v2-m3` has not been run with real weights, and
`.env.example` is not updated.

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
Flow with both flags **off** is byte-for-byte the Phase 6 flow. With `SOURCE_RAG_ENABLED=true`:

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
- Full suite: 617 tests, all pass except `tests/test_source_router.py`, which currently fails to import because of a stray ` hy` typed at the start of line 1 (uncommitted, not from this phase; 598 other tests ran and passed). It was left as is.
- Pyright on touched files: 0 errors.

## Security / Reliability Notes
- Every external call has a timeout and returns empty on failure; one failed store never fails the run. Indexing is fire-and-forget inside the graph and never raises.
- `QDRANT_API_KEY` is not logged or repr'd. `scripts/rag_manual.py` refuses the production collection name.
- Stored web content is untrusted and re-enters synthesis through the existing evidence path; prompt-injection hardening is Phase 14. Stale content is a risk: chunks keep `retrieved_at`, but freshness policy is Phase 12 (time-sensitive questions bypass stored sources).

## Known Issues
1. **Default reranker not verified with real weights** — `bge-reranker-v2-m3` (≈2.3 GB) download stalled locally (retried on 2026-09-21: still 0 bytes of the missing blob; only ≈67 MB partial cached). The code path was verified with a real small cross-encoder. To close: run Test 4 on a good connection.
2. **`.env.example` not updated** — editing it was blocked by the session's permission settings; the variables are documented in the table above and in `rag/settings.py`.
3. **Cold start**: the first retrieval with the reranker enabled loads/downloads the model inside `RAG_TIMEOUT_S`; if it exceeds it, that run falls back to live research while the load finishes in the background. Warm the model before enabling in production (e.g. one `scripts/rag_manual.py query`) or raise `RAG_TIMEOUT_S`.
4. **Gate thresholds are uncalibrated**: without the reranker, dense/hybrid scores are weakly discriminative (Test 1 shows a 0.57–0.60 spread); `RAG_RELEVANCE_THRESHOLD=0.5` is a starting point. That is why both flags default off.
5. Ingest embeds inside the request's post-answer path (in a worker thread under `RAG_TIMEOUT_S`); throughput/backgrounding is a Phase 15/16 topic.
6. Docker was not running; server-side behaviour was verified against the existing local Qdrant server on `localhost:6333` instead.
7. Local-engine shutdown prints a harmless `QdrantClient.__del__` traceback.

## Git Commit
`2ced340` `phase-7: hybrid source RAG (dense + BM25 + RRF + rerank)` on branch `phase-7-hybrid-rag`, pushed to `origin` (not merged into `main`).

## Next Phase
Phase 8 — Evidence Pipeline and Evidence Store (durable evidence metadata in PostgreSQL, searchable content in Qdrant).
