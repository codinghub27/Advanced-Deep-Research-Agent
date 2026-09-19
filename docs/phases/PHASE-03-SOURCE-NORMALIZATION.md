# PHASE-03 — Research Result and Source Normalization

## Status
Completed — implemented, verified offline (132/132 tests re-run before the checkpoint) and committed.

(No live endpoint, LLM, Tavily or PostgreSQL call was made; see Test Results.)

## Goal
Normalize source outputs into the common `ResearchResult` / `SourceDocument` model, so Web, Official Docs, GitHub, Reddit and future sources can feed the same downstream pipeline.

## Scope
- One normalization layer: raw provider result → `SourceDocument`.
- A Tavily adapter: Tavily response → `ResearchResult`.
- The two existing web search nodes use it, and still emit the state keys the graph and SSE stream already consume.
- Two provenance fields on `SourceDocument`.

## Out of Scope
- Official docs, GitHub, Reddit retrieval; source router; new providers (Phases 4–6).
- Ranking, de-duplication, quality scoring, evidence extraction, BM25/vector search, gap detection, synthesis/critic changes, caching, PostgreSQL, Docker, observability.
- LangSmith/evaluation/datasets.

## Current Implementation
How results flowed **before** this phase (from reading `state.py`):

1. `tavily_tool.ainvoke({"query": q})` returns `{"query", "answer", "results": [{"title","url","content","score",...}], "images", "response_time"}`. Two other outcomes exist: `{"error": <exception>}` (wrapper swallowed a failure) and a **plain string** (zero results: the wrapper raises `ToolException`, and `handle_tool_error=True` turns it into its message).
2. `simple_search_node` (limit 400) and `search_node` (limit 800) each held a copy-pasted block: slice `[:3]`, cut `content`, then build
   - `search_results`: one string `"Query: q\nurl\ncontent\n---\n..."` (LLM context), and
   - `sources`: `[{"url","title","snippet"}]` with `snippet = content[:150]` (SSE `sources` event).
3. `main.py` de-duplicates `sources` by exact URL and emits them. `synthesize_node` reads only `search_results`.

**Where information was lost:** full content (truncated to 400/800), Tavily `score` and any other field, the query→source link, provider, retrieval time, and any result whose URL was empty was silently un-citable. Two unrelated formats (string and dict) described the same result.

**After this phase** the nodes are: Tavily → `normalize_tavily_response` → `ResearchResult` → project back to `search_results` / `sources` (unchanged shapes) **and** carry the documents in a new `source_documents` state key.

```
TavilySearch.ainvoke ──► normalize_tavily_response ──► ResearchResult(documents=[SourceDocument...])
   (raw dict / str /            (sources/tavily.py)            │
    {"error": exc})                  uses normalize_sources    ├─► search_result_entry()  ─► state["search_results"]  (LLM context, as before)
                                     (sources/normalizer.py)   ├─► source_to_legacy()     ─► state["sources"]         (SSE, as before)
                                                               └─► the documents          ─► state["source_documents"] (new; nothing reads it yet)
```

## Planned Changes (as implemented)
1. `SourceDocument` gains `provider` and `original_url` (optional; additive per the Phase 2 stability rule).
2. New package `research_app/sources/`: `normalizer.py` (generic), `tavily.py` (envelope), `__init__.py`.
3. `domain/legacy.py`: new `search_result_entry()` (documents → the `search_results` string).
4. `agent/state.py`: `source_documents` state key; both search nodes call one shared `_web_search_update()`.
5. Tests.

### Final normalized source contract
The normalized representation **is** the Phase 2 `SourceDocument` (no second model). Fields, and where each comes from:

| Field | Meaning / how it is filled |
|---|---|
| `source_id` | `sha256(canonical_url)[:32]`. Depends on the URL only (not on content, query or type). Stable across runs; pinned by a test. Scheme/host case, default port, trailing slash and fragment do not change it; `www.`, path case and query string do. |
| `source_type` | Given by the caller/adapter: `web`, `official_docs`, `github`, `reddit` (also `rag_document`, `cache` from Phase 2). A raw result never chooses its own type. |
| `title` | Whitespace-collapsed; `None` if absent/blank. Never invented. |
| `url` | Normalized: control chars and surrounding whitespace removed, scheme/host lower-cased, default port dropped, **path/query/fragment kept**. Absolute http(s) only. |
| `domain` | Lower-cased hostname, no port, `www.` **kept** (derived by the model). |
| `content` | Full text (never truncated here). `\r\n`→`\n`, trailing spaces and runs of 3+ blank lines tidied, NFC, NBSP/zero-width/control chars removed. **Spaces inside a line are kept** so code indentation survives. `None` if absent/blank/not a string. |
| `snippet` | Provider snippet if given (single-line), else the first 150 characters of the whitespace-collapsed content. |
| `author` | If a non-blank string was given; else `None`. |
| `published_at` | From `published_at` or `published_date`: datetime, ISO-8601 or RFC 2822 → aware UTC. Unparseable → `None`, and the raw value stays in `metadata`. |
| `retrieved_at` | Batch timestamp taken when the response is normalized (aware UTC), i.e. right after the provider call. |
| `provider` (new) | Who returned it, e.g. `"tavily"`; `None` if unknown. |
| `original_url` (new) | The URL exactly as the provider returned it (surrounding whitespace only trimmed). Not validated. |
| `query`, `task_id` | The search query / task that produced it (`task_id` is `None` today: the nodes have no task ids). |
| `metadata` | Every raw key not mapped above, verbatim (e.g. Tavily's `score`). A field with the wrong type (e.g. `title: 42`) is dropped from its typed slot but kept here. |
| `credibility`, `technology`, `version` | Left at defaults. Quality scoring is Phase 8; technology/version are Phase 4. |

Provenance questions and their answers: *where from* → `provider`; *source type* → `source_type`; *original URL* → `original_url`; *original title/content/snippet* → `title`/`content`/`snippet` (whitespace-tidied only, not truncated; the raw payload is not stored twice); *retrieved when* → `retrieved_at`; *for which query* → `query`.

### Raw result vocabulary (what `normalize_source` accepts)
A mapping with `url` (required) and optionally `title`, `content`, `snippet`, `author`, `published_at`/`published_date`. Anything else goes to `metadata`. A future adapter maps its payload into this shape and passes `source_type`/`provider`.

### `ResearchResult` from Tavily
| Situation | status | error.code |
|---|---|---|
| all results normalized (including an empty list) | `ok` | — |
| some results unusable and skipped | `partial` | — |
| every result unusable | `failed` | `no_valid_results` |
| `{"error": exc}` or a string (zero results / tool error) | `failed` | `provider_error` (message ≤ 200 chars) |
| not a mapping / no `results` list | `failed` | `invalid_response` |

Normalization never raises for provider output.

## Files Expected (actual)
- Create: `research_app/sources/{__init__,normalizer,tavily}.py`, `tests/test_source_normalizer.py`, `tests/test_tavily_normalization.py`, `tests/test_search_nodes.py`
- Modify: `research_app/domain/models.py` (+2 fields), `research_app/domain/legacy.py` (+`search_result_entry`, docstring), `research_app/agent/state.py` (state key, nodes), this document, `docs/phases/PHASE-02-CONTRACTS.md` (one status note)
- Delete: None
- **Not touched:** `graph.py`, `main.py`, `schemas.py`, `db/*`, `auth/*`, `vectordb.py`, `requirements.txt`, `chat.html`.

## Architecture Decisions
- **One model.** `SourceDocument` is the normalized source. Only what Step 6 (provenance) required was added: `provider`, `original_url`. `rank`, `score`-as-quality, `raw_content` etc. were not added.
- **Generic normalizer + thin per-provider adapter.** `normalizer.py` knows a small vocabulary, not Tavily. `tavily.py` only unwraps the response envelope. Phases 4–5 add an adapter each and reuse `normalize_source(s)`.
- **`sources/` depends only on `domain/`.** It does not import LangChain or `research_app.agent` (subprocess test), so it is unit-testable without Tavily/Groq.
- **Single path, not a parallel one.** The nodes normalize once and project to the old shapes (`search_result_entry`, `source_to_legacy`) so there is no second copy of the extraction logic to drift. The two copy-pasted node bodies became one helper.
- **`source_documents` added to `ResearchState`** with the same `operator.add` reducer as `sources`, holding `SourceDocument` objects. It is the hand-off for Phase 8; **nothing reads it yet**. `main.py` builds its inputs without this key and the graph runs fine (test). If unwanted, removing it touches one line plus one dict entry.
- **A result without a valid http(s) URL is dropped (and logged), not kept.** `SourceDocument.url` is required (Phase 2) and evidence must be traceable (CLAUDE.md Rule 9). URLs with embedded credentials are rejected so they can never reach citations, logs or `original_url`.
- **URL fragment and query are kept in `url`**; only `source_id` ignores the fragment. A docs anchor (`#installation`) is useful in a citation (Phase 4/10). Tracking-parameter stripping is deliberately not done; that is de-duplication territory.
- **`www.` is not stripped from `domain`.** One derivation (the model's) keeps every path consistent; the Phase 4 registry can match `www.`/subdomains itself.
- **Text cleaning is not `sanitize_text`.** The audit flagged `sanitize_text` for destroying code indentation and non-ASCII punctuation; the normalizer keeps both (tested).
- **No de-duplication, ranking or scoring**, per scope. Two raw results with the same URL yield two documents with the same `source_id`.
- **Errors:** an unknown `source_type` raises `ValueError` (a caller bug); bad *data* raises `NormalizationError` from `normalize_source` and is skipped+logged by `normalize_sources`. Log lines carry position and reason only, never raw content.

## Dependencies
- None added or removed. Uses stdlib (`unicodedata`, `email.utils`, `urllib.parse`) and the existing `pydantic`.

## Implementation Notes
- Run tests from the repo root: `python -m unittest discover -s tests -t .`
- `import research_app.sources` (not submodules) in new code.
- `MAX_WEB_RESULTS = 3` in `state.py` mirrors `TavilySearch(max_results=3)`; it is a safety cap applied to the raw list **before** filtering, exactly like the old `[:3]`.
- `state.py` uses CRLF line endings; the edits preserve them. New files use LF (as the domain package does).

### Deliberate behaviour differences (everything else is unchanged and tested against a copy of the old logic)
1. **Whitespace in text** sent to the LLM and shown as snippets is tidied (CRLF, trailing spaces, blank-line runs, NBSP/zero-width/control characters). SSE snippets are whitespace-collapsed (newlines become spaces).
2. **Results with no usable http(s) URL are dropped.** Before: a URL-less result with content still reached the LLM prompt, and any non-empty URL (even `javascript:`) reached the SSE `sources` event and the UI.
3. **URL text in prompts/sources** now has a lower-cased scheme/host and no default port.
4. A Tavily failure or zero-result search now logs one warning (`provider_error`); the output is still the empty `"Query: q\n"` entry as before.

## Manual Test Cases
### Test 1 — Existing research flow unchanged (needs live keys)
Input: Phase 1 manual tests 4–6 (simple question, complex question, repeat for cache hit) against a running server.
Expected: same SSE sequence (`progress` → `token` → `sources` → `done`), 3 sources per search, de-duplicated by URL, answers of similar shape. Server log shows no `Skipping result` warnings for normal queries.
Actual: **Not run** (needs Groq/Tavily credentials).

### Test 2 — Normalization from a shell
Input: `python -c "from research_app.sources import normalize_tavily_response as n; r=n({'results':[{'title':'T','url':'HTTPS://Docs.Python.org:443/3/#x','content':'a\r\n\r\n\r\nb','score':.5}]}, query='q'); d=r.documents[0]; print(r.status.value, d.url, d.domain, d.source_id, d.original_url, d.metadata)"`
Expected: `ok https://docs.python.org/3/#x docs.python.org b4365d45965be1c1c1af00c627b043aa HTTPS://Docs.Python.org:443/3/#x {'score': 0.5}`
Actual: Not run by hand; the same behaviour is covered by `test_source_normalizer` (ids/URLs) and `test_tavily_normalization` (`score` → metadata).

### Test 3 — Bad provider output does not kill a search
Input: in a shell, `n({'error': RuntimeError('boom')}, query='q')`, `n('No search results found', query='q')`, `n(None, query='q')`.
Expected: three `failed` results with `provider_error`, `provider_error`, `invalid_response`; no exception.
Actual: covered by `test_tavily_normalization`; not run by hand.

## Commands Run
```text
git status; git log --oneline; read CLAUDE.md-referenced phase docs, state.py, domain/*, graph.py, tests
.venv python -m unittest discover -s tests -t .              (baseline: 53 OK)
inspect .venv/Lib/site-packages/langchain_tavily (result shape, error handling)
.venv python -m unittest tests.test_source_normalizer        (53 OK)
.venv python -m unittest tests.test_tavily_normalization tests.test_search_nodes   (26 OK)
mutation checks x3 (files backed up and restored byte-for-byte, verified with cmp)
.venv python -m unittest discover -s tests -t .              (final: 132 OK)
.venv python: import research_app.main + compiled_graph -> routes / nodes / edges
git status --short; git diff --stat
```
All Python runs used `PYTHONDONTWRITEBYTECODE=1`.

## Test Results
- New: 79 tests — `test_source_normalizer` 53, `test_tavily_normalization` 15, `test_search_nodes` 11 (9 node tests + 2 compiled-graph tests).
- Full suite: **132/132 pass** (53 existing + 79 new). Baseline before changes: 53/53.
- Coverage of the requested cases: valid result, missing title/URL/content/snippet/optional metadata, URL/domain extraction, source-type assignment, stable IDs (incl. a pinned value), metadata preservation, independent normalization of many results, and malformed input (`None`, strings, lists, ints, wrong-typed fields, junk URLs, credentials, non-string keys).
- **Equivalence test:** `test_output_matches_pre_phase_3_behaviour_for_well_formed_results` runs both nodes against a copy of the pre-Phase-3 node body for 6 response shapes (normal, >3 results, empty content, empty list, `{"error": ...}`, string) and asserts identical `search_results` and `sources`.
- **Compiled graph, offline:** `ainvoke` with the exact input dict `main.py` builds (no `source_documents` key) → 3 parallel searches → 9 `sources` and 9 `source_documents`; `astream(stream_mode=["updates"])` updates serialise with `json.dumps` as `main.py` does. Tavily, Groq and the cache are replaced by fakes.
- Mutation checks (all caught, all reverted): removing the credentials check; changing `search_node`'s 800 limit; applying the result cap after filtering instead of before.
- Import/route/graph smoke: 16 application routes and 7 graph nodes / 10 edges, identical to the Phase 1/2 baseline (also asserted by `test_domain_isolation`).
- **Not run:** any live endpoint, Groq call, real Tavily call, or PostgreSQL.

## Security / Reliability Notes
- URL scheme allow-list (http/https), credentials rejected, control characters stripped. This is a shape check only; SSRF and prompt-injection defences remain Phase 14. Content is still untrusted and still interpolated into the synthesis prompt.
- One unusable result never affects the others; provider output cannot make the normalizer raise. Failure information is now structured (`ResearchError`) rather than silently empty, but it is only logged, not surfaced to the user.
- Log lines contain result position, reason and error *codes*, not raw content. `provider_error` messages (≤ 200 chars) hold the wrapper's own text and are stored on the result, not logged by the node.

## Known Issues
- `source_documents` is written but read by nothing yet; its real-world fit is unproven until Phase 8.
- **Tavily `answer`** (`include_answer=True`) and response-level fields (`response_time`, `images`) are still discarded. The answer is LLM-generated, not a source.
- **Pre-existing, not addressed (Phase 13):** `tavily_tool.ainvoke` can still raise for transport-level failures the wrapper does not catch, which kills the run; there is still no retry/timeout on the search call.
- `task_id` is always `None` on documents: the graph has no task objects until Phase 6.
- `published_at`/`author` are only populated if a provider sends them; the current Tavily configuration (general topic) does not, so they are `None` for web results today.
- The `SourceDocument` table in `PHASE-02-CONTRACTS.md` predates the two new fields; this document is authoritative for them.
- `original_url` is not validated and is trusted to be display/provenance only.
- Carried over: `.env` has `LANGSMITH_TRACING` enabled (conflicts with CLAUDE.md); PostgreSQL path untested; Python 3.14 "Pydantic V1" warning from `langchain_core`.

## Deferred to later phases
- Phase 4: official-docs adapter, `technology`/`version`, `is_official` credibility.
- Phase 5: GitHub/Reddit adapters and their metadata.
- Phase 6: task ids on documents, source routing.
- Phase 8: consuming `source_documents`; quality filter, reranking, de-duplication, evidence store, persistence (naive-UTC conversion).
- Phase 10: citations built from `source_id` instead of `[Source, Year]`; content limits (400/800) applied only at the prompt projection today.
- Phase 13: timeouts/retries and user-visible partial-failure reporting.
- Phase 14: SSRF, prompt injection, `chat.html` href scheme check.

## Git Commit
`phase-3: source normalization` (hash omitted on purpose; see `git log`). Preceded by `a4ede46 phase-2: domain contracts`.

## Next Phase
**Phase 04 — Official Documentation Source** (`docs/phases/PHASE-04-OFFICIAL-DOCS.md`): technology intent detection, official-domain registry, an official-docs adapter that reuses `normalize_source(s)` with `source_type="official_docs"`. Start in a fresh session.
