# PHASE-04 — Official Documentation Source

## Status
Completed — implemented and verified offline (237/237 tests, 15/15 mutation checks killed); git checkpoint `phase-4: official documentation source`.

(No live endpoint, Groq or Tavily call was made; see Test Results and Manual Test Cases.)

## Goal
Add first-class official documentation retrieval and technology-to-domain routing.

## Scope
- A registry (JSON data) mapping technologies to official hosts, path prefixes and version patterns; user-extensible.
- Rule-based detection of documentation questions (technology + technical cue), no LLM call.
- `OfficialDocsAdapter`: Tavily search restricted with `include_domains`, then verification of every result URL against the registry, then `SourceDocument(source_type=official_docs, ...)`.
- Integration inside the existing search nodes (graph topology unchanged); official docs placed first and labelled; minimal conditional synthesis guidance.

## Out of Scope
- GitHub, Reddit, the source router (Phase 5/6), hybrid RAG, reranking, de-duplication, evidence store, gap detection, critic, cache changes, PostgreSQL, Docker, observability, prompt-injection defences (Phase 14), page fetching / `TavilyExtract` (explicitly declined), LangSmith/evaluation.

## Current Implementation
Before this phase: one web source (`TavilySearch`, 3 results, advanced depth). Phase 3 turned its output into `SourceDocument`s and projected them onto `search_results` / `sources` / `source_documents`. The contracts already had `SourceType.OFFICIAL_DOCS`, `SourceDocument.technology/version`, `SourceCredibility.is_official`, `SearchRequest.include_domains` and the `SourceAdapter` protocol; nothing used them.

```
question ──► plan_official_docs()            (registry aliases + technical cue; no LLM call)
              │
   not a docs question ─────► web search only  (unchanged, byte-identical output)
   docs question
              ▼
     asyncio task ── OfficialDocsAdapter.search(SearchRequest(include_domains=[registered hosts]))
                        Tavily search restricted to those hosts
                        → normalize_tavily_response (Phase 3)
                        → DocsRegistry.match_url() on EVERY result   ← security boundary
                        → SourceDocument(official_docs, technology, version, is_official, provenance)
     awaited   ── tavily_tool.ainvoke({"query"})   (web, unchanged)
              ▼
     official docs first in search_results / sources / source_documents;
     synthesis prompt gets 3 lines of guidance only if verified official docs exist
```

## Planned Changes (as implemented)
1. New package `research_app/sources/official_docs/`: `registry.py`, `detector.py`, `adapter.py`, `priority.py`, `plan.py`, `settings.py`, `__init__.py`, `data/official_docs.json`.
2. `domain/legacy.py`: `official_docs_entries`, `is_official_docs_entry`, `prioritize_search_results`.
3. `agent/state.py`: `_research_update` (web + docs, concurrent, failure-isolated) used by both search nodes; `planner_router` passes the original `question` to each sub-question search; `_official_docs_guidance` in both synthesis prompts.
4. `.env.example`: `OFFICIAL_DOCS_*` names.
5. Tests.

## Files Expected (actual)
- Create: `research_app/sources/official_docs/{__init__,registry,detector,adapter,priority,plan,settings}.py`, `research_app/sources/official_docs/data/official_docs.json`, `tests/test_docs_registry.py`, `tests/test_docs_detector.py`, `tests/test_docs_adapter.py`, `tests/test_docs_nodes.py`
- Modify: `research_app/agent/state.py`, `research_app/domain/legacy.py`, `research_app/sources/__init__.py` (docstring only), `tests/test_domain_isolation.py` (+1 test), `.env.example`, this document
- Delete: None
- **Not touched:** `graph.py`, `main.py`, `schemas.py`, `db/*`, `auth/*`, `vectordb.py`, `sources/tavily.py`, `sources/normalizer.py`, `domain/models.py`, `requirements.txt`, `chat.html`.

## Registry
`research_app/sources/official_docs/data/official_docs.json`, 16 entries. Every host and path was verified on **2026-09-19**: fetched, or (OpenAI) found through a search restricted to the official domains because direct fetch returned 403.

| id | Official site(s) | Path prefixes | Version pattern |
|---|---|---|---|
| python | docs.python.org | — | `/3.14/`, `/3/` |
| fastapi | fastapi.tiangolo.com | — | — |
| uvicorn | uvicorn.dev | — | — |
| pydantic | pydantic.dev | `/docs/validation` | `/docs/validation/2.x/` |
| sqlalchemy | docs.sqlalchemy.org | — | `/en/20/` |
| alembic | alembic.sqlalchemy.org | — | — |
| langgraph | docs.langchain.com | `/oss/python/langgraph` | — |
| langchain | docs.langchain.com | `/oss/python/langchain` | — |
| qdrant | qdrant.tech | `/documentation` | — |
| postgresql | www.postgresql.org | `/docs` | `/docs/18/` |
| docker | docs.docker.com | — | — |
| docker-compose | docs.docker.com | `/compose`, `/reference/compose-file`, `/reference/cli/docker/compose` | — |
| tavily | docs.tavily.com | — | — |
| openai-api | developers.openai.com; platform.openai.com | `/api`; `/docs` | — |
| claude-api | platform.claude.com | `/docs` | — |
| groq | console.groq.com | `/docs` | — |

Why these: the project stack (FastAPI, Uvicorn, Pydantic, SQLAlchemy, Alembic, LangGraph, LangChain, Qdrant, PostgreSQL, Tavily, Groq), the Phase 4 examples (Docker Compose, PostgreSQL, OpenAI API), and Python itself.

Sites that **moved** (found during verification; only the current location is registered, because stale legacy pages would be labelled official but outdated): LangGraph and LangChain `langchain-ai.github.io/langgraph`, `python.langchain.com` → `docs.langchain.com/oss/python/...`; Pydantic `docs.pydantic.dev` → `pydantic.dev/docs/validation/`; Anthropic `docs.anthropic.com` → `platform.claude.com/docs`. `www.uvicorn.org` could not be resolved (DNS error), so only `uvicorn.dev` is registered. Version patterns are only for layouts seen in the fetched URLs (Alembic has none: its version layout was not seen).

Registry file format (`id`, `name`, `aliases`, `weak_aliases`, `sites[{host, path_prefixes, version_patterns}]`) is validated per entry; an invalid entry is skipped and logged, never fatal.

## Detection behaviour
A documentation query needs **technology alias** + **technical cue** and must not be opinion/comparison phrasing.
- Alias: word-boundary match on normalised text (lower-case, NFKC, `-`/`_` as space). Longest alias first and non-overlapping, so `docker compose` does not also yield `docker`.
- **Weak aliases** (`python`, `compose`, `openai`, `claude`, `anthropic`) need a *specific* cue (install, configure, syntax/API/CLI/options, version/changelog/migrate/upgrade); a bare "how do I" is not enough.
- Cues: specific (install/setup, configure/config/settings, syntax/parameters/options/flags/commands/API/SDK/CLI/reference/docs/official, version/release notes/what changed/migrate/upgrade/deprecated) and generic (`how do/can/to/does ...`).
- Vetoes: vs/versus, compare, alternatives, better than, pros and cons, popular(ity), best (not "best practices"), top N, reviews, worth it, choose/which is, "between X and".
- At most 3 technologies per question, in text order; input analysed up to 2000 characters.
- The original *question* decides; the sub-question text is what is searched. `search_node` gets the question through the `Send` payload (`planner_router` adds a `question` key; optional, falls back to `query`).
- Version hints in the question ("Pydantic 2") are **not** interpreted; `SourceDocument.version` comes only from the result URL, so it is never invented.

## Retrieval flow
1. `plan_official_docs` builds a `SearchRequest(source_type=official_docs, include_domains=<hosts of the detected technologies>, max_results=3, timeout_s=OFFICIAL_DOCS_TIMEOUT_S)`.
2. `OfficialDocsAdapter.search` calls Tavily through an injected callable (`state._tavily_search`, which delegates to the same `tavily_tool`) with `{"query", "include_domains"}`. No custom HTTP fetching; no `TavilyExtract`.
3. The response goes through Phase 3's `normalize_tavily_response` (typed `web`), then each document is verified; only verified documents are re-stamped `official_docs` with `technology`, `version`, `credibility.is_official=True`, and `metadata["official_docs"] = {technology_id, host, path_prefix}`. Tavily extras such as `score` stay in `metadata`.
4. Results: OK / PARTIAL (some dropped) / FAILED (`no_official_results`, `provider_error`, `invalid_response`, `no_valid_results`, `no_domains`) / TIMEOUT. The adapter never raises for source failures.
5. In `_research_update` the docs task runs concurrently with the web call and is merged afterwards.

## Security / domain validation (`DocsRegistry.match_url`)
Deliberately strict; every rule has a test and most have a mutation check:
- https only; no credentials; port must be 443/absent.
- Hostname must **equal** a registered host (leading `www.` ignored on both sides). No suffix/subdomain matching: `fastapi.tiangolo.com.evil.com`, `evilfastapi.tiangolo.com`, `docs.fastapi.tiangolo.com`, `evil.com/fastapi.tiangolo.com`, `fastapi.tiangolo.com@evil.com`, Unicode look-alikes and IPs are all rejected.
- Shared hosts use path prefixes on **segment boundaries** (`/oss/python/langgraphevil` is not under `/oss/python/langgraph`); paths are percent-decoded once and dot-segments resolved as a client would, so `/compose/../engine/` is Docker Engine, not Compose; backslashes, control characters and paths over 2048 characters are rejected.
- `allowed_hosts` = the hosts the search was restricted to: a registered official host that was not requested is rejected too.
- Off-domain results are **dropped**, never downgraded, and never labelled official. The label cannot come from provider data (test).
- Equal specificity resolves by registry order, so matching is deterministic.

## Priority behaviour
- Verified official docs get `source_type=official_docs` and `credibility.is_official=True`.
- Within a documentation question they come first in `search_results` (one entry per document, labelled `[OFFICIAL DOCUMENTATION: <Technology>[, version <v>]]` on line 2), in `sources` (SSE) and in `source_documents`; other sources keep their order (stable). The synthesis step also moves official entries to the front across parallel searches (`prioritize_search_results`).
- Entries are capped at 1500 characters (what `synthesize_node` reads), content at 1000.
- Not a ranking or reranking and not de-duplication (Phases 7-8).

## Synthesis guidance (minimal)
Only when at least one `SourceDocument` in `source_documents` is `official_docs`, both synthesis prompts (main and fallback) get:
> OFFICIAL DOCUMENTATION GUIDANCE: Some research data blocks are labelled [OFFICIAL DOCUMENTATION: ...]. For installation, API usage, configuration, technical behavior and version-specific facts, prefer those blocks over all other sources. If another source conflicts with them on these points, follow the official documentation.

Otherwise the prompts are byte-identical to before (test). Label text inside web content does not enable it (test). No citation mapping (Phase 10).

## Failure behaviour
Official docs can only *add* results. Covered by tests: search exception, Tavily error dict, "no results" string, `None`/malformed response, empty list, malformed items, off-domain-only results, timeout (bounded by `OFFICIAL_DOCS_TIMEOUT_S`), planner/registry failure, adapter crash, merge failure, unreadable user registry (built-in still works), unreadable built-in registry (empty registry, feature off). In each case the node output equals the web-only output. A **web** failure still propagates exactly as before and cancels the docs task. `asyncio.CancelledError` is never swallowed.

## Configuration
| Variable | Default | Meaning |
|---|---|---|
| `OFFICIAL_DOCS_ENABLED` | `true` | `false/0/no/off` = web only (kill switch) |
| `OFFICIAL_DOCS_TIMEOUT_S` | `15` | 1-60; invalid values fall back to 15 with a warning |
| `OFFICIAL_DOCS_REGISTRY_PATH` | unset | JSON merged over the built-in registry (same `id` replaces, new ids appended); cached per path, so a change needs a restart |

Read at call time; a bad value never stops startup.

## Architecture Decisions
- **No new graph node / no topology change** (per decision). The integration seam is `state._research_update` + `plan_official_docs`; Phase 6 can replace those two without touching the registry, detector or adapter.
- **Adapter is request-driven** (`SearchRequest.include_domains`), stateless, and takes the search function as a dependency, so `sources/` stays LangChain-free (subprocess test extended) and Phase 6 can build requests from `ResearchTask.preferred_domains`.
- **Web-normalise then verify then re-stamp.** `sources/tavily.py` is untouched; a document only becomes `official_docs` after `match_url` accepts it. No unverified document ever carries the type.
- **Exact host match, no wildcards.** Safer and deterministic; a subdomain must be listed explicitly.
- **Legacy/moved hosts are not registered** (stale content would be labelled official).
- **Registry order is precedence**; alias conflicts keep the first owner (logged).
- **No `ResearchQuery` wiring.** `ResearchQuery.technology` / `is_documentation_query` exist but the graph does not build a `ResearchQuery`; `DocsIntent` carries the same information for when Phase 6 does.
- `technology` holds the registry display name (e.g. `Docker Compose`); the stable id is in `metadata["official_docs"]["technology_id"]`.
- No dependency added: JSON (not YAML) for the registry.

## Dependencies
None added or removed. Uses stdlib `json`/`re`/`asyncio`/`dataclasses` plus existing `pydantic` and `langchain_tavily` (`include_domains` is a supported call-time parameter of the installed 0.2.18).

## Implementation Notes
- Run tests from the repo root: `python -m unittest discover -s tests -t .`
- `import research_app.sources.official_docs` explicitly; it is not re-exported from `research_app.sources`.
- `state.py` is CRLF; it was patched with a CRLF-preserving script. New files are LF.
- Tests that touch `state.py` replace `tavily_tool` and `llm_groq` with fakes; a fake distinguishes documentation searches by `include_domains` in the payload.

## Manual Test Cases
Live runs need Tavily and Groq keys and were **not** run. "Offline" = the real detector, registry and request builder executed in a shell (no network), reported honestly below. For live runs: start `uvicorn research_app.main:app`, log in, POST `/api/research/stream`, and check the SSE `sources` event and the server log line `Documentation query: technologies=...`.

| # | Input | Expected | Offline result (executed 2026-09-19) | Live |
|---|---|---|---|---|
| 1 | How do I install FastAPI? | docs query; official docs from fastapi.tiangolo.com first, labelled official; web results after | docs=True, tech `fastapi`, cues install+how_to, `include_domains=['fastapi.tiangolo.com']` | Not run |
| 2 | How do I use LangGraph? | docs query; pages under docs.langchain.com/oss/python/langgraph | docs=True, `langgraph`, `['docs.langchain.com']` | Not run |
| 3 | How do I configure Qdrant? | docs query; qdrant.tech/documentation | docs=True, `qdrant`, `['qdrant.tech']` | Not run |
| 4 | How does PostgreSQL CREATE INDEX work? | docs query; postgresql.org/docs, version from URL (e.g. 18) | docs=True, `postgresql`, `['www.postgresql.org']` | Not run |
| 5 | How do I use Docker Compose? | docs query for Compose (not also Docker); docs.docker.com/compose | docs=True, `docker-compose` only, `['docs.docker.com']` | Not run |
| 6 | best laptops 2026 | web only, one Tavily call, output as before | docs=False, reason `no_technology` | Not run |
| 7 | FastAPI vs Flask | web only | docs=False, reason `vetoed:versus` | Not run |
| 8 | What changed in Pydantic 2? | docs query; pydantic.dev/docs/validation | docs=True, `pydantic`, cue `version`, `['pydantic.dev']` | Not run |

Things to look for in live runs: (a) whether Tavily accepts `include_domains=['www.postgresql.org']` (the host is passed exactly as registered) and returns results; (b) how many results survive prefix filtering on shared hosts (LangGraph/LangChain on docs.langchain.com, Docker/Compose on docs.docker.com); (c) that the answer's citations name the official docs; (d) latency of a docs question versus a web-only one.

## Commands Run
```text
git status; git diff --stat; python -m unittest discover -s tests -t .   (Phase 3 verification: 132 OK)
git commit "phase-3: source normalization"                               (6951783)
WebFetch / WebSearch of every candidate official host and path (2026-09-19)
python -m unittest tests.test_docs_registry tests.test_docs_detector tests.test_docs_adapter tests.test_docs_nodes tests.test_domain_isolation
python <scratchpad>/mutate.py                                            (15 mutations, files restored and hash-checked)
python -m unittest discover -s tests -t .                                (final: 237 OK)
python <scratchpad>/smoke.py   (import research_app.main, routes, compiled graph, 8 offline manual queries)
git status --short; git diff --stat
```
All Python runs used `PYTHONDONTWRITEBYTECODE=1`.

## Test Results
- Baseline before Phase 4 (after the Phase 3 commit): 132/132.
- New: `test_docs_registry` 28, `test_docs_detector` 15, `test_docs_adapter` 31, `test_docs_nodes` 30, plus 1 in `test_domain_isolation` (sources/official_docs importable without LangChain/LangGraph/agent code).
- **Full suite: 237/237 pass**, no network.
- Import/route/graph smoke: 16 application routes, 7 graph nodes, 10 edges, identical to the Phase 1-3 baseline (also asserted in `test_domain_isolation`).
- Compiled graph offline (`ainvoke` and `astream` with `main.py`-shaped inputs, fake Tavily/LLM): complex documentation question runs 3 web + 3 docs searches, 15 sources, official first in the prompt, guidance present; non-documentation question makes no docs call and its prompt has no official text; SSE updates stay JSON-serialisable.
- **Mutation checks: 15/15 killed** (exact-host matching, path-segment boundary, `allowed_hosts`, https-only, credentials, dot-segment resolution, adapter verification, opinion veto, weak-alias rule, docs-before-web ordering, guidance gating, planner failure isolation, concurrency, question-vs-sub-query detection, entry-size cap). One initially survived (entry cap); a direct formatter test was added and it is now killed. All files restored byte-for-byte (sha256 verified).
- **Not run:** any live endpoint, Groq call, real Tavily call, PostgreSQL.

## Known Issues
- **No live verification.** Tavily's handling of `include_domains` with these hosts, result quality after path-prefix filtering, and end-to-end answers are unverified.
- **Shared hosts yield fewer results.** Tavily restricts by host only; on `docs.langchain.com` and `docs.docker.com` some returned pages fall outside the technology's prefixes (they may still match a sibling entry, e.g. LangChain for a LangGraph question) or are dropped.
- **Duplicate documentation across sub-questions** (each of the up to 3 searches may return the same pages); de-duplication is Phase 8. Synthesis reads at most 12 entries and 1500 characters each; docs entries are sized for that cap (Phase 10 to lift it).
- **Added latency** on a documentation question is bounded by `OFFICIAL_DOCS_TIMEOUT_S` (docs run concurrently with web but are awaited after it). The web call itself still has no timeout (Phase 13).
- **Semantic cache** can serve a stale answer to "latest version" questions and is not source-aware (Phase 12).
- **Content is untrusted** and goes into the prompt like web content; the domain check reduces but does not remove that risk (Phase 14). The guidance text does not defend against injected instructions.
- Detection is conservative by design: paraphrases without a listed cue ("tell me about setting up X") fall back to web; queries naming a technology not in the registry are web-only until it is added.
- Registry accuracy depends on hosts staying put; a moved site returns nothing official (web still answers). `OFFICIAL_DOCS_REGISTRY_PATH` changes need a restart.
- Only `www.` is treated as equivalent to the bare host; other subdomains must be registered.
- Version comes only from the URL; unversioned sites (FastAPI, Qdrant, ...) have `version=None`.
- Carried over (not addressed): `.env` has `LANGSMITH_TRACING` enabled (conflicts with CLAUDE.md); PostgreSQL path untested; Python 3.14 "Pydantic V1" warning from `langchain_core`.

## Git Commit
Committed as `phase-4: official documentation source` (see `git log`). Phase 3 was committed first as `6951783 phase-3: source normalization`.

## Next Phase
**Phase 05 — GitHub and Reddit Sources** (`docs/phases/PHASE-05-GITHUB-REDDIT.md`): two more adapters that reuse `normalize_source(s)` and the `SourceAdapter` protocol, with source-specific metadata and no equivalence to official docs. Phase 6 then replaces `_research_update`/`plan_official_docs` with the real source router. Start in a fresh session.
