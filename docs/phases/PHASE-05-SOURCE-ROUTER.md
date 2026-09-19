# PHASE-05 — Source Router: Web + GitHub + Reddit (+ Official Docs)

## Status
In Progress — implemented and verified: 346/346 offline tests, 25/25 mutation checks killed, live Tavily manual tests done (7 questions). **Awaiting review and the git checkpoint** (nothing committed).

## Goal
The first deterministic source router. Four first-class source types — `official_docs`, `web`, `github`, `reddit` — chosen per question by rules, executed concurrently with failure isolation, all normalised into the Phase 3 `SourceDocument`.

## Scope
- Rule-based `route_sources(question) -> RoutePlan` (no LLM, no scoring).
- GitHub and Reddit as `SourceAdapter`s over the **existing single Tavily tool** (`include_domains=["github.com"]` / `["reddit.com"]`), with strict hostname verification of every result.
- Concurrent, failure-isolated execution inside the existing search nodes (graph topology unchanged).
- Minimal synthesis change: source labels + guidance, and a fair share of the 12-entry prompt window.
- Official docs (Phase 4) preserved; the router adds one case (technology + problem/troubleshooting).

## Out of Scope
Hybrid RAG, BM25, vectors, reranking, quality scoring, de-duplication, evidence store, gap detection, critic/retry, semantic cache, PostgreSQL, Redis, LangSmith, Docker, GitHub API, Reddit API, OAuth, scraping, another search provider, LLM router, prompt-injection defences (Phase 14), web timeouts/retries (Phase 13).

## Current Implementation
```
question ──► route_sources(question)         rules over the ORIGINAL question (sub-questions are searched with their own text)
              │   web: always │ official_docs: Phase 4 detector, or technology + problem/trouble
              │   github: code/repo wording, or technical problem │ reddit: experience/community, or problem/trouble
              ▼
   asyncio tasks, concurrent:   Tavily(query)                                   -> web (direct path, as before)
                                OfficialDocsAdapter (Phase 4, unchanged)         -> official_docs
                                GitHubAdapter: Tavily(query, include_domains=["github.com"])
                                RedditAdapter: Tavily(query, include_domains=["reddit.com"])
                                     └► normalize_tavily_response (Phase 3) ► hostname allowlist on EVERY result URL
                                        ► SourceDocument(source_type=github|reddit, URL-derived metadata, credibility)
              ▼
   merged: official_docs → github → reddit → web   (search_results / sources / source_documents)
   synthesis: labelled entries + guidance only for sources that returned verified documents
```

### Routing rules (`sources/routing/router.py`)
Web is always selected. Signals are regexes over the lower-cased question (first 2000 chars).

| Signal | Cues (abridged) | Adds |
|---|---|---|
| DOCS | Phase 4 `detect_docs_intent` (registry technology + technical cue, no opinion phrasing) | official_docs |
| CODE | "github", repo, pull request, source code, open source, code examples/samples; or example/sample/implementation/boilerplate **next to a registry technology** | github |
| EXPERIENCE | reddit/subreddit/r/x, "real-world experience", "what do people/users think", community opinions, "has anyone …" | reddit |
| PROBLEMS | problem/challenge/pain-point wording **and** developers/users/people | reddit; github if technical; official_docs if a registry technology is named |
| TROUBLE | error/exception/traceback/"not working"/fix wording **and** a technical anchor (registry technology incl. weak aliases, traceback-type token, or a `FooError` class name) | official_docs (non-weak registry technology), github, reddit |

Not cues: "best", "recommend", "top", opinion words alone. Weak aliases (`python`, `compose`, `openai`) never trigger documentation on their own (Phase 4 rule).

| Question | Route |
|---|---|
| How do I configure FastAPI? | official_docs + web |
| Find GitHub examples of LangGraph agents | github + web |
| What problems are developers facing with LangGraph? | official_docs + github + reddit + web |
| Why am I getting this LangGraph error? | official_docs + github + reddit + web |
| Best laptops for students | web |
| Real-world experiences using Qdrant | reddit + web |

### Domain validation (`sources/routing/hosts.py`)
Exact hostname allowlists: GitHub `github.com`, `www.github.com`; Reddit `reddit.com`, `www.reddit.com`, `old.reddit.com`. No suffix matching. Rejected: `github.com.evil.com`, `evilgithub.com`, `github.com@evil.com`, backslash/userinfo tricks, non-default ports, trailing-dot hosts, other subdomains (`docs.github.com`, `gist.github.com`, `api.github.com`), `github.io`, `githubusercontent.com`, `redd.it`, IDN/full-width look-alikes, non-http(s). Classification uses the URL only, never title or text. Off-domain results are **dropped, never relabelled**.

### Normalisation and metadata
Reuses `normalize_tavily_response` → `SourceDocument` (no second model). Each verified document keeps `source_type`, title, URL, content/snippet, domain, `provider="tavily"`, `original_url`, `query`, `task_id`, `retrieved_at`, Tavily's extra fields in `metadata`, plus `credibility=SourceCredibility(is_official=False, notes=…)` (no score). URL-derived, pattern-validated metadata: `metadata["github"] = {host, kind, owner, repo, number, ref, path}` (kind: repository / issue / pull_request / discussion / file / directory / release / wiki / commit / profile / other), `metadata["reddit"] = {host, kind, subreddit, post_id, comment_id}`. Labels are built from this metadata only, so result text cannot forge them.

### Failure isolation
- GitHub/Reddit/docs failures (exception, timeout, error dict, "no results" string, malformed items, off-domain-only, adapter crash, executor crash, request-preparation error, merge error) leave every other source and web untouched. Each is logged once by its adapter.
- A routing crash → web + Phase 4 docs check only.
- Empty results are a valid empty OK.
- **Web is unchanged**: a raised web exception still propagates (pinned by a Phase 4 test) and cancels the background searches; a swallowed Tavily error still degrades to an empty entry. Phase 13 owns web timeouts/retries.

### Synthesis (minimal)
- Entries: `[OFFICIAL DOCUMENTATION: …]` (Phase 4), `[GITHUB: owner/repo, issue #N]`, `[REDDIT: r/sub]`, and `[WEB]` — the web entry is labelled **only when GitHub/Reddit documents are present**; unlabelled = web. Web-only and docs+web outputs are byte-identical to Phase 3/4.
- `SOURCE LABEL GUIDANCE` is appended only when verified GitHub/Reddit documents exist (Reddit = anecdotal, never authority for API/config behaviour; GitHub = implementation evidence, not official support). Phase 4 guidance unchanged.
- The prompt reads at most 12 entries (9 in the fallback). Up to 12 entries: Phase 4 order (official first, then arrival order). Over the limit **and** GitHub/Reddit entries present: round-robin across official/github/reddit/web so no selected source is cut off. Otherwise exactly the Phase 4 slice.

## Files Changed
- **Created:** `research_app/sources/routing/{__init__,hosts,adapters,router,executor,settings}.py`; `tests/test_source_hosts.py`, `test_source_adapters.py`, `test_source_router.py`, `test_router_nodes.py`.
- **Modified:** `research_app/agent/state.py` (`_research_update` + helpers, guidance, `select_search_results`), `research_app/domain/legacy.py` (`community_entries`, `entry_kind`, `select_search_results`, `WEB_LABEL`, optional `label` on `search_result_entry`), `research_app/sources/official_docs/detector.py` (+ `mentioned_technologies`, additive) and its `__init__` export, `research_app/sources/__init__.py` (docstring), `.env.example`, `tests/test_domain_isolation.py` (routing package added to the import-isolation check).
- **Renamed:** `docs/phases/PHASE-05-GITHUB-REDDIT.md` → `PHASE-05-SOURCE-ROUTER.md`.
- **Not modified:** domain models/enums, the Phase 4 adapter/registry/planner, the graph, `main.py`, any existing test.
- **No new dependencies.** No new Tavily client (`tavily_tool` is the only one; a test asserts the routing package never builds one).

## Configuration
`SOURCE_ROUTER_ENABLED` (default true; false → exactly Phase 4: web + documentation detector only), `SOURCE_ROUTER_TIMEOUT_S` (default 15, 1–60, per GitHub/Reddit search). `OFFICIAL_DOCS_*` unchanged and still honoured (disabled → no documentation anywhere). Read at call time; bad values fall back to defaults.

## Architecture Decisions
1. Web always selected; the router only adds sources. A question with no signal behaves exactly as before.
2. Phase 4's docs planning call stays in `state.py` (Phase 4 tests pin `plan_official_docs`, `OfficialDocsAdapter`, `official_docs_entries`); the router *reports* the docs decision and adds only the technology+problem case, via `plan_technology_docs`.
3. No `WebAdapter`: web stays the direct Tavily path so its pinned failure semantics are unchanged. `RoutePlan` still lists web.
4. Hostnames are an exact allowlist (a miss costs nothing; a false accept gives an arbitrary site a trusted label).
5. Every sub-question of a research run re-routes from the **original** question (pure regex, no state-schema change).
6. Web label and 12-entry window changes are conditional so Phase 3/4 behaviour is byte-identical when GitHub/Reddit are absent.

## Manual Tests (live Tavily, real `_research_update`, no Groq; LangSmith forced off for the run)
| # | Question | Route | Tavily calls | Documents | Notes |
|---|---|---|---|---|---|
| 1 | Find GitHub examples of LangGraph agents | github + web | 2 (2.6 s / 2.8 s) | 3 github, 3 web | all `github.com`; kinds: topic page, notebook (file), repository |
| 2 | Real-world experiences using Qdrant | reddit + web | 2 (3.1 s / 3.7 s) | 3 reddit, 3 web | all `www.reddit.com`, kind `post`, subreddits r/Rag, r/vectordatabase |
| 3 | Why am I getting this LangGraph error? | docs + github + reddit + web | 4, concurrent, 4.4 s total | 2 official, 3 github, 3 reddit, 3 web | github kinds `issue #6446`, `#3421`; reddit r/LangGraph, r/LangChain; 1 docs result dropped (outside `/oss/python/langgraph`, Phase 4 behaviour) |
| 4 | Best laptops for students | web | 1 | 3 web | unlabelled web entry, as before |
| 5 | How do I install FastAPI? | official_docs + web | 2 | 3 official, 3 web | Phase 4 regression OK |
| 6 | GitHub Actions workflow examples on GitHub repos | github + web | 2 | 1 github, 3 web | **Tavily returned `docs.github.com` for `include_domains=["github.com"]`; 2 were dropped** (logged) — the strict allowlist works |
| 7 | Reddit discussion: is GitHub Copilot worth it | github + reddit + web | 3 | 3 github, 3 reddit, 3 web | explicit "reddit" wording |

Off-domain/lookalike community documents in every run: none. Latency of a routed question ≈ its slowest source (sources run concurrently).

## Commands Run
```text
python -m unittest discover -s tests -t .          # baseline 237 OK; final 346 OK
python -m unittest tests.test_source_hosts tests.test_source_adapters tests.test_source_router tests.test_router_nodes
mutation script (scratchpad): 25 deliberate bugs, one at a time, files restored byte-for-byte (sha256)
live script (scratchpad): python live_tavily.py [questions...]   # real Tavily via st._research_update
```

## Test Results
- **346/346** (237 existing, unchanged and green, + 109 new). No live call in the suite.
- New coverage: routing (brief examples + ~30 more, kill switches, garbage input), host validation (≈45 look-alike/garbage URLs), adapters (payload, classification, provenance, metadata, off-domain drops, empty/malformed, timeout, cancellation), executor (isolation, concurrency, cancellation), node integration (routed searches, ordering, concurrency, ten failure modes × three sources, web-failure propagation, task cleanup, forged-label resistance), synthesis (labels, guidance gating, window), compiled graph (nodes unchanged, `ainvoke`, `astream`), real API (`/api/research/stream` SSE contract, `/api/research` shape).
- **Mutation checks: 25/25 killed** (suffix host match, port/userinfo acceptance, skipped verification, missing `include_domains`, wrong `source_type`, no timeout, exceptions escaping, always-add-github, ignored kill switch, `best` as Reddit cue, sequential executor, no task cancellation, missing `[WEB]` label, wrong merge order, ignored guidance, routing by sub-query, ignored timeout, window round-robin off/always, label from result text, multi-line query).

## Security / Reliability Notes
- Domain check on every result; results outside the allowlist are dropped and logged (domains only, never content).
- Content from GitHub/Reddit is **untrusted** and enters the prompt like web content; labels are metadata-derived and the query line is single-line, but injected instructions inside page text are not defended against (Phase 14). Guidance tells the model Reddit/GitHub are not authority.
- Bounded concurrency: ≤ 4 searches per search node (≤ 12 per research run).

## Known Issues
- **Cost/latency:** a four-source question makes up to 4 Tavily *advanced* searches per node (12 per complex run vs 3 before). Kill switch: `SOURCE_ROUTER_ENABLED=false`.
- **Cross-source duplicates:** the same page can appear as `github`/`reddit` and as `web` (seen live), and GitHub discussions under two URL forms; de-duplication is Phase 8. The SSE `sources` event still de-duplicates by URL.
- **Subdomains dropped by design:** `gist.github.com`, `docs.github.com`, `old.reddit.com` allowed only where listed; Tavily does return `docs.github.com` for `include_domains=["github.com"]` (dropped; those remain available as web results).
- GitHub `orgs/...` and topic URLs get a bare `[GITHUB]` label (no owner/repo); label quality only, no ranking.
- **Keyword rules miss paraphrases** and over-trigger on some wording (e.g. "open source" → github). A miss falls back to web; tune in Phase 6.
- Routing runs per sub-question from the original question; the planner does not yet choose sources (Phase 6, now narrowed to planner integration).
- No quality scores, rerank or de-dup (later phases); web has no timeout (Phase 13); the round-robin window is a fixed 12/9.
- Carried over: `.env` may enable LangSmith tracing (conflicts with CLAUDE.md; the manual-test script forced it off); PostgreSQL path untested; Python 3.14 "Pydantic V1" warning.

## Git Commit
Not created yet. Recommended message: `phase-5: source router (web, github, reddit, official docs)`.

## Next Phase
**Phase 06 — Source Router and Planner Integration** (`docs/phases/PHASE-06-SOURCE-ROUTER.md`): the router itself now exists, so Phase 6 shrinks to letting the planner emit per-sub-question source choices/requirements and replacing `_research_update`'s call to `route_sources` with the planner's plan. Start in a fresh session.
