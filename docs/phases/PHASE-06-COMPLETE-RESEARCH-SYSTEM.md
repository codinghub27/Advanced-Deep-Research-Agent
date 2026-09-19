# PHASE-06 — Complete Research System
Planner integration + research pipeline + conversation persistence (+ the sources/router display in the chat UI)

## Status
In Progress — implemented and verified: **503/503 offline tests** (the 346 existing tests are green; 157 new), UI display logic checked in a DOM emulator, live manual tests run against real Tavily + Groq + PostgreSQL (see Manual Tests). **Awaiting review and the git checkpoint** (nothing committed for this phase; Phase 5 was committed first as `924b99e`).

## Goal
A question gets a cited, multi-source research answer; follow-ups understand the conversation; every turn is stored; the chat UI shows which sources the router searched and which sources the answer cites.

## Scope
- **A — source-aware planner:** the planner proposes sources per sub-question; a deterministic router (`route_task`) validates and enforces.
- **B — research pipeline:** in-memory evidence pool → one gap round → evidence-only synthesis with `[N]` citations → critic → at most one retry.
- **C — conversation persistence:** sessions, turns and citations in the database (Alembic), context-aware query understanding and planning, session endpoints.
- **UI:** cited-source chips, a "N sources" footer, a sources/search-plan panel and a live router panel, with built-in symbols only.

## Out of Scope
Qdrant writes, embeddings, semantic-cache redesign (Phase 12), BM25/hybrid/rerank, evidence persistence, cross-session context, multi-round research, GitHub/Reddit APIs, new providers, prompt-injection defence, streaming redesign, Docker, Redis, LangSmith.

## Current Implementation

### Graph (existing nodes reused; `synthesize_node` is no longer wired)
```
START → semantic_cache_node ─hit→ END (unchanged)
   └miss→ classify_node  (= query understanding: follow-up, resolved query, intent, technology)
            ├simple→  simple_search_node ─────────────────┐
            └complex→ planner_node ─Send→ search_node ×N ──┤   per-sub-question routing (route_task)
                                                            ▼
        evidence_collection → gap_detection ─gaps→ targeted_search ─┐   (one round)
                                   └──────────── sufficient ────────┤
                                                                    ▼
        synthesis_node → critic_node ─bad/high→ retry_node ─┐          (at most one)
                              └────────── good ─────────────┤
                                                            ▼
                     format_response → save_to_cache_node → END
```
There is no edge back to an earlier node: the gap round and the retry are bounded by the topology. `format_response` is the only node that emits `final_answer` (the SSE layer streams every one it sees). Conversation loading and saving live in the **route layer**, not the graph (decision D8).

### Part A — planner and router
- Planner output: per sub-question `text`, `source_intent` (8 values), `suggested_sources`, `technology` (`SourceAwarePlan`; all fields required so the JSON schema is strict-mode safe). Legacy output (`sub_questions: list[str]`) still works.
- `route_task(task, understanding) -> RoutingDecision` (`sources/routing/task_router.py`): pure, deterministic, no LLM. Reuses `detect_signals` / `route_sources` (Phase 5).
  - Sources come from the valid planner suggestions (origin `planner`), else the intent defaults (`policy`), else the Phase 5 route over the whole resolved question (`fallback`).
  - Rules for planner-led decisions: (1) registry technology + technical task ⇒ `official_docs` enforced; Reddit never substitutes; (2) no technical signal ⇒ web only (Reddit kept only where Phase 5 selects it); (3) `MAX_SOURCES_PER_TASK` cap trims Reddit, then GitHub — docs and web are never trimmed; (4) `SOURCE_ROUTER_ENABLED=false` / `OFFICIAL_DOCS_ENABLED=false` honoured.
  - A hint-less task gets the Phase 5 route unchanged (a four-source route is not trimmed).
  - `RoutingDecision`: `task_id`, `sub_question`, `source_intent`, ordered `sources`, `origins`, `dropped` (with reasons), `stage` (search / gap_search / retry) and, after execution, per-source `outcomes` (status, result count, latency, error code).
- A sub-question inherits the run's technology only when it has none of its own and is technical or has no intent (`task_technology`): "Best laptops" in a LangGraph session stays web-only.
- Execution: same single Tavily client, Phase 5 failure isolation, and one process-wide semaphore (`CONCURRENCY_LIMIT`) around every Tavily call.

### Part B — pipeline
- `EvidencePool` (in memory, per request): documents grouped by sub-question, exact duplicates (same URL + same text) removed while keeping the link to every sub-question that found it, coverage per sub-question (a document counts when it contains ≥ `RELEVANCE_THRESHOLD` of the sub-question's key words).
- Gap detection: zero evidence ⇒ gap ⇒ up to `MAX_TARGETED_QUERIES` keyword follow-up queries (no LLM); a sub-question answered by one source type only is a reported soft gap, not a trigger. One round.
- Synthesis: the LLM writes plain text and cites numbered sources; **code** builds the `SynthesisResult` — markers found outside code, renumbered by first use, citations listed, `confidence` derived by rule, covered/missed sub-questions from the pool. Nothing depends on JSON output from the model. No evidence ⇒ acknowledgment, low confidence, and the model is not called. Model failure ⇒ source list, low confidence.
- Critic: deterministic checks (orphan citation numbers, uncited paragraphs, technical answer citing only Reddit, retrieved-but-uncited official docs, sub-questions with no evidence) + one LLM check (coverage, unsupported claims, missing information, suggested searches). Verdict `good | needs_improvement | bad`. If the LLM check fails, the deterministic verdict stands.
- Retry (once): for `bad`, or `needs_improvement` with a high-severity issue. It searches the critic's suggested queries (if any are new) and re-writes with the critic's findings in the prompt; a retry with nothing new to search still re-writes from the same evidence. The better-reviewed attempt is kept (the retry wins ties). A second `bad` verdict returns the best result with a limitations note and low confidence.
- `format_response`: removes dead `[N]` markers from the text, appends the limitations note when the final verdict is `bad`, settles confidence, and produces the citation list.

### Part C — persistence
- **Schema (Alembic, D1/D5):** `sessions` extended (`updated_at`, `is_active`); new `research_conversations` (UUID PK, `UNIQUE(session_id, turn_number)`, `ON DELETE CASCADE`) and `research_citations`. Session ids stay integers (`users.id` is an integer). Migrations `0001_baseline` (adopts existing installs) and `0002_conversations`, both guarded and idempotent; startup runs `upgrade head` instead of `create_all`. SQLite gets `PRAGMA foreign_keys=ON`.
- **Sessions:** no `session_id` ⇒ a new session (title = first 80 characters of the question, no LLM), id returned; unknown / another user's / soft-deleted session ⇒ 404 (D3).
- **History:** the route loads the last `CONVERSATION_HISTORY_LIMIT` turns into a `ConversationContext` (answer *summaries* only, ≤ `ANSWER_SUMMARY_LENGTH` chars; topics and technologies extracted without an LLM).
- **Query understanding / planner:** one LLM call each (no extra calls). With history the prompts carry the conversation; the model is not trusted — no history ⇒ never a follow-up; an empty or runaway rewrite ⇒ the original question.
- **Saving (D8):** the legacy `queries` row is still written before the response (now with `sources = {citations, routing}`); the conversation turn and its citations are saved **after** the response by a FastAPI background task with its own session. Any failure is logged (`Conversation save FAILED …`) and never reaches the user.
- **Cache (D6):** lookup is skipped when the session has earlier turns; follow-up answers and answers without citations are not cached; the cache keeps citations + routing next to the answer, so a hit shows its sources.
- **Endpoints (D4):** `GET /sessions` (`limit`, `offset`, default 20), `GET /sessions/{id}` (turns + citations), `DELETE /sessions/{id}` (soft). `/api/sessions/*` unchanged (hidden when inactive); `/admin/clear-all-data` now deletes conversations and citations first.

### API and SSE
- `/api/research` response: the three legacy keys (`sub_questions`, `answer`, `session_id`) plus `turn_number`, `is_follow_up`, `resolved_query`, `citations`, `sources_consulted`, `subquestions`, `confidence`, `critic_verdict`, `research_metadata` (`total_sources_found`, `total_evidence_pieces`, `gap_search_performed`, `retry_performed`, `sources_failed`, `routing_decisions`, `limitations`, `critic_issues`, `cache_hit`, `latency_ms`).
- SSE: the original events are unchanged (`progress`, `cache_hit`, `token`, `sources`, `done`). Additive: `routing` (per search node, live), `citations` (before the tokens), and extra fields on `done` (`session_id`, `turn_number`, `is_follow_up`, `resolved_query`, `confidence`, `critic_verdict`).

### Chat UI (`templates/chat.html`)
- **Source citations display (add-on, replaces the first chip/popover version):** `[N]` markers in the answer become small superscript links coloured by source type (hover: title — domain; click or Enter: opens this answer's panel, scrolls to the card, flashes it for 2 s). Under each answer a **collapsible "📄 Sources (N)" panel** lists one card per citation: `[N]`, letter avatar, title, domain, coloured badge (OFFICIAL DOCS green, WEB indigo, GITHUB dark, REDDIT orange); the card opens the URL in a new tab. Collapsed by default, open when there is a single citation, absent when there are no citations. Unknown source types render as web. Panel ids and lookups are scoped per answer, so several answers in one chat do not interfere.
- Below the panel, when the router ran: a **"Search plan" button** (side drawer; bottom sheet on phones: per sub-question the sources chosen, why, results, failures, dropped suggestions) and a "Searched: Official docs · GitHub · Web" line. A **live router panel** fills in inside the progress card as each search finishes. Reloaded chats and cache hits show the same.
- **No favicon service and no request to any site (kept from the first version, decision D12):** four inline SVG symbols for the "Searched" line and coloured letter circles for sites. Text goes through `textContent`/`esc()`, URLs must be http(s), Escape closes the drawer, "Copy Report" keeps the `[N]` markers. Light/dark follows the page's theme toggle (`[data-theme]`), not `prefers-color-scheme`.
- **SSE `sources` event (additive):** still `{type:'sources', sources:[legacy list]}`; when the answer has citations it also carries `citations` (the full numbered list, once, after the last token) and `total`. Cache hits with stored citations now emit it too. With no citations the two keys are absent; with neither citations nor legacy sources no event is sent. The earlier `citations` event before the tokens is unchanged.

## Files Changed
- **Created:** `domain/pipeline.py`; `sources/routing/task_router.py`; `agent/pipeline/{__init__,settings,evidence,gaps,search,synthesis,critic,nodes,response}.py`; `conversation/{__init__,settings,context}.py`; `db/{conversations,migrate}.py`; `db/migrations/{env.py,script.py.mako,versions/0001_baseline.py,versions/0002_conversations.py}`; `alembic.ini`; `research_service.py`; `tests/{p6_helpers,test_p6_routing,test_p6_pipeline,test_p6_conversation}.py`; `tests/ui/check_sources_ui.js`.
- **Modified:** `agent/{graph,state,vectordb}.py`, `db/{models,database,crud}.py`, `domain/{enums,models,legacy,__init__}.py`, `sources/routing/{__init__,settings}.py`, `main.py`, `schemas/schemas.py`, `templates/chat.html`, `.env.example`.
- **Tests adapted (deliberately, listed so nothing is silent):** `tests/base.py` (test DB for the post-response save), `test_docs_nodes.py` (fake LLM knows the new understanding schema), `test_domain_isolation.py` and `test_router_nodes.py` (graph node set, the 3 new routes, response/SSE additions), `test_ownership.py` (no `session_id` ⇒ new session, D2).
- **Renamed:** `PHASE-06-SOURCE-ROUTER.md` (empty template) → `PHASE-06-COMPLETE-RESEARCH-SYSTEM.md`.
- **No new dependencies** (`alembic` was already in `requirements.txt` and installed).

## Configuration
`MAX_SOURCES_PER_TASK` (3), `CONCURRENCY_LIMIT` (6), `GAP_SEARCH_ENABLED` (true), `MAX_TARGETED_QUERIES` (2), `MAX_CRITIC_RETRIES` (1; 0 disables), `RELEVANCE_THRESHOLD` (0.3), `CONVERSATION_HISTORY_LIMIT` (10), `ANSWER_SUMMARY_LENGTH` (300), `SESSION_TITLE_MAX_LENGTH` (80). All read at call time; bad values fall back to defaults with a warning. Documented in `.env.example`. Phase 4/5 switches (`OFFICIAL_DOCS_*`, `SOURCE_ROUTER_*`) still apply.

## Architecture Decisions (approved)
D1 extend `sessions` (int ids), new UUID tables for turns/citations · D2 no `session_id` ⇒ new session · D3 wrong-user/missing ⇒ 404 · D4 `/sessions` routes added, soft delete · D5 Alembic replaces `create_all`, SQLite FK pragma · D6 cache bypass for history, no caching of follow-ups/uncited, extras stored · D7 LLM writes text, code assembles `SynthesisResult`; critic = deterministic + one LLM check · D8 load/save in the route layer (background task), not graph nodes · D9 cap trims Reddit then GitHub, docs + web protected · D10 additive SSE (`done` fields, `routing`, `citations`) · D11 Phase 5 committed first, old `synthesize_node` kept unwired.

Also decided while building: hint-less tasks route on the resolved question exactly as Phase 5 did (so the cap applies to planner-led decisions only); a retry may re-write without searching; "no results" codes are reported as `empty`, not `failed`; `sanitize_text` now leaves fenced code untouched (Phase 1 finding).

## Manual Tests
Run with **real Tavily + real Groq + real PostgreSQL 18** (a throwaway database, dropped afterwards; your `langGraph_db` was not touched), through the real FastAPI app (`TestClient` with startup events), LangSmith forced off. The scratch database started with the **old three-table schema and a row**, so the startup migration ran on an existing install.

**Migration on Postgres (existing install):** startup upgraded it to `0002_conversations`; the legacy user/session/query rows were kept, `sessions.updated_at` was backfilled and `is_active` defaulted to true; the two new tables were created.

| # | Question | Sub-question routing (sources) | Evidence | Gap / retry | Critic | Citations (types) | Latency |
|---|---|---|---|---|---|---|---|
| 1 (stream) | How do I install and configure FastAPI with uvicorn? | install: docs+web · configure: docs+web · troubleshooting: docs(router)+reddit+web | 15 | – / – | needs_improvement* | 11 (docs, reddit, web) | 34 s |
| 2 | Now how do I add JWT authentication to it? | docs+github+web ×2, docs+web | 24 | – / retry | needs_improvement* | 6 (docs, web) | 123 s |
| 3 | What about OAuth2 instead? | docs+github+web, docs+web, docs(router)+github+web (reddit dropped: cap) | 21 | – / – | needs_improvement* | 10 (docs, github, web) | 101 s |
| 4 | How does Docker Compose work? | docs+web ×2, docs(router)+reddit+web | 20 | – / – | needs_improvement* | 12 (docs, reddit, web) | 96 s |
| 5 | Find LangGraph agent examples and explain the ReAct pattern | docs+**github (timeout)**+web, docs+web, reddit+web | 15 | – / – | needs_improvement* | 10 (docs, reddit, web) | 85 s |
| 6 | What problems are developers having with Qdrant in production? | docs(router)+github+web ×3 (reddit dropped: cap) → **fixed, see re-run** | 22 | – / retry | needs_improvement* | 11 (docs, github, web) | 125 s |
| 7 | Best laptops for college students | web, reddit+web (the planner wrote a "Reddit threads…" sub-question), web → **fixed, see re-run** | 12 | – / – | needs_improvement* | 11 (reddit, web) | 74 s |
| 8 | Compare FastAPI and Django: setup steps and community opinions | FastAPI setup: docs+github+web · Django setup: **web only** (official_docs dropped: `technology_not_in_registry`) · opinions: reddit+web | 16 | – / – | needs_improvement* | 10 (docs, github, reddit, web) | 77 s |
| 9 | Tell me everything about LangGraph | docs+web, docs(router)+github+web, reddit+web | 20 | **gap search** fired / – | needs_improvement* | 8 (docs, github, web) | 76 s |

\* before the critic calibration described below: all nine were `needs_improvement`.

- **Test 1:** session created, `session_id` returned; SSE order verified (`routing` and `citations` before the first `token`, `sources` then `done` last, `done` carries `session_id`/`turn_number`); rows created: 1 session, 1 `queries`, 1 `research_conversations`, 11 `research_citations`; official docs cited (FastAPI tutorial).
- **Tests 2–3 (follow-ups):** `is_follow_up = true`; resolved to "How do I add JWT authentication to FastAPI?" and "How do I add OAuth2 authentication to FastAPI?" (the "instead of JWT" nuance was not kept); the planner's sub-questions were about JWT / OAuth2 only, **none re-researched installation**.
- **Test 4 (topic change):** `is_follow_up = false`, resolved query unchanged, researched independently.
- **Test 5:** GitHub timed out after 15 s: recorded as an outcome (`github:timeout`, `sources_failed`), every other source and sub-question completed, answer produced. Failure isolation confirmed live.
- **Test 8:** sub-questions routed differently as required; Django is **not in the official-docs registry**, so it was routed to web and the drop was reported (add it with `OFFICIAL_DOCS_REGISTRY_PATH` / `official_docs.json`).
- **Test 9:** one gap round ran, once.
- **Test 10 (history of Tests 1–4):** 4 turns, `turn_number` 1–4, each with answer text and citations (11, 6, 10, 12); resolved queries differ from the raw text on turns 2 and 3; the legacy `queries` rows carry the same citations for chat reload; `GET /sessions` listed the sessions.
- **Totals:** 9 turns → `research_conversations` 9, `research_citations` 89, `queries` 10 (1 legacy + 9).

**What the live run changed (fixed, unit-tested, and re-run):**
1. *Test 7:* the router let Reddit through because the planner had written "Reddit" into its own sub-question. Now Reddit is allowed for a non-technical question only if the **user's** question asks for community views.
2. *Test 6:* the cap always trimmed Reddit first, so a question literally about developers' problems lost Reddit. Now GitHub is trimmed first when the user's question asks for community experience.
3. *Critic:* nine of nine verdicts were `needs_improvement` because the LLM reviewer flags something on every long answer. One or two flagged claims and "missing information" items are now low severity (still listed, and still feed the retry queries); three flagged claims, or an answer that misses the question, stay medium/high.
4. *Outcomes:* `no_official_results` / `no_domain_results` / Tavily's "No search results found" are reported as `empty`, not `failed` (a first-run Test 1 showed a false red "failed" for docs).

RERUN_PLACEHOLDER

### Citations display: browser checks still to do (not verifiable from the CLI)
Panel collapsed by default and expands on click · badge colours per type · clicking `[N]` scrolls to and flashes the card and opens a collapsed panel · card opens the URL in a new tab · dark and light theme (toggle) · phone width (cards stack, badge wraps, avatar hidden under 480 px) · no citations → no panel · an old answer without citations renders normally · reload a session and a cache hit show the panel · two answers in one chat behave independently.

Automated: `python -m unittest discover -s tests -t .` → **510 OK** (503 + 7 in `tests/test_citations_display.py`); `node tests/ui/check_sources_ui.js` (jsdom, scratch install) → all checks pass.

## Commands Run
```text
python -m unittest discover -s tests -t .      # baseline 346 OK → 503 OK
node tests/ui/check_sources_ui.js               # 28 display checks (jsdom, not a project dependency)
manual_e2e.py (scratchpad)                      # real Tavily + Groq + PostgreSQL scratch DB; LangSmith forced off
```

## Test Results
- **503/503** offline (deterministic; no live Tavily, LLM or database). New: routing/planner/understanding (49), pipeline (61), persistence/API/cache (52+).
- Real bugs the tests caught while building: citation timestamps arrived as ISO strings and the `DateTime` column rejected them (every real save would have failed — logged, never a 500); the run-level technology leaked into unrelated sub-questions (laptops got official docs); a retry was refused when an answer's citations were all wrong; orphan `[N]` markers stayed in the final text.
- UI: 28 checks in jsdom (chips, code blocks untouched, escaping, `javascript:` URLs, panel, Escape). **Not checked in a real browser** (layout/visuals).

## Security / Reliability Notes
- Retrieved text is still untrusted and enters prompts as before (Phase 14). New UI text goes through `textContent`/`esc()`; citation URLs are http(s)-validated server-side (`HttpUrl`) and again in the page.
- Every new stage degrades instead of failing the run: critic crash ⇒ unreviewed answer, understanding/planner crash ⇒ complex/standalone/one sub-question, gap/retry search failure ⇒ skipped, save failure ⇒ logged. One source failing never affects another.
- Bounded work: ≤ `CONCURRENCY_LIMIT` searches in flight; one gap round + one retry; extra searches ≤ 2 × sources per round.

## Known Issues
- **Latency and cost:** a complex question now makes ~4–6 LLM calls (understanding, planner, synthesis, critic; + synthesis and critic on a retry) and up to ~12 Tavily searches, plus ≤ 12 more for gap/retry. Kill switches: `GAP_SEARCH_ENABLED`, `MAX_CRITIC_RETRIES=0`, `SOURCE_ROUTER_ENABLED`.
- **Strict critic:** the LLM check tends to flag medium issues on long technical answers (`needs_improvement`); only high-severity issues trigger a retry.
- **Groq calls still have no timeout** (`timeout=None`, Phase 13); the JSON-schema calls (understanding, planner, critic) fall back silently on schema failures — watch the logs for "failed (…)" warnings.
- **Relevance is lexical** (word overlap): a paraphrase can look like a gap and cause one extra targeted search; it never blocks an answer.
- **In-process cache** is still shared across users and lost on restart (Phase 12); the per-turn `confidence` shown for a cache hit is the stored one.
- **Turn numbers** are chosen when a request starts; two simultaneous requests in one session can both pick the same number — the save retries once with the next free one (the response may then show a stale `turn_number`).
- **Existing rows** written before Phase 6 have no citations or routing (the UI shows the answer only).
- **Sessions API overlaps:** the session list exists twice (`/api/sessions` and `/sessions`, the latter paginated with turn counts), and `DELETE /api/sessions/{id}` still hard-deletes while `DELETE /sessions/{id}` soft-deletes. Kept as approved (D4); consolidate later.
- Not verified in a real browser: visual layout of the new UI.
- Carried over: `.env` enables LangSmith tracing (forced off for manual runs); Python 3.14 "Pydantic V1" warning; the old `synthesize_node` is dead code kept for a later cleanup.

## Impact on later phases
Phases 8 (evidence — in-memory pool only so far), 9 (gap detection — one lightweight round done), 10 (synthesis/citations — done in a first form), 11 (critic/retry — done in a first form) and 12 (cache — additive extras only) start from this implementation and shrink accordingly; their docs need re-scoping.

## Git Commit
Not created yet. Recommended message: `phase-6: source-aware planner, evidence pipeline, conversation persistence, cited-sources UI`. Optional split into three commits: (A) router/planner, (B) pipeline, (C) persistence + API + UI.

## Next Phase
Review this phase, then re-scope Phase 7 (hybrid source RAG) — nothing here depends on it.
