# P1 — Claim Quality, Freshness and Evidence Hardening

Tracks the 14-part "P1" directive layered on top of the numbered CLAUDE.md phases (it targets
gaps found after Phase 7: citation entailment, source-type consistency, freshness, durable
evidence, provider-failure visibility, minimum security hardening). This is a separate tracking
document, not a renumbering of `PHASE-01`..`PHASE-19`; those keep their own status.

## Status
In Progress — P1.1, P1.2, P1.4, P1.5 and P1.6 implemented and verified (691/691 offline tests, 0
pyright errors on touched files). P1.3, P1.7–P1.14 not started.

## Goal
See the accepted P1 directive (approved by the user). Summary: freshness classification and date
extraction, source-type validation, claim-to-evidence/claim-to-citation mapping, citation
verification, minimal durable evidence persistence, and explicit insufficient/stale/uncertain
evidence disclosure — without redesigning the existing architecture (Qdrant, PostgreSQL, Tavily,
GitHub, Reddit, the hybrid RAG pipeline, or the LangGraph structure).

## Scope (this update)
- **P1.2 — Date extraction and normalization**: extend `sources/normalizer.py` to extract a
  second date (`updated_at`, distinct from `published_at`), record `date_confidence` and
  `date_source`, and parse simple relative-date expressions ("3 days ago", "yesterday", "today").
- **P1.1 — Freshness classification**: `agent/temporal.py` gains `FreshnessCategory` (via
  `domain/enums.py`), `FreshnessPolicy` and `classify_freshness()`, extending (not replacing)
  `is_time_sensitive`. Also `evaluate_source_freshness()`: one source's freshness evaluated
  against a request's `FreshnessPolicy`, never a single global threshold.
- **P1.4 — Source-type classification**: new `research_app/sources/classification.py`,
  `classify_content()`: a finer `ContentClassification` (13 values) + `classification_confidence`
  + `classification_reason` + `AuthorityLevel` + `is_primary_source` + `independently_verified`
  on top of the existing (unchanged) `SourceType` routing bucket.
- **P1.6 — Claim/evidence model**: new `Claim` domain model (`domain/models.py`) with
  `ClaimType`/`ClaimSupportStatus`/`ClaimVerificationStatus` (`domain/enums.py`); `Citation`
  gains a stable `citation_id` (distinct from the renumberable `marker`) for a claim to
  reference; `ResearchRun.claims` added.
- **P1.5 — Minimal durable evidence persistence**: Alembic migration `0003_evidence_claims`
  (revises `0002_conversations`) adding `research_evidence` (every source collected for a turn,
  cited or rejected, with P1.2/P1.4 metadata attached) and `research_claims` (mirrors `Claim`).
  `db/conversations.py:save_turn()` gains optional `evidence=`/`claims=` kwargs, with loaders
  `load_evidence()`/`load_claims()`.

## Out of Scope (this update)
- P1.3 (wiring the freshness policy into `semantic_cache_node`/the RAG gate — the cache and RAG
  gate still behave exactly as at the end of Phase 7). `SourceDocument.freshness_status` is a
  field, not yet ever set by any node.
- Wiring `classify_content()` into any graph node, the synthesis prompt, or the citation/source
  labels (`agent/state.py`, `domain/legacy.py`) — deferred to P1.8/P1.9 (synthesis/critic
  updates), which are the natural place to make the existing source-authority rules *depend on*
  the recorded classification, per the P1 directive's own phrasing.
- Nothing yet *builds* a `Claim` from a real synthesized answer. `synthesis.py`'s marker-finding
  code (`_citation`, Phase 6) is unchanged; extracting claims from the model's free text is
  P1.7/P1.8 work, once citation verification exists to feed `support_status`/`verification_status`.
- Nothing yet *calls* `save_turn(..., evidence=, claims=)` from the live route layer
  (`research_service.py`). The schema and CRUD are ready and tested against real SQLite rows;
  wiring the call happens once P1.7/P1.8 actually produce evidence/claim data to pass in --
  calling it now would only ever persist empty lists.
- P1.7, P1.10–P1.14, and anything in CLAUDE.md Phases 8–19.

## Current Implementation
Before this update: only `agent/temporal.py:is_time_sensitive` existed (a boolean, no category);
`sources/normalizer.py:parse_datetime` handled ISO-8601/RFC 2822 only, no relative dates, no
second date field, no confidence/source bookkeeping.

## Planned Changes (as implemented)
1. `domain/enums.py`: `FreshnessCategory` (stable, current_information, recent, as_of_date,
   version_dependent, historical, unknown), `DateConfidence` (exact, approximate, unknown),
   `SourceFreshnessStatus` (fresh, stale, unknown, future_invalid, date_conflict).
2. `domain/models.py`: `SourceDocument` gains `updated_at`, `date_confidence`, `date_source`,
   `freshness_status` — all optional, additive (Phase 2 contract-stability rule).
3. `domain/__init__.py`: re-exports the three new enums.
4. `sources/normalizer.py`: `parse_relative_date()`, `_extract_date_field()` (shared by
   `published_at` and the new `updated_at` search, preserving the exact original per-key loop
   semantics — first parseable value wins and is consumed; an unparseable value is left in
   `metadata` under its own key, never dropped, never invented); `normalize_source()` now also
   looks for `updated_at`/`updated_date`/`last_updated`/`modified_at`.
5. `agent/temporal.py`: `FreshnessPolicy` (frozen dataclass), `classify_freshness()`,
   `evaluate_source_freshness()`, `_extract_cutoff_year()`, plus `_VERSION_CUE`/`_HISTORICAL_CUE`/
   `_AS_OF_CUE` regexes. `is_time_sensitive`, `recency_params`, `refresh_stale_years`, `date_line`
   are unchanged and are called from the new code, not duplicated.
6. Tests: `tests/test_date_freshness.py` (30 tests).

### Freshness category → policy (defaults)
| Category | cache_allowed | rag_allowed | live_search_required | official_verification_required | window |
|---|---|---|---|---|---|
| `stable` (default) | yes | yes | no | no | none |
| `current_information` | no | no | yes | no | 7 days |
| `recent` | no | no | yes | no | 30 days |
| `as_of_date` | no | yes | yes | yes | none (cutoff year if extractable) |
| `version_dependent` | no | yes | yes | yes | none |
| `historical` | yes | yes | no | no | none |
| `unknown` (ambiguous/empty question) | no | yes | no | no | none |

Detection order: version cue (only alongside `is_documentation_query=True`, passed by the caller
from the existing Phase 4 docs detector — a bare "version" never triggers this alone) → historical
cue → `is_time_sensitive` (current vs. recent split reuses the exact Phase 6 `_SHORT`/`_MEDIUM`/
`_RECENT` regexes) → as-of cue → empty text → stable default. "Release notes"/"changelog" fall
under the existing `_MEDIUM` regex and so classify as `current_information` (7-day window), not
`recent` (30-day) — a deliberate reading of the existing Phase 6 cue set, not a new rule.

### Per-source freshness evaluation
`evaluate_source_freshness(source, policy, now=...)`:
1. `updated_at` present and more than a day older than `published_at` → `DATE_CONFLICT`.
2. No `updated_at` and no `published_at` → `UNKNOWN` (never invented).
3. Reference date (`updated_at` if present, else `published_at`) more than a day in the future →
   `FUTURE_INVALID`.
4. No `preferred_window_days` on the policy (stable/historical/version/as-of/unknown) → `FRESH`
   (an old authoritative source is not penalized by age alone; "Stable questions may use older
   authoritative sources" per the P1 directive).
5. Otherwise: within the window → `FRESH`, else `STALE`.

### Content classification (P1.4)
`classify_content(doc) -> SourceDocument` (a copy, pure): dispatches on the existing
`source_type` first, then narrows using URL-derived metadata/domain/content length already on
the document — no new network or LLM call.

| `source_type` | Signal | `ContentClassification` | `AuthorityLevel` | primary | verified |
|---|---|---|---|---|---|
| `official_docs` | changelog/blog path or "announcing" title | `official_announcement` | `official` | yes | yes |
| `official_docs` | otherwise | `official_documentation` | `official` | yes | yes |
| `github` | `metadata.github.kind` in issue/pull_request/discussion | `github_issue_or_discussion` | `community` | no | yes |
| `github` | kind in repository/file/directory/release/commit/wiki | `github_repository` | `community` | yes | yes |
| `github` | kind missing/unrecognised (`profile`, `other`) | `unknown` (confidence 0.3) | `community` | — | yes |
| `reddit` | `metadata.reddit.kind` in post/comment | `reddit_post_or_discussion` | `community` | no | yes |
| `reddit` | kind missing/unrecognised | `unknown` (confidence 0.3) | `community` | no | yes |
| `web` | domain in `_PREPRINT_DOMAINS` (arxiv.org, biorxiv.org, ...) | `preprint` | `academic` | yes | **no** |
| `web` | domain in `_RESEARCH_PAPER_DOMAINS` (nature.com, dl.acm.org, ...) | `research_paper` | `academic` | yes | yes |
| `web` | domain in `_NEWS_DOMAINS` (reuters.com, bbc.com, ...) | `news_report` | `established` | no | yes |
| `web` | domain in `_BLOG_DOMAINS` or `/blog/` path | `expert_blog` | `community` | no | no |
| `web` | `.gov` domain | `technical_report` (confidence 0.5) | `established` | yes | no |
| `web` | none of the above, content < 200 chars or absent | `search_snippet` | `unknown` | — | no |
| `web` | none of the above, full content present | `unknown` (confidence 0.3) | `unknown` | — | no |
| `rag_document` / `cache` | — | `unknown` (confidence 0.0) | `unknown` | — | no |

Domain lists (`_PREPRINT_DOMAINS`, `_RESEARCH_PAPER_DOMAINS`, `_NEWS_DOMAINS`, `_BLOG_DOMAINS`) are
illustrative and extensible module-level sets, not an exhaustive or verified registry (unlike
Phase 4's official-docs registry, which is checked against fetched URLs) — a domain absent from a
list falls through to `unknown` rather than being guessed. `www.` is ignored on both sides for
matching (same convention as Phase 4/5's registries), matching only, `SourceDocument.domain`
itself is untouched.

### Claim model (P1.6)
`Claim`: `claim_id`, `run_id`, `text`, `claim_type` (fact/inference/recommendation/opinion/
estimate/summary), `evidence_ids`, `source_ids`, `excerpts`, `citation_id`, `support_status`
(directly/partially/indirectly supported, unsupported, contradicted, not_applicable),
`evidence_strength` (0-1), `freshness_status` (reuses `SourceFreshnessStatus` from P1.2 --
one claim can be flagged stale even if the overall run is not), `conflict_status` (bool),
`verification_status` (unverified/verified/failed/revised/removed -- P1.7 sets this).
`Citation.citation_id` (new, `default_factory=new_id`) gives a claim something stable to point
to, since `Citation.marker` is renumbered by synthesis (Phase 6) and is not a durable key.
`ResearchRun.claims: list[Claim]` added alongside the existing `citations`/`evidence` lists.

### Evidence/claim persistence (P1.5)
No separate "research_runs" table: `research_conversations.id` (Phase 6, one row per turn)
already *is* the run id both new tables reference.

- `research_evidence`: one row per source collected for a turn, cited or not. Columns mirror
  `SourceDocument` (P1.2's `published_at`/`content_updated_at`/`date_confidence`/`date_source`/
  `freshness_status`, P1.4's `content_classification`/`classification_confidence`/
  `authority_level`/`is_primary_source`/`independently_verified`), plus `included` (bool) and
  `rejection_reason` (text) so accepted and rejected sources live in one auditable table instead
  of two.
- `research_claims`: one row per `Claim` (P1.6), same fields, `evidence_ids`/`source_ids`/
  `excerpts` as JSON lists of domain-level string ids (not `research_evidence.id` foreign keys --
  those ids are stable across runs, DB primary keys are not meaningful across them).
- `db/conversations.py`: `save_turn(..., evidence=(), claims=())` -- both new kwargs default to
  empty and are additive; every existing caller (Phase 6, unchanged) keeps working exactly as
  before. `_build_evidence`/`_build_claim` convert a plain mapping (what a pipeline node would
  produce) into the ORM row, reusing `_naive_utc` for date fields exactly like citations already
  do. `load_evidence()`/`load_claims()` read them back, retrieval order.
- Cascade: deleting a session deletes its conversations (existing), which deletes their evidence
  and claims (new `cascade="all, delete-orphan"` relationships) -- no orphaned rows.

## Files Changed
- **Modified:** `research_app/domain/enums.py`, `research_app/domain/models.py`,
  `research_app/domain/__init__.py`, `research_app/sources/normalizer.py`,
  `research_app/agent/temporal.py`, `research_app/db/models.py`, `research_app/db/conversations.py`,
  `tests/test_domain_isolation.py` (+1 module in the sources-package isolation check, no behavior
  change), `tests/test_p6_conversation.py` (pinned head-revision string updated from
  `0002_conversations` to `0003_evidence_claims`, same pattern as `test_domain_isolation`'s
  node-set snapshot -- no behavioral change, just the new expected value).
- **Created:** `research_app/sources/classification.py`,
  `research_app/db/migrations/versions/0003_evidence_claims.py`, `tests/test_date_freshness.py`,
  `tests/test_content_classification.py`, `tests/test_claim_model.py`,
  `tests/test_p1_evidence_persistence.py`, this document.
- **Not touched:** `agent/state.py`, `agent/graph.py`, `main.py`, `research_service.py`, `rag/*`,
  any prompt, any existing test file's assertions besides the one pinned revision string above.

## Architecture Decisions
- **`classify_freshness`/`evaluate_source_freshness` live in `agent/temporal.py`**, not in
  `domain/` or `sources/`, because they call `is_time_sensitive` (already in `agent/temporal.py`)
  and the P1 directive says to extend, not duplicate, the existing detection. `domain/` and
  `sources/` stay framework-free per the Phase 2/3 isolation rule; `agent/temporal.py` already
  sits outside that boundary (it imports `research_app.conversation.settings`).
- **Not wired into any graph node yet.** `FreshnessPolicy`/`evaluate_source_freshness` are pure
  functions nothing calls at runtime, same pattern as Phase 2's domain contracts before Phase 3
  wired them. `SourceDocument.freshness_status` exists as a field but no code sets it. This keeps
  the diff reviewable and avoids touching the pinned `test_domain_isolation` graph-node snapshot
  or the SSE contract in this slice; P1.3 does the wiring (semantic cache gate, RAG gate).
- **`is_documentation_query` is a parameter, not an import.** `classify_freshness` does not import
  `sources.official_docs.detector` itself; the caller (state.py, in P1.3) passes the already-computed
  docs-intent boolean. Keeps `agent/temporal.py`'s dependency surface unchanged in this slice.
- **`updated_at` is generic**, not split into GitHub-commit-date/Reddit-post-date/version-date
  fields as the directive's prose lists them: Tavily (the only provider today) does not return
  structured per-source-type date fields, and `SourceDocument.metadata` already carries
  provider-specific extras (`metadata["github"]`, `metadata["reddit"]`). A generic `updated_at` +
  `date_source` (which raw key it came from) covers "last updated / commit / release date"
  without inventing fields no adapter can currently fill.
- **Relative-date parsing is deliberately narrow** (`N days/weeks/months/years ago`, `yesterday`,
  `today`/`just now`, exact match only): scanning free text for embedded dates risks false
  positives and is out of scope for this slice; only an explicit, whole-value relative expression
  in a date field is accepted.
- **`DATE_CONFLICT`/`FUTURE_INVALID` use a 1-day tolerance** for clock skew between the
  normalization batch and a provider's own timestamp, not zero tolerance.
- **`classify_content` lives in `sources/`, not `domain/`**, mirroring `normalizer.py`: it is a
  transform over a `SourceDocument`, not a contract. It stays framework-free (added to the
  `test_domain_isolation` sources-package isolation check).
- **`SourceType` is not touched or extended.** A new, separate `ContentClassification` enum keeps
  the load-bearing routing/synthesis/DB enum stable (Rule 3); `SourceType.RAG_DOCUMENT`/`CACHE`
  are explicitly out of scope (Phase 7: a chunk keeps its original `source_type`, so there is
  nothing new to classify there without re-deriving from the original document).
- **A preprint is never `independently_verified`** even though `is_primary_source=True`: the
  preprint server does not itself verify claims (P1.4's explicit "do not label preprints as
  peer-reviewed" rule) — primary-source status and independent verification are tracked as two
  separate booleans specifically so this distinction is representable.
- **`official_docs`/`github`/`reddit` all get `independently_verified=True`** because Phase 4's
  registry match and Phase 5's hostname allowlist already are the independent verification (the
  URL was checked against a maintained list before the document was ever labelled that
  `source_type`); P1.4 does not re-verify, it reads that existing guarantee.
- **Unrecognised metadata never upgrades confidence.** A GitHub/Reddit result with a missing or
  unrecognised URL-derived `kind` becomes `unknown` at confidence 0.3, not a best-guess
  `github_repository`/`reddit_post_or_discussion` — directly implements "do not treat... repository
  lists as proof of importance".
- **`Claim` is a plain data model with no validators enforcing "opinions get NOT_APPLICABLE".**
  The P1 directive's ranking is guidance for the code that *creates* claims (P1.7/P1.8), not an
  invariant of the shape itself -- an opinion claim quoting a source that states that exact
  opinion verbatim legitimately can be `DIRECTLY_SUPPORTED`. Forcing it in the model would make
  that case impossible to represent.
- **`Citation.citation_id` is new, `Citation.marker` is unchanged.** `marker` stays the
  answer-facing "[1]" text (renumbered by synthesis, Phase 6); `citation_id` is a stable key
  nothing currently sets differently per-request (each `Citation()` call gets a fresh UUID via
  `default_factory`), so persisting a `Claim.citation_id` in P1.5 will need the *same* `Citation`
  object's id at both write times, not two independently-constructed citations.
- **No claim-extraction code yet.** Building `Claim`s from a real answer means walking the
  synthesized text and its markers, deciding `claim_type` and initial `support_status` -- that
  logic belongs with P1.7 (citation verification), which needs the same text-walk to check
  entailment; building the model without it first, as a separate change, keeps this diff to a
  pure, reviewable data-shape addition (Rule 5).

## Dependencies
None added or removed. Stdlib only (`dataclasses`, `datetime.timedelta`) plus the existing
`pydantic`/domain machinery.

## Manual Tests
Not run live (no LLM/Tavily/database call in this slice — pure functions only). Recommended once
P1.3 wires this in: repeat the P1.13 matrix's freshness rows (latest query with fresh/stale/
unknown-date source, historical query, version query with outdated docs, conflicting dates,
missing publication date) against a live server.

## Commands Run
```text
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python -W ignore -m unittest discover -s tests -t .   # 620 OK (baseline, dirty tree) -> 650 OK (P1.1/P1.2) -> 668 OK (P1.4) -> 681 OK (P1.6) -> 691 OK (P1.5)
npx -y pyright --pythonpath .venv/Scripts/python.exe research_app/domain/enums.py research_app/domain/models.py research_app/domain/__init__.py research_app/sources/normalizer.py research_app/agent/temporal.py research_app/sources/classification.py research_app/db/models.py research_app/db/conversations.py research_app/db/migrations/versions/0003_evidence_claims.py tests/test_content_classification.py tests/test_claim_model.py tests/test_p1_evidence_persistence.py   # 0 errors
```

## Test Results
- New: `tests/test_date_freshness.py` (30 tests), `tests/test_content_classification.py`
  (18 tests), `tests/test_claim_model.py` (13 tests: safe defaults, validation of empty
  text/unknown fields/out-of-range strength, every `ClaimType`/`ClaimSupportStatus` value is
  constructible, `Citation.citation_id` is stable and distinct per-instance, `ResearchRun.claims`
  round-trips), and `tests/test_p1_evidence_persistence.py` (10 tests: migration `0003` runs
  cleanly and idempotently on an empty DB, downgrade removes the new tables, `save_turn` persists
  evidence+claims with all P1.2/P1.4 fields intact, included-vs-rejected evidence is
  distinguished, cascade delete removes evidence/claims with their session, unparseable/missing
  dates land as `NULL` not an invented value, and a backward-compatibility test that `save_turn`
  without `evidence=`/`claims=` behaves exactly as every existing Phase 6 caller expects).
- Full suite: **691/691 pass** (620 pre-existing incl. the dirty working tree at session start,
  +30 P1.1/P1.2, +18 P1.4, +13 P1.6, +10 P1.5). `test_domain_isolation` (extended, still green),
  `test_domain_models`, and `test_source_normalizer` (the Phase 3 pinned equivalence test) all
  still pass. One pre-existing pinned test (`test_p6_conversation.py`'s migration-head check) was
  updated from `0002_conversations` to `0003_evidence_claims` -- the new expected head revision,
  not a behavior change (same pattern as `test_domain_isolation`'s node-set snapshot).
- Pyright on all touched/created source and test files: 0 errors, 0 warnings (a few incidental
  `Score`/`Optional[float]`-vs-`float` comparison errors in test files were fixed with explicit
  `float(...)` casts; two pre-existing `BaseRoute.path` errors in `test_domain_isolation.py` and
  several pre-existing `ResearchState`-partial-dict errors in `test_p6_conversation.py`, unrelated
  to the lines this update touched, are untouched baseline noise).

## Known Issues
- **Not yet load-bearing.** Nothing in the running graph calls `classify_freshness`,
  `evaluate_source_freshness`, or reads `SourceDocument.freshness_status`/`date_confidence`/
  `date_source`/`updated_at` yet. P1.3 must wire this into `semantic_cache_node` (cache-hit
  freshness check) and the RAG gate (`rag_gate_router`/`source_rag_node`) before it affects any
  answer.
- **`updated_at` is rarely populated today.** Tavily's general web search mostly returns
  `published_date` only (per Phase 6's real-time-research finding); `updated_at` will show up for
  the minority of results that carry a distinguishable field, and `date_confidence`/`date_source`
  will correctly read `None`/absent for the rest — this is expected, not a bug.
- **`as_of_date` cutoff extraction is a bare 4-digit year only** (`_YEAR` regex, reused from
  Phase 6). "as of March 2020" extracts `2020-01-01`, not the month; adequate for a cutoff signal,
  not a precise date.
- **No content-based date extraction** (og:published_time, JSON-LD, visible page text): out of
  scope, since the app never fetches raw pages today, only Tavily-mediated results (P1.11/SSRF
  territory if that ever changes).
- **`classify_content` is also not yet load-bearing.** Nothing in the running graph calls it or
  reads `content_classification`/`authority_level`/`is_primary_source`/`independently_verified`.
  The existing Phase 4/5 synthesis guidance text is untouched — "official docs preferred",
  "Reddit/GitHub are not authority" are still driven by `source_type` + hard-coded prompt text,
  not by this new field. P1.8/P1.9 should switch that guidance to read `authority_level` instead,
  which is the "make the existing source-authority rules depend on the recorded source type" part
  of the P1 directive that this slice intentionally left undone.
- **Domain lists are hand-curated, not fetched/verified** (unlike Phase 4's official-docs
  registry, whose hosts were individually checked against live pages on 2026-09-19). A domain
  could be miscategorized or a real venue could be missing; low `classification_confidence`
  (0.3–0.6) on the weaker branches (`technical_report`, `expert_blog`, `unknown`) reflects this.
  No live verification was performed for this slice.
- **News/paper/blog detection is domain-only**, not content-based: a personal blog hosted on a
  news domain's subdomain, or a paper mirrored off its recognised venue, is not detected.
- **`Claim` is an empty shape with nothing populating it.** No code anywhere constructs a `Claim`
  from a real research run yet; `ResearchRun.claims` will be `[]` for every run until P1.7/P1.8.
- **`save_turn(evidence=, claims=)` is not called anywhere live.** Schema, ORM models and CRUD
  are complete and tested against real SQLite rows, but `research_service.py` never passes these
  kwargs, so every `research_evidence`/`research_claims` table stays empty until P1.7/P1.8 wire
  a real call. No fine-grained per-provider status table yet either (P1.10).
- Carried over from Phase 7 (untouched by this slice): `SOURCE_RAG_ENABLED=false` (not
  production-ready), PostgreSQL path only manually verified once, `.env.example` still not
  updated for Phase 7's RAG variables (blocked by session permissions in that earlier session).

## Git Commit
- `1e6d3ec` — p1.1-p1.2: freshness classification and date extraction/normalization
- `b148901` — p1.4: source-type (content) classification
- `2df7376` — p1.6: claim/evidence domain model
- Pending — recommended message: `p1.5: minimal durable evidence and claim persistence`. (Per
  CLAUDE.md Rule 4, one architectural change per commit; each P1 sub-part above is its own commit
  on `phase-7-hybrid-rag`.)

## Next Phase
P1.7 — citation verification: for each citation, check the source was retrieved this run, the
excerpt exists and supports the claim, the claim isn't stronger than the evidence, the source
type/date satisfy the freshness requirement, and the same source isn't reused to fake evidence
diversity. This is the first sub-phase that actually *builds* `Claim` rows from a real answer and
is the natural point to wire `save_turn(evidence=, claims=)` into the live route layer. P1.3
(wiring `classify_freshness`/`evaluate_source_freshness` into the cache and RAG gates) and P1.8/
P1.9 (synthesis/critic reading `authority_level`/`content_classification`) remain open alongside it.
