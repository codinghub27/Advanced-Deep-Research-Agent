# P1 — Claim Quality, Freshness and Evidence Hardening

Tracks the 14-part "P1" directive layered on top of the numbered CLAUDE.md phases (it targets
gaps found after Phase 7: citation entailment, source-type consistency, freshness, durable
evidence, provider-failure visibility, minimum security hardening). This is a separate tracking
document, not a renumbering of `PHASE-01`..`PHASE-19`; those keep their own status.

## Status
In Progress — P1.1 and P1.2 implemented and verified (650/650 offline tests, 0 pyright errors on
touched files). P1.3–P1.14 not started.

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

## Out of Scope (this update)
- P1.3 (wiring the policy into `semantic_cache_node`/the RAG gate — the cache and RAG gate still
  behave exactly as at the end of Phase 7), P1.4–P1.14, and anything in CLAUDE.md Phases 8–19.
  `SourceDocument.freshness_status` is a field, not yet ever set by any node.

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

## Files Changed
- **Modified:** `research_app/domain/enums.py`, `research_app/domain/models.py`,
  `research_app/domain/__init__.py`, `research_app/sources/normalizer.py`,
  `research_app/agent/temporal.py`.
- **Created:** `tests/test_date_freshness.py`, this document.
- **Not touched:** `agent/state.py`, `agent/graph.py`, `main.py`, `db/*`, `rag/*`, any prompt,
  any existing test file (all existing tests pass unchanged).

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
PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python -W ignore -m unittest discover -s tests -t .   # 620 OK (baseline, dirty tree) -> 650 OK (with this slice)
npx -y pyright --pythonpath .venv/Scripts/python.exe research_app/domain/enums.py research_app/domain/models.py research_app/domain/__init__.py research_app/sources/normalizer.py research_app/agent/temporal.py   # 0 errors
```

## Test Results
- New: `tests/test_date_freshness.py`, 30 tests (freshness classification, per-source evaluation,
  relative-date parsing, normalizer date extraction including an explicit equivalence test that a
  result with no date fields normalizes exactly as before this change).
- Full suite: **650/650 pass** (620 pre-existing incl. the dirty working tree at session start,
  +30 new). `test_domain_isolation`, `test_domain_models`, and `test_source_normalizer` (the
  Phase 3 pinned equivalence test) all still pass unchanged.
- Pyright on the 5 touched source files: 0 errors, 0 warnings.

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
- Carried over from Phase 7 (untouched by this slice): `SOURCE_RAG_ENABLED=false` (not
  production-ready), PostgreSQL path only manually verified once, `.env.example` still not
  updated for Phase 7's RAG variables (blocked by session permissions in that earlier session).

## Git Commit
Pending — recommended message: `p1.1-p1.2: freshness classification and date extraction/normalization`.

## Next Phase
P1.4 — source-type classification (confidence, reason, authority level, primary/secondary,
independently-verified), reusing the existing `sources/official_docs/registry.py` and
`sources/routing/hosts.py` registries. Then P1.6 (claim/evidence model) before P1.5 (persistence),
per the inspection report's recommended sequence.
