# CLAUDE.md — Advanced Production-Grade Deep Research Agent

## 1. Project Mission

Upgrade the existing Deep Research Agent incrementally into a production-grade, reliable, multi-source research system.

The existing working system already contains important functionality. **Do not replace it with a new application.** Preserve working behavior and API contracts while adding capabilities phase by phase.

### Existing stack to preserve

- Python
- FastAPI
- LangGraph
- LangChain
- PostgreSQL
- Qdrant
- Semantic cache
- RAG
- Planner
- Parallel research execution
- Tavily/web extraction
- Synthesis
- Critic
- Authentication/session functionality

### Explicitly excluded from this upgrade

Do NOT implement:

- LangSmith
- automated evaluation pipelines
- evaluation datasets
- evaluator LLMs
- benchmark dashboards
- training pipelines

Evaluation will be handled manually now and added later.

---

# 2. Target Architecture

The final system should follow this high-level flow:

USER QUESTION
    |
    v
QUERY UNDERSTANDING
    - intent
    - complexity
    - time sensitivity
    - technology/documentation intent
    |
    v
LAYER 1 — SEMANTIC CACHE
    - Qdrant answer cache
    - similarity search
    - freshness/validity decision
    |
    +---- CACHE HIT + FRESH --------------------> RETURN CACHED ANSWER
    |
    +---- CACHE MISS / STALE
                    |
                    v
LAYER 2 — SOURCE DOCUMENT RAG
    - Qdrant dense retrieval
    - BM25 keyword retrieval
    - reciprocal-rank / score fusion
    - reranking
    |
    +---- RELEVANT ------------------------------> SYNTHESIS
    |                                               |
    |                                               v
    |                                             CRITIC
    |                                               |
    |                                               v
    |                                          FINAL ANSWER
    |
    +---- NOT RELEVANT
                    |
                    v
FULL RESEARCH AGENT
    |
    v
PLANNER
    - research strategy
    - subquestions
    - source requirements
    |
    v
SOURCE SELECTION / ROUTER
    |
    +---- Official Documentation
    |       - technology installation
    |       - API usage
    |       - framework/library usage
    |       - configuration
    |       - version-specific technical facts
    |
    +---- Tavily / Web
    |
    +---- GitHub
    |
    +---- Reddit / Community
    |
    v
PARALLEL RESEARCH
    |
    v
RESULT NORMALIZATION
    |
    v
SOURCE QUALITY FILTER
    |
    v
RELEVANCE RERANKING
    |
    v
DEDUPLICATION
    |
    v
EVIDENCE STORE
    - Qdrant
    - PostgreSQL metadata
    |
    v
GAP DETECTION
    |
    +---- GAP EXISTS ----> TARGETED SEARCH ----> Evidence pipeline
    |
    +---- SUFFICIENT
                    |
                    v
                 SYNTHESIS
                    |
                    v
                  CRITIC
                    |
             +------+------+
             |             |
            GOOD          BAD
             |             |
             v             v
       SAVE TO STORES   TARGETED RETRY
                           |
                           +----> SYNTHESIS
    |
    v
FINAL ANSWER
    - grounded claims
    - citations
    - source list
    - source metadata

STORAGE
- Qdrant:
    1. answer_cache
    2. source_chunks
    3. research_evidence (if useful as a separate collection)
- PostgreSQL:
    users
    sessions
    queries
    research_runs
    sources
    evidence metadata
    learning/history metadata
- BM25 / keyword index:
    source document keyword retrieval

OBSERVABILITY
- structured application logging
- request/research run IDs
- latency/error metrics
- provider/tool tracing through application logs

Do not add LangSmith in this phase.

---

# 3. Critical Source-Selection Requirement

This is a mandatory feature.

## Official Documentation Source

The source-selection layer MUST detect questions where official documentation is the authoritative source.

Examples:

- "How do I install FastAPI?"
- "How do I use LangGraph?"
- "How does PostgreSQL CREATE INDEX work?"
- "How do I configure Docker Compose?"
- "How do I use the OpenAI API?"
- "What is the official syntax for X?"
- "How do I configure Qdrant?"
- "How do I use a Python package?"
- "What changed in version X?"

For these questions, the router should prioritize the official documentation for the relevant technology.

### Required behavior

1. Detect technology/documentation intent.
2. Identify the technology/entity.
3. Resolve the official documentation domain.
4. Search official documentation with domain restrictions where possible.
5. Extract the relevant documentation content.
6. Store normalized evidence with:
   - source_type = official_docs
   - domain
   - URL
   - title
   - technology
   - version if available
   - retrieved_at
   - content/chunk
7. Give official documentation higher source priority for technical/how-to claims.
8. Cite the exact official documentation source in the final answer.
9. Do not treat Reddit/community content as equivalent authority for official API/configuration behavior.

The architecture should support an extensible official-documentation registry, for example:

technology -> official domains -> documentation URL patterns

Do not hard-code all technologies inside the planner. Use a configurable registry.

---

# 4. Engineering Rules

## Rule 1 — Preserve the existing project

Before changing code:

1. Inspect repository structure.
2. Inspect current entry points.
3. Inspect LangGraph state/schema.
4. Inspect nodes and edges.
5. Inspect API routes.
6. Inspect database models.
7. Inspect Qdrant integration.
8. Inspect current tools.
9. Inspect configuration/environment handling.
10. Run existing tests or smoke tests.

Never assume the current implementation.

## Rule 2 — No giant rewrite

Never rewrite the whole repository.

Each phase must produce a small, reviewable change.

Prefer:

- new module
- adapter
- interface
- router
- service
- test
- migration

over replacing working code.

## Rule 3 — Keep contracts stable

Do not break:

- existing API request formats
- existing API response formats
- authentication
- session behavior
- existing graph invocation contract

If a contract must change, document it first and provide backward compatibility where practical.

## Rule 4 — Small commits

Each phase should normally produce:

- one architectural change
- supporting tests
- documentation
- one or a few focused commits

Do not mix unrelated cleanup with feature work.

## Rule 5 — Type-safe boundaries

Use clear Pydantic/domain models for important boundaries.

Especially standardize:

- ResearchQuery
- ResearchTask
- ResearchResult
- SourceDocument
- Evidence
- Citation
- ResearchRun
- CriticResult
- GapAnalysis
- SearchRequest

## Rule 6 — Async and parallelism

Use async I/O for external sources.

Parallelize independent searches.

Do not create unbounded concurrency.

Use:

- concurrency limits
- timeouts
- retries
- cancellation
- failure isolation

## Rule 7 — Configuration

Secrets and environment-specific settings belong in environment variables/configuration.

Never hard-code API keys.

Separate:

- development
- test/manual validation
- production

configuration.

## Rule 8 — Reliability

Every external dependency must have:

- timeout
- retry policy where appropriate
- error handling
- logging
- graceful degradation

One failed source should not automatically kill the whole research run.

## Rule 9 — Evidence first

The final answer must be generated from collected evidence.

Avoid allowing the LLM to silently invent unsupported facts.

Every important factual claim should be traceable to evidence/source metadata.

## Rule 10 — Manual testing only for now

At the end of every phase, create manual test cases.

Do not build LangSmith/evaluation infrastructure.

---

# 5. Claude Code Operating Procedure

Claude Code must follow this sequence for EVERY phase.

## Step A — Read project instructions

Read:

- CLAUDE.md
- current phase file
- relevant previous phase files

Do not read the entire repository blindly.

## Step B — Inspect

Use focused inspection:

- repository tree
- relevant modules
- imports
- interfaces
- tests
- configuration
- graph definition

## Step C — Plan

Before editing, explain:

1. current implementation
2. target change
3. files to create
4. files to modify
5. dependencies
6. migration impact
7. tests
8. risks

Wait for approval if the change is architectural or destructive.

## Step D — Implement

Implement only the current phase.

Do not start future phases.

## Step E — Test

Run the smallest relevant test/smoke-test set first.

Then run broader tests if necessary.

## Step F — Review

Check:

- regressions
- error handling
- type consistency
- logging
- security
- performance
- backwards compatibility

## Step G — Document

Update the phase file with:

- implemented changes
- files changed
- commands run
- test results
- decisions
- known issues
- next phase

## Step H — Git checkpoint

Create a clean checkpoint after the phase is verified.

Suggested commit:

`phase-N: <short description>`

---

# 6. Token-Efficient Claude Code Rules

The goal is high-quality implementation without wasting Claude Code context.

### Do

- Start a fresh Claude Code session for each phase.
- Give Claude the current phase file.
- Inspect only relevant files first.
- Ask for a plan before implementation.
- Implement one phase at a time.
- Run focused tests.
- Reuse existing code.
- Ask Claude to summarize completed work into the phase file.
- Keep architecture decisions in docs instead of repeatedly explaining them.

### Avoid

- "Read my entire project and improve everything."
- giant one-shot prompts
- asking Claude to implement all phases at once
- unnecessary refactoring
- repeated full-repository scans
- rewriting working modules
- adding dependencies without justification
- asking Claude to evaluate the entire system after every tiny edit

### Preferred prompt pattern

"Read CLAUDE.md and docs/phases/PHASE-XX-*.md. Inspect only the files relevant to this phase. First give me an implementation plan and list the exact files you expect to change. Do not modify code yet."

After approval:

"Implement only this phase. Preserve existing behavior and API contracts. Do not start later phases. Run focused tests, review the diff, and update the phase document."

---

# 7. Phase Completion Definition

A phase is complete only when:

- implementation exists
- focused tests/smoke tests pass
- existing behavior is not unintentionally broken
- configuration is documented
- errors are handled
- phase document is updated
- known limitations are documented
- Git checkpoint is created

---

# 8. Phase Order

## Phase 01 — Repository Audit and Baseline

Goal:
Understand the existing system before modifying it.

Tasks:
- map repository
- identify current graph
- identify APIs
- identify database/Qdrant code
- identify tools
- identify cache
- identify RAG
- identify tests
- establish baseline smoke tests
- document technical debt

Output:
`docs/phases/PHASE-01-AUDIT.md`

No feature rewrite.

---

## Phase 02 — Architecture and Domain Contracts

Goal:
Create clean boundaries without changing behavior.

Introduce/standardize domain models:

- ResearchQuery
- ResearchTask
- ResearchResult
- SourceDocument
- Evidence
- Citation
- ResearchRun
- CriticResult
- GapAnalysis

Create service interfaces where useful.

Output:
`docs/phases/PHASE-02-CONTRACTS.md`

---

## Phase 03 — Research Result and Source Normalization

Goal:
Normalize all external search results into one internal format.

Every source should become a common structure.

Include:

- source_type
- title
- URL
- domain
- author if available
- published_at if available
- retrieved_at
- snippet
- content
- credibility metadata
- query/subquestion
- source-specific metadata

Implement adapters for current sources.

---

## Phase 04 — Official Documentation Source

Goal:
Add first-class official documentation retrieval.

Implement:

- technology intent detection
- technology identification
- official-domain registry
- official-doc search adapter
- documentation extraction
- version metadata
- source priority
- citations

Official docs should be prioritized for technical/how-to/configuration/API questions.

Do not let this break general web research.

---

## Phase 05 — GitHub and Reddit Sources

Goal:
Add specialized sources.

GitHub:
- repositories
- README/docs
- issues/discussions where appropriate
- code/search metadata

Reddit:
- discussions
- practical experiences
- troubleshooting/community context

Use source-specific metadata.

Do not treat these as equivalent to official documentation.

---

## Phase 06 — Source Router and Planner Integration

Goal:
Make the planner choose appropriate sources based on intent.

Example:

technical installation:
    official_docs + web

library usage:
    official_docs + GitHub

current community issue:
    web + Reddit + GitHub

general current research:
    web + specialized sources

code implementation:
    official_docs + GitHub + web

The router must support multiple sources for one task.

---

## Phase 07 — Hybrid Source RAG

Goal:
Upgrade retrieval from dense-only to hybrid retrieval.

Implement:

- Qdrant dense retrieval
- BM25 keyword retrieval
- score/rank fusion
- configurable top-k
- metadata filtering
- reranking

Keep answer cache separate from source knowledge.

Suggested Qdrant separation:

`answer_cache`
`source_chunks`

Optional later:

`research_evidence`

---

## Phase 08 — Evidence Pipeline and Evidence Store

Goal:
Make evidence a first-class object.

Pipeline:

raw result
 -> normalize
 -> quality filter
 -> relevance
 -> rerank
 -> deduplicate
 -> evidence

Store:

- claim/evidence text
- source
- URL
- source type
- retrieval timestamp
- query/subquestion
- ranking
- confidence/quality metadata

PostgreSQL stores durable metadata; Qdrant stores searchable semantic content.

---

## Phase 09 — Gap Detection and Iterative Research

Goal:
Move from one-shot search to evidence-driven research.

Implement:

- evidence coverage analysis
- missing subquestion detection
- unsupported claim detection
- contradictory evidence detection where feasible
- targeted follow-up queries

Flow:

research
 -> evidence
 -> gap detection
 -> targeted search
 -> evidence
 -> gap detection
 -> synthesis

Use bounded iteration.

Never create infinite research loops.

---

## Phase 10 — Synthesis, Citations, and Grounding

Goal:
Generate high-quality grounded answers.

The synthesis layer should:

- use evidence only
- distinguish facts from uncertainty
- preserve source attribution
- generate citation mappings
- avoid unsupported claims
- handle conflicting sources
- provide source list

For technical questions, cite official documentation where it was used.

---

## Phase 11 — Critic and Targeted Retry

Goal:
Make the critic operational rather than decorative.

Critic checks:

- question coverage
- evidence support
- citation correctness
- source quality
- contradictions
- missing important information
- unsupported claims

If bad:

critic feedback
 -> targeted retry
 -> additional evidence
 -> synthesis
 -> critic

Bound retry count.

---

## Phase 12 — Semantic Cache and Freshness

Goal:
Make caching production-safe.

Implement:

- semantic similarity
- cache hit threshold
- query normalization
- freshness metadata
- TTL by query/source type
- invalidation
- stale detection
- source-aware cache policy

Do not return stale answers for time-sensitive questions.

Examples of time-sensitive queries:

- current versions
- latest releases
- current pricing
- current incidents
- recent news
- current APIs

---

## Phase 13 — Reliability and Fault Tolerance

Goal:
Handle real-world failures.

Implement:

- timeouts
- bounded retries
- exponential backoff where appropriate
- circuit-breaker-like protection where justified
- per-source failure isolation
- partial result handling
- fallback behavior
- structured error states

One failed source must not necessarily fail the entire research run.

---

## Phase 14 — Security and API Hardening

Goal:
Prepare the API for production use.

Review:

- JWT/authentication
- authorization
- input validation
- rate limiting
- request size limits
- SSRF protection
- URL validation
- prompt-injection defenses for retrieved content
- secret management
- CORS
- logging of sensitive data
- database safety

Treat external web content as untrusted input.

---

## Phase 15 — Performance and Cost Optimization

Goal:
Reduce latency and unnecessary LLM/tool calls.

Implement where justified:

- parallel source retrieval
- async I/O
- connection pooling
- embedding batching
- caching
- search result reuse
- configurable model selection
- token limits
- early stopping
- bounded research iterations

Measure before optimizing.

Do not optimize by removing required research quality.

---

## Phase 16 — API, Streaming, and Long-Running Research

Goal:
Make research usable as a production API.

Review:

- request/response schemas
- research run IDs
- status tracking
- streaming/progress events where useful
- cancellation
- timeout behavior
- background/long-running execution
- result retrieval

Preserve existing clients where possible.

---

## Phase 17 — Docker and Production Configuration

Goal:
Containerize the system cleanly.

Prepare:

- Dockerfile
- docker-compose for local development
- PostgreSQL
- Qdrant
- environment configuration
- health checks
- startup/shutdown handling
- persistent volumes
- production configuration

Do not put secrets in Docker images.

---

## Phase 18 — Production Observability

Goal:
Add application-level observability without LangSmith.

Implement:

- request IDs
- research run IDs
- structured logs
- source latency
- tool errors
- cache hit/miss logs
- research iteration counts
- token/cost metadata if provider exposes it
- database/Qdrant operation timing

Keep logs useful and privacy-conscious.

---

## Phase 19 — Final Manual Validation and Release

Goal:
Validate the complete system manually.

Create a manual matrix covering:

1. simple factual question
2. complex research
3. comparison
4. multi-hop
5. current information
6. technical installation
7. official documentation lookup
8. GitHub/code question
9. Reddit/community troubleshooting
10. cache hit
11. stale cache
12. cache miss
13. insufficient evidence
14. conflicting sources
15. source failure
16. timeout
17. invalid input
18. prompt-injection content
19. long-running research
20. repeated research

Record:

- expected behavior
- actual behavior
- latency
- citations
- source quality
- errors
- observations

Do not build automated evaluation yet.

---

# 9. Definition of Final Production Architecture

The completed system should have these major layers:

1. API Layer
2. Query Understanding
3. Semantic Cache
4. Source Document RAG
5. Research Planner
6. Source Router
7. Official Documentation Source
8. Web Source
9. GitHub Source
10. Reddit Source
11. Result Normalization
12. Source Quality Filter
13. Hybrid Retrieval
14. Reranking
15. Deduplication
16. Evidence Store
17. Gap Detection
18. Targeted Search
19. Synthesis
20. Critic
21. Targeted Retry
22. Citation/Grounding
23. Persistence
24. Reliability
25. Security
26. Performance
27. Observability
28. Docker/Deployment

---

# 10. What Claude Code Must NOT Do

Never:

- implement all phases in one session
- rewrite the application from scratch
- delete working features without approval
- add LangSmith
- build evaluation datasets
- create automated evaluation pipelines
- invent unsupported architecture
- introduce Redis/Celery/etc. merely because they are common production tools
- add dependencies without checking whether the existing stack already solves the problem
- expose API keys
- trust arbitrary retrieved web content
- create infinite agent loops
- silently change API contracts
- perform unrelated refactoring

---

# 11. Phase State

Each phase document must contain:

```text
Status:
Not Started | In Progress | Blocked | Completed

Goal:
...

Scope:
...

Out of Scope:
...

Current Implementation:
...

Planned Changes:
...

Files Expected:
...

Dependencies:
...

Manual Tests:
...

Results:
...

Known Issues:
...

Architecture Decisions:
...

Git Commit:
...

Next Phase:
...
```

Never mark a phase Completed if the implementation or focused tests are unfinished.
