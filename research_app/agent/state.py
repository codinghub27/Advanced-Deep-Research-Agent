import asyncio
import operator
import os
import logging
import re
import time
import weakref
from typing import Any, Optional, TypedDict, Annotated
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.types import Send
from research_app.agent.vectordb import get_cache_extras, lookup_cache, store_cache, store_cache_extras
from langgraph.graph import END
from research_app.agent.llms import llm_groq
from research_app.agent.pipeline.settings import PipelineSettings
from research_app.domain import (
    ORIGIN_FALLBACK,
    ResearchTask,
    ResultStatus,
    RoutingDecision,
    SourceIntent,
    SourceOutcome,
    SourceType,
)
from research_app.domain.legacy import (
    WEB_LABEL,
    community_entries,
    is_official_docs_entry,
    official_docs_entries,
    search_result_entry,
    select_search_results,
    source_to_legacy,
)
from research_app.sources import normalize_tavily_response
from research_app.sources.official_docs import OfficialDocsAdapter, plan_official_docs, prioritize
from research_app.sources.routing import (
    WEB_ONLY,
    RouterSettings,
    adapter_for,
    build_request,
    plan_technology_docs,
    route_sources,
    route_task,
    run_adapters,
    task_technology,
)

logger = logging.getLogger(__name__)

_CODE_FENCE = re.compile(r"(```.*?```)", re.S)


def sanitize_text(text: str) -> str:
    """Remove special characters, emojis, and formatting artifacts from text. Fenced code
    blocks are left exactly as written (collapsing their spaces destroyed indentation)."""
    if not text:
        return text
    parts = _CODE_FENCE.split(text)
    return "".join(part if index % 2 else _sanitize_prose(part) for index, part in enumerate(parts)).strip()


def _sanitize_prose(text: str) -> str:
    # Remove common special char pairs: 【】, ❌, ✅, ✓, 🔍, ⚡, 💾, ❌, 📊, etc.
    text = re.sub(r'【|】', '', text)
    text = re.sub(r'[❌✅✓✔✖🔍⚡💾📊📈🎯🚀💡🔬🧪]', '', text)
    
    # Remove other common emoji patterns
    text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
    
    # Remove multiple consecutive spaces/newlines
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove isolated special chars that might appear
    text = re.sub(r'[^\w\s\-.,;:!?()\[\]{}@#$%&*+=/<>|\\~`"\'\n]', '', text)

    return text

tavily_api = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearch(
    max_results=3,
    search_depth="advanced",
    include_answer=True,
    tavily_api_key=tavily_api
)

class ResearchState(TypedDict):
    question: str
    final_answer: str
    messages: Annotated[list, operator.add]
    step_count: int
    sources: Annotated[list, operator.add]
    sub_questions: list
    search_results: Annotated[list, operator.add]
    source_documents: Annotated[list, operator.add]  # normalized SourceDocument objects (Phase 3)
    cache_hit: bool
    api_limit_reached: bool
    critic_score: int
    critic_feedback: str
    is_simple: bool
    history: list
    # ---- Phase 6 ----------------------------------------------------------------------
    run_id: str
    session_id: Optional[int]
    turn_number: int
    conversation_context: Any  # ConversationContext, loaded by the route before the graph runs
    understanding: dict  # query understanding: intent, technology, time sensitivity, ...
    is_follow_up: bool
    resolved_query: str  # the question with references resolved; researched instead of ``question``
    research_tasks: list  # ResearchTask per sub-question (planner output, hints included)
    routing_decisions: Annotated[list, operator.add]  # RoutingDecision (+ outcomes) per search
    evidence_pool: Any
    gap_analysis: Any  # GapAnalysis
    gap_search_performed: bool
    confidence: str
    cached_routing: list  # routing report stored with a cached answer
    synthesis: Any  # SynthesisResult
    critic_result: Any  # CriticResult
    retry_performed: bool
    limitations: list
    citations: list  # what the UI and the database receive (list of dicts)
    rag_hit: bool  # Phase 7: answered from stored sources (source_chunks)
    rag_top_score: float  # Phase 7: best retrieval score of the lookup
    started_at: float

class SearchPlan(BaseModel):
    """Legacy planner output (Phase 1-5): plain sub-question strings. Still accepted."""
    sub_questions: list[str] = Field(description="3 targeted search queries")

class QuestionType(BaseModel):
    is_simple: bool

# The two models below are what the LLM is asked for. Every field is required (no defaults)
# so the JSON schema is valid for strict structured-output modes; "" / [] mean "none".

class QueryUnderstanding(BaseModel):
    is_simple: bool = Field(description="true only for a single factual answer")
    is_follow_up: bool = Field(description="true if the question depends on the earlier conversation")
    resolved_query: str = Field(description="the question rewritten to stand alone")
    intent: str = Field(description="unknown | factual | research | comparison | technical_howto | troubleshooting | code | current_events")
    technology: str = Field(description="main technology the question is about, or empty")
    time_sensitivity: str = Field(description="unknown | evergreen | recent | current")
    context_topics: list[str] = Field(description="earlier topics this question relies on")

class PlannedSubquestion(BaseModel):
    text: str = Field(description="a specific search query")
    source_intent: str = Field(description="technical_howto | library_usage | code_implementation | troubleshooting | community_experience | general_research | comparison | current_events")
    suggested_sources: list[str] = Field(description="subset of: official_docs, github, reddit, web")
    technology: str = Field(description="technology this sub-question is about, or empty")

class SourceAwarePlan(BaseModel):
    tasks: list[PlannedSubquestion] = Field(description="3 targeted sub-questions")

class Evaluate(BaseModel):
    score: int
    feedback: str

# ========== NODES ==========

def semantic_cache_node(state: ResearchState):
    """Check if question is cached, return answer if found."""
    try:
        question = state.get("question", "").strip()
        if not question:
            return {"cache_hit": False, "final_answer": "", "messages": []}
        
        context = state.get("conversation_context")
        if context is not None and getattr(context, "turns", None):
            # The cache is keyed by the raw question and shared by everyone: "deploy it" after a
            # FastAPI turn must not be answered with what someone else's "deploy it" meant.
            logger.info("Cache bypassed: the session has earlier turns")
            return {"cache_hit": False, "final_answer": "", "messages": []}

        logger.info(f"Checking cache: '{question[:70]}...'")
        cache_hit, cached_answer = lookup_cache(question)

        if cache_hit and cached_answer:
            logger.info(f"Cache hit - Returning cached answer")
            extras = get_cache_extras(cached_answer) or {}
            cached_answer = sanitize_text(cached_answer)
            update = {
                "cache_hit": True,
                "final_answer": cached_answer,
                "messages": [],
                "search_results": [],
                "sources": [],
                "sub_questions": ["(from cache)"],
            }
            if extras:  # Phase 6: the sources the answer was written from
                update["citations"] = extras.get("citations", [])
                update["cached_routing"] = extras.get("routing", [])
                update["confidence"] = extras.get("confidence", "")
            return update
        
        logger.info(f"Cache miss - proceeding to research")
        return {"cache_hit": False, "final_answer": "", "messages": []}
        
    except Exception as e:
        logger.error(f"Cache error: {e}")
        return {"cache_hit": False, "final_answer": "", "messages": []}

def cache_router(state: ResearchState):
    """Route to END if cache hit, otherwise classify."""
    if state.get("cache_hit"):
        logger.info("Cache hit - routing to END")
        return END
    return "classify_node"

MAX_RESOLVED_QUERY_CHARS = 500
_VALID_INTENTS = {"unknown", "factual", "research", "comparison", "technical_howto",
                  "troubleshooting", "code", "current_events"}
_VALID_TIME = {"unknown", "evergreen", "recent", "current"}


def _conversation_block(context) -> str:
    """The conversation-so-far section of the understanding / planner prompts. Empty when
    there is no history. Answers appear only as the short summaries the context holds."""
    turns = list(getattr(context, "turns", None) or [])
    if not turns:
        return ""
    lines = []
    for turn in turns:
        line = f'Turn {turn.turn_number}: the user asked "{turn.query_text}"'
        if turn.resolved_query and turn.resolved_query != turn.query_text:
            line += f' (understood as "{turn.resolved_query}")'
        if turn.answer_summary:
            line += f"; the answer began: {turn.answer_summary}"
        lines.append(line)
    techs = ", ".join(getattr(context, "technologies_mentioned", None) or [])
    tail = f"\nTechnologies discussed: {techs}" if techs else ""
    return "CONVERSATION SO FAR:\n" + "\n".join(lines) + tail + "\n\n"


def finalize_understanding(result, question: str, context) -> dict:
    """Turn the model's answer (or a failure: ``result`` is anything) into the understanding
    dict the graph stores. The model is not trusted: with no conversation history there is
    nothing to be a follow-up of, an empty or runaway rewrite falls back to the question, and
    a question that is not a follow-up is never rewritten."""
    has_history = bool(getattr(context, "turns", None))
    understood = isinstance(result, QueryUnderstanding)
    is_follow_up = bool(understood and result.is_follow_up and has_history)
    resolved = " ".join((result.resolved_query if understood else "").split())
    if not is_follow_up or not resolved or len(resolved) > MAX_RESOLVED_QUERY_CHARS:
        resolved = question
    intent = result.intent.strip().lower() if understood else "unknown"
    time_sensitivity = result.time_sensitivity.strip().lower() if understood else "unknown"
    return {
        "is_simple": bool(understood and result.is_simple),
        "is_follow_up": is_follow_up,
        "resolved_query": resolved,
        "intent": intent if intent in _VALID_INTENTS else "unknown",
        "technology": (result.technology.strip()[:100] if understood else ""),
        "time_sensitivity": time_sensitivity if time_sensitivity in _VALID_TIME else "unknown",
        "context_topics": [t.strip()[:120] for t in (result.context_topics if understood else [])
                           if isinstance(t, str) and t.strip()][:5] if is_follow_up else [],
    }


async def classify_node(state: ResearchState):
    """Query understanding: simple or complex, plus (Phase 6) intent, technology, time
    sensitivity and, when the session has history, whether this is a follow-up and what it
    refers to. One LLM call; any failure degrades to "complex, standalone" as before."""
    question = state["question"]
    context = state.get("conversation_context")
    try:
        classifier = llm_groq.with_structured_output(QueryUnderstanding, method="json_schema")
        result = await classifier.ainvoke(
            f"""Analyse the user's question.

{_conversation_block(context)}Question: {question}

Return ONLY JSON with these fields:
- is_simple: true only for a single factual answer (e.g. "capital of France?"); false when research is needed (e.g. "best laptops 2026?").
- is_follow_up: true if the question depends on the conversation above: pronouns (it, that, this, them), an implied subject ("now add authentication"), an explicit return ("going back to FastAPI"), or a comparison ("how does it compare to Django?"). false if it stands on its own or there is no conversation above.
- resolved_query: the question rewritten so it stands alone, replacing references with what they refer to, using ONLY the conversation above (e.g. "deploy it" after a FastAPI turn -> "deploy FastAPI"). If it is not a follow-up, repeat the question unchanged.
- intent: one of unknown, factual, research, comparison, technical_howto, troubleshooting, code, current_events.
- technology: the main technology, library or product the question is about (after resolving references), or "".
- time_sensitivity: one of unknown, evergreen, recent, current.
- context_topics: earlier topics the question relies on, or []."""
        )
        understanding = finalize_understanding(result, question, context)
    except Exception as exc:
        logger.warning("Query understanding failed (%s); treating the question as complex and standalone",
                       type(exc).__name__)
        understanding = finalize_understanding(None, question, context)
    logger.info("Query understanding: follow_up=%s simple=%s intent=%s technology=%s resolved=%r",
                understanding["is_follow_up"], understanding["is_simple"], understanding["intent"],
                understanding["technology"] or "-", understanding["resolved_query"][:80])
    return {
        "is_simple": understanding["is_simple"],
        "is_follow_up": understanding["is_follow_up"],
        "resolved_query": understanding["resolved_query"],
        "understanding": understanding,
    }

def classify_router(state: ResearchState):
    return "simple_search_node" if state["is_simple"] else "planner_node"

MAX_WEB_RESULTS = 3  # same cap as TavilySearch(max_results=3) above


def _web_search_update(query: str, response, *, content_limit: int, sink: Optional[list] = None) -> dict:
    """Normalize a Tavily response, then project it onto the state keys the rest
    of the graph (and the SSE ``sources`` event) already consume. ``sink`` (Phase 6)
    receives the normalization result, for the per-source outcome report."""
    result = normalize_tavily_response(response, query=query, max_results=MAX_WEB_RESULTS)
    if sink is not None:
        sink.append(result)
    if result.error:
        logger.warning("Web search for '%s' returned no usable results: %s",
                       query[:70], result.error.code)
    docs = result.documents
    return {
        "search_results": [search_result_entry(query, docs, content_limit=content_limit)],
        "sources": [source_to_legacy(d) for d in docs],
        "source_documents": docs,
    }

OFFICIAL_DOCS_CONTENT_LIMIT = 1000  # per document; the entry is also capped at 1500 characters


# One slot per in-flight source search, process-wide (per event loop), so fan-out across
# sub-questions and sources is bounded by CONCURRENCY_LIMIT instead of multiplying.
_search_slots: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def _search_slot() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    slot = _search_slots.get(loop)
    if slot is None:
        slot = asyncio.Semaphore(PipelineSettings.from_env().concurrency_limit)
        _search_slots[loop] = slot
    return slot


async def _tavily_invoke(payload: dict):
    """Every Tavily call (web, documentation, GitHub, Reddit) goes through here.
    ``tavily_tool`` is looked up at call time so it can be replaced (tests)."""
    async with _search_slot():
        return await tavily_tool.ainvoke(payload)


async def _tavily_search(payload: dict):
    """Search callable for the official-docs and community adapters."""
    return await _tavily_invoke(payload)


def _plan_docs_search(question: str, query: str):
    """An official-docs search plan, or None (= web only). Never raises: a detection or
    registry problem must not stop research."""
    try:
        return plan_official_docs(question, query)
    except Exception as exc:
        logger.warning("Official docs planning failed (%s); using web search only", type(exc).__name__)
        return None


async def _run_docs_search(plan):
    """Run the official-docs adapter; it returns a failed result instead of raising, and
    this guards the rest."""
    try:
        return await OfficialDocsAdapter(_tavily_search, plan.registry).search(plan.request)
    except Exception as exc:
        logger.warning("Official docs search crashed (%s); continuing with web results", type(exc).__name__)
        return None


def _add_official_docs(update: dict, docs_result, query: str) -> dict:
    """Put verified official documentation ahead of the web results in ``update``."""
    if docs_result is None or not docs_result.documents:
        return update  # failure/empty was already logged by the adapter
    try:
        docs = prioritize(list(docs_result.documents) + update["source_documents"])
        return {
            **update,
            "search_results": official_docs_entries(
                query, docs_result.documents, content_limit=OFFICIAL_DOCS_CONTENT_LIMIT
            ) + update["search_results"],
            "sources": [source_to_legacy(d) for d in docs],
            "source_documents": docs,
        }
    except Exception as exc:
        logger.warning("Could not merge official docs (%s); continuing with web results", type(exc).__name__)
        return update


COMMUNITY_CONTENT_LIMIT = 1000  # per GitHub/Reddit document; the entry is also capped at 1500 characters


def _route_for(text: str):
    """The source route for ``text`` (Phase 5). Never raises: if routing fails the question
    is researched with web (plus the Phase 4 documentation check) only."""
    try:
        return route_sources(text)
    except Exception as exc:
        logger.warning("Source routing failed (%s); using web search only", type(exc).__name__)
        return WEB_ONLY


def _plan_technology_docs_search(question: str, query: str):
    """Official-docs plan for a problem/troubleshooting question that names a registered
    technology, or None. Never raises."""
    try:
        return plan_technology_docs(question, query)
    except Exception as exc:
        logger.warning("Technology documentation planning failed (%s); skipping official docs", type(exc).__name__)
        return None


def _community_jobs(route, query: str) -> list:
    """(adapter, request) for each of GitHub/Reddit that the route selected. Never raises."""
    jobs = []
    try:
        timeout_s = RouterSettings.from_env().timeout_s
        for source_type in (SourceType.GITHUB, SourceType.REDDIT):
            if route.includes(source_type):
                request = build_request(source_type, query, timeout_s=timeout_s)
                if request is not None:
                    jobs.append((adapter_for(source_type, _tavily_search), request))
    except Exception as exc:
        logger.warning("Could not prepare GitHub/Reddit searches (%s); continuing without them", type(exc).__name__)
    return jobs


async def _community_results(task) -> list:
    """Results of the GitHub/Reddit task; ``run_adapters`` isolates source failures and
    this guards the rest."""
    try:
        return await task
    except Exception as exc:
        logger.warning("GitHub/Reddit search crashed (%s); continuing without them", type(exc).__name__)
        return []


def _add_community(update: dict, results: list, query: str, *, content_limit: int) -> dict:
    """Put GitHub and Reddit documents between official documentation and web results in
    ``update``, and label the web entry ``[WEB]`` so synthesis can tell the sources apart.
    Returns ``update`` unchanged when there is nothing to add or the merge fails."""
    try:
        github = [d for r in results if r.source_type == SourceType.GITHUB for d in r.documents]
        reddit = [d for r in results if r.source_type == SourceType.REDDIT for d in r.documents]
        if not (github or reddit):
            return update  # failures/empties were already logged by the adapters
        existing = update["source_documents"]
        official = [d for d in existing if d.source_type == SourceType.OFFICIAL_DOCS]
        web = [d for d in existing if d.source_type == SourceType.WEB]
        official_entries = [e for e in update["search_results"] if is_official_docs_entry(e)]
        web_entries = [e for e in update["search_results"] if not is_official_docs_entry(e)]
        if any(d.content for d in web):
            web_entries = [search_result_entry(query, web, content_limit=content_limit, label=WEB_LABEL)]
        docs = official + github + reddit + web
        return {
            **update,
            "search_results": official_entries
            + community_entries(query, github + reddit, content_limit=COMMUNITY_CONTENT_LIMIT)
            + web_entries,
            "sources": [source_to_legacy(d) for d in docs],
            "source_documents": docs,
        }
    except Exception as exc:
        logger.warning("Could not merge GitHub/Reddit results (%s); continuing without them", type(exc).__name__)
        return update


# QueryIntent value (from query understanding) -> the sub-question source intent. Values with no
# entry (factual / research / unknown) leave the intent unset, so the Phase 5 rules decide.
_SOURCE_INTENT_FOR = {
    "technical_howto": SourceIntent.TECHNICAL_HOWTO,
    "troubleshooting": SourceIntent.TROUBLESHOOTING,
    "code": SourceIntent.CODE_IMPLEMENTATION,
    "comparison": SourceIntent.COMPARISON,
    "current_events": SourceIntent.CURRENT_EVENTS,
}


def _decision_from_plan(task: ResearchTask, plan) -> RoutingDecision:
    """Report a question-level ``RoutePlan`` (used when task routing failed) as a decision."""
    sources = list(plan.sources)
    return RoutingDecision(
        task_id=task.task_id,
        sub_question=task.sub_question,
        sources=sources,
        origins={s.value: ORIGIN_FALLBACK for s in sources},
        docs_from_technology=bool(getattr(plan, "docs_from_technology", False)),
    )


def _decide_route(task: ResearchTask, understanding, text: str):
    """(route, decision) for one task. ``route`` has ``includes()`` and ``docs_from_technology``
    (a ``RoutingDecision``, or the Phase 5 ``RoutePlan`` if task routing failed). Never raises."""
    try:
        decision = route_task(task, understanding)
        return decision, decision
    except Exception as exc:
        logger.warning("Task routing failed (%s); using the question-level route", type(exc).__name__)
        plan = _route_for(text)
        return plan, _decision_from_plan(task, plan)


# The search itself worked but nothing usable came back: an empty result, not a failure.
_EMPTY_CODES = {"no_official_results", "no_domain_results"}


def _is_empty_not_failed(result) -> bool:
    error = result.error
    if error is None:
        return False
    if error.code in _EMPTY_CODES:
        return True
    return error.code == "provider_error" and error.message.lower().startswith("no search results")


def _outcome_from_result(source: SourceType, result, *, latency_ms: Optional[float] = None) -> SourceOutcome:
    if result is None:
        return SourceOutcome(source=source, status="failed", error_code="no_result")
    if result.status == ResultStatus.TIMEOUT:
        status = "timeout"
    elif result.status == ResultStatus.FAILED:
        status = "empty" if _is_empty_not_failed(result) else "failed"
    else:
        status = "ok" if result.documents else "empty"
    latency = result.latency_ms if result.latency_ms is not None else latency_ms
    return SourceOutcome(
        source=source, status=status, result_count=len(result.documents),
        latency_ms=round(latency, 1) if latency is not None else None,
        error_code=result.error.code if result.error else None,
    )


def _finish_task_update(update: dict, task: ResearchTask, decision: RoutingDecision, *, web_results: list,
                        web_ms: Optional[float], docs_selected: bool, docs_planned: bool, docs_result,
                        community_jobs: list, community_results: list) -> dict:
    """Stamp the documents with the task and attach the routing decision with what each selected
    source returned. Never raises: the report must not cost the research."""
    try:
        outcomes: list[SourceOutcome] = []
        for source in decision.sources:
            if source == SourceType.WEB:
                outcomes.append(_outcome_from_result(source, web_results[0] if web_results else None,
                                                     latency_ms=web_ms))
            elif source == SourceType.OFFICIAL_DOCS:
                if not docs_planned:
                    outcomes.append(SourceOutcome(source=source, status="skipped", error_code="no_docs_plan"))
                else:
                    outcomes.append(_outcome_from_result(source, docs_result))
            else:
                ran = any(getattr(req, "source_type", None) == source for _adapter, req in community_jobs)
                match = next((r for r in community_results if r.source_type == source), None)
                if not ran:
                    outcomes.append(SourceOutcome(source=source, status="skipped", error_code="not_run"))
                else:
                    outcomes.append(_outcome_from_result(source, match))
        documents = [d if d.task_id else d.model_copy(update={"task_id": task.task_id})
                     for d in update["source_documents"]]
        return {
            **update,
            "source_documents": documents,
            "routing_decisions": [decision.model_copy(update={"outcomes": outcomes})],
        }
    except Exception as exc:
        logger.warning("Could not build the routing report (%s); continuing without it", type(exc).__name__)
        return update


async def _research_update(query: str, *, question: str | None, content_limit: int,
                           task: Optional[ResearchTask] = None, understanding: Optional[dict] = None) -> dict:
    """Route, then search the selected sources concurrently: web always, plus official
    documentation, GitHub and/or Reddit when the router picked them. The optional sources can
    only add results, never remove or fail the web side. A raised web failure propagates as it
    always has (and cancels the others); Phase 13 owns web retries and timeouts.

    Without ``task`` this is the Phase 5 behaviour exactly (routing by the question text). With a
    ``task`` (Phase 6) the sub-question is routed by ``route_task`` and the update also carries the
    routing decision with per-source outcomes, and the documents are stamped with the task id."""
    text = question or query
    decision = None
    docs_selected = False
    if task is None:
        route = _route_for(text)
        plan = _plan_docs_search(text, query)
        if plan is None and route.docs_from_technology:
            plan = _plan_technology_docs_search(text, query)
    else:
        route, decision = _decide_route(task, understanding, text)
        if isinstance(route, RoutingDecision):
            docs_selected = route.includes(SourceType.OFFICIAL_DOCS)
            plan = None
            if docs_selected:
                docs_text = " ".join(
                    part for part in (query, (understanding or {}).get("resolved_query"),
                                      task_technology(task, understanding)) if part)
                plan = _plan_docs_search(docs_text, query) or _plan_technology_docs_search(docs_text, query)
        else:  # routing failed: the question-level Phase 4/5 behaviour
            plan = _plan_docs_search(text, query)
            if plan is None and route.docs_from_technology:
                plan = _plan_technology_docs_search(text, query)
            docs_selected = plan is not None
    jobs = _community_jobs(route, query)

    docs_task = asyncio.create_task(_run_docs_search(plan)) if plan is not None else None
    community_task = asyncio.create_task(run_adapters(jobs)) if jobs else None
    background = [t for t in (docs_task, community_task) if t is not None]
    web_results: list = []
    docs_result = None
    community_results: list = []
    web_ms: Optional[float] = None
    try:
        started = time.perf_counter()
        results = await _tavily_invoke({"query": query})
        web_ms = (time.perf_counter() - started) * 1000
        update = _web_search_update(query, results, content_limit=content_limit, sink=web_results)
        if docs_task is not None:
            docs_result = await docs_task
            update = _add_official_docs(update, docs_result, query)
        if community_task is not None:
            community_results = await _community_results(community_task)
            update = _add_community(update, community_results, query, content_limit=content_limit)
    except BaseException:
        for task_ in background:
            task_.cancel()
        raise
    if task is not None:
        update = _finish_task_update(
            update, task, decision, web_results=web_results, web_ms=web_ms, docs_selected=docs_selected,
            docs_planned=plan is not None, docs_result=docs_result, community_jobs=jobs,
            community_results=community_results)
    return update

async def simple_search_node(state: ResearchState):
    """Single search for simple questions."""
    understanding = state.get("understanding")
    if understanding is None:  # not routed through query understanding: the original behaviour
        query = state["question"]
        update = await _research_update(query, question=query, content_limit=400)
        update["sub_questions"] = [query]
        return update
    query = state.get("resolved_query") or state["question"]
    task = ResearchTask(
        sub_question=query,
        source_intent=_SOURCE_INTENT_FOR.get(understanding.get("intent")),
        technology=understanding.get("technology") or None,
    )
    update = await _research_update(query, question=query, content_limit=400, task=task,
                                    understanding=understanding)
    update["sub_questions"] = [query]
    update["research_tasks"] = [task]
    return update

MAX_SUB_QUESTIONS = 3
_MAX_HINT_CHARS = 40
_MAX_HINTS = 8


def _clean_hints(raw) -> list[str]:
    """Planner source suggestions as plain strings; anything else is dropped (the router
    validates the values themselves and reports the ones it rejects)."""
    if not isinstance(raw, (list, tuple)):
        return []
    hints = [h.strip()[:_MAX_HINT_CHARS] for h in raw if isinstance(h, str) and h.strip()]
    if len(hints) != len(raw):
        logger.warning("Planner returned %d malformed source suggestion(s); dropped", len(raw) - len(hints))
    return hints[:_MAX_HINTS]


def _parse_source_intent(value) -> Optional[SourceIntent]:
    try:
        return SourceIntent(value.strip().lower()) if isinstance(value, str) else None
    except ValueError:
        logger.warning("Planner gave an unknown source_intent; ignored")
        return None


def tasks_from_plan(result, question: str, technology: Optional[str] = None) -> list[ResearchTask]:
    """Planner output -> ``ResearchTask``s. Accepts the Phase 6 shape (``tasks`` with source
    hints) and the legacy shape (``sub_questions``: plain strings, no hints). Blank and duplicate
    sub-questions are dropped, at most ``MAX_SUB_QUESTIONS`` kept, and an empty plan falls back to
    the question itself."""
    tasks: list[ResearchTask] = []
    seen: set[str] = set()

    def add(text, **fields) -> None:
        text = " ".join(text.split()) if isinstance(text, str) else ""
        if not text or text.lower() in seen or len(tasks) >= MAX_SUB_QUESTIONS:
            return
        seen.add(text.lower())
        tasks.append(ResearchTask(sub_question=text[:500], **fields))

    for item in getattr(result, "tasks", None) or []:
        add(getattr(item, "text", ""),
            source_intent=_parse_source_intent(getattr(item, "source_intent", None)),
            suggested_sources=_clean_hints(getattr(item, "suggested_sources", None)),
            technology=(getattr(item, "technology", "") or "").strip()[:100] or None)
    if not tasks:
        for text in getattr(result, "sub_questions", None) or []:
            add(text)
    return tasks or [ResearchTask(sub_question=question, technology=technology or None)]


def _planner_prompt(question: str, original: str, context, understanding: dict) -> str:
    conversation = _conversation_block(context)
    rules = ""
    if conversation:
        rules = """RULES FOR THE CONVERSATION:
1. Research the resolved question above, not any ambiguous wording.
2. Do not re-research what earlier turns already covered; focus on the NEW information needed.
3. Build on what the conversation established, and only use facts stated in it.
4. If the question changes topic, plan independent sub-questions and do not force a connection.

"""
    wording = f"\n(The user's original wording: {original})" if original != question else ""
    technology = (understanding or {}).get("technology")
    hint = f"\nMain technology: {technology}" if technology else ""
    return f"""Generate {MAX_SUB_QUESTIONS} specific web search sub-questions for the question below, and say where each is best answered.

{conversation}Question: {question}{wording}{hint}

{rules}For each sub-question return:
- text: a specific search query
- source_intent: one of technical_howto, library_usage, code_implementation, troubleshooting, community_experience, general_research, comparison, current_events
- suggested_sources: any of official_docs, github, reddit, web
- technology: the technology it is about, or ""

Source purposes:
- official_docs: authoritative API, configuration, install and version facts
- github: code, repos, implementation examples, issues and pull requests
- reddit: real-world experience, troubleshooting stories, opinions
- web: general information
Only suggest sources that fit; a non-technical question needs only web.
Return ONLY JSON: {{"tasks": [{{"text": "...", "source_intent": "...", "suggested_sources": ["web"], "technology": ""}}]}}"""


async def planner_node(state: ResearchState):
    """Generate search sub-questions for complex questions, each with a source intent, suggested
    sources and technology (Phase 6). The suggestions are hints: the router validates them."""
    try:
        question = state.get("resolved_query") or state["question"]
        understanding = state.get("understanding") or {}
        planner = llm_groq.with_structured_output(SourceAwarePlan, method="json_schema")
        try:
            results = await planner.ainvoke(
                _planner_prompt(question, state["question"], state.get("conversation_context"), understanding))
            tasks = tasks_from_plan(results, question, understanding.get("technology"))
        except Exception as exc:
            logger.warning("Planner failed (%s); researching the question itself", type(exc).__name__)
            tasks = [ResearchTask(sub_question=question, technology=understanding.get("technology") or None)]

        sub_questions = [t.sub_question for t in tasks]
        sub_q_text = "\n".join(f"- {q}" for q in sub_questions)
        msg = HumanMessage(content=f"Search:\n{sub_q_text}")
        return {"messages": [msg], "sub_questions": sub_questions, "research_tasks": tasks}
    except Exception as e:
        logger.warning(f"Planner error: {e}")
        return {"messages": [], "sub_questions": [state["question"]]}

def planner_router(state: ResearchState):
    tasks = state.get("research_tasks")
    if not tasks:  # legacy state: no tasks, only sub-question strings
        return [Send("search_node", {"query": q, "question": state["question"]}) for q in state["sub_questions"]]
    return [
        Send("search_node", {"query": t.search_query or t.sub_question, "question": state["question"],
                             "task": t, "understanding": state.get("understanding")})
        for t in tasks
    ]

async def search_node(state: dict):
    """Run web search for a query."""
    query = state["query"]
    return await _research_update(query, question=state.get("question"), content_limit=800,
                                  task=state.get("task"), understanding=state.get("understanding"))

OFFICIAL_DOCS_GUIDANCE = """

OFFICIAL DOCUMENTATION GUIDANCE:
Some research data blocks are labelled [OFFICIAL DOCUMENTATION: ...]. For installation, API usage, configuration, technical behavior and version-specific facts, prefer those blocks over all other sources. If another source conflicts with them on these points, follow the official documentation."""


def _official_docs_guidance(state) -> str:
    """Empty (prompts unchanged) unless verified official documentation was retrieved."""
    docs = state.get("source_documents") or []
    if any(getattr(d, "source_type", None) == SourceType.OFFICIAL_DOCS for d in docs):
        return OFFICIAL_DOCS_GUIDANCE
    return ""

COMMUNITY_GUIDANCE = """

SOURCE LABEL GUIDANCE:
Research data blocks are labelled by source. [OFFICIAL DOCUMENTATION: ...] is authoritative documentation. [GITHUB: ...] is implementation and code evidence (repositories, examples, issues, pull requests): it shows how things are built and what problems are reported, not what is officially supported. [REDDIT: ...] is community experience and opinion: treat it as anecdotal, and never as authority for API or configuration behavior. [WEB] or unlabelled blocks are general web information. When a claim rests on GitHub or Reddit evidence, say so."""


def _community_guidance(state) -> str:
    """Empty (prompts unchanged) unless verified GitHub or Reddit documents were retrieved."""
    docs = state.get("source_documents") or []
    if any(getattr(d, "source_type", None) in (SourceType.GITHUB, SourceType.REDDIT) for d in docs):
        return COMMUNITY_GUIDANCE
    return ""

async def synthesize_node(state: ResearchState):
    """Generate final answer from search results."""
    try:
        logger.info("Generating professional report...")
        
        trimmed = [r[:1500] for r in select_search_results(state["search_results"], 12)]
        guidance = _official_docs_guidance(state) + _community_guidance(state)
        all_results = "\n\n".join(trimmed)
        
        prompt = f"""Write a comprehensive professional research report answering: {state['question']}

RESEARCH DATA:
{all_results}{guidance}

REPORT STRUCTURE:

Title: Professional descriptive title (not repeating the question)

Introduction
2-3 sentences explaining context and why this topic matters

1. Main Topic Section
Write 150-200 words. Include specific facts, numbers, names, and years from the data. Use citations format: [Source, Year]

2. Second Topic Section
Write 150-200 words with specific details

3. Third Topic Section  
Write 150-200 words with specific details

Comparison Section
If comparing multiple options, create a comparison showing each option's key features and metrics

Conclusion
150-200 word summary with key takeaways and recommendations

CRITICAL REQUIREMENTS:
- Write complete paragraphs and sentences (NO bullet points except in tables)
- Include specific numbers: percentages, fees, years, counts, rankings
- Use citations: [Source Name, Year]
- Minimum 1000 words
- Professional and analytical tone
- NO raw URLs embedded in text
- NO phrases like "Based on data" or "It appears that"
- Ground all claims in provided data
- Make comparisons between options if multiple exist

Now write the complete professional report:"""
        
        result = await llm_groq.ainvoke(prompt)
        answer = result.content if isinstance(result.content, str) else str(result.content)
        answer = answer.strip()
        answer = sanitize_text(answer)
        
        if not answer or len(answer) < 500:
            logger.info("Answer too short, using fallback")
            return await _synthesize_fallback(state)
        
        url_count = answer.count("http")
        words = len(answer.split())
        if url_count > 5 and words < 500:
            logger.info("Too many URLs relative to content, using fallback")
            return await _synthesize_fallback(state)
        
        logger.info(f"Report generated: {len(answer)} chars")
        return {"final_answer": answer}
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return await _synthesize_fallback(state)

async def _synthesize_fallback(state: ResearchState):
    """Fallback synthesis when main fails."""
    try:
        trimmed = [r[:1000] for r in select_search_results(state["search_results"], 9)]
        guidance = _official_docs_guidance(state) + _community_guidance(state)
        all_results = "\n\n".join(trimmed)
        
        prompt = f"""Question: {state['question']}

Data:
{all_results}{guidance}

Write a detailed professional report with title, introduction, 3 main sections with specific facts, comparison if relevant, and conclusion. Minimum 800 words. Use [Source, Year] format for citations. Complete sentences only."""
        
        result = await llm_groq.ainvoke(prompt)
        answer = result.content if isinstance(result.content, str) else str(result.content)
        answer = answer.strip()
        answer = sanitize_text(answer)
        
        if answer and len(answer) > 300:
            logger.info(f"Fallback report generated: {len(answer)} chars")
            return {"final_answer": answer}
        else:
            return {"final_answer": "Unable to generate comprehensive report. Please try again."}
            
    except Exception as e:
        logger.error(f"Fallback synthesis error: {e}")
        return {"final_answer": "System error generating report. Please try again."}


def save_to_cache_node(state: ResearchState):
    """Save answer to cache."""
    try:
        if state.get("cache_hit"):
            return {}
        
        final_answer = state.get("final_answer", "").strip()
        question = state.get("question", "").strip()
        
        if not question or not final_answer or len(final_answer) < 50:
            return {}
        if state.get("is_follow_up"):
            # Its meaning depends on the conversation, so it is not an answer to the raw wording.
            logger.info("Not caching a follow-up answer")
            return {}
        if "synthesis" in state and not state.get("citations"):
            logger.info("Not caching an answer without citations")
            return {}

        logger.info("Saving to cache...")
        store_cache(question, final_answer)
        if state.get("citations"):
            store_cache_extras(final_answer, {
                "citations": state["citations"],
                "routing": [d.model_dump(mode="json") for d in state.get("routing_decisions") or []],
                "confidence": state.get("confidence", ""),
            })
        
    except Exception as e:
        logger.error(f"Cache save error: {e}")
    
    return {}
