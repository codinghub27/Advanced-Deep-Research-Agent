import asyncio
import operator
import os
import logging
import re
from typing import TypedDict, Annotated
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.types import Send
from research_app.agent.vectordb import lookup_cache, store_cache
from langgraph.graph import END
from research_app.agent.llms import llm_groq
from research_app.domain import SourceType
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
    run_adapters,
)

logger = logging.getLogger(__name__)

def sanitize_text(text: str) -> str:
    """Remove special characters, emojis, and formatting artifacts from text."""
    if not text:
        return text
    
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
    
    return text.strip()

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

class SearchPlan(BaseModel):
    sub_questions: list[str] = Field(description="3 targeted search queries")

class QuestionType(BaseModel):
    is_simple: bool

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
        
        logger.info(f"Checking cache: '{question[:70]}...'")
        cache_hit, cached_answer = lookup_cache(question)
        
        if cache_hit and cached_answer:
            logger.info(f"Cache hit - Returning cached answer")
            cached_answer = sanitize_text(cached_answer)
            return {
                "cache_hit": True,
                "final_answer": cached_answer,
                "messages": [],
                "search_results": [],
                "sources": [],
                "sub_questions": ["(from cache)"],
            }
        
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

async def classify_node(state: ResearchState):
    """Classify if question is simple or complex."""
    try:
        classifier = llm_groq.with_structured_output(QuestionType, method="json_schema")
        result = await classifier.ainvoke(
            f"""Classify as simple or complex.
Question: {state['question']}

SIMPLE: single factual answer (e.g., "capital of France?")
COMPLEX: needs research (e.g., "best laptops 2026?")"""
        )
        return {"is_simple": result.is_simple}
    except:
        return {"is_simple": False}

def classify_router(state: ResearchState):
    return "simple_search_node" if state["is_simple"] else "planner_node"

MAX_WEB_RESULTS = 3  # same cap as TavilySearch(max_results=3) above


def _web_search_update(query: str, response, *, content_limit: int) -> dict:
    """Normalize a Tavily response, then project it onto the state keys the rest
    of the graph (and the SSE ``sources`` event) already consume."""
    result = normalize_tavily_response(response, query=query, max_results=MAX_WEB_RESULTS)
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


async def _tavily_search(payload: dict):
    """Search callable for the official-docs adapter. ``tavily_tool`` is looked up at
    call time so it can be replaced (tests)."""
    return await tavily_tool.ainvoke(payload)


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


async def _research_update(query: str, *, question: str | None, content_limit: int) -> dict:
    """Route the question (Phase 5), then search the selected sources concurrently: web
    always, plus official documentation, GitHub and/or Reddit when the router picked them.
    The optional sources can only add results, never remove or fail the web side. A raised
    web failure propagates as it always has (and cancels the others); Phase 13 owns web
    retries and timeouts."""
    text = question or query
    route = _route_for(text)
    plan = _plan_docs_search(text, query)
    if plan is None and route.docs_from_technology:
        plan = _plan_technology_docs_search(text, query)
    jobs = _community_jobs(route, query)

    docs_task = asyncio.create_task(_run_docs_search(plan)) if plan is not None else None
    community_task = asyncio.create_task(run_adapters(jobs)) if jobs else None
    background = [t for t in (docs_task, community_task) if t is not None]
    try:
        results = await tavily_tool.ainvoke({"query": query})
        update = _web_search_update(query, results, content_limit=content_limit)
        if docs_task is not None:
            update = _add_official_docs(update, await docs_task, query)
        if community_task is not None:
            update = _add_community(update, await _community_results(community_task), query,
                                    content_limit=content_limit)
    except BaseException:
        for task in background:
            task.cancel()
        raise
    return update

async def simple_search_node(state: ResearchState):
    """Single search for simple questions."""
    query = state["question"]
    update = await _research_update(query, question=query, content_limit=400)
    update["sub_questions"] = [query]
    return update

async def planner_node(state: ResearchState):
    """Generate 3 search queries for complex questions."""
    try:
        planner = llm_groq.with_structured_output(SearchPlan, method="json_schema")
        prompt = f"""Generate 3 specific web search queries for: {state['question']}
Return ONLY JSON: {{"sub_questions": ["q1", "q2", "q3"]}}"""
        
        try:
            results = await planner.ainvoke(prompt)
            sub_questions = results.sub_questions[:3]
        except:
            # Fallback to original question
            sub_questions = [state["question"]]
        
        sub_q_text = "\n".join(f"- {q}" for q in sub_questions)
        msg = HumanMessage(content=f"Search:\n{sub_q_text}")
        return {"messages": [msg], "sub_questions": sub_questions}
    except Exception as e:
        logger.warning(f"Planner error: {e}")
        return {"messages": [], "sub_questions": [state["question"]]}

def planner_router(state: ResearchState):
    return [Send("search_node", {"query": q, "question": state["question"]}) for q in state["sub_questions"]]

async def search_node(state: dict):
    """Run web search for a query."""
    query = state["query"]
    return await _research_update(query, question=state.get("question"), content_limit=800)

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
        
        logger.info("Saving to cache...")
        store_cache(question, final_answer)
        
    except Exception as e:
        logger.error(f"Cache save error: {e}")
    
    return {}
