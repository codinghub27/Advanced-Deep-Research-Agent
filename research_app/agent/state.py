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

async def simple_search_node(state: ResearchState):
    """Single search for simple questions."""
    query = state["question"]
    results = await tavily_tool.ainvoke({"query": query})
    
    condensed = []
    sources = []
    result_list = results.get("results", []) if isinstance(results, dict) else []
    for r in result_list[:3]:
        content = r.get("content", "")[:400]
        url = r.get("url", "")
        title = r.get("title", "")
        if content:
            condensed.append(f"{url}\n{content}")
        if url:
            sources.append({"url": url, "title": title, "snippet": content[:150]})
    
    return {
        "search_results": [f"Query: {query}\n" + "\n---\n".join(condensed)],
        "sub_questions": [query],
        "sources": sources,
    }

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
    return [Send("search_node", {"query": q}) for q in state["sub_questions"]]

async def search_node(state: dict):
    """Run web search for a query."""
    query = state["query"]
    results = await tavily_tool.ainvoke({"query": query})
    
    condensed = []
    sources = []
    result_list = results.get("results", []) if isinstance(results, dict) else []
    for r in result_list[:3]:
        content = r.get("content", "")[:800]
        url = r.get("url", "")
        title = r.get("title", "")
        if content:
            condensed.append(f"{url}\n{content}")
        if url:
            sources.append({"url": url, "title": title, "snippet": content[:150]})
    
    return {
        "search_results": [f"Query: {query}\n" + "\n---\n".join(condensed)],
        "sources": sources,
    }

async def synthesize_node(state: ResearchState):
    """Generate final answer from search results."""
    try:
        logger.info("Generating professional report...")
        
        trimmed = [r[:1500] for r in state["search_results"][:12]]
        all_results = "\n\n".join(trimmed)
        
        prompt = f"""Write a comprehensive professional research report answering: {state['question']}

RESEARCH DATA:
{all_results}

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
        trimmed = [r[:1000] for r in state["search_results"][:9]]
        all_results = "\n\n".join(trimmed)
        
        prompt = f"""Question: {state['question']}

Data:
{all_results}

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
