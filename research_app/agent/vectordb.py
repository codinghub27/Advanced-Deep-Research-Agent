import hashlib
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_SCORE_THRESHOLD = 0.85

_exact_cache: dict[str, str] = {}
# Phase 6: what was learned while producing an answer (citations, routing report), kept next to
# the answer so a cache hit can show its sources. Keyed by the answer text, additive to the dict
# above, and cleared with it.
_extras: dict[str, dict] = {}


def _answer_key(answer: str) -> str:
    return hashlib.sha256(answer.strip().encode("utf-8")).hexdigest()


def _normalize(question: str) -> str:
    cleaned = question.lower().strip()
    cleaned = cleaned.strip("?.!,;:\"'")
    return " ".join(cleaned.split())


def _calculate_similarity(text1: str, text2: str) -> float:
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if not words1 or not words2:
        return 0.0
    
    overlap = len(words1 & words2)
    total = len(words1 | words2)
    return overlap / total if total > 0 else 0.0


def lookup_cache(question: str) -> tuple[bool, str]:
    try:
        normalized = _normalize(question)
        
        logger.info(f"Cache lookup for: {normalized[:50]}")
        
        if normalized in _exact_cache:
            answer = _exact_cache[normalized]
            logger.info(f"Cache hit (exact) - {len(answer)} chars")
            return True, answer
        
        best_match = None
        best_score = 0.0
        
        for cached_q, cached_answer in _exact_cache.items():
            similarity = _calculate_similarity(normalized, cached_q)
            
            if similarity > best_score and similarity >= CACHE_SCORE_THRESHOLD:
                best_score = similarity
                best_match = (cached_q, cached_answer)
        
        if best_match:
            cached_q, cached_answer = best_match
            logger.info(f"Cache hit (semantic) score {best_score:.2f}")
            return True, cached_answer
        
        logger.info(f"Cache miss")
        return False, ""
        
    except Exception as e:
        logger.error(f"Cache lookup error: {e}")
        return False, ""


def store_cache(question: str, answer: str) -> None:
    try:
        normalized = _normalize(question)
        
        if not normalized or not answer:
            return
        
        answer_cleaned = answer.strip()
        if len(answer_cleaned) < 50:
            return
        
        _exact_cache[normalized] = answer_cleaned
        logger.info(f"Cache stored - {len(answer_cleaned)} chars, total items: {len(_exact_cache)}")
        
    except Exception as e:
        logger.error(f"Cache store error: {e}")


def store_cache_extras(answer: str, extras: dict) -> None:
    """Attach ``extras`` (citations etc.) to a cached answer."""
    if answer and answer.strip() and extras:
        _extras[_answer_key(answer)] = extras


def get_cache_extras(answer: str) -> Optional[dict]:
    """The extras stored with ``answer``, or None."""
    if not answer or not answer.strip():
        return None
    return _extras.get(_answer_key(answer))


def clear_cache():
    global _exact_cache
    _exact_cache.clear()
    _extras.clear()
    logger.info("Cache cleared")


def get_cache_stats():
    return {
        "total_items": len(_exact_cache),
        "items": list(_exact_cache.keys()),
    }
