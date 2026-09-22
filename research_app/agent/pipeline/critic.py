"""Critic (Phase 6): is the answer covered, supported, cited correctly and well sourced?

Two layers. Deterministic checks (no LLM) catch what code can see for certain: citation numbers
that name no source, paragraphs with no citation, a technical answer resting only on Reddit,
official documentation that was retrieved but not cited, sub-questions with no evidence. One LLM
call judges what only reading can: does the answer address the question, and do the cited
sources really support its claims. If that call fails the deterministic verdict stands.

Verdict
  bad                two or more high-severity issues, or no citations although evidence exists
  needs_improvement  any high or medium issue
  good               nothing, or only low-severity issues
A retry is worthwhile for ``bad``, or ``needs_improvement`` with a high-severity issue.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

from pydantic import BaseModel, Field

import research_app.agent.state as st
from research_app.agent import temporal
from research_app.agent.pipeline.evidence import EvidencePool
from research_app.agent.pipeline.gaps import keyword_query, normalize_query
from research_app.agent.pipeline.synthesis import (
    MAX_ANSWER_CHARS_FOR_CRITIC,
    _CODE,
    build_sources,
    find_markers,
    is_technical,
)
from research_app.domain import (
    CriticIssue,
    CriticIssueKind,
    CriticResult,
    CriticSeverity,
    CriticVerdict,
    SourceType,
    SynthesisResult,
    canonical_url,
)

logger = logging.getLogger(__name__)

MAX_SUGGESTIONS = 3
CRITIC_SOURCE_CHARS = 500
MIN_SUBSTANTIVE_PARAGRAPH = 100
MANY_UNSUPPORTED = 3
HIGH, MEDIUM = CriticSeverity.HIGH, CriticSeverity.MEDIUM


class CriticReport(BaseModel):
    """What the LLM is asked for. All fields required (strict structured-output friendly)."""
    covers_question: bool = Field(description="true if the answer addresses what the question asks")
    unsupported_claims: list[str] = Field(description="up to 3 claims in the answer the cited sources do not support")
    missing_information: list[str] = Field(description="important aspects of the question the answer leaves out")
    suggested_searches: list[str] = Field(description="up to 2 short web search queries that would fill the gaps")


# --------------------------------------------------------------------------- deterministic

def _paragraphs(text: str) -> list[str]:
    stripped = _CODE.sub("", text)
    return [p.strip() for p in stripped.split("\n\n") if p.strip()]


def deterministic_issues(synthesis: SynthesisResult, pool: EvidencePool, understanding: Optional[dict]) -> list[CriticIssue]:
    issues: list[CriticIssue] = []
    text = synthesis.answer_text or ""
    citations = synthesis.citations
    have_evidence = len(pool) > 0

    if not text.strip():
        return [CriticIssue(kind=CriticIssueKind.MISSING_COVERAGE, severity=HIGH,
                            description="The answer is empty.")]

    # Citation validity: every [N] must name a listed citation.
    listed = {c.index for c in citations}
    orphans = sorted({n for n in find_markers(text) if n not in listed})
    if orphans:
        issues.append(CriticIssue(
            kind=CriticIssueKind.ORPHAN_CITATION, severity=HIGH,
            description="The answer cites source number(s) that do not exist: " + ", ".join(str(n) for n in orphans)))

    # Evidence support (structural): are claims cited at all?
    if have_evidence and not citations:
        issues.append(CriticIssue(kind=CriticIssueKind.UNSUPPORTED_CLAIM, severity=HIGH,
                                  description="No claim in the answer cites a source."))
    elif citations:
        substantive = [p for p in _paragraphs(text)
                       if len(p) >= MIN_SUBSTANTIVE_PARAGRAPH and not p.lstrip().startswith("#")]
        uncited = [p for p in substantive if not find_markers(p)]
        if len(uncited) >= 2 and len(uncited) * 2 > len(substantive):
            issues.append(CriticIssue(
                kind=CriticIssueKind.UNSUPPORTED_CLAIM, severity=MEDIUM,
                description=f"{len(uncited)} of {len(substantive)} paragraphs make claims without a citation."))

    # Source appropriateness: technical facts need better than Reddit.
    cited_types = {c.source_type for c in citations}
    numbered = build_sources(pool)
    technical = is_technical(understanding, pool.tasks, numbered)
    if technical and citations:
        if cited_types == {SourceType.REDDIT}:
            issues.append(CriticIssue(
                kind=CriticIssueKind.WEAK_SOURCE, severity=HIGH,
                description="A technical answer rests only on Reddit; community posts are not authority "
                            "for API, configuration or installation facts."))
        elif (SourceType.OFFICIAL_DOCS in {s.source_type for s in numbered}
              and SourceType.OFFICIAL_DOCS not in cited_types):
            issues.append(CriticIssue(
                kind=CriticIssueKind.WEAK_SOURCE, severity=MEDIUM,
                description="Official documentation was retrieved but the answer does not cite it."))

    # Completeness: sub-questions with no evidence.
    missed = synthesis.subquestions_missed
    if missed and have_evidence:
        total = len(missed) + len(synthesis.subquestions_covered)
        issues.append(CriticIssue(
            kind=CriticIssueKind.INCOMPLETE, severity=HIGH if len(missed) * 2 > total else MEDIUM,
            description="No evidence was found for: " + "; ".join(missed)))
    elif missed and not have_evidence:
        issues.append(CriticIssue(kind=CriticIssueKind.INCOMPLETE, severity=HIGH,
                                  description="No evidence was found for any part of the question."))
    return issues


# --------------------------------------------------------------------------- LLM check

def _critic_prompt(question: str, synthesis: SynthesisResult, pool: EvidencePool) -> str:
    by_url = {canonical_url(s.url): s for s in build_sources(pool)}
    blocks = []
    for c in synthesis.citations:
        source = by_url.get(canonical_url(c.url))
        body = (source.content if source else c.snippet or "")[:CRITIC_SOURCE_CHARS]
        blocks.append(f"[{c.index}] {c.title} ({c.source_type.value})\n{body}")
    subquestions = "\n".join(f"- {t.sub_question}" for t in pool.tasks)
    return f"""You are a strict reviewer of a research answer.

{temporal.date_line()}
If the answer presents something as the latest or current although the cited sources show it is more than a year old, count that as an unsupported claim.

QUESTION: {question}

SUB-QUESTIONS THE RESEARCH TRIED TO ANSWER:
{subquestions}

CITED SOURCES (the only evidence the answer may rely on):
{chr(10).join(blocks)}

ANSWER UNDER REVIEW:
{synthesis.answer_text[:MAX_ANSWER_CHARS_FOR_CRITIC]}

Judge it and return ONLY JSON with:
- covers_question: true if the answer addresses what the question asks.
- unsupported_claims: up to 3 specific claims that the cited sources above do not support (quote briefly). [] if none.
- missing_information: important aspects of the question the answer leaves out. [] if none.
- suggested_searches: up to 2 short web search queries that would find the missing information. [] if none."""


async def _llm_report(question: str, synthesis: SynthesisResult, pool: EvidencePool) -> Optional[CriticReport]:
    try:
        judge = st.llm_groq.with_structured_output(CriticReport, method="json_schema")
        report = await judge.ainvoke(_critic_prompt(question, synthesis, pool))
        return report if isinstance(report, CriticReport) else None
    except Exception as exc:
        logger.warning("Critic LLM check failed (%s); using the deterministic checks only", type(exc).__name__)
        return None


# --------------------------------------------------------------------------- verdict

def _verdict(issues: Sequence[CriticIssue], synthesis: SynthesisResult, have_evidence: bool) -> CriticVerdict:
    highs = [i for i in issues if i.severity == HIGH]
    unsupported_all = have_evidence and not synthesis.citations
    if len(highs) >= 2 or unsupported_all:
        return CriticVerdict.BAD
    if any(i.severity in (HIGH, MEDIUM) for i in issues):
        return CriticVerdict.NEEDS_IMPROVEMENT
    return CriticVerdict.GOOD


def _suggestions(report: Optional[CriticReport], synthesis: SynthesisResult) -> list[str]:
    """Queries a retry could run: the model's, then key terms of what was missing, then of the
    sub-questions that had no evidence. De-duplicated, in that order."""
    candidates: list[str] = []
    if report:
        candidates += [q.strip()[:200] for q in report.suggested_searches if isinstance(q, str) and q.strip()]
        candidates += [keyword_query(m) for m in report.missing_information
                       if isinstance(m, str) and m.strip()]
    candidates += [keyword_query(q) for q in synthesis.subquestions_missed]
    out: list[str] = []
    seen: set[str] = set()
    for query in candidates:
        key = normalize_query(query)
        if key and key not in seen:
            seen.add(key)
            out.append(query)
    return out[:MAX_SUGGESTIONS + 2]


async def run_critic(state, synthesis: SynthesisResult) -> CriticResult:
    """Review ``synthesis`` against ``state['evidence_pool']``. Never raises."""
    try:
        pool: EvidencePool = state["evidence_pool"]
        understanding = state.get("understanding")
        question = state.get("resolved_query") or state["question"]
        issues = deterministic_issues(synthesis, pool, understanding)
        report = await _llm_report(question, synthesis, pool) if synthesis.citations else None
        if report is not None:
            if not report.covers_question:
                issues.append(CriticIssue(kind=CriticIssueKind.MISSING_COVERAGE, severity=HIGH,
                                          description="The answer does not address the question."))
            # Calibrated on live runs: a model reviewer finds something to say about every long answer, so
            # its findings alone must not make "good" unreachable. One or two flagged claims are noted
            # (low); three, or an answer that misses the question, are a real problem (medium / high).
            claims = [c for c in report.unsupported_claims if isinstance(c, str) and c.strip()][:3]
            claim_severity = MEDIUM if len(claims) >= MANY_UNSUPPORTED else CriticSeverity.LOW
            for claim in claims:
                issues.append(CriticIssue(kind=CriticIssueKind.UNSUPPORTED_CLAIM, severity=claim_severity,
                                          description=f"Not supported by the cited sources: {claim.strip()[:200]}"))
            for item in [m for m in report.missing_information if isinstance(m, str) and m.strip()][:3]:
                issues.append(CriticIssue(kind=CriticIssueKind.INCOMPLETE, severity=CriticSeverity.LOW,
                                          description=f"Missing: {item.strip()[:200]}"))
        verdict = _verdict(issues, synthesis, len(pool) > 0)
        return CriticResult(
            passed=verdict == CriticVerdict.GOOD,
            verdict=verdict,
            issues=issues,
            suggestions=_suggestions(report, synthesis),
            feedback="; ".join(i.description for i in issues)[:1000],
        )
    except Exception as exc:
        logger.warning("Critic failed (%s); the answer is returned unreviewed", type(exc).__name__)
        return CriticResult(passed=True, verdict=None, feedback="critic unavailable")


async def critic_node(state):
    result = await run_critic(state, state["synthesis"])
    logger.info("Critic: verdict=%s issues=%s", result.verdict.value if result.verdict else "unavailable",
                [f"{i.kind.value}:{i.severity.value if i.severity else '-'}" for i in result.issues])
    return {"critic_result": result}


def worth_retrying(result: Optional[CriticResult]) -> bool:
    """A retry can help when the verdict is bad, or needs improvement with a high-severity issue."""
    if result is None or result.verdict is None:
        return False
    if result.verdict == CriticVerdict.BAD:
        return True
    return result.verdict == CriticVerdict.NEEDS_IMPROVEMENT and any(i.severity == HIGH for i in result.issues)


VERDICT_RANK = {CriticVerdict.GOOD: 3, CriticVerdict.NEEDS_IMPROVEMENT: 2, None: 2, CriticVerdict.BAD: 1}
