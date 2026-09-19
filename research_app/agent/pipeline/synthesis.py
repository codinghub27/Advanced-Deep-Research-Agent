"""Evidence-only synthesis with numbered citations (Phase 6).

The LLM writes plain text and cites the numbered sources it was given as ``[1]``, ``[2]``. Code
does the rest: it finds the markers (never inside code), renumbers them by first use, builds the
``Citation`` list, and derives ``confidence`` and the covered / missed sub-questions from the
evidence. Nothing structural depends on the model producing JSON.

The sources shown to the model are the pool's evidence, one numbered block per URL, most
authoritative first (official documentation, GitHub, Reddit, web). The model is told to use
nothing else and to say so when the evidence does not cover something.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import research_app.agent.state as st
from research_app.agent.pipeline.evidence import EvidencePool
from research_app.domain import (
    Citation,
    ResearchTask,
    SourceDocument,
    SourceType,
    SynthesisResult,
    canonical_url,
)
from research_app.domain.legacy import source_label
from research_app.sources.routing import SOURCE_ORDER, TECHNICAL_INTENTS

logger = logging.getLogger(__name__)

MAX_PROMPT_SOURCES = 12
OFFICIAL_CONTENT_CHARS = 1500
OTHER_CONTENT_CHARS = 1000
SNIPPET_CHARS = 300
MAX_ANSWER_CHARS_FOR_CRITIC = 6000

_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
# [1] or [1, 2]; not a markdown link "[1](url)" or a reference definition "[1]: url".
_MARKER = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\](?![(:])")


# --------------------------------------------------------------------------- markers

def _outside_code(text: str, transform) -> str:
    """Apply ``transform`` to the parts of ``text`` that are not code."""
    out, last = [], 0
    for match in _CODE.finditer(text):
        out.append(transform(text[last:match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(transform(text[last:]))
    return "".join(out)


def normalize_markers(text: str) -> str:
    """``[1, 2]`` -> ``[1][2]`` (outside code), so every marker is a single number."""
    def expand(segment: str) -> str:
        return _MARKER.sub(
            lambda m: "".join(f"[{int(n)}]" for n in re.split(r"\s*,\s*", m.group(1))), segment)
    return _outside_code(text, expand)


def find_markers(text: str) -> list[int]:
    """Citation numbers in order of appearance (with repeats), ignoring code."""
    found: list[int] = []

    def collect(segment: str) -> str:
        for m in _MARKER.finditer(segment):
            found.extend(int(n) for n in re.split(r"\s*,\s*", m.group(1)))
        return segment

    _outside_code(text, collect)
    return found


def renumber_markers(text: str, mapping: dict[int, int]) -> str:
    """Rewrite single-number markers through ``mapping``; unmapped numbers are left as they are."""
    def swap(segment: str) -> str:
        return _MARKER.sub(lambda m: f"[{mapping.get(int(m.group(1)), int(m.group(1)))}]", segment)
    return _outside_code(text, swap)


# --------------------------------------------------------------------------- sources

@dataclass
class NumberedSource:
    index: int
    source_type: SourceType
    title: str
    url: str
    domain: str
    source_id: str
    retrieved_at: object
    snippet: str
    content: str
    label: str
    documents: list[SourceDocument] = field(default_factory=list)


def build_sources(pool: EvidencePool, limit: int = MAX_PROMPT_SOURCES) -> list[NumberedSource]:
    """One numbered source per URL. Over ``limit`` the slots are shared round-robin between the
    source types (so a selected source is not cut off), then numbered official-first."""
    grouped: dict[str, list[SourceDocument]] = {}
    for doc in pool.all_evidence():
        grouped.setdefault(canonical_url(doc.url), []).append(doc)

    chosen = list(grouped.values())
    if len(chosen) > limit:
        by_type: dict[SourceType, list[list[SourceDocument]]] = {}
        for docs in chosen:
            by_type.setdefault(docs[0].source_type, []).append(docs)
        picked: list[list[SourceDocument]] = []
        for position in range(max(len(v) for v in by_type.values())):
            for source in SOURCE_ORDER:
                bucket = by_type.get(source, [])
                if position < len(bucket) and len(picked) < limit:
                    picked.append(bucket[position])
        rank = {s: i for i, s in enumerate(SOURCE_ORDER)}
        chosen = sorted(picked, key=lambda docs: rank.get(docs[0].source_type, len(rank)))

    sources: list[NumberedSource] = []
    for number, docs in enumerate(chosen, start=1):
        first = docs[0]
        cap = OFFICIAL_CONTENT_CHARS if first.source_type == SourceType.OFFICIAL_DOCS else OTHER_CONTENT_CHARS
        texts: list[str] = []
        for doc in docs:
            text = (doc.content or doc.snippet or "").strip()
            if text and text not in texts:
                texts.append(text)
        content = "\n...\n".join(texts)[:cap]
        sources.append(NumberedSource(
            index=number, source_type=first.source_type,
            title=(first.title or first.domain or first.url).strip(),
            url=first.url, domain=first.domain or "", source_id=first.source_id or "",
            retrieved_at=first.retrieved_at,
            snippet=(first.snippet or content)[:SNIPPET_CHARS],
            content=content, label=source_label(first), documents=docs,
        ))
    return sources


def _citation(source: NumberedSource, index: int) -> Citation:
    return Citation(
        marker=f"[{index}]", index=index, source_id=source.source_id, url=source.url,
        title=source.title, source_type=source.source_type, domain=source.domain,
        snippet=source.snippet, retrieved_at=source.retrieved_at,
    )


# --------------------------------------------------------------------------- prompt

def _guidance(sources: Sequence[NumberedSource]) -> str:
    types = {s.source_type for s in sources}
    text = ""
    if SourceType.OFFICIAL_DOCS in types:
        text += st.OFFICIAL_DOCS_GUIDANCE
    if types & {SourceType.GITHUB, SourceType.REDDIT}:
        text += st.COMMUNITY_GUIDANCE
    return text


def _subquestion_lines(pool: EvidencePool, sources: Sequence[NumberedSource]) -> str:
    number_of = {canonical_url(s.url): s.index for s in sources}
    lines = []
    for position, (task_id, docs) in enumerate(pool.by_subquestion().items(), start=1):
        task = pool.task(task_id)
        numbers = sorted({number_of[canonical_url(d.url)] for d in docs if canonical_url(d.url) in number_of})
        evidence = ", ".join(f"[{n}]" for n in numbers) if numbers else "NO EVIDENCE FOUND"
        lines.append(f"{position}. {task.sub_question if task else task_id} - evidence: {evidence}")
    return "\n".join(lines)


def build_prompt(question: str, original: str, sources: Sequence[NumberedSource], pool: EvidencePool,
                 context, feedback: str = "") -> str:
    blocks = "\n\n".join(f"[{s.index}] {s.label}\n{s.url}\n{s.content}" for s in sources)
    wording = f"\n(The user's original wording: {original})" if original and original != question else ""
    conversation = st._conversation_block(context)
    if conversation:
        conversation = ("EARLIER IN THIS CONVERSATION (context only: it is not evidence, do not cite it "
                        "and do not repeat it):\n" + conversation.split("\n", 1)[1])
    review = (f"\nA REVIEWER FOUND THESE PROBLEMS WITH A PREVIOUS ANSWER; do not repeat them: {feedback}\n"
              if feedback else "")
    return f"""Write a clear, well-organised answer to the question, using ONLY the numbered sources below.

QUESTION: {question}{wording}
{review}
{conversation}SUB-QUESTIONS AND THEIR EVIDENCE:
{_subquestion_lines(pool, sources)}

SOURCES:
{blocks}{_guidance(sources)}

RULES:
- Use only the sources above. Do not add facts from memory. If the sources do not cover part of the question, say so plainly instead of guessing.
- Cite every factual claim with the number of the source that supports it, at the end of the sentence or bullet: [1], or [1][3] for several. Never put a citation inside a code block. Never cite a number that is not listed above.
- Source authority: official documentation is the highest authority for installation, API, configuration and version facts; GitHub is implementation evidence; Reddit is community experience; web is general information. If sources conflict on a technical fact, prefer official documentation over web over GitHub/Reddit, and mention the conflict.
- Structure the answer naturally (headings, lists and tables only where they help). Put code in fenced code blocks. No raw URLs in the text. No title line that repeats the question.

Write the answer now:"""


# --------------------------------------------------------------------------- post-processing

def assess_confidence(citations: Sequence[Citation], missed: Sequence[str], technical: bool,
                      fallback: bool = False) -> str:
    if fallback or not citations:
        return "low"
    types = {c.source_type for c in citations}
    domains = {c.domain for c in citations if c.domain}
    solid = len(citations) >= 3 and len(domains) >= 2 and not missed
    if solid and (not technical or SourceType.OFFICIAL_DOCS in types):
        return "high"
    return "medium"


def is_technical(understanding: Optional[dict], tasks: Iterable[ResearchTask], sources: Sequence[NumberedSource] = ()) -> bool:
    intent = (understanding or {}).get("intent")
    if intent in ("technical_howto", "troubleshooting", "code"):
        return True
    if any(t.source_intent in TECHNICAL_INTENTS for t in tasks):
        return True
    return any(s.source_type == SourceType.OFFICIAL_DOCS for s in sources)


def _no_evidence_answer(missed: Sequence[str]) -> str:
    text = ("I could not find reliable sources for this question, so I can't give a grounded answer.")
    if missed:
        text += "\n\nNothing was found for:\n" + "\n".join(f"- {q}" for q in missed)
    return text + "\n\nTry rephrasing the question or making it more specific."


def _source_list_answer(sources: Sequence[NumberedSource]) -> str:
    lines = "\n".join(f"- {s.title} [{s.index}]" for s in sources[:5])
    return ("I found sources for this question but could not write the summary (the language model "
            "did not respond). The most relevant ones are:\n\n" + lines)


def strip_orphan_markers(text: str, valid: Iterable[int]) -> tuple[str, int]:
    """Remove ``[N]`` markers (outside code) whose number names no citation. Returns the text and
    how many were removed. The critic has already seen them; the reader should not."""
    valid = set(valid)
    removed = 0

    def strip(segment: str) -> str:
        nonlocal removed

        def drop(match: "re.Match[str]") -> str:
            nonlocal removed
            if int(match.group(1)) in valid:
                return match.group(0)
            removed += 1
            return ""
        return re.sub(r" ?" + _MARKER.pattern, drop, segment)

    return _outside_code(text, strip), removed


def finish_answer(raw: str, sources: Sequence[NumberedSource]) -> tuple[str, list[Citation]]:
    """Clean the model's text, renumber its valid citations by first use, and list them. A
    marker that names no source is left in the text so the critic can flag it."""
    text = st.sanitize_text((raw or "").strip())
    text = normalize_markers(text)
    valid = {s.index: s for s in sources}
    order: list[int] = []
    for number in find_markers(text):
        if number in valid and number not in order:
            order.append(number)
    mapping = {old: new for new, old in enumerate(order, start=1)}
    text = renumber_markers(text, mapping)
    citations = [_citation(valid[old], new) for old, new in mapping.items()]
    return text, citations


async def run_synthesis(state) -> SynthesisResult:
    """One synthesis pass over ``state['evidence_pool']``. Never raises."""
    pool: EvidencePool = state["evidence_pool"]
    tasks = pool.tasks
    covered_ids, missing_ids = pool.coverage()
    text_of = {t.task_id: t.sub_question for t in tasks}
    covered = [text_of[i] for i in covered_ids if i in text_of]
    missed = [text_of[i] for i in missing_ids if i in text_of]
    sources = build_sources(pool)
    technical = is_technical(state.get("understanding"), tasks, sources)

    if not sources:
        return SynthesisResult(answer_text=_no_evidence_answer(missed), confidence="low",
                               subquestions_covered=covered, subquestions_missed=missed)

    question = state.get("resolved_query") or state["question"]
    fallback = False
    try:
        prompt = build_prompt(question, state["question"], sources, pool, state.get("conversation_context"),
                              state.get("retry_feedback") or "")
        reply = await st.llm_groq.ainvoke(prompt)
        raw = reply.content if isinstance(reply.content, str) else str(reply.content)
        text, citations = finish_answer(raw, sources)
        if len(text) < 50:
            raise ValueError("answer too short")
    except Exception as exc:
        logger.warning("Synthesis failed (%s); returning the source list", type(exc).__name__)
        fallback = True
        text = _source_list_answer(sources)
        text, citations = finish_answer(text, sources)

    return SynthesisResult(
        answer_text=text,
        citations=citations,
        confidence=assess_confidence(citations, missed, technical, fallback),
        source_types_used=[t.value for t in SOURCE_ORDER if t in {c.source_type for c in citations}],
        subquestions_covered=covered,
        subquestions_missed=missed,
    )


async def synthesis_node(state):
    """Graph node. Does not emit ``final_answer``: ``format_response`` does, once, after the
    critic (the streaming layer sends every ``final_answer`` it sees)."""
    logger.info("Synthesis: %d evidence pieces", len(state["evidence_pool"]) if state.get("evidence_pool") else 0)
    result = await run_synthesis(state)
    logger.info("Synthesis done: confidence=%s citations=%d missed=%d", result.confidence,
                len(result.citations), len(result.subquestions_missed))
    return {"synthesis": result}
