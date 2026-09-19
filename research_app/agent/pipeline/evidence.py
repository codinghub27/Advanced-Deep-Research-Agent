"""Evidence collection for one research run (Phase 6). In memory only.

A plain Python object that lives for one request: no database, no vector store, no embeddings.
It gathers the normalised ``SourceDocument``s of every sub-question, drops exact duplicates,
and answers "which sub-questions have evidence and which have none".

Relevance is deliberately crude: a document counts as evidence for a sub-question when it has
text and contains at least ``RELEVANCE_THRESHOLD`` of the sub-question's content words. It only
decides coverage (gap detection); nothing is scored, ranked or dropped for being "weak".
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterable, Optional, Sequence

from research_app.domain import ResearchTask, SourceDocument, SourceType, canonical_url
from research_app.sources.routing import SOURCE_ORDER

logger = logging.getLogger(__name__)

DEFAULT_RELEVANCE_THRESHOLD = 0.3

_STOPWORDS = frozenset(
    "a an and are as at be but by can could do does for from had has have how i if in into is it its "
    "me my of on or our should so than that the their them then there these they this those to us "
    "was we were what when where which who why will with would you your about also any some such "
    "using use used".split()
)
_WORD = re.compile(r"[a-z0-9][a-z0-9+#._-]*")
UNASSIGNED = "unassigned"  # documents that carry no task id (states not built by the pipeline)


def content_words(text: Optional[str]) -> set[str]:
    """Lower-cased words of length >= 3 that are not stopwords."""
    if not text:
        return set()
    return {w.strip("._-") for w in _WORD.findall(text.lower())
            if len(w.strip("._-")) >= 3 and w not in _STOPWORDS}


def _doc_text(doc: SourceDocument) -> str:
    return doc.content or doc.snippet or ""


def _content_key(doc: SourceDocument) -> tuple[str, str]:
    normalized = " ".join(_doc_text(doc).lower().split())
    return canonical_url(doc.url), hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def relevance(sub_question: str, doc: SourceDocument) -> float:
    """Share of the sub-question's content words that appear in the document (0..1)."""
    wanted = content_words(sub_question)
    if not wanted:
        return 1.0  # nothing to compare against: do not manufacture a gap
    have = content_words(f"{doc.title or ''} {doc.url} {_doc_text(doc)}")
    return len(wanted & have) / len(wanted)


class EvidencePool:
    def __init__(self, tasks: Sequence[ResearchTask] = (), *, relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD):
        self._tasks: dict[str, ResearchTask] = {t.task_id: t for t in tasks}
        self._threshold = relevance_threshold
        self._docs: list[SourceDocument] = []
        # duplicate key -> every task id the page was found for (the copy is dropped, the link kept)
        self._owners: dict[tuple[str, str], set[str]] = {}

    # ------------------------------------------------------------------ building
    def add_documents(self, documents: Iterable[SourceDocument]) -> int:
        """Add documents with text; skip exact duplicates (same URL and same text). Returns how
        many were added. Documents without text are sources but not evidence, so they are left out."""
        added = 0
        for doc in documents:
            if not _doc_text(doc).strip():
                continue
            key = _content_key(doc)
            owner = doc.task_id or UNASSIGNED
            if key in self._owners:
                self._owners[key].add(owner)
                continue
            self._owners[key] = {owner}
            self._docs.append(doc)
            added += 1
        return added

    def add_task(self, task: ResearchTask) -> None:
        self._tasks.setdefault(task.task_id, task)

    # ------------------------------------------------------------------ reading
    def __len__(self) -> int:
        return len(self._docs)

    @property
    def tasks(self) -> list[ResearchTask]:
        return list(self._tasks.values())

    def all_evidence(self) -> list[SourceDocument]:
        """Every document, most authoritative source type first, arrival order within a type."""
        rank = {source: i for i, source in enumerate(SOURCE_ORDER)}
        return sorted(self._docs, key=lambda d: rank.get(d.source_type, len(rank)))

    def source_types(self) -> list[SourceType]:
        present = {d.source_type for d in self._docs}
        return [s for s in SOURCE_ORDER if s in present]

    def evidence_for(self, task_id: str) -> list[SourceDocument]:
        """Documents attached to the task that are relevant to its sub-question."""
        task = self._tasks.get(task_id)
        docs = [d for d in self._docs if task_id in self._owners.get(_content_key(d), ())]
        if task is None:
            return docs
        return [d for d in docs if relevance(task.sub_question, d) >= self._threshold]

    def by_subquestion(self) -> dict[str, list[SourceDocument]]:
        return {task_id: self.evidence_for(task_id) for task_id in self._tasks}

    def coverage(self) -> tuple[list[str], list[str]]:
        """(covered task ids, task ids with no evidence), in task order."""
        covered, gaps = [], []
        for task_id, docs in self.by_subquestion().items():
            (covered if docs else gaps).append(task_id)
        return covered, gaps

    def source_types_for(self, task_id: str) -> set[SourceType]:
        return {d.source_type for d in self.evidence_for(task_id)}

    def task(self, task_id: str) -> Optional[ResearchTask]:
        return self._tasks.get(task_id)


def build_pool(tasks: Sequence[ResearchTask], documents: Iterable[SourceDocument],
               relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD) -> EvidencePool:
    pool = EvidencePool(tasks, relevance_threshold=relevance_threshold)
    pool.add_documents(documents)
    return pool
