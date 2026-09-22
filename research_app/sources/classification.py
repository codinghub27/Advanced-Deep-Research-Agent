"""Content-type classification for ``SourceDocument`` (P1.4).

Rule-based and deterministic (no LLM call), same style as the official-docs detector and
the source router (Phase 4/5). ``SourceType`` (web/official_docs/github/reddit/...) is the
existing routing bucket and is unchanged -- it stays load-bearing everywhere else (routing,
synthesis labels, citations, the ``research_citations`` DB column). This module adds a finer,
optional classification on top of it, so a preprint is never presented as peer-reviewed, a
blog as an academic paper, a product announcement as independent research, a search snippet
as a complete source, or a repository list as proof of importance.

Must stay importable without ``research_app.agent`` / LangChain, like the rest of ``sources/``
(see ``tests/test_domain_isolation.py``).
"""
from __future__ import annotations

from urllib.parse import urlsplit

from research_app.domain import AuthorityLevel, ContentClassification, SourceDocument, SourceType

_GITHUB_DISCUSSION_KINDS = {"issue", "pull_request", "discussion"}
_GITHUB_REPO_KINDS = {"repository", "file", "directory", "release", "commit", "wiki"}

_REDDIT_DISCUSSION_KINDS = {"post", "comment"}

# Preprint servers: never peer-reviewed by the server itself, whatever the paper's eventual fate.
_PREPRINT_DOMAINS = {"arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "openreview.net", "techrxiv.org"}

# Recognised peer-reviewed venues or their indexes. Illustrative, not exhaustive -- extend as
# needed (a configurable registry, like Phase 4's, is future work). A domain absent from this
# list is never auto-upgraded to RESEARCH_PAPER; it falls through to UNKNOWN instead.
_RESEARCH_PAPER_DOMAINS = {
    "nature.com", "science.org", "dl.acm.org", "ieeexplore.ieee.org", "link.springer.com",
    "aclanthology.org", "pubmed.ncbi.nlm.nih.gov", "pnas.org", "jmlr.org",
}

# Established news organizations. Illustrative, not exhaustive.
_NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com", "bloomberg.com",
    "cnbc.com", "theverge.com", "techcrunch.com", "arstechnica.com", "wired.com", "forbes.com",
    "theguardian.com", "washingtonpost.com", "wsj.com",
}

# Independent publishing platforms: a post there is a blog, never a paper or a news report.
_BLOG_DOMAINS = {"medium.com", "substack.com", "dev.to", "hashnode.dev"}
_BLOG_PATH_HINTS = ("/blog/", "/blogs/")

_ANNOUNCEMENT_PATH_HINTS = ("/blog/", "/news/", "/changelog", "/release")
_ANNOUNCEMENT_TITLE_HINTS = ("announcing", "release notes", "changelog")

# Below this (or with no content at all) there is no "full source" retrieved, only a snippet.
_SNIPPET_MIN_CONTENT_CHARS = 200


def classify_content(doc: SourceDocument) -> SourceDocument:
    """Return a copy of ``doc`` with ``content_classification``, ``classification_confidence``,
    ``classification_reason``, ``authority_level``, ``is_primary_source`` and
    ``independently_verified`` filled in. Deterministic and pure; never calls an LLM or the
    network, and never changes ``source_type``, ``url``, ``content`` or any other field."""
    classification, confidence, reason, authority, primary, verified = _classify(doc)
    return doc.model_copy(update={
        "content_classification": classification,
        "classification_confidence": confidence,
        "classification_reason": reason,
        "authority_level": authority,
        "is_primary_source": primary,
        "independently_verified": verified,
    })


def _bare(domain: str) -> str:
    """Leading ``www.`` is ignored for classification matching, same convention as the
    Phase 4/5 registries (``domain`` itself is not stripped: Phase 3 keeps it as derived)."""
    return domain[4:] if domain.startswith("www.") else domain


def _classify(doc: SourceDocument):
    domain = _bare((doc.domain or "").lower())
    has_full_content = bool(doc.content) and len(doc.content) >= _SNIPPET_MIN_CONTENT_CHARS

    if doc.source_type == SourceType.OFFICIAL_DOCS:
        # Phase 4 verified this URL against the official-docs registry before it was ever
        # labelled official_docs -- that verification is the independent check here.
        classification = (
            ContentClassification.OFFICIAL_ANNOUNCEMENT if _looks_like_announcement(doc)
            else ContentClassification.OFFICIAL_DOCUMENTATION
        )
        return (
            classification, 0.95,
            "source_type=official_docs, verified against the official-docs registry (Phase 4)",
            AuthorityLevel.OFFICIAL, True, True,
        )

    if doc.source_type == SourceType.GITHUB:
        kind = str((doc.metadata.get("github") or {}).get("kind", ""))
        if kind in _GITHUB_DISCUSSION_KINDS:
            return (
                ContentClassification.GITHUB_ISSUE_OR_DISCUSSION, 0.9,
                f"github metadata kind={kind!r}", AuthorityLevel.COMMUNITY, False, True,
            )
        if kind in _GITHUB_REPO_KINDS:
            return (
                ContentClassification.GITHUB_REPOSITORY, 0.85,
                f"github metadata kind={kind!r}", AuthorityLevel.COMMUNITY, True, True,
            )
        return (
            ContentClassification.UNKNOWN, 0.3,
            "source_type=github but no (or unrecognised) URL-derived github metadata",
            AuthorityLevel.COMMUNITY, None, True,
        )

    if doc.source_type == SourceType.REDDIT:
        kind = str((doc.metadata.get("reddit") or {}).get("kind", ""))
        if kind in _REDDIT_DISCUSSION_KINDS:
            return (
                ContentClassification.REDDIT_POST_OR_DISCUSSION, 0.9,
                f"reddit metadata kind={kind!r}", AuthorityLevel.COMMUNITY, False, True,
            )
        return (
            ContentClassification.UNKNOWN, 0.3,
            "source_type=reddit but no (or unrecognised) URL-derived reddit metadata",
            AuthorityLevel.COMMUNITY, False, True,
        )

    if doc.source_type == SourceType.WEB:
        if domain in _PREPRINT_DOMAINS:
            return (
                ContentClassification.PREPRINT, 0.85,
                f"domain {domain!r} is a preprint server (not peer-reviewed by the server itself)",
                AuthorityLevel.ACADEMIC, True, False,
            )
        if domain in _RESEARCH_PAPER_DOMAINS:
            return (
                ContentClassification.RESEARCH_PAPER, 0.8,
                f"domain {domain!r} is a recognised peer-reviewed venue or index",
                AuthorityLevel.ACADEMIC, True, True,
            )
        if domain in _NEWS_DOMAINS:
            return (
                ContentClassification.NEWS_REPORT, 0.75,
                f"domain {domain!r} is a recognised news organization",
                AuthorityLevel.ESTABLISHED, False, True,
            )
        if domain in _BLOG_DOMAINS or _looks_like_blog_path(doc.url):
            return (
                ContentClassification.EXPERT_BLOG, 0.5,
                "independent publishing platform or /blog/ URL path (author expertise not verified)",
                AuthorityLevel.COMMUNITY, False, False,
            )
        if domain.endswith(".gov"):
            return (
                ContentClassification.TECHNICAL_REPORT, 0.5,
                f"domain {domain!r} is a government domain (weak signal)",
                AuthorityLevel.ESTABLISHED, True, False,
            )
        if not has_full_content:
            return (
                ContentClassification.SEARCH_SNIPPET, 0.6,
                "no full content retrieved (snippet only); not a complete source on its own",
                AuthorityLevel.UNKNOWN, None, False,
            )
        return (
            ContentClassification.UNKNOWN, 0.3,
            "web result with no recognised domain/content signal",
            AuthorityLevel.UNKNOWN, None, False,
        )

    # rag_document / cache: a chunk keeps its original source_type (Phase 7); re-classifying
    # the underlying document is out of scope for P1.4.
    return (
        ContentClassification.UNKNOWN, 0.0,
        f"source_type={doc.source_type.value!r} is not classified by P1.4",
        AuthorityLevel.UNKNOWN, None, False,
    )


def _looks_like_announcement(doc: SourceDocument) -> bool:
    path = urlsplit(doc.url).path.lower()
    title = (doc.title or "").lower()
    return (any(hint in path for hint in _ANNOUNCEMENT_PATH_HINTS)
            or any(hint in title for hint in _ANNOUNCEMENT_TITLE_HINTS))


def _looks_like_blog_path(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return any(hint in path for hint in _BLOG_PATH_HINTS)
