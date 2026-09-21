"""Manual check tool for Phase 7 hybrid source RAG. Never touches production data.

    python scripts/rag_manual.py seed                      # index a tiny corpus
    python scripts/rag_manual.py query "how do I install fastapi" --mode dense
    python scripts/rag_manual.py query "CREATE INDEX CONCURRENTLY" --mode sparse
    python scripts/rag_manual.py query "install fastapi" --mode hybrid --rerank
    python scripts/rag_manual.py query "server" --source-type reddit --technology fastapi
    python scripts/rag_manual.py query "fastapi" --after-days 30
    python scripts/rag_manual.py stats | drop

It uses its own collection (default ``source_chunks_manual_test``) and refuses to run against the
configured production collection. The store is qdrant-client's embedded engine in ``.rag_manual/``
unless ``--url`` (or QDRANT_URL) points at a server. Models download on first use.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research_app.domain import RetrievalFilter, SourceDocument, SourceType, utc_now  # noqa: E402
from research_app.rag.chunking import chunk_document  # noqa: E402
from research_app.rag.retriever import HybridRetriever, build_store  # noqa: E402
from research_app.rag.settings import RagSettings  # noqa: E402

PRODUCTION_COLLECTION = "source_chunks"
DEFAULT_COLLECTION = "source_chunks_manual_test"


def _doc(url: str, title: str, content: str, source_type: SourceType, technology: str | None,
         days_old: int = 0) -> SourceDocument:
    return SourceDocument(url=url, title=title, content=content, source_type=source_type,
                          technology=technology, provider="seed", retrieved_at=utc_now() - timedelta(days=days_old))


CORPUS = [
    _doc("https://fastapi.tiangolo.com/tutorial/", "FastAPI tutorial - first steps",
         "To install FastAPI run pip install \"fastapi[standard]\". Create a file main.py with an app = FastAPI() "
         "instance and start the development server with the fastapi dev command.",
         SourceType.OFFICIAL_DOCS, "fastapi"),
    _doc("https://fastapi.tiangolo.com/tutorial/dependencies/", "FastAPI dependencies",
         "Dependency injection in FastAPI lets a path operation declare what it needs with Depends(). "
         "Dependencies can be functions or classes and are resolved per request.",
         SourceType.OFFICIAL_DOCS, "fastapi"),
    _doc("https://www.postgresql.org/docs/current/sql-createindex.html", "PostgreSQL: CREATE INDEX",
         "CREATE INDEX CONCURRENTLY builds an index without taking a lock that prevents concurrent inserts, "
         "updates or deletes on the table. It requires more total work and cannot run inside a transaction block.",
         SourceType.OFFICIAL_DOCS, "postgresql"),
    _doc("https://www.postgresql.org/docs/current/indexes-types.html", "PostgreSQL: index types",
         "PostgreSQL provides several index types: B-tree, Hash, GiST, SP-GiST, GIN and BRIN. "
         "Each uses a different algorithm suited to different kinds of queries.",
         SourceType.OFFICIAL_DOCS, "postgresql"),
    _doc("https://qdrant.tech/documentation/concepts/hybrid-queries/", "Qdrant: hybrid queries",
         "Qdrant supports hybrid search by combining dense and sparse vectors with prefetch and fusion. "
         "Reciprocal Rank Fusion merges the ranked lists of each leg into a single ranking.",
         SourceType.OFFICIAL_DOCS, "qdrant"),
    _doc("https://www.reddit.com/r/FastAPI/comments/x1/uvicorn_slow_under_load/", "Uvicorn feels slow under load",
         "My FastAPI service got sluggish with lots of users. Running several uvicorn workers behind gunicorn "
         "fixed most of it, and moving blocking database calls to a thread pool helped too.",
         SourceType.REDDIT, "fastapi"),
    _doc("https://www.reddit.com/r/docker/comments/x2/qdrant_data_lost/", "Qdrant data lost after restart",
         "I lost my Qdrant collections after recreating the container because I forgot to mount a volume "
         "for /qdrant/storage. Use a named volume in docker compose.",
         SourceType.REDDIT, "qdrant"),
    _doc("https://blog.example.com/fastapi-0-60", "FastAPI 0.60 release notes (old)",
         "FastAPI 0.60 release notes: minor fixes for the older routing internals and dependency caching.",
         SourceType.WEB, "fastapi", days_old=400),
    _doc("https://cooking.example.com/tomato-soup", "Tomato soup",
         "A simple tomato soup with basil and garlic, simmered for twenty minutes.", SourceType.WEB, None),
]


def make_settings(args: argparse.Namespace) -> RagSettings:
    if args.url:
        os.environ["QDRANT_URL"] = args.url
    elif not os.environ.get("QDRANT_URL"):
        os.environ["QDRANT_URL"] = str(ROOT / ".rag_manual")
    os.environ["SOURCE_CHUNKS_COLLECTION"] = args.collection
    os.environ.setdefault("RAG_TIMEOUT_S", "600")  # the first run downloads models (reranker ~2.3 GB)
    if getattr(args, "mode", None):
        os.environ["RAG_MODE"] = args.mode
    if getattr(args, "rerank", None) is not None:
        os.environ["RERANK_ENABLED"] = "true" if args.rerank else "false"
    if getattr(args, "rrf_k", None):
        os.environ["RAG_RRF_K"] = str(args.rrf_k)
    settings = RagSettings.from_env()
    if settings.collection == PRODUCTION_COLLECTION:
        sys.exit(f"Refusing to use the production collection {PRODUCTION_COLLECTION!r}. Pass another --collection.")
    return settings


def cmd_seed(settings: RagSettings) -> None:
    store = build_store(settings)
    chunks = [c for d in CORPUS for c in chunk_document(d, settings.chunk_size, settings.chunk_overlap)]
    written = store.upsert(chunks)
    print(f"Indexed {written} chunks from {len(CORPUS)} documents into {settings.collection!r} "
          f"({settings.qdrant_url}); collection now has {store.count()} points.")


def cmd_stats(settings: RagSettings) -> None:
    store = build_store(settings)
    names = [c.name for c in store.client.get_collections().collections]
    print("collections:", names)
    for name in names:
        print(f"  {name}: {store.client.count(name, exact=True).count} points")


def cmd_drop(settings: RagSettings) -> None:
    build_store(settings).client.delete_collection(settings.collection)
    print(f"Dropped {settings.collection!r}.")


def cmd_query(settings: RagSettings, args: argparse.Namespace) -> None:
    flt = RetrievalFilter(
        source_types=[SourceType(t) for t in args.source_type or []],
        domains=args.domain or [],
        technology=args.technology,
        retrieved_after=utc_now() - timedelta(days=args.after_days) if args.after_days else None,
    )
    retriever = HybridRetriever(settings)
    docs = asyncio.run(retriever.retrieve(args.query, flt, args.top_k))
    print(f"mode={settings.mode.value} rerank={settings.rerank_enabled} rrf_k={settings.rrf_k} "
          f"filter={'none' if flt.is_empty else flt.model_dump(exclude_defaults=True, mode='json')}")
    if not docs:
        print("(no results)")
    for d in docs:
        m = d.metadata
        print(f"#{m['rag_rank']} score={m['rag_score']:.3f} fused={m['rag_fused_score']:.4f} "
              f"reranked={m['rag_reranked']} [{d.source_type.value}] {d.domain} :: {d.title}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--url", help="Qdrant URL or a local folder (default ./.rag_manual)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("seed")
    sub.add_parser("stats")
    sub.add_parser("drop")
    q = sub.add_parser("query")
    q.add_argument("query")
    q.add_argument("--mode", choices=["dense", "sparse", "hybrid"])
    q.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    q.add_argument("--rrf-k", type=int)
    q.add_argument("--top-k", type=int)
    q.add_argument("--source-type", action="append", choices=[t.value for t in SourceType])
    q.add_argument("--domain", action="append")
    q.add_argument("--technology")
    q.add_argument("--after-days", type=int, help="only chunks retrieved in the last N days")
    args = parser.parse_args()
    settings = make_settings(args)
    {"seed": lambda: cmd_seed(settings), "stats": lambda: cmd_stats(settings),
     "drop": lambda: cmd_drop(settings), "query": lambda: cmd_query(settings, args)}[args.command]()


if __name__ == "__main__":
    main()
