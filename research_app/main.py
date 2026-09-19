from fastapi import BackgroundTasks, FastAPI, Depends, HTTPException, Query as QueryParam, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import text, delete
import json
import os
import logging

from research_app.db.database import engine, get_db, get_session_factory, Base
from research_app.db.crud import (
    hash_password, verify_password, get_or_create_session, get_history,
    get_user_by_username, create_user, save_query, create_session,
    get_sessions, get_session_queries, get_user_session
)
from research_app.auth.auth import get_curr_user, create_access_token, require_admin
from research_app.schemas.schemas import RequestRegister, RequestResearch, NODE_LABELS
from research_app.agent.graph import compiled_graph
from research_app.db.models import (
    Session as ChatSession, User, Query, ResearchCitation, ResearchConversation,
)
from research_app.db import conversations as conversations_repo
from research_app.db.conversations import make_session_title
from research_app.db.migrate import upgrade_to_head
from research_app.conversation.settings import ConversationSettings
from research_app.agent.pipeline.response import build_response
from research_app.domain import ResearchResponse
from research_app.research_service import (
    build_inputs, elapsed_ms, legacy_sources_payload, load_conversation, merge_update,
    response_body, save_conversation_turn, turn_payload,
)
from research_app.logging_config import configure_logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Research Agent")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Bring the database schema up to date on startup (Alembic: creates what is missing and
    adds the columns an existing install lacks, which create_all never did)."""
    configure_logging()
    upgrade_to_head(engine)
    logger.info("Database schema is up to date")


# ── Basic Endpoints ──────────────────────────────────────────────────────────

@app.get('/')
def index():
    return {"status": "FastAPI is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test-db")
def test_db():
    """Test database connection"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT version();"))
            version = result.scalar()
        return {
            "status": "Database Connected",
            "postgres_version": version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/research/ui", response_class=HTMLResponse)
async def chat_ui():
    """Serve the research UI"""
    return FileResponse(os.path.join(TEMPLATES_DIR, "chat.html"), media_type="text/html")


# ── Admin Endpoints ──────────────────────────────────────────────────────────

@app.delete("/admin/clear-all-data")
def clear_all_data(user: str = Depends(require_admin), db: DBSession = Depends(get_db)):
    """
    Clear all questions, answers, and sessions from database.
    WARNING: This operation cannot be undone!
    """
    try:
        db_user = get_user_by_username(user, db)
        if not db_user:
            raise HTTPException(status_code=404, detail="User not found")

        # Delete children first (respects foreign keys; bulk deletes bypass ORM cascades)
        db.query(ResearchCitation).delete()
        conversation_count = db.query(ResearchConversation).delete()
        query_count = db.query(Query).delete()

        # Delete sessions
        session_count = db.query(ChatSession).delete()

        db.commit()

        logger.info(f"Cleared {query_count} queries and {session_count} sessions")

        return {
            "status": "success",
            "queries_deleted": query_count,
            "sessions_deleted": session_count,
            "conversations_deleted": conversation_count,
            "message": "All data cleared. Refresh your browser to see changes."
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error clearing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/db-stats")
def get_db_stats(user: str = Depends(require_admin), db: DBSession = Depends(get_db)):
    """Check current database record counts"""
    try:
        queries_count = db.query(Query).count()
        sessions_count = db.query(ChatSession).count()
        users_count = db.query(User).count()

        return {
            "queries": queries_count,
            "sessions": sessions_count,
            "users": users_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Authentication Endpoints ─────────────────────────────────────────────────

@app.post("/auth/register")
def register(request: RequestRegister, db: DBSession = Depends(get_db)):
    """Register a new user"""
    user = get_user_by_username(request.username, db)
    if user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )

    hashed_pwd = hash_password(request.password)
    create_user(request.username, hashed_pwd, db)

    return {"success": "Username and password saved"}


@app.post("/auth/login")
def login(request: RequestRegister, db: DBSession = Depends(get_db)):
    """Login user and return access token"""
    user = get_user_by_username(request.username, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Username does not exist"
        )

    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    access_token = create_access_token(data={"sub": user.username})
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


# ── Research Endpoints ───────────────────────────────────────────────────────

def _resolve_session(request: RequestResearch, db_user, db: DBSession):
    """The session this request runs in. With a ``session_id``: that session if it is the caller's
    and active, else 404. Without one (Phase 6): a NEW session titled from the question; the
    response and the ``done`` event return its id."""
    if request.session_id:
        session = get_user_session(request.session_id, db_user.id, db)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    title = make_session_title(request.question, ConversationSettings.from_env().title_max_length)
    return create_session(db_user.id, title, db)


def _prepare_run(request: RequestResearch, db_user, db: DBSession):
    """(session, turn_number, graph inputs) for a research request."""
    session = _resolve_session(request, db_user, db)
    raw_history = get_history(db_user.id, db)
    history = [{"question": q.question, "answer": q.answer} for q in raw_history]
    context, turn_number = load_conversation(db, session.id)
    inputs = build_inputs(
        request.question, history, session_id=session.id, turn_number=turn_number, context=context)
    logger.info("Research run %s: session=%s turn=%s history_turns=%d", inputs["run_id"], session.id,
                turn_number, len(context.turns))
    return session, turn_number, inputs


def _describe(state: dict, session_id: int, turn_number: int):
    """The ``ResearchResponse`` for a finished run. It describes the run; it must never be the
    reason a request fails, so a bug in it degrades to the bare answer."""
    try:
        return build_response(state, session_id=session_id, turn_number=turn_number,
                              latency_ms=elapsed_ms(state))
    except Exception as exc:
        logger.error("Could not build the research response (%s); returning the bare answer", type(exc).__name__)
        return ResearchResponse(session_id=session_id, turn_number=turn_number,
                                answer=state.get("final_answer", ""), resolved_query=state.get("question", ""))


def _persist(background_tasks: BackgroundTasks, session_factory, *, db: DBSession, session_id: int,
             turn_number: int, question: str, state: dict, response: ResearchResponse) -> None:
    """Save the legacy ``queries`` row (as always, before the response) and schedule the
    conversation turn to be saved after it."""
    save_query(session_id=session_id, question=question, answer=state.get("final_answer", ""), db=db,
               sources=legacy_sources_payload(response))
    background_tasks.add_task(
        save_conversation_turn, session_factory, session_id=session_id, turn_number=turn_number,
        payload=turn_payload(question, state, response), citations=list(state.get("citations") or []),
        run_id=state.get("run_id", ""))


@app.post('/api/research')
async def research(
        request: RequestResearch,
        background_tasks: BackgroundTasks,
        db: DBSession = Depends(get_db),
        user: str = Depends(get_curr_user),
        session_factory=Depends(get_session_factory),
):
    """Research endpoint - returns full answer at once"""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    session, turn_number, inputs = _prepare_run(request, db_user, db)

    result = await compiled_graph.ainvoke(inputs)
    response = _describe(result, session.id, turn_number)

    if not result.get("api_limit_reached"):
        _persist(background_tasks, session_factory, db=db, session_id=session.id, turn_number=turn_number,
                 question=request.question, state=result, response=response)

    return response_body(response, result['sub_questions'])


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _routing_payload(items) -> list:
    return [i.model_dump(mode="json") if hasattr(i, "model_dump") else i for i in items or []]


@app.post('/api/research/stream')
async def research_stream(
        request: RequestResearch,
        background_tasks: BackgroundTasks,
        db: DBSession = Depends(get_db),
        user: str = Depends(get_curr_user),
        session_factory=Depends(get_session_factory),
):
    """Streaming research endpoint - returns tokens as they're generated"""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    session, turn_number, inputs = _prepare_run(request, db_user, db)

    async def event_generator():
        final_answer = ""
        api_limit_reached = False
        collected_sources = []
        cache_hit_flag = False
        state = dict(inputs)  # the graph's state as it builds up, for the response and the save

        async for mode, chunk in compiled_graph.astream(
                inputs,
                stream_mode=["updates"]
        ):
            if mode == "updates":
                node_name = list(chunk.keys())[0]
                node_data = chunk[node_name]

                label = NODE_LABELS.get(node_name, node_name)
                yield _sse({'type': 'progress', 'node': node_name, 'label': label})

                if node_data:
                    merge_update(state, node_data)

                    # Track cache hit
                    if node_data.get("cache_hit"):
                        cache_hit_flag = True
                        logger.info("Cache hit detected")
                        yield _sse({'type': 'cache_hit', 'message': 'Answer found in cache!'})

                    # Collect sources (skip if cache hit)
                    if not cache_hit_flag:
                        new_sources = node_data.get("sources", [])
                        if new_sources:
                            collected_sources.extend(new_sources)

                    if node_data.get("api_limit_reached"):
                        api_limit_reached = True

                    # Phase 6: which sources the router searched (live), then the numbered
                    # citations, both before the answer's tokens. New event types only.
                    routing = node_data.get("routing_decisions") or node_data.get("cached_routing")
                    if routing:
                        yield _sse({'type': 'routing', 'cached': bool(node_data.get("cached_routing")),
                                    'stage': node_name, 'decisions': _routing_payload(routing)})
                    if node_data.get("citations"):
                        yield _sse({'type': 'citations', 'citations': node_data["citations"]})

                    answer = node_data.get("final_answer", "")
                    if answer and answer.strip():
                        final_answer = answer
                        logger.info(f"Got answer from {node_name}: {len(answer)} chars")

                        # Stream answer as chunks
                        words = answer.split(" ")
                        chunk_size = 3
                        payload = {
                            "type": "token",
                            "content": "",
                            "cache_hit": cache_hit_flag,
                        }
                        for i in range(0, len(words), chunk_size):
                            piece = " ".join(words[i:i + chunk_size])
                            if i + chunk_size < len(words):
                                piece += " "
                            payload["content"] = piece
                            yield _sse(payload)

        # Emit sources after tokens. ``sources`` is the legacy list (unchanged). The numbered
        # citations are added once, complete, only when there are some (cache hits included).
        unique_sources = []
        if collected_sources and not cache_hit_flag:
            seen_urls = set()
            for s in collected_sources:
                if s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    unique_sources.append(s)
        final_citations = list(state.get("citations") or [])
        if unique_sources or final_citations:
            sources_event = {'type': 'sources', 'sources': unique_sources}
            if final_citations:
                sources_event.update(citations=final_citations, total=len(final_citations))
            yield _sse(sources_event)

        response = _describe(state, session.id, turn_number)

        # Save to database
        if final_answer and not api_limit_reached:
            _persist(background_tasks, session_factory, db=db, session_id=session.id,
                     turn_number=turn_number, question=request.question, state=state, response=response)

        # ``done`` keeps its type; the extra fields are additive (Phase 6).
        yield _sse({'type': 'done', 'session_id': session.id, 'turn_number': turn_number,
                    'is_follow_up': response.is_follow_up, 'resolved_query': response.resolved_query,
                    'confidence': response.confidence, 'critic_verdict': response.critic_verdict})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get('/api/research/history')
def get_history_(user: str = Depends(get_curr_user), db: DBSession = Depends(get_db)):
    """Get research history for current user"""
    user_obj = get_user_by_username(user, db)
    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    history = get_history(user_obj.id, db)
    return {
        "history": [
            {
                "question": q.question,
                "answer": q.answer,
                "created_at": q.created_at.isoformat()
            }
            for q in history
        ]
    }


# ── Session Endpoints ────────────────────────────────────────────────────────

@app.post('/api/sessions')
def create_session_(
        user: str = Depends(get_curr_user),
        db: DBSession = Depends(get_db)
):
    """Create a new research session"""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    session = create_session(db_user.id, "New Research Session", db)
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat()
    }


@app.get('/api/sessions')
def get_sessions_(
        user: str = Depends(get_curr_user),
        db: DBSession = Depends(get_db)
):
    """Get all sessions for current user"""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    sessions = get_sessions(db_user.id, db)
    return {
        "sessions": [
            {"id": s.id, "title": s.title, "created_at": s.created_at.isoformat()}
            for s in sessions
        ]
    }


@app.get('/api/sessions/{session_id}/queries')
def get_session_queries_(
        session_id: int,
        user: str = Depends(get_curr_user),
        db: DBSession = Depends(get_db)
):
    """Get all queries in a session"""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    if not get_user_session(session_id, db_user.id, db):
        raise HTTPException(status_code=404, detail="Session not found")

    queries = get_session_queries(session_id, db)
    return {
        "queries": [
            {
                "question": q.question,
                "answer": q.answer,
                "created_at": q.created_at.isoformat(),
                # Phase 6 (additive): the sources and router report saved with the answer
                "citations": q.sources.get("citations", []) if isinstance(q.sources, dict) else [],
                "routing": q.sources.get("routing", []) if isinstance(q.sources, dict) else [],
            }
            for q in queries
        ]
    }


@app.patch('/api/sessions/{session_id}')
def rename_session_(
        session_id: int,
        body: dict,
        user: str = Depends(get_curr_user),
        db: DBSession = Depends(get_db)
):
    """Rename a session"""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    session = get_user_session(session_id, db_user.id, db)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.title = body.get("title", session.title)
    db.commit()

    return {
        "id": session.id,
        "title": session.title
    }


@app.delete('/api/sessions/{session_id}')
def delete_session_(
        session_id: int,
        user: str = Depends(get_curr_user),
        db: DBSession = Depends(get_db)
):
    """Delete a session and all its queries"""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    session = db.query(ChatSession).filter(
        ChatSession.id == session_id,
        ChatSession.user_id == db_user.id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    db.delete(session)
    db.commit()

    return {"deleted": session_id}


# ── Conversation Endpoints (Phase 6) ─────────────────────────────────────────

def _turn_json(turn: ResearchConversation) -> dict:
    return {
        "turn_number": turn.turn_number,
        "query_text": turn.query_text,
        "query_type": turn.query_type,
        "is_follow_up": turn.is_follow_up,
        "resolved_query": turn.resolved_query,
        "answer_text": turn.answer_text,
        "confidence": turn.confidence,
        "critic_verdict": turn.critic_verdict,
        "sources_consulted": turn.sources_consulted,
        "subquestions": turn.subquestions,
        "research_metadata": turn.research_metadata,
        "created_at": turn.created_at.isoformat() if turn.created_at else None,
        "citations": [
            {
                "index": c.citation_index,
                "source_type": c.source_type,
                "title": c.title,
                "url": c.url,
                "domain": c.domain,
                "snippet": c.snippet,
                "retrieved_at": c.retrieved_at.isoformat() if c.retrieved_at else None,
            }
            for c in turn.citations
        ],
    }


def _session_json(session: ChatSession, turn_count: int = 0) -> dict:
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None,
        "is_active": session.is_active,
        "turn_count": turn_count,
    }


@app.get('/sessions')
def list_sessions_(
        limit: int = QueryParam(20, ge=1, le=100),
        offset: int = QueryParam(0, ge=0),
        user: str = Depends(get_curr_user),
        db: DBSession = Depends(get_db)
):
    """The caller's active sessions, most recently updated first (paginated)."""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    rows, total = conversations_repo.list_user_sessions(db, db_user.id, limit, offset)
    counts = conversations_repo.turn_counts(db, [s.id for s in rows])
    return {
        "sessions": [_session_json(s, counts.get(s.id, 0)) for s in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get('/sessions/{session_id}')
def get_session_(
        session_id: int,
        user: str = Depends(get_curr_user),
        db: DBSession = Depends(get_db)
):
    """One session with every stored turn and its citations."""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    session = get_user_session(session_id, db_user.id, db)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    turns = conversations_repo.load_all_turns(db, session.id)
    return {**_session_json(session, len(turns)), "turns": [_turn_json(t) for t in turns]}


@app.delete('/sessions/{session_id}')
def soft_delete_session_(
        session_id: int,
        user: str = Depends(get_curr_user),
        db: DBSession = Depends(get_db)
):
    """Soft delete: the session is hidden (is_active = false); nothing is removed."""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    session = get_user_session(session_id, db_user.id, db)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    conversations_repo.soft_delete_session(db, session)
    return {"deleted": session_id, "soft": True}
