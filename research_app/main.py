from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from sqlalchemy.orm import Session as DBSession
from sqlalchemy import text, delete
import json
import os
import logging

from research_app.db.database import engine, get_db, Base
from research_app.db.crud import (
    hash_password, verify_password, get_or_create_session, get_history,
    get_user_by_username, create_user, save_query, create_session,
    get_sessions, get_session_queries, get_user_session
)
from research_app.auth.auth import get_curr_user, create_access_token, require_admin
from research_app.schemas.schemas import RequestRegister, RequestResearch, NODE_LABELS
from research_app.agent.graph import compiled_graph
from research_app.db.models import Session as ChatSession, User, Query
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
    """Create all tables on startup"""
    configure_logging()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


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

        # Delete queries first (respects foreign key)
        query_count = db.query(Query).delete()

        # Delete sessions
        session_count = db.query(ChatSession).delete()

        db.commit()

        logger.info(f"Cleared {query_count} queries and {session_count} sessions")

        return {
            "status": "success",
            "queries_deleted": query_count,
            "sessions_deleted": session_count,
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

@app.post('/api/research')
async def research(
        request: RequestResearch,
        db: DBSession = Depends(get_db),
        user: str = Depends(get_curr_user)
):
    """Research endpoint - returns full answer at once"""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get or create session
    if request.session_id:
        session = get_user_session(request.session_id, db_user.id, db)
    else:
        session = db.query(ChatSession).filter(
            ChatSession.user_id == db_user.id
        ).order_by(ChatSession.created_at.desc()).first()
        if not session:
            session = create_session(db_user.id, "Research Session", db)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get history for context
    raw_history = get_history(db_user.id, db)
    history = [{"question": q.question, "answer": q.answer} for q in raw_history]

    inputs = {
        "question": request.question,
        "messages": [],
        "step_count": 0,
        "final_answer": "",
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "cache_hit": False,
        "api_limit_reached": False,
        "critic_score": 0,
        "critic_feedback": "",
        "is_simple": False,
        "history": history
    }

    result = await compiled_graph.ainvoke(inputs)

    if not result.get("api_limit_reached"):
        save_query(
            session_id=session.id,
            question=request.question,
            answer=result['final_answer'],
            db=db
        )

    return {
        "sub_questions": result['sub_questions'],
        "answer": result["final_answer"],
        "session_id": session.id
    }


@app.post('/api/research/stream')
async def research_stream(
        request: RequestResearch,
        db: DBSession = Depends(get_db),
        user: str = Depends(get_curr_user)
):
    """Streaming research endpoint - returns tokens as they're generated"""
    db_user = get_user_by_username(user, db)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get or create session
    if request.session_id:
        session = get_user_session(request.session_id, db_user.id, db)
    else:
        session = db.query(ChatSession).filter(
            ChatSession.user_id == db_user.id
        ).order_by(ChatSession.created_at.desc()).first()
        if not session:
            session = create_session(db_user.id, "Research Session", db)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get history for context
    raw_history = get_history(db_user.id, db)
    history = [{"question": q.question, "answer": q.answer} for q in raw_history]

    inputs = {
        "question": request.question,
        "messages": [],
        "step_count": 0,
        "final_answer": "",
        "sub_questions": [],
        "search_results": [],
        "sources": [],
        "cache_hit": False,
        "api_limit_reached": False,
        "critic_score": 0,
        "critic_feedback": "",
        "is_simple": False,
        "history": history,
    }

    async def event_generator():
        final_answer = ""
        api_limit_reached = False
        collected_sources = []
        cache_hit_flag = False

        async for mode, chunk in compiled_graph.astream(
                inputs,
                stream_mode=["updates"]
        ):
            if mode == "updates":
                node_name = list(chunk.keys())[0]
                node_data = chunk[node_name]

                label = NODE_LABELS.get(node_name, node_name)
                yield f"data: {json.dumps({'type': 'progress', 'node': node_name, 'label': label})}\n\n"

                if node_data:
                    # Track cache hit
                    if node_data.get("cache_hit"):
                        cache_hit_flag = True
                        logger.info("Cache hit detected")
                        yield f"data: {json.dumps({'type': 'cache_hit', 'message': 'Answer found in cache!'})}\n\n"

                    # Collect sources (skip if cache hit)
                    if not cache_hit_flag:
                        new_sources = node_data.get("sources", [])
                        if new_sources:
                            collected_sources.extend(new_sources)

                    if node_data.get("api_limit_reached"):
                        api_limit_reached = True

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
                            yield f"data: {json.dumps(payload)}\n\n"

        # Emit sources after tokens
        if collected_sources and not cache_hit_flag:
            seen_urls = set()
            unique_sources = []
            for s in collected_sources:
                if s["url"] not in seen_urls:
                    seen_urls.add(s["url"])
                    unique_sources.append(s)
            yield f"data: {json.dumps({'type': 'sources', 'sources': unique_sources})}\n\n"

        # Save to database
        if final_answer and not api_limit_reached:
            save_query(
                session_id=session.id,
                question=request.question,
                answer=final_answer,
                db=db
            )

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

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
                "created_at": q.created_at.isoformat()
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
