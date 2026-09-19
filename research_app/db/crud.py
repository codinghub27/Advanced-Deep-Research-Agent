
import bcrypt


from sqlalchemy.orm import Session

from research_app.db.models import User, Session as ChatSession, Query






# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password: str) -> str:
    """
    Hash a plain-text password using bcrypt.
    """

    return bcrypt.hashpw(
        password[:72].encode(),
        bcrypt.gensalt()
    ).decode()


def verify_password(
    plain: str,
    hashed: str
) -> bool:
    """
    Verify a plain-text password against
    a stored bcrypt hash.
    """

    return bcrypt.checkpw(
        plain[:72].encode(),
        hashed.encode()
    )





# ============================================================
# USER FUNCTIONS
# ============================================================

def get_user_by_username(
    username: str,
    db: Session
):
    """
    Find a user using username.
    """

    user = (
        db.query(User)
        .filter(User.username == username)
        .first()
    )

    return user


def create_user(
    username: str,
    hashed_password: str,
    db: Session
):
    """
    Create and save a new user.
    """

    new_user = User(
        username=username,
        hashed_password=hashed_password
    )

    db.add(new_user)

    db.commit()

    db.refresh(new_user)

    return new_user


# ============================================================
# SESSION FUNCTIONS
# ============================================================

def get_or_create_session(
    user_id: int,
    db: Session
):
    """
    Get the user's existing session.

    If no session exists, create a new one.
    """

    chat_session = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .first()
    )

    if not chat_session:

        chat_session = ChatSession(
            user_id=user_id,
            title="New Research Session"
        )

        db.add(chat_session)

        db.commit()

        db.refresh(chat_session)

    return chat_session

def create_session(user_id: int, title: str, db: Session):
    session = ChatSession(user_id=user_id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def get_sessions(user_id: int, db: Session):
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id, ChatSession.is_active.is_(True))
        .order_by(ChatSession.created_at.desc())
        .all()
    )


def get_user_session(session_id: int, user_id: int, db: Session):
    """
    Return the session only if it belongs to the given user and has not been
    soft-deleted (Phase 6 ``is_active``), else None.
    """
    return (
        db.query(ChatSession)
        .filter(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
            ChatSession.is_active.is_(True),
        )
        .first()
    )



# ============================================================
# QUERY FUNCTIONS
# ============================================================

def get_session_queries(session_id:int,db:Session):
    return (
        db.query(Query).filter(Query.session_id==session_id).order_by(Query.created_at.asc()).all()
    )


def save_query(
    session_id: int,
    question: str,
    answer: str,
    db: Session,
    sources: dict = None
):
    query = Query(
        session_id=session_id,
        question=question,
        answer=answer,
        sources=sources or {}
    )


    db.add(query)

    db.commit()

    db.refresh(query)

    return query


def get_history(
    user_id: int,
    db: Session,
    limit: int = 20
):
    """
    Get the last N queries across ALL sessions of a user,
    ordered chronologically — used as context for the planner.
    """
    # Get all session IDs for this user
    session_ids = [
        s.id for s in db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .all()
    ]
    if not session_ids:
        return []

    history = (
        db.query(Query)
        .filter(Query.session_id.in_(session_ids))
        .order_by(Query.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(history))