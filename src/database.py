"""
database.py — SQLAlchemy engine, session factory, init helpers
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from src.config import SQLITE_DB_PATH
from src.models import Base, User

_ENGINE = None
_SessionLocal = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = create_engine(
            f"sqlite:///{SQLITE_DB_PATH}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
    return _ENGINE


def init_db():
    """Create all tables and seed default users."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    # seed synthetic users for the assignment demo
    with get_session() as session:
        defaults = ["Alice", "Bob", "Carol"]
        for name in defaults:
            exists = session.query(User).filter_by(display_name=name).first()
            if not exists:
                session.add(User(display_name=name))
        session.commit()


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionLocal


@contextmanager
def get_session() -> Session:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_all_users():
    """Return list of (user_id, display_name) tuples."""
    with get_session() as session:
        users = session.query(User).all()
        return [(u.user_id, u.display_name) for u in users]
