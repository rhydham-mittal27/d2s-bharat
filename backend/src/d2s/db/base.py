"""Engine, session factory and unit of work.

Production runs on Supabase Postgres; tests use SQLite with the same models and repositories.
Vectors are not stored here: they live in ChromaDB (``d2s.vector``).

Supabase notes
* use the *session* pooler (port 5432) or the direct connection for a long-lived API server;
* the *transaction* pooler (port 6543) does not support prepared statements, so they are disabled;
* TLS is required: ``sslmode=require`` is added when the URL does not set it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(UTC)


def normalise_url(url: str) -> str:
    """Accept Supabase's ``postgres://`` / ``postgresql://`` URLs and use the psycopg 3 driver."""
    u = make_url(url)
    if u.drivername in ("postgres", "postgresql", "postgresql+psycopg2"):
        u = u.set(drivername="postgresql+psycopg")
    if u.drivername.startswith("postgresql") and "sslmode" not in u.query and u.host not in (
            "localhost", "127.0.0.1", None):
        u = u.update_query_dict({"sslmode": "require"})
    return u.render_as_string(hide_password=False)


def make_engine(url: str, echo: bool = False) -> Engine:
    url = normalise_url(url)
    u = make_url(url)
    if u.drivername.startswith("sqlite"):
        kw = {"connect_args": {"check_same_thread": False}}
        if u.database in (None, "", ":memory:"):
            kw["poolclass"] = StaticPool  # one shared in-memory DB across threads
        engine = create_engine(url, echo=echo, **kw)

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

        return engine
    connect_args = {}
    if u.port == 6543:  # Supabase transaction pooler: no server-side prepared statements
        connect_args["prepare_threshold"] = None
    return create_engine(url, echo=echo, pool_pre_ping=True, pool_size=5, max_overflow=5,
                         pool_recycle=1800, connect_args=connect_args)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """One unit of work: commit on success, roll back on any error."""
    s = factory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
