"""Database wiring for the API: one engine per process, a session per request, and helpers that pick
the database when it is configured and the analysis files when it is not."""

import logging
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from d2s.config import get_settings
from d2s.db import CourseRepository, make_engine, make_session_factory, session_scope
from d2s.services import store

log = logging.getLogger(__name__)


CATALOGUE_TTL_S = 300  # safety net for edits made outside this API process (e.g. Supabase dashboard)


class Database:
    """A remote database (Supabase) costs ~0.4 s per round trip from India, so the request path avoids
    it: the catalogue is cached in memory and plan runs are written by a background thread."""

    def __init__(self):
        self.engine = None
        self.factory: sessionmaker[Session] | None = None
        self.error: str | None = None
        self._catalogue: tuple[float, list] | None = None
        self._cat_lock = threading.Lock()
        self._writer: ThreadPoolExecutor | None = None
        self._pending: list[Future] = []

    @property
    def enabled(self) -> bool:
        return self.factory is not None

    def configure(self, url: str | None, echo: bool = False) -> None:
        self.dispose()
        if not url:
            return
        try:
            self.engine = make_engine(url, echo=echo)
            with self.engine.connect() as c:
                c.execute(text("select 1"))
            self.factory = make_session_factory(self.engine)
        except Exception as exc:  # reported by /api/health and /api/db/status; files still work
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("database connection failed")
            self.factory = None

    def configure_from_settings(self) -> None:
        s = get_settings()
        self.configure(s.database_url, s.database_echo)

    def dispose(self) -> None:
        self.drain()
        if self._writer is not None:
            self._writer.shutdown(wait=True)
        if self.engine is not None:
            self.engine.dispose()
        self.engine, self.factory, self.error, self._writer = None, None, None, None
        self.invalidate_catalogue()

    # catalogue cache
    def courses(self) -> list:
        with self._cat_lock:
            if self._catalogue and time.monotonic() - self._catalogue[0] < CATALOGUE_TTL_S:
                return self._catalogue[1]
        with session_scope(self.factory) as s:
            courses = CourseRepository(s).courses()
        with self._cat_lock:
            self._catalogue = (time.monotonic(), courses)
        return courses

    def invalidate_catalogue(self) -> None:
        with self._cat_lock:
            self._catalogue = None

    # background writes
    def submit(self, fn: Callable[[Session], None], what: str) -> None:
        """Run fn(session) in its own unit of work on the writer thread; errors are logged."""
        if self._writer is None:
            self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db-writer")

        def run():
            try:
                with session_scope(self.factory) as s:
                    fn(s)
            except Exception:
                log.exception("background write failed: %s", what)

        self._pending = [f for f in self._pending if not f.done()] + [self._writer.submit(run)]

    def drain(self, timeout: float = 30) -> None:
        """Wait for queued writes (tests, shutdown)."""
        for f in list(self._pending):
            f.result(timeout=timeout)
        self._pending = []

    def status(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "error": self.error,
                    "hint": None if self.error else "set D2S_DATABASE_URL to the Supabase connection string"}
        return {"enabled": True, "dialect": self.engine.dialect.name,
                "host": self.engine.url.host, "database": self.engine.url.database}


DB = Database()


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one unit of work per request (commit on success, rollback on error)."""
    if not DB.enabled:
        raise HTTPException(503, f"database not configured{': ' + DB.error if DB.error else ''}; "
                                 "set D2S_DATABASE_URL and run scripts/db_init.py")
    with session_scope(DB.factory) as s:
        yield s


def catalogue_source():
    """The active course catalogue: the database's when configured and seeded, else the CSV."""
    if DB.enabled:
        courses = DB.courses()
        if courses:
            return courses
    return store.CATALOGUE
