"""Persistence: SQLAlchemy 2 models + repositories over Supabase Postgres (vectors: d2s.vector).

The database is optional. Without ``D2S_DATABASE_URL`` the API runs from the analysis files as
before; with it, the catalogue, cohorts, plan history and semantic skill search use the database.
"""

from d2s.db.base import (
    Base,
    make_engine,
    make_session_factory,
    normalise_url,
    session_scope,
)
from d2s.db.repositories import (
    AgentRunRepository,
    CohortRepository,
    Conflict,
    CourseRepository,
    NotFound,
    PlanRunRepository,
    UserRepository,
)
from d2s.db.schema import create_schema, drop_schema

__all__ = [
    "AgentRunRepository", "Base",
    "CohortRepository",
    "Conflict",
    "CourseRepository",
    "NotFound",
    "PlanRunRepository",
    "UserRepository",
    "create_schema",
    "drop_schema",
    "make_engine",
    "make_session_factory",
    "normalise_url",
    "session_scope",
]
