"""Create the schema. Idempotent.

A hackathon prototype uses ``create_all``; a production deployment would move these tables to
Alembic migrations (or Supabase migrations) without changing the models or repositories.
"""

from sqlalchemy import Engine, inspect, text

from d2s.db import models  # noqa: F401  (registers the tables on Base.metadata)
from d2s.db.base import Base


def create_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    if engine.dialect.name == "postgresql":
        _lock_down(engine)


def _lock_down(engine: Engine) -> None:
    """Supabase exposes the public schema through its REST API (PostgREST) to the anon and
    authenticated roles. Our API connects as the table owner, which bypasses row-level security, so
    enabling RLS with no policies closes that side door (users.password_hash included) without
    affecting the app."""
    with engine.begin() as c:
        for table in Base.metadata.tables:
            c.execute(text(f'alter table "{table}" enable row level security'))


# Columns added after the first deploy: create_all() does not alter existing tables.
_ADDED = {
    "cohorts": [("user_id", "varchar(36) references users(id) on delete cascade")],
    "plan_runs": [("user_id", "varchar(36) references users(id) on delete cascade")],
}


def _add_missing_columns(engine: Engine) -> None:
    insp = inspect(engine)
    with engine.begin() as c:
        for table, cols in _ADDED.items():
            have = {col["name"] for col in insp.get_columns(table)}
            for name, ddl in cols:
                if name not in have:
                    c.execute(text(f"alter table {table} add column {name} {ddl}"))
                    c.execute(text(f"create index if not exists ix_{table}_{name} on {table} ({name})"))


def drop_schema(engine: Engine) -> None:
    Base.metadata.drop_all(engine)
