"""Create the D2S schema on Supabase (or any Postgres/SQLite URL) and load the course catalogue.

    uv run python scripts/db_init.py                      # uses D2S_DATABASE_URL
    uv run python scripts/db_init.py --url "postgresql://..." --reseed-courses

Steps: create tables (adds new columns to existing ones), enable row-level security on Postgres, load
the course catalogue (only if empty, unless --reseed-courses). Skill vectors live in ChromaDB:
see scripts/build_vector_index.py (the API also builds that index on first start).
"""

import argparse
import sys
import time

from sqlalchemy import inspect

from d2s.config import get_settings
from d2s.db import create_schema, make_engine, make_session_factory, session_scope
from d2s.db.seed import seed_courses


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="database URL (default: D2S_DATABASE_URL)")
    ap.add_argument("--reseed-courses", action="store_true", help="overwrite the catalogue from the CSV")
    args = ap.parse_args()
    url = args.url or get_settings().database_url
    if not url:
        print("No database URL: pass --url or set D2S_DATABASE_URL (Supabase: Project Settings > Database "
              "> Connection string, session pooler).", file=sys.stderr)
        return 2
    engine = make_engine(url)
    print(f"connecting to {engine.url.render_as_string(hide_password=True)}")
    t = time.perf_counter()
    create_schema(engine)
    print(f"schema ready: {sorted(inspect(engine).get_table_names())}")
    with session_scope(make_session_factory(engine)) as s:
        n = seed_courses(s, overwrite=args.reseed_courses)
        print(f"courses: {'loaded ' + str(n) if n else 'already present (use --reseed-courses to overwrite)'}")
    print(f"done in {time.perf_counter() - t:.1f}s")
    engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
