"""Repository layer and DB-backed API. SQLite stands in for Supabase Postgres: same models and
repositories. (Vector search is ChromaDB: tests/test_vector.py.)"""

import io

import numpy as np
import pytest
from sqlalchemy.engine import make_url
from sqlalchemy.exc import StatementError

from d2s.db import (
    CohortRepository,
    CourseRepository,
    NotFound,
    PlanRunRepository,
    create_schema,
    make_engine,
    make_session_factory,
    normalise_url,
    session_scope,
)
from d2s.engine.schemas import Course


@pytest.fixture
def factory():
    engine = make_engine("sqlite://")
    create_schema(engine)
    yield make_session_factory(engine)
    engine.dispose()


def test_supabase_urls_are_normalised():
    u = make_url(normalise_url("postgres://postgres.abc:pw@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"))
    assert u.drivername == "postgresql+psycopg" and u.query["sslmode"] == "require" and u.password == "pw"
    local = make_url(normalise_url("postgresql://u:p@localhost:5432/d2s"))
    assert "sslmode" not in local.query
    assert normalise_url("sqlite://") == "sqlite://"


def test_course_repository_roundtrip_and_guards(factory):
    with session_scope(factory) as s:
        repo = CourseRepository(s)
        repo.upsert(Course(id="py", name="Python", fixed_cost=100, max_seats=30, coverage={"coding_skills": 100}))
        repo.upsert(Course(id="ml", name="ML", fixed_cost=200, max_seats=20, coverage={"ai_and_ml_skills": 100},
                           prerequisites=["py"]), assumption_basis="trainer x 40h")
    with session_scope(factory) as s:
        repo = CourseRepository(s)
        courses = {c.id: c for c in repo.courses()}
        assert courses["ml"].prerequisites == ["py"] and courses["py"].coverage == {"coding_skills": 100}
        assert repo.get("ml").assumption_basis == "trainer x 40h"
        with pytest.raises(ValueError, match="prerequisite of ml"):
            repo.delete("py")
        repo.upsert(courses["ml"], active=False)  # deactivated: kept, but not offered to the optimiser
        assert [c.id for c in repo.courses()] == ["py"] and repo.count() == 2
        repo.upsert(courses["ml"])
        repo.upsert(courses["py"], active=False)  # ml needs py, so ml cannot run either
        assert repo.courses() == [] and repo.count() == 2
    with pytest.raises(NotFound), session_scope(factory) as s:
        CourseRepository(s).get("nope")


def test_unit_of_work_rolls_back_on_error(factory):
    with pytest.raises(RuntimeError), session_scope(factory) as s:
        CourseRepository(s).upsert(Course(id="x", fixed_cost=1, max_seats=1))
        raise RuntimeError("boom")
    with session_scope(factory) as s:
        assert CourseRepository(s).count() == 0


def test_cohorts_and_plan_runs(factory):
    with session_scope(factory) as s:
        c = CohortRepository(s).add("batch-1", {"n_learners": 3, "n_rejected": 1}, {"coding_skills": 2})
        runs = PlanRunRepository(s)
        plan = {"status": "optimal", "objective": 10.5, "total_cost": 900, "total_trainer_hours": 40,
                "courses": [{"course_id": "py"}, {"course_id": "ml"}]}
        r1 = runs.add("full", {"budget": 1000}, {"coding_skills": 2}, plan, {"x": 1}, cohort_id=c.id, label="A")
        runs.add("alternatives", {"budget": 500}, {"coding_skills": 2}, plan, {"x": 2})
        cid, rid = c.id, r1.id
    with session_scope(factory) as s:
        runs = PlanRunRepository(s)
        assert len(runs.list()) == 2 and [r.id for r in runs.list(cohort_id=cid)] == [rid]
        r = runs.get(rid)
        assert r.courses == ["py", "ml"] and r.total_cost == 900 and r.cohort.name == "batch-1"
        assert [x.kind for x in runs.list(kind="alternatives")] == ["alternatives"]
        CohortRepository(s).delete(cid)
    with session_scope(factory) as s:
        assert PlanRunRepository(s).get(rid).cohort_id is None  # history survives cohort deletion


# ---- API with the database switched on ---------------------------------------------------------------------
def _ready() -> bool:
    from d2s.services import store

    try:
        store.artifact_path("hike_model.pkl")
        store.table("rq1/triangulation_rq1_rq2.csv")
        return True
    except FileNotFoundError:
        return False


api = pytest.mark.skipif(not _ready(), reason="analysis outputs / artifacts not built")


@pytest.fixture(scope="module")
def db_client():
    from fastapi.testclient import TestClient

    from d2s.api.db import DB
    from d2s.api.main import app
    from d2s.db.seed import seed_courses

    with TestClient(app) as c:
        DB.configure("sqlite://")
        create_schema(DB.engine)
        with session_scope(DB.factory) as s:
            seed_courses(s)
        yield c
        DB.dispose()


@api
def test_api_db_catalogue_runs_and_cohorts(db_client):
    c = db_client
    assert c.get("/api/health").json()["database"] == "connected"
    st = c.get("/api/db/status").json()
    assert st["enabled"] and st["courses"] == 9
    assert len(c.get("/api/courses").json()) == 9

    full = c.post("/api/plan/full", json={"label": "baseline"}).json()
    assert full["run_id"] and full["result"]["plan"]["total_cost"] == 352_900
    chosen = [x["course_id"] for x in full["result"]["plan"]["courses"]]

    # edit the catalogue: deactivate a chosen course -> the next solve must not use it
    course = c.get(f"/api/courses/{chosen[0]}").json()
    upd = {k: v for k, v in course.items() if k not in ("id", "updated_at")} | {"active": False}
    assert c.put(f"/api/courses/{chosen[0]}", json=upd).status_code == 200
    again = c.post("/api/plan/solve", json={}).json()
    assert chosen[0] not in [x["course_id"] for x in again["plan"]["courses"]]
    c.put(f"/api/courses/{chosen[0]}", json=upd | {"active": True})
    assert c.put("/api/courses/x", json=upd | {"coverage": {"nope": 5}}).status_code == 422
    assert c.put("/api/courses/x", json=upd | {"prerequisites": ["ghost"]}).status_code == 422

    alt = c.post("/api/plan/alternatives", json={})
    assert alt.headers.get("X-Plan-Run-Id")
    runs = c.get("/api/plan/runs").json()
    assert {r["kind"] for r in runs} == {"full", "alternatives"}
    run = c.get(f"/api/plan/runs/{full['run_id']}").json()
    assert run["label"] == "baseline" and run["result"]["brief"]["faithful"]
    assert c.get("/api/plan/runs/does-not-exist").status_code == 404

    csv = ("id,coding_skills,maths-stats_skills,ai_and_ml_skills,big_data_skills,dashboard_and_storytelling_skills\n"
           "a,2,2,2,2,2\nb,3,3,3,3,3\nc,5,5,5,5,5\n")
    prof = c.post("/api/cohort/profile", files={"file": ("batch.csv", io.BytesIO(csv.encode()), "text/csv")}).json()
    cid = prof["cohort_id"]
    assert cid and c.get("/api/cohorts").json()[0]["name"] == "batch.csv"
    small = c.post("/api/plan/full", json={"cohort_id": cid}).json()
    assert small["result"]["gaps"] == c.get(f"/api/cohorts/{cid}").json()["gaps"]
    assert c.get("/api/plan/runs", params={"cohort_id": cid}).json()[0]["id"] == small["run_id"]


def test_db_routes_report_when_not_configured():
    from fastapi.testclient import TestClient

    from d2s.api.db import DB
    from d2s.api.main import app

    DB.dispose()
    r = TestClient(app).get("/api/courses")
    assert r.status_code == 503 and "D2S_DATABASE_URL" in r.json()["detail"]
