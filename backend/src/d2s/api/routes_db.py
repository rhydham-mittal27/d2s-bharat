"""Database-backed endpoints: course catalogue CRUD, saved cohorts and the plan-run history."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from d2s.analysis.plan import AREAS
from d2s.api.auth import OwnerId
from d2s.api.db import DB, get_session
from d2s.db import CohortRepository, CourseRepository, PlanRunRepository
from d2s.engine.schemas import Course

router = APIRouter(prefix="/api")
SessionDep = Annotated[Session, Depends(get_session)]


@router.get("/db/status")
def db_status(s: SessionDep):
    return {**DB.status(), "courses": CourseRepository(s).count()}


# ---- courses ---------------------------------------------------------------------------------------------
class CourseIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    fixed_cost: int = Field(..., ge=0)
    cost_per_seat: int = Field(0, ge=0)
    min_batch: int = Field(1, ge=0)
    max_seats: int = Field(..., ge=1)
    trainer_hours: int = Field(0, ge=0)
    coverage: dict[str, int] = Field(..., description="skill area -> % of a seat's learners whose gap it closes")
    prerequisites: list[str] = []
    exclusive_group: str | None = None
    mandatory: bool = False
    assumption_basis: str | None = None
    active: bool = True


class CourseOut(CourseIn):
    id: str
    updated_at: datetime


def _out(r) -> CourseOut:
    return CourseOut(id=r.id, name=r.name, fixed_cost=r.fixed_cost, cost_per_seat=r.cost_per_seat,
                     min_batch=r.min_batch, max_seats=r.max_seats, trainer_hours=r.trainer_hours,
                     coverage=r.coverage or {}, prerequisites=r.prerequisites or [],
                     exclusive_group=r.exclusive_group, mandatory=r.mandatory,
                     assumption_basis=r.assumption_basis, active=r.active, updated_at=r.updated_at)


@router.get("/courses", response_model=list[CourseOut])
def list_courses(s: SessionDep, include_inactive: bool = False):
    return [_out(r) for r in CourseRepository(s).list(include_inactive)]


@router.get("/courses/{course_id}", response_model=CourseOut)
def get_course(course_id: str, s: SessionDep):
    return _out(CourseRepository(s).get(course_id))


@router.put("/courses/{course_id}", response_model=CourseOut)
def put_course(course_id: str, body: CourseIn, s: SessionDep):
    """Create or replace a course. The optimiser uses the updated catalogue on the next solve."""
    unknown = set(body.coverage) - set(AREAS)
    if unknown:
        raise ValueError(f"unknown skill areas in coverage: {sorted(unknown)}; expected {AREAS}")
    repo = CourseRepository(s)
    known = {r.id for r in repo.list(include_inactive=True)} | {course_id}
    missing = [p for p in body.prerequisites if p not in known]
    if missing:
        raise ValueError(f"unknown prerequisite courses: {missing}")
    if course_id in body.prerequisites:
        raise ValueError("a course cannot be its own prerequisite")
    course = Course(id=course_id, **body.model_dump(exclude={"assumption_basis", "active"}))
    row = repo.upsert(course, assumption_basis=body.assumption_basis, active=body.active)
    s.commit()  # commit before invalidating, so the next solve reloads the new catalogue
    DB.invalidate_catalogue()
    return _out(row)


@router.delete("/courses/{course_id}", status_code=204)
def delete_course(course_id: str, s: SessionDep):
    CourseRepository(s).delete(course_id)
    s.commit()
    DB.invalidate_catalogue()


# ---- cohorts ---------------------------------------------------------------------------------------------
class CohortSummary(BaseModel):
    id: str
    name: str
    source: str
    n_learners: int
    n_rejected: int
    gaps: dict[str, int]
    created_at: datetime


def _cohort(r) -> CohortSummary:
    return CohortSummary(id=r.id, name=r.name, source=r.source, n_learners=r.n_learners,
                         n_rejected=r.n_rejected, gaps=r.gaps, created_at=r.created_at)


@router.get("/cohorts", response_model=list[CohortSummary])
def list_cohorts(s: SessionDep, owner: OwnerId, limit: int = Query(50, ge=1, le=500)):
    return [_cohort(r) for r in CohortRepository(s, owner).list(limit)]


@router.get("/cohorts/{cohort_id}")
def get_cohort(cohort_id: str, s: SessionDep, owner: OwnerId):
    r = CohortRepository(s, owner).get(cohort_id)
    return {**_cohort(r).model_dump(), "profile": r.profile}


@router.delete("/cohorts/{cohort_id}", status_code=204)
def delete_cohort(cohort_id: str, s: SessionDep, owner: OwnerId):
    CohortRepository(s, owner).delete(cohort_id)


# ---- plan runs -------------------------------------------------------------------------------------------
class RunSummary(BaseModel):
    id: str
    kind: str
    label: str | None
    cohort_id: str | None
    status: str
    objective: float | None
    total_cost: int | None
    total_trainer_hours: int | None
    courses: list[str]
    request: dict
    created_at: datetime


def _run(r) -> RunSummary:
    return RunSummary(id=r.id, kind=r.kind, label=r.label, cohort_id=r.cohort_id, status=r.status,
                      objective=r.objective, total_cost=r.total_cost, total_trainer_hours=r.total_trainer_hours,
                      courses=r.courses or [], request=r.request, created_at=r.created_at)


@router.get("/plan/runs", response_model=list[RunSummary])
def list_runs(s: SessionDep, owner: OwnerId, limit: int = Query(50, ge=1, le=500), cohort_id: str | None = None,
              kind: str | None = Query(None, pattern="^(full|alternatives)$")):
    DB.drain()  # include runs this process is still writing in the background
    return [_run(r) for r in PlanRunRepository(s, owner).list(limit, cohort_id, kind)]


@router.get("/plan/runs/{run_id}")
def get_run(run_id: str, s: SessionDep, owner: OwnerId):
    DB.drain()
    r = PlanRunRepository(s, owner).get(run_id)
    return {**_run(r).model_dump(), "gaps": r.gaps, "result": r.result}


@router.delete("/plan/runs/{run_id}", status_code=204)
def delete_run(run_id: str, s: SessionDep, owner: OwnerId):
    PlanRunRepository(s, owner).delete(run_id)
