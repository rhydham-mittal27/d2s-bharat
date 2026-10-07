"""Repository layer: the only code that touches ORM rows. Services and the API work with domain
objects (engine ``Course``, pydantic results, plain dicts); repositories translate in both directions.

Each repository wraps one ``Session``; the caller owns the transaction (``session_scope`` or the
FastAPI ``get_session`` dependency), so several repositories can share one unit of work.
"""

from __future__ import annotations  # methods named `list` would shadow the builtin in annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from d2s.db.models import AgentRunRow, CohortRow, CourseRow, PlanRunRow, UserRow
from d2s.engine.schemas import Course


class NotFound(LookupError):
    pass


class Conflict(ValueError):
    pass


# ---- users ------------------------------------------------------------------------------------------------
class UserRepository:
    def __init__(self, session: Session):
        self.s = session

    def by_email(self, email: str) -> UserRow | None:
        return self.s.scalar(select(UserRow).where(UserRow.email == email.strip().lower()))

    def get(self, user_id: str) -> UserRow:
        r = self.s.get(UserRow, user_id)
        if r is None:
            raise NotFound("user not found")
        return r

    def create(self, email: str, name: str, password_hash: str) -> UserRow:
        email = email.strip().lower()
        if self.by_email(email) is not None:
            raise Conflict("an account with this email already exists")
        r = UserRow(email=email, name=name.strip(), password_hash=password_hash)
        self.s.add(r)
        self.s.flush()
        return r


class _Owned:
    """Rows that belong to a user. With ``owner`` set, every read is filtered to that user and every
    write is stamped with it; another user's id behaves exactly like a missing id (no existence leak).
    ``owner=None`` (auth switched off) means unscoped."""

    model: type

    def __init__(self, session: Session, owner: str | None = None):
        self.s = session
        self.owner = owner

    def _scoped(self, q):
        return q.where(self.model.user_id == self.owner) if self.owner else q

    def _get(self, row_id: str, what: str):
        r = self.s.get(self.model, row_id)
        if r is None or (self.owner and r.user_id != self.owner):
            raise NotFound(f"{what} {row_id} not found")
        return r


# ---- courses ----------------------------------------------------------------------------------------------
class CourseRepository:
    def __init__(self, session: Session):
        self.s = session

    @staticmethod
    def to_domain(r: CourseRow) -> Course:
        return Course(id=r.id, name=r.name, fixed_cost=r.fixed_cost, cost_per_seat=r.cost_per_seat,
                      min_batch=r.min_batch, max_seats=r.max_seats, trainer_hours=r.trainer_hours,
                      coverage=dict(r.coverage or {}), prerequisites=list(r.prerequisites or []),
                      exclusive_group=r.exclusive_group, mandatory=r.mandatory)

    def list(self, include_inactive: bool = False) -> list[CourseRow]:
        q = select(CourseRow).order_by(CourseRow.id)
        if not include_inactive:
            q = q.where(CourseRow.active.is_(True))
        return list(self.s.scalars(q))

    def courses(self) -> list[Course]:
        """Active catalogue as engine courses (what the optimiser consumes). A course whose
        prerequisite is inactive cannot run, so it is left out too (transitively)."""
        courses = {r.id: self.to_domain(r) for r in self.list()}
        changed = True
        while changed:
            blocked = [cid for cid, c in courses.items() if any(p not in courses for p in c.prerequisites)]
            changed = bool(blocked)
            for cid in blocked:
                del courses[cid]
        return list(courses.values())

    def get(self, course_id: str) -> CourseRow:
        r = self.s.get(CourseRow, course_id)
        if r is None:
            raise NotFound(f"course {course_id} not found")
        return r

    def upsert(self, course: Course, assumption_basis: str | None = None, active: bool = True) -> CourseRow:
        r = self.s.get(CourseRow, course.id) or CourseRow(id=course.id)
        r.name = course.name or course.id
        r.fixed_cost, r.cost_per_seat = course.fixed_cost, course.cost_per_seat
        r.min_batch, r.max_seats, r.trainer_hours = course.min_batch, course.max_seats, course.trainer_hours
        r.coverage, r.prerequisites = dict(course.coverage), list(course.prerequisites)
        r.exclusive_group, r.mandatory, r.active = course.exclusive_group, course.mandatory, active
        if assumption_basis is not None:
            r.assumption_basis = assumption_basis
        self.s.add(r)
        self.s.flush()
        return r

    def delete(self, course_id: str) -> None:
        r = self.get(course_id)
        dependants = [c.id for c in self.list(include_inactive=True) if course_id in (c.prerequisites or [])]
        if dependants:
            raise ValueError(f"course {course_id} is a prerequisite of {', '.join(dependants)}; remove that first")
        self.s.delete(r)
        self.s.flush()

    def count(self) -> int:
        return self.s.scalar(select(func.count()).select_from(CourseRow)) or 0


# ---- cohorts ----------------------------------------------------------------------------------------------
class CohortRepository(_Owned):
    model = CohortRow

    def add(self, name: str, profile: dict, gaps: dict[str, int], source: str = "upload") -> CohortRow:
        r = CohortRow(user_id=self.owner, name=name, source=source, n_learners=int(profile["n_learners"]),
                      n_rejected=int(profile.get("n_rejected", 0)), gaps=gaps, profile=profile)
        self.s.add(r)
        self.s.flush()
        return r

    def get(self, cohort_id: str) -> CohortRow:
        return self._get(cohort_id, "cohort")

    def list(self, limit: int = 50) -> list[CohortRow]:
        q = self._scoped(select(CohortRow)).order_by(CohortRow.created_at.desc()).limit(limit)
        return list(self.s.scalars(q))

    def delete(self, cohort_id: str) -> None:
        self.s.delete(self.get(cohort_id))
        self.s.flush()


# ---- plan runs ----------------------------------------------------------------------------------------------
class PlanRunRepository(_Owned):
    model = PlanRunRow

    def add(self, kind: str, request: dict, gaps: dict[str, int], plan: dict, result: dict,
            cohort_id: str | None = None, label: str | None = None, run_id: str | None = None) -> PlanRunRow:
        """``plan`` is the headline Plan (dict); ``result`` the full response that was returned.
        ``run_id`` lets the caller hand the id out before the row is written (async save)."""
        r = PlanRunRow(id=run_id, user_id=self.owner, kind=kind, cohort_id=cohort_id, label=label, request=request, gaps=gaps,
                       status=plan.get("status", "unknown"), objective=plan.get("objective"),
                       total_cost=plan.get("total_cost"), total_trainer_hours=plan.get("total_trainer_hours"),
                       courses=[c["course_id"] for c in plan.get("courses", [])], result=result)
        self.s.add(r)
        self.s.flush()
        return r

    def get(self, run_id: str) -> PlanRunRow:
        return self._get(run_id, "plan run")

    def list(self, limit: int = 50, cohort_id: str | None = None, kind: str | None = None) -> list[PlanRunRow]:
        q = self._scoped(select(PlanRunRow)).order_by(PlanRunRow.created_at.desc()).limit(limit)
        if cohort_id:
            q = q.where(PlanRunRow.cohort_id == cohort_id)
        if kind:
            q = q.where(PlanRunRow.kind == kind)
        return list(self.s.scalars(q))

    def delete(self, run_id: str) -> None:
        self.s.delete(self.get(run_id))
        self.s.flush()


# ---- agent runs ---------------------------------------------------------------------------------------------
class AgentRunRepository(_Owned):
    model = AgentRunRow

    def add(self, agent: str, input: dict, output: dict, trace: list, ms: float, path: str | None = None,
            run_id: str | None = None) -> AgentRunRow:
        r = AgentRunRow(id=run_id, user_id=self.owner, agent=agent, input=input, output=output, trace=trace,
                        ms=ms, path=path)
        self.s.add(r)
        self.s.flush()
        return r

    def get(self, run_id: str) -> AgentRunRow:
        return self._get(run_id, "agent run")

    def list(self, limit: int = 50, agent: str | None = None) -> list[AgentRunRow]:
        q = self._scoped(select(AgentRunRow)).order_by(AgentRunRow.created_at.desc()).limit(limit)
        if agent:
            q = q.where(AgentRunRow.agent == agent)
        return list(self.s.scalars(q))

