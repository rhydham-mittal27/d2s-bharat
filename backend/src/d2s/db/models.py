"""ORM tables.

agent_runs     every agent run (question, path, steps, answer) per user
users          accounts (email + Argon2 password hash); owners of cohorts and plan runs
courses        the institution's course catalogue (editable; the optimiser reads it)
cohorts        uploaded learner cohorts with their computed gap profile
plan_runs      every saved decision: request, gaps, result (the intervention history)
"""

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from d2s.db.base import Base, utcnow


def _uuid() -> str:
    return str(uuid.uuid4())


class UserRow(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)  # stored lower-case
    name: Mapped[str] = mapped_column(String(120))
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CourseRow(Base):
    __tablename__ = "courses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    fixed_cost: Mapped[int] = mapped_column(Integer)
    cost_per_seat: Mapped[int] = mapped_column(Integer, default=0)
    min_batch: Mapped[int] = mapped_column(Integer, default=0)
    max_seats: Mapped[int] = mapped_column(Integer)
    trainer_hours: Mapped[int] = mapped_column(Integer, default=0)
    coverage: Mapped[dict] = mapped_column(JSON, default=dict)  # area -> % of a seat's learners helped
    prerequisites: Mapped[list] = mapped_column(JSON, default=list)
    exclusive_group: Mapped[str | None] = mapped_column(String(64))
    mandatory: Mapped[bool] = mapped_column(Boolean, default=False)
    assumption_basis: Mapped[str | None] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class CohortRow(Base):
    __tablename__ = "cohorts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    source: Mapped[str] = mapped_column(String(32), default="upload")  # upload | sas-sample
    n_learners: Mapped[int] = mapped_column(Integer)
    n_rejected: Mapped[int] = mapped_column(Integer, default=0)
    gaps: Mapped[dict] = mapped_column(JSON)  # area -> learners short
    profile: Mapped[dict] = mapped_column(JSON)  # full CohortProfile
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    runs: Mapped[list["PlanRunRow"]] = relationship(back_populates="cohort")


class PlanRunRow(Base):
    __tablename__ = "plan_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))  # full | alternatives
    cohort_id: Mapped[str | None] = mapped_column(ForeignKey("cohorts.id", ondelete="SET NULL"))
    label: Mapped[str | None] = mapped_column(String(200))
    request: Mapped[dict] = mapped_column(JSON)
    gaps: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16))
    objective: Mapped[float | None] = mapped_column(Float)
    total_cost: Mapped[int | None] = mapped_column(Integer)
    total_trainer_hours: Mapped[int | None] = mapped_column(Integer)
    courses: Mapped[list] = mapped_column(JSON, default=list)  # chosen course ids
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    cohort: Mapped[CohortRow | None] = relationship(back_populates="runs")


class AgentRunRow(Base):
    """One agent run: what was asked, which path it took, every step, and the answer."""

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    agent: Mapped[str] = mapped_column(String(32))  # copilot | stress_test | intake
    input: Mapped[dict] = mapped_column(JSON)
    path: Mapped[str | None] = mapped_column(String(16))  # copilot: fast | llm | none
    trace: Mapped[list] = mapped_column(JSON, default=list)
    output: Mapped[dict] = mapped_column(JSON)
    ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
