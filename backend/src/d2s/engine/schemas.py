"""Input and output contracts of the decision engine.

Money is in whole rupees, coverage in integer percent, so every solve is integer-exact.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class SkillGap(BaseModel):
    skill_id: str
    learners_short: int = Field(ge=0, description="Learners in the cohort lacking this skill")
    demand_weight: float = Field(ge=0, description="Market value of closing the gap (from forecasts)")
    min_closure_pct: int = Field(0, ge=0, le=100, description="Equity floor: share of gap that must close")


class Course(BaseModel):
    id: str
    name: str = ""
    fixed_cost: int = Field(0, ge=0, description="Cost to run the course at all (trainer, setup), ₹")
    cost_per_seat: int = Field(0, ge=0, description="Marginal cost per learner, ₹")
    min_batch: int = Field(1, ge=0)
    max_seats: int = Field(ge=0)
    trainer_hours: int = Field(0, ge=0)
    coverage: dict[str, int] = Field(
        default_factory=dict, description="skill_id -> % of a skill gap one seat closes (0-100)"
    )
    prerequisites: list[str] = Field(default_factory=list)
    exclusive_group: str | None = None
    mandatory: bool = False

    @model_validator(mode="after")
    def _check(self) -> "Course":
        if self.min_batch > self.max_seats:
            raise ValueError(f"course {self.id}: min_batch > max_seats")
        for skill, pct in self.coverage.items():
            if not 0 <= pct <= 100:
                raise ValueError(f"course {self.id}: coverage for {skill} must be 0-100")
        return self


class Constraints(BaseModel):
    budget: int = Field(ge=0)
    trainer_hours: int | None = Field(None, ge=0)
    max_total_seats: int | None = Field(None, ge=0)


class SolverOptions(BaseModel):
    time_limit_s: float | None = None
    workers: int | None = None
    gap_limit: float | None = None
    seed: int | None = None
    hint: dict[str, int] | None = Field(None, description="course_id -> seats from a previous plan")


class PlanningProblem(BaseModel):
    skills: list[SkillGap]
    courses: list[Course]
    constraints: Constraints
    forced_in: list[str] = Field(default_factory=list, description="courses that must run (what-if)")
    forced_out: list[str] = Field(default_factory=list, description="courses that must not run")


class SolveStatus(StrEnum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class CourseDecision(BaseModel):
    course_id: str
    seats: int
    cost: int
    trainer_hours: int


class SkillOutcome(BaseModel):
    skill_id: str
    learners_short: int
    learners_closed: float
    closure_pct: float
    weighted_value: float


class Plan(BaseModel):
    status: SolveStatus
    objective: float = 0.0
    best_bound: float = 0.0
    gap: float = 0.0
    courses: list[CourseDecision] = Field(default_factory=list)
    skills: list[SkillOutcome] = Field(default_factory=list)
    total_cost: int = 0
    total_trainer_hours: int = 0
    total_seats: int = 0
    wall_time_s: float = 0.0
    messages: list[str] = Field(default_factory=list)

    @property
    def is_solution(self) -> bool:
        return self.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE)

    def seats(self) -> dict[str, int]:
        return {c.course_id: c.seats for c in self.courses}


class ProgressEvent(BaseModel):
    objective: float
    best_bound: float
    wall_time_s: float
    solutions: int
