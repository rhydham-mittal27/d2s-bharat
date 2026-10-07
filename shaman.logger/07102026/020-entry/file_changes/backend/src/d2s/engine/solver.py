"""Solve a PlanningProblem with CP-SAT and translate the result into a Plan."""

import time
from collections.abc import Callable

from ortools.sat.python import cp_model

from d2s.config import get_settings
from d2s.engine.model import BuiltModel, ModelValidationError, build_model
from d2s.engine.schemas import (
    CourseDecision,
    Plan,
    PlanningProblem,
    ProgressEvent,
    SkillOutcome,
    SolverOptions,
    SolveStatus,
)

ProgressFn = Callable[[ProgressEvent], None]

_STATUS = {
    cp_model.OPTIMAL: SolveStatus.OPTIMAL,
    cp_model.FEASIBLE: SolveStatus.FEASIBLE,
    cp_model.INFEASIBLE: SolveStatus.INFEASIBLE,
    cp_model.MODEL_INVALID: SolveStatus.INVALID,
    cp_model.UNKNOWN: SolveStatus.UNKNOWN,
}


class _ProgressCallback(cp_model.CpSolverSolutionCallback):
    """Reports every improving solution; CP-SAT calls this from its search threads."""

    def __init__(self, unscale: float, on_progress: ProgressFn):
        super().__init__()
        self._unscale = unscale
        self._on_progress = on_progress
        self._count = 0

    def on_solution_callback(self) -> None:
        self._count += 1
        self._on_progress(
            ProgressEvent(
                objective=self.objective_value * self._unscale,
                best_bound=self.best_objective_bound * self._unscale,
                wall_time_s=self.wall_time,
                solutions=self._count,
            )
        )


def make_solver(options: SolverOptions | None = None) -> cp_model.CpSolver:
    s = get_settings()
    o = options or SolverOptions()
    solver = cp_model.CpSolver()
    p = solver.parameters
    p.max_time_in_seconds = o.time_limit_s if o.time_limit_s is not None else s.solver_time_limit_s
    p.num_workers = o.workers if o.workers is not None else s.solver_workers
    p.relative_gap_limit = o.gap_limit if o.gap_limit is not None else s.solver_gap_limit
    p.random_seed = o.seed if o.seed is not None else s.solver_seed
    return solver


def apply_hint(built: BuiltModel, hint: dict[str, int] | None) -> None:
    if not hint:
        return
    built.model.clear_hints()
    for cid, seats in hint.items():
        if cid in built.seats:
            built.model.add_hint(built.seats[cid], seats)
            built.model.add_hint(built.run[cid], int(seats > 0))


def solve_built(
    built: BuiltModel,
    problem: PlanningProblem,
    options: SolverOptions | None = None,
    on_progress: ProgressFn | None = None,
) -> Plan:
    apply_hint(built, options.hint if options else None)
    solver = make_solver(options)
    started = time.perf_counter()
    callback = _ProgressCallback(built.weight_unscale, on_progress) if on_progress else None
    raw_status = solver.solve(built.model, callback)
    status = _STATUS.get(raw_status, SolveStatus.UNKNOWN)
    plan = Plan(status=status, wall_time_s=time.perf_counter() - started, messages=list(built.warnings))
    if status not in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE):
        if status is SolveStatus.INFEASIBLE:
            plan.messages.append(
                "no plan satisfies all constraints; relax budget, trainer hours, equity floors "
                "or forced courses"
            )
        elif status is SolveStatus.UNKNOWN:
            plan.messages.append("time limit reached before any plan was found; increase the limit")
        return plan
    return _extract_plan(solver, built, problem, plan)


def _extract_plan(
    solver: cp_model.CpSolver, built: BuiltModel, problem: PlanningProblem, plan: Plan
) -> Plan:
    courses = []
    for c in problem.courses:
        seats = solver.value(built.seats[c.id])
        if solver.value(built.run[c.id]):
            courses.append(
                CourseDecision(
                    course_id=c.id,
                    seats=seats,
                    cost=c.fixed_cost + c.cost_per_seat * seats,
                    trainer_hours=c.trainer_hours,
                )
            )

    skills = []
    objective = 0.0
    for s in problem.skills:
        closed = solver.value(built.closed[s.skill_id]) / 100
        value = s.demand_weight * closed
        objective += value
        skills.append(
            SkillOutcome(
                skill_id=s.skill_id,
                learners_short=s.learners_short,
                learners_closed=closed,
                closure_pct=100 * closed / s.learners_short if s.learners_short else 100.0,
                weighted_value=value,
            )
        )

    plan.objective = objective
    raw_obj = solver.objective_value
    raw_bound = solver.best_objective_bound
    # Report the bound in real units, scaled by how the exact objective relates to the integer one.
    ratio = objective / raw_obj if raw_obj else built.weight_unscale
    plan.best_bound = max(raw_bound * ratio, objective)
    plan.gap = 0.0 if plan.status is SolveStatus.OPTIMAL or not raw_bound else max(
        0.0, (raw_bound - raw_obj) / raw_bound
    )
    plan.courses = courses
    plan.skills = skills
    plan.total_cost = sum(c.cost for c in courses)
    plan.total_trainer_hours = sum(c.trainer_hours for c in courses)
    plan.total_seats = sum(c.seats for c in courses)
    return plan


def solve(
    problem: PlanningProblem,
    options: SolverOptions | None = None,
    on_progress: ProgressFn | None = None,
) -> Plan:
    """Build and solve. Input errors come back as an INVALID plan with a readable message."""
    try:
        built = build_model(problem)
    except ModelValidationError as exc:
        return Plan(status=SolveStatus.INVALID, messages=[str(exc)])
    return solve_built(built, problem, options, on_progress)
