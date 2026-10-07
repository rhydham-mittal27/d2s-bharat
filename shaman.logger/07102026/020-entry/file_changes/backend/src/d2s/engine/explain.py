""""Why wasn't course X chosen?": re-solve with X forced in and report the cost of that choice."""

from pydantic import BaseModel

from d2s.engine.schemas import Plan, PlanningProblem, SolverOptions
from d2s.engine.solver import solve


class WhyNot(BaseModel):
    course_id: str
    feasible: bool
    objective_delta: float | None = None
    objective_delta_pct: float | None = None
    reason: str
    alternative_plan: Plan | None = None


def why_not(
    problem: PlanningProblem,
    base_plan: Plan,
    course_id: str,
    options: SolverOptions | None = None,
) -> WhyNot:
    courses = {c.id: c for c in problem.courses}
    if course_id not in courses:
        raise ValueError(f"unknown course {course_id}")
    if course_id in base_plan.seats():
        return WhyNot(course_id=course_id, feasible=True, objective_delta=0.0,
                      objective_delta_pct=0.0, reason="course is already in the plan")

    course = courses[course_id]
    if course_id in problem.forced_out:
        return WhyNot(course_id=course_id, feasible=False, reason="course was excluded by the user")

    blockers = _static_blockers(problem, course_id)
    variant = problem.model_copy(update={"forced_in": [*problem.forced_in, course_id]})
    opts = (options or SolverOptions()).model_copy(update={"hint": base_plan.seats()})
    alt = solve(variant, opts)
    if not alt.is_solution:
        reason = "; ".join(blockers) if blockers else "; ".join(alt.messages) or "no feasible plan"
        return WhyNot(course_id=course_id, feasible=False, reason=reason)

    delta = alt.objective - base_plan.objective
    pct = 100 * delta / base_plan.objective if base_plan.objective else 0.0
    covers = [s for s, v in course.coverage.items() if v]
    reason = (
        f"forcing it in changes impact by {delta:+.2f} ({pct:+.1f}%); "
        f"the budget it uses closes more demand-weighted gap elsewhere"
        if delta < 0
        else "an equally good plan exists that includes it (tie)"
    )
    if not covers:
        reason = "it does not cover any skill with a current gap"
    return WhyNot(course_id=course_id, feasible=True, objective_delta=delta,
                  objective_delta_pct=pct, reason=reason, alternative_plan=alt)


def _static_blockers(problem: PlanningProblem, course_id: str) -> list[str]:
    """Cheap explanations that don't need a solve."""
    courses = {c.id: c for c in problem.courses}
    course = courses[course_id]
    out = []
    for p in course.prerequisites:
        if p in problem.forced_out:
            out.append(f"prerequisite {p} is excluded")
    if course.exclusive_group:
        rivals = [
            c.id for c in problem.courses
            if c.exclusive_group == course.exclusive_group and c.id != course_id
            and (c.mandatory or c.id in problem.forced_in)
        ]
        if rivals:
            out.append(f"mutually exclusive with required course(s) {rivals}")
    floor = course.fixed_cost + course.cost_per_seat * course.min_batch
    if floor > problem.constraints.budget:
        out.append(f"minimum cost ₹{floor:,} exceeds the whole budget")
    return out
