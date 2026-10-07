"""Cost-vs-impact Pareto frontier via the ε-constraint method.

For each budget level b (the ε on cost):
  1. maximize impact subject to cost ≤ b
  2. fix impact ≥ that optimum and minimize cost, so the point is not "lazy" (no wasted money)
Dominated points are dropped. Each level is warm-started from the previous plan.
"""

from pydantic import BaseModel

from d2s.engine.model import ModelValidationError, build_model
from d2s.engine.schemas import Plan, PlanningProblem, SolverOptions, SolveStatus
from d2s.engine.solver import solve_built


class ParetoPoint(BaseModel):
    budget_cap: int
    cost: int
    impact: float
    plan: Plan


def _solve_lexicographic(
    problem: PlanningProblem, options: SolverOptions
) -> Plan | None:
    built = build_model(problem)
    best = solve_built(built, problem, options)
    if not best.is_solution:
        return None
    # Stage 2: keep the impact, spend as little as possible.
    built.model.add(built.impact_expr >= _impact_int(built, best))
    built.model.clear_objective()
    built.model.minimize(built.cost_expr)
    cheaper = solve_built(built, problem, options.model_copy(update={"hint": best.seats()}))
    if cheaper.is_solution and cheaper.total_cost <= best.total_cost:
        cheaper.status = best.status if best.status is SolveStatus.FEASIBLE else cheaper.status
        return cheaper
    return best


def _impact_int(built, plan: Plan) -> int:
    closed = {s.skill_id: round(s.learners_closed * 100) for s in plan.skills}
    return sum(built.int_weights[sid] * v for sid, v in closed.items())


def frontier(
    problem: PlanningProblem,
    levels: int = 8,
    min_budget: int | None = None,
    options: SolverOptions | None = None,
) -> list[ParetoPoint]:
    """Return non-dominated (cost, impact) plans for budgets from min_budget up to the problem budget."""
    if levels < 2:
        raise ValueError("levels must be at least 2")
    top = problem.constraints.budget
    low = min_budget if min_budget is not None else max(1, top // levels)
    opts = options or SolverOptions()
    points: list[ParetoPoint] = []
    hint: dict[str, int] | None = None

    for i in range(levels):
        cap = round(low + (top - low) * i / (levels - 1))
        variant = problem.model_copy(
            update={"constraints": problem.constraints.model_copy(update={"budget": cap})}
        )
        try:
            plan = _solve_lexicographic(variant, opts.model_copy(update={"hint": hint}))
        except ModelValidationError:
            continue  # budget below mandatory floor at this level
        if plan is None:
            continue
        hint = plan.seats()
        points.append(ParetoPoint(budget_cap=cap, cost=plan.total_cost, impact=plan.objective, plan=plan))

    return _non_dominated(points)


def _non_dominated(points: list[ParetoPoint]) -> list[ParetoPoint]:
    result: list[ParetoPoint] = []
    for p in sorted(points, key=lambda p: (p.cost, -p.impact)):
        if result and p.impact <= result[-1].impact + 1e-9:
            continue  # costs at least as much for no more impact
        result.append(p)
    return result
