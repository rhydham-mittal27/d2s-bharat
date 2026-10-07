"""Marginal value of resources: "what is one more rupee / trainer-hour / seat worth?"

CP-SAT is an integer solver and has no duals, so we combine two estimates:

* **LP shadow prices** from the continuous relaxation, solved with GLOP. Instant; an
  optimistic estimate because the relaxation can run fractional courses.
* **Exact finite differences**: re-solve the integer model with the resource raised by
  ``delta`` (warm-started from the base plan). Costs one solve per resource, but it's the
  real answer.
"""

from ortools.linear_solver import pywraplp
from pydantic import BaseModel

from d2s.engine.model import validate
from d2s.engine.schemas import Plan, PlanningProblem, SolverOptions
from d2s.engine.solver import solve


class ShadowPrices(BaseModel):
    lp_objective: float
    budget_per_rupee: float
    trainer_hours_per_hour: float | None
    total_seats_per_seat: float | None
    course_reduced_costs: dict[str, float]
    """How much a course's `run` variable would change the objective per unit; negative = unattractive."""


class MarginalValue(BaseModel):
    resource: str
    delta: float
    base_objective: float
    new_objective: float
    per_unit: float
    lp_estimate_per_unit: float | None


class SensitivityReport(BaseModel):
    shadow_prices: ShadowPrices
    marginals: list[MarginalValue]


def lp_shadow_prices(problem: PlanningProblem) -> ShadowPrices:
    validate(problem)
    lp = pywraplp.Solver.CreateSolver("GLOP")
    if lp is None:  # pragma: no cover - GLOP ships with OR-Tools
        raise RuntimeError("GLOP solver unavailable")
    cons = problem.constraints
    inf = lp.infinity()

    run, seats = {}, {}
    for c in problem.courses:
        lo = 1.0 if (c.mandatory or c.id in problem.forced_in) else 0.0
        hi = 0.0 if c.id in problem.forced_out else 1.0
        run[c.id] = lp.NumVar(lo, hi, f"run[{c.id}]")
        seats[c.id] = lp.NumVar(0, c.max_seats, f"seats[{c.id}]")
        lp.Add(seats[c.id] - c.max_seats * run[c.id] <= 0)
        lp.Add(seats[c.id] - c.min_batch * run[c.id] >= 0)
        for p in c.prerequisites:
            lp.Add(run[c.id] - run[p] <= 0)

    groups: dict[str, list[str]] = {}
    for c in problem.courses:
        if c.exclusive_group:
            groups.setdefault(c.exclusive_group, []).append(c.id)
    for members in groups.values():
        lp.Add(sum(run[m] for m in members) <= 1)

    budget_ct = lp.Add(
        sum(c.fixed_cost * run[c.id] + c.cost_per_seat * seats[c.id] for c in problem.courses)
        <= cons.budget
    )
    hours_ct = None
    if cons.trainer_hours is not None:
        hours_ct = lp.Add(
            sum(c.trainer_hours * run[c.id] for c in problem.courses) <= cons.trainer_hours
        )
    seats_ct = None
    if cons.max_total_seats is not None:
        seats_ct = lp.Add(sum(seats.values()) <= cons.max_total_seats)

    objective = lp.Objective()
    for s in problem.skills:
        closed = lp.NumVar(0, 100 * s.learners_short, f"closed[{s.skill_id}]")
        supply = [c.coverage[s.skill_id] * seats[c.id] for c in problem.courses if c.coverage.get(s.skill_id)]
        if supply:
            lp.Add(closed - sum(supply) <= 0)
        else:
            lp.Add(closed <= 0)
        if s.min_closure_pct:
            lp.Add(closed >= s.min_closure_pct * s.learners_short)
        objective.SetCoefficient(closed, s.demand_weight / 100)
    objective.SetMaximization()

    status = lp.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise ValueError("LP relaxation is infeasible; no shadow prices available")

    # GLOP reports duals of a maximization with ≤ constraints as non-negative "value per unit".
    return ShadowPrices(
        lp_objective=objective.Value(),
        budget_per_rupee=abs(budget_ct.dual_value()),
        trainer_hours_per_hour=abs(hours_ct.dual_value()) if hours_ct is not None else None,
        total_seats_per_seat=abs(seats_ct.dual_value()) if seats_ct is not None else None,
        course_reduced_costs={cid: v.reduced_cost() for cid, v in run.items()},
    )


def _with_resource(problem: PlanningProblem, resource: str, delta: float) -> PlanningProblem:
    cons = problem.constraints.model_copy()
    current = getattr(cons, resource)
    setattr(cons, resource, int(current + delta))
    return problem.model_copy(update={"constraints": cons})


def analyze(
    problem: PlanningProblem,
    base_plan: Plan,
    budget_delta: int = 100_000,
    hours_delta: int = 10,
    seats_delta: int = 10,
    options: SolverOptions | None = None,
) -> SensitivityReport:
    """Shadow prices plus exact marginal values for each constrained resource."""
    prices = lp_shadow_prices(problem)
    opts = (options or SolverOptions()).model_copy(update={"hint": base_plan.seats()})
    checks: list[tuple[str, float, float | None]] = [("budget", budget_delta, prices.budget_per_rupee)]
    if problem.constraints.trainer_hours is not None:
        checks.append(("trainer_hours", hours_delta, prices.trainer_hours_per_hour))
    if problem.constraints.max_total_seats is not None:
        checks.append(("max_total_seats", seats_delta, prices.total_seats_per_seat))

    marginals = []
    for resource, delta, lp_est in checks:
        bumped = solve(_with_resource(problem, resource, delta), opts)
        new_obj = bumped.objective if bumped.is_solution else base_plan.objective
        marginals.append(
            MarginalValue(
                resource=resource,
                delta=delta,
                base_objective=base_plan.objective,
                new_objective=new_obj,
                per_unit=(new_obj - base_plan.objective) / delta,
                lp_estimate_per_unit=lp_est,
            )
        )
    return SensitivityReport(shadow_prices=prices, marginals=marginals)
