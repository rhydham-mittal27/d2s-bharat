"""Plan A / B / C: the k best *structurally different* course portfolios.

Plan A is the optimum. Plan B is the optimum among plans whose course set differs from A's in at
least ``min_diff`` courses; Plan C differs from both A and B, and so on. The "differs" condition is a
no-good cut on the run literals of each earlier plan P:

    Σ_{c ∈ P} (1 - run_c) + Σ_{c ∉ P} run_c  ≥  min_diff

so seat re-shuffles of the same courses never count as an alternative. Each plan is solved
lexicographically (max impact, then min cost at that impact) so no plan wastes money.
"""

from pydantic import BaseModel

from d2s.engine.model import ModelValidationError, build_model
from d2s.engine.pareto import _impact_int
from d2s.engine.schemas import Plan, PlanningProblem, SolverOptions
from d2s.engine.solver import solve_built


class Alternative(BaseModel):
    label: str  # "Plan A", "Plan B", ...
    plan: Plan
    impact_pct_of_best: float  # 100 for Plan A
    cost_delta: int  # vs Plan A (negative = cheaper)
    hours_delta: int
    added: list[str]  # courses not in Plan A
    dropped: list[str]  # Plan A courses not in this plan
    tradeoff: str


def _with_cuts(problem: PlanningProblem, previous: list[set[str]], min_diff: int):
    built = build_model(problem)
    for chosen in previous:
        flips = [1 - built.run[c] for c in chosen] + [built.run[c] for c in built.run if c not in chosen]
        built.model.add(sum(flips) >= min_diff)
    return built


def _solve_cheapest_best(problem, previous, min_diff, options) -> Plan | None:
    built = _with_cuts(problem, previous, min_diff)
    best = solve_built(built, problem, options)
    if not best.is_solution:
        return None
    built.model.add(built.impact_expr >= _impact_int(built, best))
    built.model.clear_objective()
    built.model.minimize(built.cost_expr)
    cheaper = solve_built(built, problem, options.model_copy(update={"hint": best.seats()}))
    if cheaper.is_solution and cheaper.total_cost <= best.total_cost and cheaper.objective >= best.objective - 1e-6:
        cheaper.status = best.status
        return cheaper
    return best


def _tradeoff(label: str, best: Plan, plan: Plan, added: list[str], dropped: list[str], names: dict[str, str]) -> str:
    if label == "Plan A":
        return "Highest demand-weighted impact under every constraint (proven optimal)."
    loss = 100 * (best.objective - plan.objective) / best.objective if best.objective else 0.0
    dc = plan.total_cost - best.total_cost
    swap = []
    if dropped:
        swap.append("drops " + ", ".join(names.get(c, c) for c in dropped))
    if added:
        swap.append("adds " + ", ".join(names.get(c, c) for c in added))
    money = f"costs Rs {abs(dc):,} {'less' if dc < 0 else 'more'}" if dc else "costs the same"
    a = {s.skill_id: s.closure_pct for s in best.skills}
    gains = [s.skill_id for s in plan.skills if s.closure_pct > a.get(s.skill_id, 0) + 0.5]
    tail = f"; closes more of {', '.join(names.get(g, g) for g in gains)}" if gains else ""
    text = "; ".join(swap)
    return f"{text[:1].upper()}{text[1:]}: {loss:.1f}% less impact, {money}{tail}."


def alternatives(
    problem: PlanningProblem,
    k: int = 3,
    min_diff: int = 1,
    options: SolverOptions | None = None,
    names: dict[str, str] | None = None,
) -> list[Alternative]:
    """names: display names for course ids and skill ids (used in the trade-off text only)."""
    if k < 1 or min_diff < 1:
        raise ValueError("k and min_diff must be >= 1")
    opts = options or SolverOptions()
    names = names or {}
    try:
        build_model(problem)
    except ModelValidationError as exc:
        raise ValueError(str(exc)) from exc
    found: list[Plan] = []
    sets: list[set[str]] = []
    for _ in range(k):
        hint = opts.model_copy(update={"hint": found[-1].seats()}) if found else opts
        plan = _solve_cheapest_best(problem, sets, min_diff, hint)
        if plan is None or (found and (not plan.courses or plan.objective <= 1e-9)):
            break  # no further structurally different plan with any impact exists
        found.append(plan)
        sets.append(set(plan.seats()))
    if not found:
        return []
    best = found[0]
    out = []
    for i, plan in enumerate(found):
        label = f"Plan {chr(ord('A') + i)}"
        added = sorted(sets[i] - sets[0])
        dropped = sorted(sets[0] - sets[i])
        out.append(Alternative(
            label=label, plan=plan,
            impact_pct_of_best=round(100 * plan.objective / best.objective, 2) if best.objective else 100.0,
            cost_delta=plan.total_cost - best.total_cost,
            hours_delta=plan.total_trainer_hours - best.total_trainer_hours,
            added=added, dropped=dropped,
            tradeoff=_tradeoff(label, best, plan, added, dropped, names),
        ))
    return out
