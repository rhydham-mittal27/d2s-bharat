"""CP-SAT model for choosing courses and seats that close the most demand-weighted skill gap.

    max  Σ_s w_s · closed_s
    s.t. closed_s = min(100·short_s, Σ_c cov[c,s]·seats_c)     (saturation: no value past the gap)
         min_batch_c·run_c ≤ seats_c ≤ max_seats_c·run_c
         Σ_c fixed_c·run_c + per_seat_c·seats_c ≤ budget
         Σ_c hours_c·run_c ≤ trainer_hours
         run_c ≤ run_p  for every prerequisite p of c
         Σ_{c∈g} run_c ≤ 1  for every exclusive group g
         closed_s ≥ min_closure_pct_s · short_s                (equity floor)

``closed_s`` is in "percent-learners" (learners × 100) so coverage percentages stay integral.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from d2s.engine.schemas import PlanningProblem

WEIGHT_RESOLUTION = 1000  # largest demand weight maps to this integer


class ModelValidationError(ValueError):
    """Input that can never produce a plan; reported to the user verbatim."""


@dataclass
class BuiltModel:
    model: cp_model.CpModel
    run: dict[str, cp_model.IntVar]
    seats: dict[str, cp_model.IntVar]
    closed: dict[str, cp_model.IntVar]
    int_weights: dict[str, int]
    weight_unscale: float  # objective_int * weight_unscale = objective in real units
    cost_expr: cp_model.LinearExpr
    impact_expr: cp_model.LinearExpr
    budget_ct: cp_model.Constraint
    warnings: list[str] = field(default_factory=list)
    # label -> assumption literal guarding a relaxable constraint group (with_assumptions=True only)
    assumptions: dict[str, cp_model.IntVar] = field(default_factory=dict)


def validate(problem: PlanningProblem, check_budget_floor: bool = True) -> list[str]:
    """Raise on fatal input errors, return non-fatal warnings."""
    warnings: list[str] = []
    if not problem.courses:
        raise ModelValidationError("no candidate courses given")
    course_ids = [c.id for c in problem.courses]
    if len(set(course_ids)) != len(course_ids):
        raise ModelValidationError("duplicate course ids")
    skill_ids = {s.skill_id for s in problem.skills}
    if len(skill_ids) != len(problem.skills):
        raise ModelValidationError("duplicate skill ids")

    known = set(course_ids)
    for c in problem.courses:
        missing = [p for p in c.prerequisites if p not in known]
        if missing:
            raise ModelValidationError(f"course {c.id}: unknown prerequisites {missing}")
        unknown_skills = [s for s in c.coverage if s not in skill_ids]
        if unknown_skills:
            warnings.append(f"course {c.id}: ignoring coverage of skills without a gap {unknown_skills}")

    for cid in (*problem.forced_in, *problem.forced_out):
        if cid not in known:
            raise ModelValidationError(f"forced course {cid} does not exist")
    clash = set(problem.forced_in) & set(problem.forced_out)
    if clash:
        raise ModelValidationError(f"courses both forced in and out: {sorted(clash)}")

    _check_prerequisite_cycles(problem)

    if not check_budget_floor:
        return warnings
    mandatory_floor = sum(
        c.fixed_cost + c.cost_per_seat * c.min_batch
        for c in problem.courses
        if c.mandatory or c.id in problem.forced_in
    )
    if mandatory_floor > problem.constraints.budget:
        raise ModelValidationError(
            f"mandatory/forced courses need at least ₹{mandatory_floor:,} "
            f"but budget is ₹{problem.constraints.budget:,}"
        )
    return warnings


def _check_prerequisite_cycles(problem: PlanningProblem) -> None:
    graph = {c.id: c.prerequisites for c in problem.courses}
    state: dict[str, int] = {}  # 1 = visiting, 2 = done

    def visit(node: str, path: list[str]) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            cycle = path[path.index(node):] + [node]
            raise ModelValidationError(f"prerequisite cycle: {' -> '.join(cycle)}")
        state[node] = 1
        for p in graph[node]:
            visit(p, path + [node])
        state[node] = 2

    for cid in graph:
        visit(cid, [])


def build_model(problem: PlanningProblem, with_assumptions: bool = False) -> BuiltModel:
    """with_assumptions=True guards every relaxable constraint group with an assumption literal,
    so that an infeasible model can report a small conflicting subset (see d2s.xai.plan)."""
    warnings = validate(problem, check_budget_floor=not with_assumptions)
    m = cp_model.CpModel()
    cons = problem.constraints
    skills = {s.skill_id: s for s in problem.skills}
    assumptions: dict[str, cp_model.IntVar] = {}

    def guard(ct, label: str):
        if with_assumptions:
            lit = assumptions.get(label)
            if lit is None:
                lit = assumptions[label] = m.new_bool_var(f"assume[{label}]")
            ct.only_enforce_if(lit)
        return ct

    run: dict[str, cp_model.IntVar] = {}
    seats: dict[str, cp_model.IntVar] = {}
    for c in problem.courses:
        run[c.id] = m.new_bool_var(f"run[{c.id}]")
        seats[c.id] = m.new_int_var(0, c.max_seats, f"seats[{c.id}]")
        m.add(seats[c.id] <= c.max_seats * run[c.id])
        m.add(seats[c.id] >= c.min_batch * run[c.id])
        if c.mandatory or c.id in problem.forced_in:
            guard(m.add(run[c.id] == 1), f"{'mandatory' if c.mandatory else 'forced in'}: {c.id}")
        if c.id in problem.forced_out:
            guard(m.add(run[c.id] == 0), f"forced out: {c.id}")
    for c in problem.courses:  # after all run vars exist: prerequisites may appear in any order
        for p in c.prerequisites:
            m.add_implication(run[c.id], run[p])

    groups: dict[str, list[str]] = defaultdict(list)
    for c in problem.courses:
        if c.exclusive_group:
            groups[c.exclusive_group].append(c.id)
    for members in groups.values():
        m.add_at_most_one(run[cid] for cid in members)

    cost_expr = sum(c.fixed_cost * run[c.id] + c.cost_per_seat * seats[c.id] for c in problem.courses)
    budget_ct = guard(m.add(cost_expr <= cons.budget), "budget")
    if cons.trainer_hours is not None:
        guard(m.add(sum(c.trainer_hours * run[c.id] for c in problem.courses) <= cons.trainer_hours),
              "trainer-hours")
    if cons.max_total_seats is not None:
        guard(m.add(sum(seats.values()) <= cons.max_total_seats), "total seats")

    closed: dict[str, cp_model.IntVar] = {}
    for sid, s in skills.items():
        cap = 100 * s.learners_short
        terms = [c.coverage[sid] * seats[c.id] for c in problem.courses if c.coverage.get(sid)]
        closed[sid] = m.new_int_var(0, cap, f"closed[{sid}]")
        if not terms:
            m.add(closed[sid] == 0)
        else:
            supply_ub = sum(c.coverage[sid] * c.max_seats for c in problem.courses if c.coverage.get(sid))
            supply = m.new_int_var(0, supply_ub, f"supply[{sid}]")
            m.add(supply == sum(terms))
            m.add_min_equality(closed[sid], [supply, cap])
        if s.min_closure_pct:
            guard(m.add(closed[sid] >= s.min_closure_pct * s.learners_short),
                  f"equity floor: {sid} >= {s.min_closure_pct}%")

    max_w = max((s.demand_weight for s in problem.skills), default=0.0)
    if max_w <= 0:
        int_weights = {sid: 1 for sid in skills}
        unscale = 0.0
        warnings.append("all demand weights are zero; maximizing raw learners closed")
    else:
        int_weights = {
            sid: max(1, round(WEIGHT_RESOLUTION * s.demand_weight / max_w)) if s.demand_weight > 0 else 0
            for sid, s in skills.items()
        }
        unscale = max_w / WEIGHT_RESOLUTION / 100
    impact_expr = sum(int_weights[sid] * closed[sid] for sid in skills)
    m.maximize(impact_expr)
    if with_assumptions:
        m.add_assumptions(list(assumptions.values()))

    return BuiltModel(
        model=m,
        run=run,
        seats=seats,
        closed=closed,
        int_weights=int_weights,
        weight_unscale=unscale,
        cost_expr=cost_expr,
        impact_expr=impact_expr,
        budget_ct=budget_ct,
        warnings=warnings,
        assumptions=assumptions,
    )
