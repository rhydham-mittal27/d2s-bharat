"""Explaining training plans (CP-SAT).

* constraint report : every constraint with usage, slack and whether it is tight
* course what-if    : the smallest increase in budget, trainer-hours or evidence weight at which an
                      excluded course enters the optimal plan (search by doubling + bisection; each
                      probe is an exact CP-SAT solve)
* infeasibility     : when no plan exists, a small set of conflicting constraint groups from
                      CP-SAT's sufficient_assumptions_for_infeasibility(), with suggested relaxations
"""

from ortools.sat.python import cp_model

from d2s.engine import PlanningProblem, SolverOptions, solve
from d2s.engine.model import ModelValidationError, build_model
from d2s.engine.schemas import Plan
from d2s.xai.schemas import Explanation, Factor

OPTS = SolverOptions(time_limit_s=10, workers=8, gap_limit=0.0)


def _rs(x: float) -> str:
    return f"Rs {x:,.0f}"


def _gain_if_relaxed(problem: PlanningProblem, plan: Plan, field: str, factor: float) -> float:
    cons = problem.constraints
    relaxed = problem.model_copy(update={"constraints": cons.model_copy(
        update={field: int(getattr(cons, field) * factor) + 1})})
    alt = solve(relaxed, OPTS.model_copy(update={"hint": plan.seats()}))
    return (alt.objective - plan.objective) if alt.is_solution else 0.0


def constraint_report(problem: PlanningProblem, plan: Plan, relax_factor: float = 1.25) -> Explanation:
    """A constraint is *binding* if relaxing it alone (by 25%) improves the optimum; small slack is not
    enough to tell, because courses come in whole blocks (e.g. Rs 15k left but no course fits)."""
    c = problem.constraints
    courses = {x.id: x for x in problem.courses}
    chosen = plan.seats()
    f: list[Factor] = []
    tight = []

    def add(name, field, used, limit, unit):
        slack = limit - used
        gain = _gain_if_relaxed(problem, plan, field, relax_factor)
        binding = gain > 1e-6
        f.append(Factor(name=name, value=f"{used:,.0f} / {limit:,.0f}", contribution=slack, unit=f"{unit} slack",
                        source="cp-sat.relax-and-resolve",
                        note=(f"binding: +{relax_factor - 1:.0%} would add {gain:,.0f} impact" if binding
                              else f"not binding: +{relax_factor - 1:.0%} adds nothing")))
        if binding:
            tight.append(name)

    add("budget", "budget", plan.total_cost, c.budget, "Rs")
    if c.trainer_hours is not None:
        add("trainer-hours", "trainer_hours", plan.total_trainer_hours, c.trainer_hours, "hours")
        if "budget" not in tight and "trainer-hours" not in tight:
            # neither binds alone; test both together (a course may need more money AND more hours)
            both = problem.model_copy(update={"constraints": c.model_copy(update={
                "budget": int(c.budget * relax_factor) + 1, "trainer_hours": int(c.trainer_hours * relax_factor) + 1})})
            alt = solve(both, OPTS.model_copy(update={"hint": plan.seats()}))
            gain = (alt.objective - plan.objective) if alt.is_solution else 0.0
            if gain > 1e-6:
                tight.append("budget and trainer-hours together")
                f.append(Factor(name="budget + trainer-hours together", value=f"+{relax_factor - 1:.0%} of both",
                                contribution=round(gain, 2), unit="impact gained", source="cp-sat.relax-and-resolve",
                                note=f"binding together: +{relax_factor - 1:.0%} of both adds {gain:,.0f} impact, "
                                     "although neither alone adds anything"))
    if c.max_total_seats is not None:
        add("total seats", "max_total_seats", plan.total_seats, c.max_total_seats, "seats")
    at_cap = [cid for cid, s in chosen.items() if s >= courses[cid].max_seats]
    for cid in at_cap:
        f.append(Factor(name=f"seat cap: {courses[cid].name or cid}", value=f"{chosen[cid]} / {courses[cid].max_seats}",
                        contribution=0, unit="seats slack", source="cp-sat.slack", note="tight"))
    saturated = [s.skill_id for s in plan.skills if s.learners_short and s.closure_pct >= 99.9]
    groups = {}
    for x in problem.courses:
        if x.exclusive_group:
            groups.setdefault(x.exclusive_group, []).append(x.id)
    for g, members in groups.items():
        pick = [m for m in members if m in chosen]
        f.append(Factor(name=f"either/or group '{g}'", value=", ".join(pick) or "none chosen",
                        source="cp-sat.structure", note=f"members: {', '.join(members)}"))
    caps = [f"seat cap of {courses[x].name or x}" for x in at_cap]
    summary = (f"Binding constraints: {', '.join(tight + caps) or 'none'}. "
               f"Gaps fully closed: {len(saturated)} of {len(plan.skills)}; extra seats there would add no value.")
    return Explanation(subject="plan", question="why", summary=summary, factors=f,
                       method="slack in the optimal CP-SAT solution + exact re-solve with each limit relaxed by 25%",
                       details={"tight": tight, "seat_caps_reached": at_cap, "saturated_gaps": saturated})


def _chosen_with(problem: PlanningProblem, course_id: str, hint) -> tuple[bool, Plan]:
    p = solve(problem, OPTS.model_copy(update={"hint": hint}))
    return (p.is_solution and course_id in p.seats()), p


def _search(make, lo: float, hi_start: float, hi_max: float, resolution: float, course_id: str, hint):
    """Smallest value v in (lo, hi_max] with course chosen under make(v); None if never."""
    hi = hi_start
    while True:
        ok, plan = _chosen_with(make(hi), course_id, hint)
        if ok:
            break
        if hi >= hi_max:
            return None, None
        lo, hi = hi, min(hi * 2 if hi > 0 else resolution, hi_max)
    best_plan = plan
    while hi - lo > resolution:
        mid = (lo + hi) / 2
        ok, plan = _chosen_with(make(mid), course_id, hint)
        if ok:
            hi, best_plan = mid, plan
        else:
            lo = mid
    return hi, best_plan


def course_what_if(problem: PlanningProblem, plan: Plan, course_id: str, resource: str) -> Explanation:
    """resource: 'budget' | 'trainer_hours' | 'weight' (evidence weight of the course's main area)."""
    courses = {x.id: x for x in problem.courses}
    if course_id not in courses:
        raise ValueError(f"unknown course {course_id}")
    c = courses[course_id]
    name = c.name or course_id
    if course_id in plan.seats():
        return Explanation(subject=f"course:{course_id}", question="what_if", summary=f"{name} is already in the plan.",
                           method="n/a")
    cons = problem.constraints
    hint = plan.seats()
    if resource == "budget":
        def make(v):
            return problem.model_copy(update={"constraints": cons.model_copy(update={"budget": int(cons.budget + v)})})
        v, alt = _search(make, 0, 25_000, 5 * max(cons.budget, 100_000), 5_000, course_id, hint)
        unit, fmt = "Rs", _rs
    elif resource == "trainer_hours":
        if cons.trainer_hours is None:
            raise ValueError("this plan has no trainer-hour limit")

        def make(v):
            return problem.model_copy(update={"constraints": cons.model_copy(update={"trainer_hours": int(cons.trainer_hours + v)})})
        v, alt = _search(make, 0, 10, 10 * max(cons.trainer_hours, 50), 2, course_id, hint)
        unit, fmt = "trainer-hours", lambda x: f"{x:,.0f} hours"
    elif resource == "weight":
        area = max(c.coverage, key=c.coverage.get) if c.coverage else None
        if area is None:
            raise ValueError("course covers no skill area")

        def make(v):
            skills = [s.model_copy(update={"demand_weight": s.demand_weight * (1 + v)}) if s.skill_id == area else s
                      for s in problem.skills]
            return problem.model_copy(update={"skills": skills})
        v, alt = _search(make, 0, 0.25, 20.0, 0.02, course_id, hint)
        unit, fmt = "relative increase in evidence weight", lambda x: f"+{x:.0%} weight on {area}"
    else:
        raise ValueError("resource must be budget, trainer_hours or weight")

    if v is None:
        summary = (f"Raising {resource.replace('_', '-')} alone (up to the search limit) never brings {name} into "
                   f"the optimal plan; another constraint also blocks it.")
        return Explanation(subject=f"course:{course_id}", question="what_if", summary=summary,
                           method="doubling + bisection over exact CP-SAT re-solves",
                           details={"resource": resource, "found": False})
    alt_obj = alt.objective if alt else None
    summary = (f"{name} enters the optimal plan with {fmt(v)} more"
               + (f" ({resource.replace('_', '-')})" if resource != "weight" else "")
               + (f"; total impact would rise from {plan.objective:,.0f} to {alt_obj:,.0f}." if alt_obj else "."))
    return Explanation(
        subject=f"course:{course_id}", question="what_if", summary=summary,
        factors=[Factor(name=resource, value=round(v, 4), unit=unit, source="cp-sat.search")],
        counterfactuals=[summary],
        method="doubling + bisection over exact CP-SAT re-solves (course not forced; the solver chooses it)",
        details={"resource": resource, "found": True, "minimal_increase": v,
                 "new_plan_courses": alt.seats() if alt else {}, "new_objective": alt_obj},
    )


_RELAX = {"budget": "increase the budget", "trainer-hours": "add trainer-hours", "total seats": "allow more seats"}


def infeasibility(problem: PlanningProblem) -> Explanation:
    try:
        built = build_model(problem, with_assumptions=True)
    except ModelValidationError as exc:
        return Explanation(subject="plan", question="is_feasible", summary=f"The input itself is invalid: {exc}.",
                           method="input validation")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20
    solver.parameters.num_workers = 1  # assumption cores are reported by the sequential search
    status = solver.solve(built.model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Explanation(subject="plan", question="is_feasible", summary="A feasible plan exists; nothing conflicts.",
                           method="CP-SAT with assumption literals", details={"feasible": True})
    if status != cp_model.INFEASIBLE:
        return Explanation(subject="plan", question="is_feasible",
                           summary="Feasibility could not be decided within the time limit.",
                           method="CP-SAT with assumption literals", details={"feasible": None})
    index_to_label = {lit.index: label for label, lit in built.assumptions.items()}
    core = [index_to_label[i] for i in solver.sufficient_assumptions_for_infeasibility() if i in index_to_label]
    fixes = []
    for label in core:
        key = label.split(":")[0]
        if label in _RELAX:
            fixes.append(_RELAX[label])
        elif key == "equity floor":
            fixes.append(f"lower the {label}")
        elif key in ("forced in", "mandatory"):
            fixes.append(f"drop the requirement '{label}'")
        elif key == "forced out":
            fixes.append(f"allow '{label.split(': ')[1]}' again")
    summary = ("No plan satisfies all requirements. These cannot all hold together: "
               + "; ".join(core) + ". Relaxing any one of them may restore feasibility: " + "; ".join(fixes) + ".")
    return Explanation(subject="plan", question="is_feasible", summary=summary,
                       factors=[Factor(name=label, source="cp-sat.assumption-core") for label in core],
                       counterfactuals=fixes,
                       method="CP-SAT sufficient_assumptions_for_infeasibility over guarded constraint groups",
                       details={"feasible": False, "conflicting": core})
