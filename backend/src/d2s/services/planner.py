"""What-If Simulator: optimal training plan for a cohort, with binding-constraint analysis,
why-not explanations, scenario robustness and the cost-impact frontier.

Gaps come from the cohort profile (or the JDS sample); gap values from the RQ1+RQ2 evidence;
courses from the documented catalogue. Every solve is CP-SAT, proven optimal or flagged.
"""

import pandas as pd
from pydantic import BaseModel, Field

from d2s.analysis import plan as planlib
from d2s.engine import PlanningProblem, SolverOptions, alternatives, frontier, solve, why_not
from d2s.engine.schemas import Plan
from d2s.models.hike import LABELS

OPTS = SolverOptions(time_limit_s=10, workers=8, gap_limit=0.0)


class PlanRequest(BaseModel):
    budget: int = Field(400_000, ge=0)
    trainer_hours: int | None = Field(200, ge=0)
    scheme: str = Field("balanced", pattern="^(balanced|market-led|outcome-led)$")
    bound: str = Field("point", pattern="^(low|point|high)$")
    forced_in: list[str] = []
    forced_out: list[str] = []
    min_closure_pct: dict[str, int] = Field(default_factory=dict,
                                            description="equity floors: area -> minimum % of its gap to close")


class Binding(BaseModel):
    budget_slack: int
    hours_slack: int | None
    gain_extra_budget: float  # exact objective gain from +step_budget
    gain_extra_hours: float  # exact objective gain from +step_hours
    gain_both: float
    step_budget: int
    step_hours: int
    verdict: str


class PlanResult(BaseModel):
    request: PlanRequest
    plan: Plan
    course_names: dict[str, str]
    weights: dict[str, float]
    gaps: dict[str, int]
    binding: Binding
    why_not: list[dict]


def _problem(gaps: pd.DataFrame, tri: pd.DataFrame, courses, req: PlanRequest) -> PlanningProblem:
    w = planlib.evidence_weights(tri, req.scheme, req.bound)
    p = planlib.build_problem(gaps, w, courses, req.budget, req.trainer_hours)
    unknown = set(req.min_closure_pct) - {s.skill_id for s in p.skills}
    if unknown:
        raise ValueError(f"unknown skill areas in min_closure_pct: {sorted(unknown)}")
    skills = [s.model_copy(update={"min_closure_pct": int(req.min_closure_pct.get(s.skill_id, 0))}) for s in p.skills]
    return p.model_copy(update={"skills": skills, "forced_in": req.forced_in, "forced_out": req.forced_out})


def _binding(problem: PlanningProblem, base: Plan, step_budget: int = 150_000, step_hours: int = 40) -> Binding:
    c = problem.constraints
    hint = OPTS.model_copy(update={"hint": base.seats()})

    def gain(budget, hours):
        alt = solve(problem.model_copy(update={"constraints": c.model_copy(update={"budget": budget, "trainer_hours": hours})}), hint)
        return (alt.objective - base.objective) if alt.is_solution else 0.0

    hrs = c.trainer_hours
    g_b = gain(c.budget + step_budget, hrs)
    g_h = gain(c.budget, None if hrs is None else hrs + step_hours)
    g_bh = gain(c.budget + step_budget, None if hrs is None else hrs + step_hours)
    eps = 1e-6
    if g_b > eps and g_h <= eps:
        verdict = "budget is the binding constraint"
    elif g_h > eps and g_b <= eps:
        verdict = "trainer capacity is the binding constraint"
    elif g_b > eps and g_h > eps:
        verdict = "both budget and trainer capacity limit the plan"
    elif g_bh > eps:
        verdict = "the next course needs more budget AND more trainer-hours together"
    else:
        verdict = "all valuable gaps are covered; more resources add nothing"
    return Binding(budget_slack=c.budget - base.total_cost,
                   hours_slack=None if hrs is None else hrs - base.total_trainer_hours,
                   gain_extra_budget=g_b, gain_extra_hours=g_h, gain_both=g_bh,
                   step_budget=step_budget, step_hours=step_hours, verdict=verdict)


def optimise(gaps: pd.DataFrame, tri: pd.DataFrame, catalogue_path, req: PlanRequest,
             explain_excluded: bool = True) -> PlanResult:
    courses = planlib.load_catalogue(catalogue_path)
    problem = _problem(gaps, tri, courses, req)
    best = solve(problem, OPTS)
    names = {c.id: c.name for c in courses}
    if not best.is_solution:
        raise ValueError("; ".join(best.messages) or f"no plan ({best.status})")
    binding = _binding(problem, best)
    wn = []
    if explain_excluded:
        for c in courses:
            if c.id in best.seats() or c.id in req.forced_out:
                continue
            w = why_not(problem, best, c.id, OPTS)
            wn.append({"course_id": c.id, "name": names[c.id], "feasible": w.feasible,
                       "objective_delta": w.objective_delta,
                       "objective_delta_pct": w.objective_delta_pct, "reason": w.reason})
    weights = dict(zip(planlib.AREAS, planlib.evidence_weights(tri, req.scheme, req.bound)["weight"].round(2), strict=True))
    return PlanResult(request=req, plan=best, course_names=names, weights=weights,
                      gaps=dict(zip(gaps["area"], gaps["learners_short"].astype(int), strict=True)),
                      binding=binding, why_not=wn)


def robustness(gaps: pd.DataFrame, tri: pd.DataFrame, catalogue_path, req: PlanRequest) -> dict:
    """Re-optimise under 3 weighting schemes x 3 CI bounds; report each course's selection rate."""
    courses = planlib.load_catalogue(catalogue_path)
    seats = {}
    for scheme in planlib.SCHEMES:
        for bound in ("low", "point", "high"):
            r = req.model_copy(update={"scheme": scheme, "bound": bound})
            p = solve(_problem(gaps, tri, courses, r), OPTS)
            seats[f"{scheme}/{bound}"] = p.seats() if p.is_solution else {}
    out = []
    for c in courses:
        picks = [s.get(c.id, 0) for s in seats.values()]
        out.append({"course_id": c.id, "name": c.name, "chosen_in": sum(v > 0 for v in picks),
                    "scenarios": len(picks), "seats_min": min(picks), "seats_max": max(picks)})
    core = [o["course_id"] for o in out if o["chosen_in"] == len(seats)]
    return {"courses": out, "safe_core": core, "scenarios": list(seats)}


def cost_frontier(gaps: pd.DataFrame, tri: pd.DataFrame, catalogue_path, req: PlanRequest,
                  max_budget: int = 800_000, levels: int = 9) -> list[dict]:
    courses = planlib.load_catalogue(catalogue_path)
    p = _problem(gaps, tri, courses, req)
    p = p.model_copy(update={"constraints": p.constraints.model_copy(update={"budget": max_budget})})
    pts = frontier(p, levels=levels, min_budget=max(50_000, max_budget // levels), options=OPTS)
    return [{"budget_cap": x.budget_cap, "cost": x.cost, "impact": round(x.impact, 2),
             "courses": [c.course_id for c in x.plan.courses]} for x in pts]


def alternative_plans(gaps: pd.DataFrame, tri: pd.DataFrame, catalogue_path, req: PlanRequest,
                      k: int = 3, min_diff: int = 1) -> list[dict]:
    """Plan A (optimal) plus the next-best plans that each differ in at least min_diff courses."""
    courses = planlib.load_catalogue(catalogue_path)
    names = {c.id: c.name for c in courses}
    labels = {**names, **LABELS}
    alts = alternatives(_problem(gaps, tri, courses, req), k=k, min_diff=min_diff, options=OPTS, names=labels)
    if not alts:
        raise ValueError("no feasible plan under these constraints")
    return [{**a.model_dump(), "course_names": {c: names[c] for c in a.plan.seats()}} for a in alts]


# ---- confidence ---------------------------------------------------------------------------------------------
class CourseConfidence(BaseModel):
    course_id: str
    name: str
    chosen_in: int  # evidence readings in which the optimiser still picks this course
    scenarios: int
    level: str  # high | medium | low
    seats_range: tuple[int, int]


class Confidence(BaseModel):
    level: str  # plan-level: high | medium | low
    score: float  # mean share of evidence readings that keep each recommended course (0-1)
    same_plan_share: float  # readings that give exactly the same set of courses
    scenarios: list[str]
    courses: list[CourseConfidence]
    explanation: str


def _level(share: float) -> str:
    return "high" if share >= 0.85 else "medium" if share >= 0.5 else "low"


def confidence(gaps: pd.DataFrame, tri: pd.DataFrame, catalogue_path, req: PlanRequest) -> Confidence:
    """How much the recommendation depends on how the evidence is read: re-optimise under the 3 weighting
    schemes x the low / point / high ends of every 95% interval (9 readings) and count how often each
    recommended course survives."""
    courses = planlib.load_catalogue(catalogue_path)
    names = {c.id: c.name for c in courses}
    base = solve(_problem(gaps, tri, courses, req), OPTS)
    if not base.is_solution:
        raise ValueError("; ".join(base.messages) or "no feasible plan")
    chosen = base.seats()
    runs: dict[str, dict[str, int]] = {}
    for scheme in planlib.SCHEMES:
        for bound in ("low", "point", "high"):
            p = solve(_problem(gaps, tri, courses, req.model_copy(update={"scheme": scheme, "bound": bound})),
                      OPTS.model_copy(update={"hint": chosen}))
            runs[f"{scheme}/{bound}"] = p.seats() if p.is_solution else {}
    n = len(runs)
    per = []
    for cid in chosen:
        k = sum(cid in r for r in runs.values())
        seats = [r[cid] for r in runs.values() if cid in r] or [0]
        per.append(CourseConfidence(course_id=cid, name=names.get(cid, cid), chosen_in=k, scenarios=n,
                                    level=_level(k / n), seats_range=(min(seats), max(seats))))
    per.sort(key=lambda c: -c.chosen_in)
    score = sum(c.chosen_in for c in per) / (n * len(per)) if per else 0.0
    same = sum(set(r) == set(chosen) for r in runs.values()) / n
    weak = [c.name for c in per if c.level == "low"]
    expl = (f"{sum(c.level == 'high' for c in per)} of {len(per)} recommended courses are chosen under at least "
            f"85% of the {n} ways of reading the evidence; the same set of courses comes out in "
            f"{round(same * n)} of {n}.")
    if weak:
        expl += f" Least certain: {', '.join(weak)}. These depend on how the evidence is weighted."
    return Confidence(level=_level(score), score=round(score, 3), same_plan_share=round(same, 3),
                      scenarios=list(runs), courses=per, explanation=expl)


# ---- naive baselines ------------------------------------------------------------------------------------------
class BaselineComparison(BaseModel):
    optimiser: dict
    baselines: list[dict]
    best_naive: str  # best rule *in hindsight* for these exact settings
    uplift_vs_best_pct: float  # vs that hindsight-best rule
    uplift_vs_average_pct: float  # vs the average rule: a planner does not know in advance which rule wins
    summary: str
    sweep: list[dict] | None = None  # optional: each rule's % of the optimum at other budgets
    worst_case_pct: dict[str, float] | None = None  # rule -> lowest % of the optimum across the sweep


def compare_baselines(problem: PlanningProblem, plan: Plan, tri: pd.DataFrame,
                      names: dict[str, str]) -> BaselineComparison:
    from d2s.engine.baselines import baselines, score

    share = dict(zip(tri["area"], tri["demand_share"], strict=True))
    opt_impact, opt_closure = score(problem, plan.seats())
    rows = baselines(problem, share)
    best = max(rows, key=lambda b: b.impact)

    def pct(x: float) -> float:
        return round(100 * x / opt_impact, 1) if opt_impact else 0.0

    def row(b):
        return {**b.model_dump(), "impact_pct_of_optimiser": pct(b.impact),
                "courses": [names.get(c, c) for c in b.seats]}

    uplift = round(100 * (opt_impact - best.impact) / best.impact, 1) if best.impact else 0.0
    avg = sum(r.impact for r in rows) / len(rows)
    uplift_avg = round(100 * (opt_impact - avg) / avg, 1) if avg else 0.0
    summary = (f"With the same budget and trainer-hours, the optimiser closes {uplift_avg}% more demand-weighted "
               f"skill gap than a simple rule on average. ")
    summary += (f"The best rule for these exact settings, \"{best.name}\", happens to reach the optimum, but "
                f"which rule wins changes with the budget." if uplift <= 0.05 else
                f"Even the best rule for these exact settings, \"{best.name}\", closes {uplift}% less, and "
                f"which rule wins changes with the budget.")
    return BaselineComparison(
        optimiser={"impact": opt_impact, "cost": plan.total_cost, "trainer_hours": plan.total_trainer_hours,
                   "closure_pct": opt_closure, "gaps_fully_closed": sum(v >= 99.5 for v in opt_closure.values()),
                   "courses": [names.get(c, c) for c in plan.seats()]},
        baselines=[row(b) for b in rows], best_naive=best.name, uplift_vs_best_pct=uplift,
        uplift_vs_average_pct=uplift_avg, summary=summary)


SWEEP = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def baseline_sweep(gaps: pd.DataFrame, tri: pd.DataFrame, catalogue_path, req: PlanRequest) -> BaselineComparison:
    """compare_baselines at today's settings, plus every rule's share of the optimum across budgets from
    25% to 200% of today's: a rule that matches the optimiser at one budget can fall far short at another."""
    from d2s.engine.baselines import baselines, score

    courses = planlib.load_catalogue(catalogue_path)
    names = {c.id: c.name for c in courses}
    share = dict(zip(tri["area"], tri["demand_share"], strict=True))
    problem = _problem(gaps, tri, courses, req)
    plan = solve(problem, OPTS)
    if not plan.is_solution:
        raise ValueError("; ".join(plan.messages) or "no feasible plan")
    out = compare_baselines(problem, plan, tri, names)
    sweep, worst = [], {}
    for f in SWEEP:
        b = max(int(req.budget * f), 1)
        p = problem.model_copy(update={"constraints": problem.constraints.model_copy(update={"budget": b})})
        opt = solve(p, OPTS)
        if not opt.is_solution or opt.objective <= 0:
            continue
        opt_impact = score(p, opt.seats())[0]
        rules = {r.id: round(100 * r.impact / opt_impact, 1) for r in baselines(p, share)}
        sweep.append({"budget": b, "rules": rules})
        for k, v in rules.items():
            worst[k] = min(worst.get(k, 100.0), v)
    out.sweep, out.worst_case_pct = sweep, worst
    return out
