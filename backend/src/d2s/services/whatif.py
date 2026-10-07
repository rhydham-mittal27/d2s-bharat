"""'What happens if...' simulator: change the world, re-optimise, and compare with today's plan.

A scenario combines any of: budget change (%), trainer-hours change, courses dropped or required,
course prices change (%), demand for a skill area shifts (%), the cohort's gaps grow or shrink (%).
For each scenario we report
  * the re-optimised plan and what changed (courses added/dropped, seat moves, gap closure per area);
  * whether today's plan would still fit, and what it would be worth under the new conditions
    (so the value of re-planning is visible, not just the new plan);
  * if nothing fits, which requirements conflict (CP-SAT assumption core).
"""

import pandas as pd
from pydantic import BaseModel, Field

from d2s.analysis import plan as planlib
from d2s.engine import solve
from d2s.engine.baselines import score
from d2s.engine.schemas import Plan, PlanningProblem
from d2s.models.hike import LABELS
from d2s.services.planner import OPTS, PlanRequest, _problem


class Scenario(BaseModel):
    budget_pct: int = Field(0, ge=-95, le=500, description="change in training budget, %")
    trainer_hours_delta: int = Field(0, ge=-2000, le=5000, description="trainer-hours added (+) or lost (-)")
    remove_courses: list[str] = Field(default_factory=list, description="courses no longer available")
    require_courses: list[str] = Field(default_factory=list, description="courses that must run")
    cost_pct: int = Field(0, ge=-90, le=500, description="change in every course's price, %")
    demand_pct: dict[str, int] = Field(default_factory=dict, description="area -> change in its evidence weight, %")
    cohort_pct: int = Field(0, ge=-90, le=500, description="change in learners short in every area, %")


class PlanSummary(BaseModel):
    feasible: bool
    courses: dict[str, int]  # course id -> seats
    cost: int
    trainer_hours: int
    seats: int
    impact: float
    closure_pct: dict[str, float]
    learners_closed: dict[str, float]


class WhatIfResult(BaseModel):
    scenario: Scenario
    title: str
    today: PlanSummary
    after: PlanSummary
    today_still_fits: bool
    today_impact_after: float | None  # today's plan valued under the new conditions (if it still fits)
    added: list[str]
    dropped: list[str]
    seat_changes: dict[str, int]  # courses in both plans whose seats changed: id -> delta
    impact_change_pct: float | None  # re-optimised vs today (same units only when weights unchanged)
    replanning_gain_pct: float | None  # re-optimised vs keeping today's plan, both under new conditions
    narrative: list[str]
    conflicts: list[str]
    course_names: dict[str, str]


PRESETS = [
    {"id": "budget_cut", "title": "Budget is cut by 30%", "scenario": {"budget_pct": -30}},
    {"id": "budget_up", "title": "We get 50% more budget", "scenario": {"budget_pct": 50}},
    {"id": "trainer_leaves", "title": "A trainer leaves (-60 trainer-hours)", "scenario": {"trainer_hours_delta": -60}},
    {"id": "prices_up", "title": "Course prices rise 20%", "scenario": {"cost_pct": 20}},
    {"id": "ai_boom", "title": "Demand for AI & ML jumps 50%", "scenario": {"demand_pct": {"ai_and_ml_skills": 50}}},
    {"id": "bigger_cohort", "title": "Next batch has 40% more learners short", "scenario": {"cohort_pct": 40}},
    {"id": "lose_stats", "title": "The statistics course is unavailable", "scenario": {"remove_courses": ["stats"]}},
    {"id": "must_bigdata", "title": "Management insists on the Big-Data course",
     "scenario": {"require_courses": ["bigdata"]}},
]


def _apply(problem: PlanningProblem, s: Scenario) -> PlanningProblem:
    known = {c.id for c in problem.courses}
    areas = {sk.skill_id for sk in problem.skills}
    bad = [c for c in s.remove_courses + s.require_courses if c not in known]
    if bad:
        raise ValueError(f"unknown courses: {bad}")
    if set(s.remove_courses) & set(s.require_courses):
        raise ValueError("a course cannot be both removed and required")
    bad_areas = set(s.demand_pct) - areas
    if bad_areas:
        raise ValueError(f"unknown skill areas: {sorted(bad_areas)}")
    for a, v in s.demand_pct.items():
        if not -100 <= v <= 500:
            raise ValueError(f"demand change for {a} must be between -100% and +500%")

    c = problem.constraints
    hours = None if c.trainer_hours is None else max(0, c.trainer_hours + s.trainer_hours_delta)
    cons = c.model_copy(update={"budget": max(0, round(c.budget * (1 + s.budget_pct / 100))),
                                "trainer_hours": hours})
    k = 1 + s.cost_pct / 100
    courses = [x.model_copy(update={"fixed_cost": round(x.fixed_cost * k), "cost_per_seat": round(x.cost_per_seat * k)})
               for x in problem.courses] if s.cost_pct else problem.courses
    skills = [sk.model_copy(update={
        "demand_weight": sk.demand_weight * (1 + s.demand_pct.get(sk.skill_id, 0) / 100),
        "learners_short": max(0, round(sk.learners_short * (1 + s.cohort_pct / 100)))}) for sk in problem.skills]
    return problem.model_copy(update={
        "constraints": cons, "courses": courses, "skills": skills,
        "forced_out": sorted(set(problem.forced_out) | set(s.remove_courses)),
        "forced_in": sorted(set(problem.forced_in) | set(s.require_courses))})


def _summary(problem: PlanningProblem, plan: Plan) -> PlanSummary:
    if not plan.is_solution:
        return PlanSummary(feasible=False, courses={}, cost=0, trainer_hours=0, seats=0, impact=0.0,
                           closure_pct={}, learners_closed={})
    return PlanSummary(feasible=True, courses=plan.seats(), cost=plan.total_cost,
                       trainer_hours=plan.total_trainer_hours, seats=plan.total_seats, impact=plan.objective,
                       closure_pct={s.skill_id: round(s.closure_pct, 1) for s in plan.skills},
                       learners_closed={s.skill_id: round(s.learners_closed, 1) for s in plan.skills})


def _fits(problem: PlanningProblem, seats: dict[str, int]) -> bool:
    by_id = {c.id: c for c in problem.courses}
    c = problem.constraints
    if any(cid in problem.forced_out for cid in seats):
        return False
    if any(cid not in seats for cid in problem.forced_in):
        return False
    cost = sum(by_id[i].fixed_cost + by_id[i].cost_per_seat * n for i, n in seats.items())
    hours = sum(by_id[i].trainer_hours for i in seats)
    return cost <= c.budget and (c.trainer_hours is None or hours <= c.trainer_hours)


def _plain(label: str, names: dict[str, str]) -> str:
    """CP-SAT constraint-group label -> words a planner uses."""
    key, _, rest = label.partition(": ")
    if label == "budget":
        return "the budget"
    if label == "trainer-hours":
        return "the trainer-hours available"
    if label == "total seats":
        return "the seat limit"
    if key in ("forced in", "mandatory"):
        return f"must run {names.get(rest, rest)}"
    if key == "forced out":
        return f"{names.get(rest, rest)} is unavailable"
    if key == "equity floor":
        area, _, pct = rest.partition(" >= ")
        return f"close at least {pct} of the {LABELS.get(area, area)} gap"
    return label


def _why_unchanged(problem: PlanningProblem, plan: Plan, s: Scenario) -> list[str]:
    """When extra resources change nothing, find out exactly which other limit is in the way by re-solving
    with that limit relaxed as well (+50%)."""
    c = problem.constraints
    if all(sk.closure_pct >= 99.5 for sk in plan.skills if sk.learners_short):
        return ["Every gap the catalogue can address is already closed."]

    def gains(update: dict) -> bool:
        p = problem.model_copy(update={"constraints": c.model_copy(update=update)})
        alt = solve(p, OPTS.model_copy(update={"hint": plan.seats()}))
        return alt.is_solution and alt.objective > plan.objective + 1e-6

    more_money = s.budget_pct > 0 or s.cost_pct < 0
    more_hours = s.trainer_hours_delta > 0
    if more_money and c.trainer_hours is not None and gains({"trainer_hours": int(c.trainer_hours * 1.5) + 1}):
        return [("More money alone does not help: trainer capacity is the limit. With more trainer-hours as "
                 "well, the extra budget would add impact.")]
    if more_hours and gains({"budget": int(c.budget * 1.5) + 1}):
        return [("More trainer-hours alone do not help: the budget is the limit. With more money as well, the "
                 "extra hours would add impact.")]
    if more_money or more_hours:
        return [("Neither more money nor more trainer-hours on their own add impact here: the remaining "
                 "courses are blocked by prerequisites, either/or choices or seat caps.")]
    return []


def _title(s: Scenario, names: dict[str, str]) -> str:
    parts = []
    if s.budget_pct:
        parts.append(f"budget {s.budget_pct:+d}%")
    if s.trainer_hours_delta:
        parts.append(f"trainer-hours {s.trainer_hours_delta:+d}")
    if s.cost_pct:
        parts.append(f"course prices {s.cost_pct:+d}%")
    for a, v in s.demand_pct.items():
        if v:
            parts.append(f"{LABELS.get(a, a)} demand {v:+d}%")
    if s.cohort_pct:
        parts.append(f"learners short {s.cohort_pct:+d}%")
    parts += [f"without {names.get(c, c)}" for c in s.remove_courses]
    parts += [f"must run {names.get(c, c)}" for c in s.require_courses]
    return "What if " + (", ".join(parts) if parts else "nothing changes") + "?"


def simulate(gaps: pd.DataFrame, tri: pd.DataFrame, catalogue, req: PlanRequest, s: Scenario) -> WhatIfResult:
    courses = planlib.load_catalogue(catalogue)
    names = {c.id: c.name for c in courses}
    base_problem = _problem(gaps, tri, courses, req)
    today_plan = solve(base_problem, OPTS)
    if not today_plan.is_solution:
        raise ValueError("today's settings have no feasible plan; adjust them first")
    new_problem = _apply(base_problem, s)
    after_plan = solve(new_problem, OPTS.model_copy(update={"hint": today_plan.seats()}))
    today, after = _summary(base_problem, today_plan), _summary(new_problem, after_plan)

    still_fits = _fits(new_problem, today.courses)
    today_after = score(new_problem, today.courses)[0] if still_fits else None
    if still_fits and after.feasible and after.impact <= today_after + 1e-6 * max(1.0, abs(today_after)):
        # Today's plan is still optimal (the solver may have returned a different, equally good plan):
        # never recommend a change that gains nothing.
        impact, closure = score(new_problem, today.courses)
        by_skill = {sk.skill_id: sk for sk in new_problem.skills}
        after = today.model_copy(update={
            "impact": impact, "closure_pct": {k: round(v, 1) for k, v in closure.items()},
            "learners_closed": {k: round(v / 100 * by_skill[k].learners_short, 1) for k, v in closure.items()}})
    weights_same = not s.demand_pct and not s.cohort_pct
    impact_change = (round(100 * (after.impact - today.impact) / today.impact, 1)
                     if after.feasible and weights_same and today.impact else None)
    gain = (round(100 * (after.impact - today_after) / today_after, 1)
            if after.feasible and today_after else None)

    added = [c for c in after.courses if c not in today.courses]
    dropped = [c for c in today.courses if c not in after.courses]
    moved = {c: after.courses[c] - today.courses[c] for c in after.courses
             if c in today.courses and after.courses[c] != today.courses[c]}

    narrative, conflicts = [], []
    if not after.feasible:
        from d2s.xai.plan import infeasibility

        exp = infeasibility(new_problem)
        conflicts = [_plain(c, names) for c in (exp.details or {}).get("conflicting", [])]
        narrative.append("No plan can satisfy these conditions.")
        if conflicts:
            narrative.append("These cannot all hold together: " + "; ".join(conflicts) + ".")
            narrative.append("Relaxing any one of them may restore a plan.")
    else:
        if not added and not dropped and not moved:
            narrative.append("The best plan does not change.")
            narrative += _why_unchanged(new_problem, after_plan, s)
        if dropped:
            narrative.append("Drop " + ", ".join(names[c] for c in dropped) + ".")
        if added:
            narrative.append("Add " + ", ".join(names[c] for c in added) + ".")
        for c, d in moved.items():
            narrative.append(f"{names[c]}: {abs(d)} {'more' if d > 0 else 'fewer'} seats.")
        if impact_change is not None and (added or dropped or moved):
            narrative.append(f"Demand-weighted impact changes by {impact_change:+.1f}%.")
        for a in after.closure_pct:
            d = after.closure_pct[a] - today.closure_pct.get(a, 0)
            if abs(d) >= 5:
                narrative.append(f"{LABELS.get(a, a)}: {today.closure_pct.get(a, 0):.0f}% -> "
                                 f"{after.closure_pct[a]:.0f}% of the gap closed.")
        if not still_fits:
            narrative.append("Today's plan no longer fits these conditions, so re-planning is required.")
        elif gain is not None and gain > 0.05:
            narrative.append(f"Keeping today's plan would leave {gain:.1f}% of the achievable impact on the table.")
        elif gain is not None:
            narrative.append("Today's plan is still the best choice under these conditions.")
    return WhatIfResult(scenario=s, title=_title(s, names), today=today, after=after, today_still_fits=still_fits,
                        today_impact_after=today_after, added=added, dropped=dropped, seat_changes=moved,
                        impact_change_pct=impact_change, replanning_gain_pct=gain, narrative=narrative,
                        conflicts=conflicts, course_names=names)
