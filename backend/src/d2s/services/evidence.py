"""Evidence Trail: Market Signal -> Skill Gap -> Constraint -> Optimisation -> Intervention,
for every course in a plan. Each link is a number traceable to a table or a solve.
"""

import pandas as pd
from pydantic import BaseModel

from d2s.analysis import plan as planlib
from d2s.engine import solve
from d2s.services.planner import OPTS, PlanResult, _problem


class AreaEvidence(BaseModel):
    area: str
    label: str
    coverage_pct: int
    market_demand_share: float
    market_demand_ci: tuple[float, float]
    market_pay_or: float
    market_pay_ci: tuple[float, float]
    junior_hike_or: float
    junior_hike_ci: tuple[float, float]
    gap_value_weight: float
    learners_short: int
    learners_closed_by_course: float


class CourseTrail(BaseModel):
    course_id: str
    name: str
    seats: int
    cost: int
    trainer_hours: int
    areas: list[AreaEvidence]
    impact_lost_if_removed_pct: float
    constraint_context: str


def build(result: PlanResult, gaps: pd.DataFrame, tri: pd.DataFrame, catalogue_path) -> list[CourseTrail]:
    courses = {c.id: c for c in planlib.load_catalogue(catalogue_path)}
    problem = _problem(gaps, tri, list(courses.values()), result.request)
    t = tri.set_index("area")
    chosen = result.plan.seats()
    closed = {s.skill_id: s.learners_closed for s in result.plan.skills}
    # attribute each area's closed learners to courses in proportion to the supply they provide
    supply = {a: sum(courses[c].coverage.get(a, 0) * n / 100 for c, n in chosen.items()) for a in planlib.AREAS}
    trails = []
    for d in result.plan.courses:
        c = courses[d.course_id]
        areas = []
        for a, cov in c.coverage.items():
            r = t.loc[a]
            own = cov * d.seats / 100
            areas.append(AreaEvidence(
                area=a, label=planlib.LABEL[a], coverage_pct=cov,
                market_demand_share=float(r["demand_share"]),
                market_demand_ci=(float(r["demand_ci_low"]), float(r["demand_ci_high"])),
                market_pay_or=float(r["odds_ratio"]), market_pay_ci=(float(r["or_ci_low"]), float(r["or_ci_high"])),
                junior_hike_or=float(r["rq2_odds_ratio"]),
                junior_hike_ci=(float(r["rq2_or_ci_low"]), float(r["rq2_or_ci_high"])),
                gap_value_weight=float(result.weights[a]), learners_short=int(result.gaps[a]),
                learners_closed_by_course=round(closed[a] * own / supply[a], 1) if supply[a] else 0.0,
            ))
        without = problem.model_copy(update={"forced_out": [*problem.forced_out, d.course_id],
                                             "forced_in": [x for x in problem.forced_in if x != d.course_id]})
        alt = solve(without, OPTS.model_copy(update={"hint": chosen}))
        lost = (result.plan.objective - alt.objective) / result.plan.objective * 100 if alt.is_solution else 100.0
        trails.append(CourseTrail(course_id=d.course_id, name=c.name, seats=d.seats, cost=d.cost,
                                  trainer_hours=d.trainer_hours, areas=areas,
                                  impact_lost_if_removed_pct=round(lost, 1),
                                  constraint_context=result.binding.verdict))
    return trails
