"""Naive baselines: how institutions often pick courses without an optimiser.

Each baseline orders the catalogue by a simple rule and fills it greedily under the *same* constraints
as the optimiser (budget, trainer-hours, total seats, seat caps, minimum batch, prerequisites,
either/or groups). Every plan, the optimiser's included, is scored with the same objective:

    impact = Σ_s weight_s · min(short_s, Σ_c cov[c,s]/100 · seats_c)

so "the optimiser closes X% more demand-weighted gap than the best naive rule" is a like-for-like
comparison. Baselines are deliberately sensible (they stop adding seats once the gaps a course
serves are closed, and run mandatory courses first), not straw men. They ignore equity floors.
"""

import math

from pydantic import BaseModel

from d2s.engine.schemas import Course, PlanningProblem


class BaselinePlan(BaseModel):
    id: str
    name: str
    rule: str
    seats: dict[str, int]
    cost: int
    trainer_hours: int
    impact: float
    closure_pct: dict[str, float]  # skill -> % of its gap closed
    gaps_fully_closed: int


def score(problem: PlanningProblem, seats: dict[str, int]) -> tuple[float, dict[str, float]]:
    """Objective value of a seat allocation, and % of each gap it closes (the optimiser's formula)."""
    courses = {c.id: c for c in problem.courses}
    impact, closure = 0.0, {}
    for s in problem.skills:
        supply = sum(courses[cid].coverage.get(s.skill_id, 0) / 100 * n for cid, n in seats.items())
        closed = min(s.learners_short, supply)
        impact += s.demand_weight * closed
        closure[s.skill_id] = 100 * closed / s.learners_short if s.learners_short else 100.0
    return impact, closure


def _greedy(problem: PlanningProblem, order: list[Course]) -> dict[str, int]:
    cons = problem.constraints
    budget, hours = cons.budget, cons.trainer_hours
    seats_left = cons.max_total_seats
    remaining = {s.skill_id: float(s.learners_short) for s in problem.skills}
    chosen: dict[str, int] = {}
    groups_used: set[str] = set()
    blocked = set(problem.forced_out)

    for c in order:
        if c.id in blocked or c.id in chosen:
            continue
        if any(p not in chosen for p in c.prerequisites):
            continue
        if c.exclusive_group and c.exclusive_group in groups_used:
            continue
        if hours is not None and c.trainer_hours > hours:
            continue
        # seats worth buying: enough to close the largest remaining gap this course serves
        useful = max((math.ceil(remaining[s] * 100 / pct) for s, pct in c.coverage.items()
                      if pct > 0 and remaining.get(s, 0) > 0), default=0)
        if useful == 0:
            continue
        n = min(c.max_seats, useful)
        if seats_left is not None:
            n = min(n, seats_left)
        if c.cost_per_seat:
            n = min(n, (budget - c.fixed_cost) // c.cost_per_seat)
        elif c.fixed_cost > budget:
            n = 0
        n = max(n, c.min_batch) if n > 0 else 0
        cost = c.fixed_cost + c.cost_per_seat * n
        if n <= 0 or n < c.min_batch or cost > budget:
            continue
        chosen[c.id] = n
        budget -= cost
        if hours is not None:
            hours -= c.trainer_hours
        if seats_left is not None:
            seats_left -= n
        if c.exclusive_group:
            groups_used.add(c.exclusive_group)
        for s, pct in c.coverage.items():
            if s in remaining:
                remaining[s] = max(0.0, remaining[s] - pct / 100 * n)
    return chosen


def _prereqs_first(problem: PlanningProblem, ranked: list[Course]) -> list[Course]:
    """Keep the rule's ranking but put each course's prerequisites just before it."""
    by_id = {c.id: c for c in problem.courses}
    out: list[Course] = []

    def add(c: Course, seen: frozenset = frozenset()):
        if c in out or c.id in seen:
            return
        for p in c.prerequisites:
            if p in by_id:
                add(by_id[p], seen | {c.id})
        out.append(c)

    for c in ranked:
        add(c)
    return out


def _area_led(problem: PlanningProblem, area_rank: list[str]) -> list[Course]:
    """For each area in rank order, its best-covering course (cheaper per seat breaks ties)."""
    ranked: list[Course] = []
    for a in area_rank:
        cands = sorted((c for c in problem.courses if c.coverage.get(a, 0) > 0),
                       key=lambda c: (-c.coverage[a], c.cost_per_seat + c.fixed_cost / max(c.max_seats, 1)))
        ranked += [c for c in cands if c not in ranked]
    return _prereqs_first(problem, ranked)


RULES = {
    "biggest_gap": ("Biggest gap first", "fund the skill area with the most learners short, then the next"),
    "most_demanded": ("Most in-demand first", "fund the skill area employers ask for most, then the next"),
    "cheapest_seat": ("Cheapest seats first", "buy the lowest cost-per-learner courses until the money runs out"),
}


def baselines(problem: PlanningProblem, demand_share: dict[str, float]) -> list[BaselinePlan]:
    """demand_share: area -> share of postings (the 'popularity' a naive planner would look at)."""
    gaps = sorted(problem.skills, key=lambda s: -s.learners_short)
    orders = {
        "biggest_gap": _area_led(problem, [s.skill_id for s in gaps]),
        "most_demanded": _area_led(problem, sorted((s.skill_id for s in problem.skills),
                                                   key=lambda a: -demand_share.get(a, 0))),
        "cheapest_seat": _prereqs_first(problem, sorted(
            problem.courses, key=lambda c: c.cost_per_seat + c.fixed_cost / max(c.max_seats, 1))),
    }
    by_id = {c.id: c for c in problem.courses}
    must = [c for c in problem.courses if c.mandatory or c.id in problem.forced_in]
    out = []
    for key, order in orders.items():
        seats = _greedy(problem, _prereqs_first(problem, must + order))  # required courses go first
        impact, closure = score(problem, seats)
        name, rule = RULES[key]
        out.append(BaselinePlan(
            id=key, name=name, rule=rule, seats=seats,
            cost=sum(by_id[c].fixed_cost + by_id[c].cost_per_seat * n for c, n in seats.items()),
            trainer_hours=sum(by_id[c].trainer_hours for c in seats),
            impact=impact, closure_pct=closure,
            gaps_fully_closed=sum(v >= 99.5 for v in closure.values())))
    return out
