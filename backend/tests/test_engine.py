import itertools

import pytest

from d2s.engine import (
    Constraints,
    Course,
    PlanningProblem,
    SkillGap,
    SolveStatus,
    analyze,
    frontier,
    lp_shadow_prices,
    solve,
    why_not,
)


def check_feasible(problem: PlanningProblem, plan) -> None:
    """Invariants every returned plan must satisfy, independent of the solver."""
    courses = {c.id: c for c in problem.courses}
    chosen = plan.seats()
    assert plan.total_cost <= problem.constraints.budget
    assert plan.total_cost == sum(courses[c].fixed_cost + courses[c].cost_per_seat * s for c, s in chosen.items())
    if problem.constraints.trainer_hours is not None:
        assert plan.total_trainer_hours <= problem.constraints.trainer_hours
    for cid, seats in chosen.items():
        c = courses[cid]
        assert c.min_batch <= seats <= c.max_seats
        for p in c.prerequisites:
            assert p in chosen, f"{cid} chosen without prerequisite {p}"
    groups = {}
    for cid in chosen:
        g = courses[cid].exclusive_group
        if g:
            groups[g] = groups.get(g, 0) + 1
    assert all(n <= 1 for n in groups.values())
    for s in plan.skills:
        assert 0 <= s.learners_closed <= s.learners_short


def test_solve_returns_feasible_optimal_plan(problem):
    plan = solve(problem)
    assert plan.status is SolveStatus.OPTIMAL
    check_feasible(problem, plan)
    assert plan.objective > 0
    assert plan.best_bound >= plan.objective - 1e-6


def test_solution_matches_brute_force_on_tiny_problem():
    """Exhaustive check: enumerate all run/seat choices on a tiny instance."""
    p = PlanningProblem(
        skills=[SkillGap(skill_id="a", learners_short=5, demand_weight=3),
                SkillGap(skill_id="b", learners_short=4, demand_weight=2)],
        courses=[
            Course(id="x", fixed_cost=10, cost_per_seat=2, min_batch=1, max_seats=5, coverage={"a": 100}),
            Course(id="y", fixed_cost=5, cost_per_seat=3, min_batch=2, max_seats=4, coverage={"b": 100, "a": 50}),
            Course(id="z", fixed_cost=1, cost_per_seat=1, min_batch=1, max_seats=3, coverage={"b": 50},
                   prerequisites=["x"]),
        ],
        constraints=Constraints(budget=30),
    )
    best = 0.0
    ranges = [range(0, c.max_seats + 1) for c in p.courses]
    for seats in itertools.product(*ranges):
        alloc = dict(zip([c.id for c in p.courses], seats, strict=True))
        ok, cost = True, 0
        for c in p.courses:
            s = alloc[c.id]
            if s:
                if s < c.min_batch or any(alloc[q] == 0 for q in c.prerequisites):
                    ok = False
                cost += c.fixed_cost + c.cost_per_seat * s
        if not ok or cost > p.constraints.budget:
            continue
        value = 0.0
        for sk in p.skills:
            supply = sum(c.coverage.get(sk.skill_id, 0) * alloc[c.id] for c in p.courses) / 100
            value += sk.demand_weight * min(sk.learners_short, supply)
        best = max(best, value)
    plan = solve(p)
    assert plan.status is SolveStatus.OPTIMAL
    assert plan.objective == pytest.approx(best)


def test_prerequisite_and_exclusive_groups_respected(problem):
    plan = solve(problem)
    chosen = plan.seats()
    if "ml201" in chosen:
        assert "py101" in chosen
    assert not ("excel_basic" in chosen and "excel_adv" in chosen)


def test_saturation_never_trains_past_the_gap(problem):
    rich = problem.model_copy(update={"constraints": Constraints(budget=10_000_000, trainer_hours=10_000)})
    plan = solve(rich)
    for s in plan.skills:
        assert s.learners_closed <= s.learners_short
    # With unlimited money every gap that some course covers should close completely.
    assert all(s.closure_pct == pytest.approx(100) for s in plan.skills)


def test_forced_in_and_out(problem):
    forced = problem.model_copy(update={"forced_in": ["excel_adv"], "forced_out": ["py101"]})
    plan = solve(forced)
    chosen = plan.seats()
    assert "excel_adv" in chosen
    assert "py101" not in chosen and "ml201" not in chosen  # ml201 needs py101


def test_equity_floor_is_enforced(problem):
    skills = [s.model_copy(update={"min_closure_pct": 80}) if s.skill_id == "excel" else s
              for s in problem.skills]
    plan = solve(problem.model_copy(update={"skills": skills}))
    excel = next(s for s in plan.skills if s.skill_id == "excel")
    assert excel.closure_pct >= 80


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda p: p.courses[0].prerequisites.append("nope"), "unknown prerequisites"),
        (lambda p: p.courses[0].prerequisites.append("ml201"), "prerequisite cycle"),
        (lambda p: p.forced_in.append("ghost"), "does not exist"),
        (lambda p: setattr(p.constraints, "budget", 1_000), "budget"),
    ],
)
def test_invalid_input_gives_readable_message(problem, mutate, fragment):
    p = problem.model_copy(deep=True)
    if fragment == "budget":
        p.courses[0].mandatory = True
    mutate(p)
    plan = solve(p)
    assert plan.status is SolveStatus.INVALID
    assert any(fragment in m for m in plan.messages)


def test_infeasible_reports_reason(problem):
    skills = [s.model_copy(update={"min_closure_pct": 100}) for s in problem.skills]
    p = problem.model_copy(update={"skills": skills, "constraints": Constraints(budget=60_000)})
    plan = solve(p)
    assert plan.status is SolveStatus.INFEASIBLE
    assert plan.messages


def test_progress_callback_streams_solutions(problem):
    events = []
    solve(problem, on_progress=events.append)
    assert events
    assert events[-1].objective == pytest.approx(solve(problem).objective, rel=0.01)


def test_hint_does_not_change_optimum(problem):
    base = solve(problem)
    from d2s.engine import SolverOptions
    hinted = solve(problem, SolverOptions(hint=base.seats()))
    assert hinted.objective == pytest.approx(base.objective)


def test_shadow_prices_and_exact_marginals(problem):
    tight = problem.model_copy(update={"constraints": Constraints(budget=150_000, trainer_hours=200)})
    plan = solve(tight)
    prices = lp_shadow_prices(tight)
    assert prices.budget_per_rupee > 0  # budget is binding
    assert prices.lp_objective >= plan.objective - 1e-6  # LP relaxation bounds the integer optimum
    report = analyze(tight, plan, budget_delta=50_000)
    budget = next(m for m in report.marginals if m.resource == "budget")
    assert budget.new_objective >= budget.base_objective
    assert budget.per_unit >= 0


def test_pareto_frontier_is_monotone(problem):
    points = frontier(problem, levels=5)
    assert len(points) >= 2
    costs = [p.cost for p in points]
    impacts = [p.impact for p in points]
    assert costs == sorted(costs)
    assert impacts == sorted(impacts)
    assert all(p.cost <= p.budget_cap for p in points)


def test_why_not_explains_excluded_course(problem):
    plan = solve(problem)
    excluded = [c.id for c in problem.courses if c.id not in plan.seats()]
    assert excluded
    ans = why_not(problem, plan, excluded[0])
    if ans.feasible:
        assert ans.objective_delta is not None and ans.objective_delta <= 1e-6
    assert ans.reason


def test_alternatives_are_distinct_ranked_and_true(problem):
    from d2s.engine import alternatives
    from d2s.engine.solver import solve as solve_

    alts = alternatives(problem, k=3)
    assert [a.label for a in alts][:2] == ["Plan A", "Plan B"]
    assert abs(alts[0].plan.objective - solve_(problem).objective) < 1e-6
    sets = [frozenset(a.plan.seats()) for a in alts]
    assert len(set(sets)) == len(sets)  # every plan has a different course set
    objs = [a.plan.objective for a in alts]
    assert objs == sorted(objs, reverse=True)
    for a in alts:  # each one is feasible under the real constraints
        assert a.plan.total_cost <= problem.constraints.budget
        assert a.plan.total_trainer_hours <= problem.constraints.trainer_hours
    # Plan B is truly the best plan that differs from A: forcing out any A course can't beat it
    best_without = max(solve_(problem.model_copy(update={"forced_out": [c]})).objective for c in sets[0])
    assert alts[1].plan.objective >= best_without - 1e-6
    assert alts[1].tradeoff and alts[1].impact_pct_of_best <= 100


def test_alternatives_min_diff_and_exhaustion(problem):
    from d2s.engine import alternatives

    alts = alternatives(problem, k=3, min_diff=2)
    a = set(alts[0].plan.seats())
    for alt in alts[1:]:
        assert len(a ^ set(alt.plan.seats())) >= 2
    tiny = problem.model_copy(update={"courses": problem.courses[:1]})
    assert len(alternatives(tiny, k=3)) == 1  # the empty plan is not offered as an alternative


def test_prerequisite_listed_after_dependant(problem):
    """Course order must not matter (the database returns courses sorted by id)."""
    reordered = problem.model_copy(update={"courses": list(reversed(problem.courses))})
    assert abs(solve(reordered).objective - solve(problem).objective) < 1e-6
