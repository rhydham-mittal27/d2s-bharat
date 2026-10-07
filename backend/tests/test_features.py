"""Confidence, naive baselines and the 'What happens if...' simulator."""

import pytest
from tests_support import artifacts_ready

from d2s.engine import Constraints, solve
from d2s.engine.baselines import baselines, score

real = pytest.mark.skipif(not artifacts_ready(), reason="analysis outputs / artifacts not built")


# ---- baselines (engine level, fixture problem) -------------------------------------------------------------
def _feasible(problem, seats):
    by_id = {c.id: c for c in problem.courses}
    cons = problem.constraints
    cost = sum(by_id[c].fixed_cost + by_id[c].cost_per_seat * n for c, n in seats.items())
    hours = sum(by_id[c].trainer_hours for c in seats)
    groups = [by_id[c].exclusive_group for c in seats if by_id[c].exclusive_group]
    return (cost <= cons.budget and (cons.trainer_hours is None or hours <= cons.trainer_hours)
            and all(by_id[c].min_batch <= n <= by_id[c].max_seats for c, n in seats.items())
            and all(p in seats for c in seats for p in by_id[c].prerequisites)
            and len(groups) == len(set(groups)))


def test_score_matches_the_optimiser_objective(problem):
    plan = solve(problem)
    assert abs(score(problem, plan.seats())[0] - plan.objective) < 1e-6


@pytest.mark.parametrize("budget,hours", [(60_000, 200), (150_000, 120), (250_000, 200), (400_000, 200),
                                          (400_000, 90), (900_000, 400)])
def test_baselines_are_feasible_and_never_beat_the_optimiser(problem, budget, hours):
    p = problem.model_copy(update={"constraints": Constraints(budget=budget, trainer_hours=hours)})
    opt = solve(p)
    share = {"python": 0.4, "sql": 0.3, "ml": 0.2, "excel": 0.1}
    rows = baselines(p, share)
    assert {b.id for b in rows} == {"biggest_gap", "most_demanded", "cheapest_seat"}
    for b in rows:
        assert _feasible(p, b.seats), (b.id, b.seats)
        assert b.impact <= opt.objective + 1e-6, (b.id, b.impact, opt.objective)


def test_baselines_respect_mandatory_and_excluded_courses(problem):
    p = problem.model_copy(update={"forced_in": ["excel_adv"], "forced_out": ["py101"]})
    for b in baselines(p, {}):
        assert "excel_adv" in b.seats and "py101" not in b.seats and "ml201" not in b.seats  # ml201 needs py101


# ---- service level, real SAS data -----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sas():
    from d2s.api.state import AppState
    from d2s.services import store

    st = AppState()
    st.load()
    return st, store.CATALOGUE


@real
def test_confidence_counts_survival_across_nine_readings(sas):
    from d2s.analysis.plan import load_catalogue
    from d2s.services import planner

    st, cat = sas
    c = planner.confidence(st.sample_gaps, st.tri, cat, planner.PlanRequest())
    base = planner.solve(planner._problem(st.sample_gaps, st.tri, load_catalogue(cat), planner.PlanRequest()),
                         planner.OPTS)
    assert len(c.scenarios) == 9 and {x.course_id for x in c.courses} == set(base.seats())
    for x in c.courses:
        assert x.scenarios == 9 and 0 <= x.chosen_in <= 9
        assert x.level == ("high" if x.chosen_in / 9 >= 0.85 else "medium" if x.chosen_in / 9 >= 0.5 else "low")
    assert c.level in ("high", "medium", "low") and 0 <= c.same_plan_share <= 1


@real
def test_baseline_comparison_and_sweep(sas):
    from d2s.services import planner

    st, cat = sas
    b = planner.baseline_sweep(st.sample_gaps, st.tri, cat, planner.PlanRequest())
    assert b.uplift_vs_average_pct >= b.uplift_vs_best_pct >= 0
    assert all(r["impact_pct_of_optimiser"] <= 100.0 for r in b.baselines)
    assert b.sweep and all(v <= 100.0 for r in b.sweep for v in r["rules"].values())
    assert set(b.worst_case_pct) == {"biggest_gap", "most_demanded", "cheapest_seat"}


@real
def test_what_if_scenarios(sas):
    from d2s.services import planner, whatif

    st, cat = sas
    req = planner.PlanRequest()

    def run(**kw):
        return whatif.simulate(st.sample_gaps, st.tri, cat, req, whatif.Scenario(**kw))

    cut = run(budget_pct=-30)
    assert cut.after.feasible and cut.after.cost <= 280_000 and not cut.today_still_fits
    assert cut.dropped and "re-planning is required" in " ".join(cut.narrative)

    same = run()
    assert same.after.courses == same.today.courses and "does not change" in same.narrative[0]
    assert same.replanning_gain_pct == 0.0

    gone = run(remove_courses=["stats"])
    assert "stats" not in gone.after.courses and "stats" in gone.dropped
    forced = run(require_courses=["bigdata"])
    assert "bigdata" in forced.after.courses

    grow = run(cohort_pct=40)  # weights change -> today's plan is re-valued, not compared raw
    assert grow.impact_change_pct is None and grow.today_impact_after is not None
    assert grow.after.impact >= grow.today_impact_after - 1e-6  # re-planning never does worse

    money = run(budget_pct=50)
    assert any("trainer capacity is the limit" in n for n in money.narrative)

    impossible = run(budget_pct=-95, require_courses=["bootcamp"])
    assert not impossible.after.feasible and impossible.conflicts

    with pytest.raises(ValueError, match="unknown courses"):
        run(remove_courses=["nope"])
    with pytest.raises(ValueError, match="both removed and required"):
        run(remove_courses=["stats"], require_courses=["stats"])
    assert {p["id"] for p in whatif.PRESETS} >= {"budget_cut", "trainer_leaves", "ai_boom"}


@real
def test_endpoints(sas):
    from fastapi.testclient import TestClient

    from d2s.api.main import app

    with TestClient(app) as c:
        full = c.post("/api/plan/full", json={}).json()
        assert full["baselines"]["uplift_vs_average_pct"] >= 0
        assert c.post("/api/plan/confidence", json={}).json()["courses"]
        assert c.post("/api/plan/baselines", json={}).json()["sweep"]
        assert len(c.get("/api/plan/what-if/presets").json()) >= 6
        r = c.post("/api/plan/what-if", json={"scenario": {"trainer_hours_delta": -60}}).json()
        assert r["title"].startswith("What if") and r["after"]["trainer_hours"] <= 140
        assert c.post("/api/plan/what-if", json={"scenario": {"budget_pct": -200}}).status_code == 422
