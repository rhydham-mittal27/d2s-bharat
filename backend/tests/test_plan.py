import pandas as pd

from d2s.analysis import plan
from d2s.engine import Course, solve


def _jds():
    rows = []
    for i in range(20):
        hike = int(i >= 10)
        base = 5.0 if hike else 3.0
        rows.append({a: base for a in plan.AREAS} | {"salary_hike_high_or_low": hike})
    return pd.DataFrame(rows)


def _tri():
    return pd.DataFrame({
        "area": plan.AREAS,
        "demand_share": [0.4, 0.16, 0.16, 0.1, 0.12], "demand_ci_low": [0.39, 0.15, 0.15, 0.09, 0.11],
        "demand_ci_high": [0.41, 0.17, 0.17, 0.11, 0.13],
        "odds_ratio": [1.5, 1.56, 1.44, 1.33, 1.14], "or_ci_low": [1.3, 1.3, 1.2, 1.1, 0.98],
        "or_ci_high": [1.7, 1.8, 1.7, 1.6, 1.33],
        "rq2_odds_ratio": [1.67, 3.0, 2.0, 1.76, 2.72], "rq2_or_ci_low": [1.1, 2.0, 1.2, 1.2, 1.8],
        "rq2_or_ci_high": [2.8, 4.6, 3.2, 2.7, 4.7],
    })


def test_cohort_gaps_count_learners_below_high_group_median():
    g = plan.cohort_gaps(_jds()).set_index("area")
    assert (g["target_score"] == 5.0).all()
    assert (g["learners_short"] == 10).all()  # the 10 low-hike juniors are below target


def test_evidence_weights_scaled_and_or_below_one_gets_no_market_value():
    for scheme in plan.SCHEMES:
        for bound in ("low", "point", "high"):
            w = plan.evidence_weights(_tri(), scheme, bound)
            assert abs(w["weight"].max() - 100) < 1e-9
            assert (w["weight"] >= 0).all()
    low = plan.evidence_weights(_tri(), "market-led", "low").set_index("area")
    assert low.loc["dashboard_and_storytelling_skills", "market_c"] == 0  # OR CI low 0.98 -> clipped


def test_build_problem_solves_and_respects_budget():
    courses = [Course(id=f"c_{a}", fixed_cost=1000, cost_per_seat=10, min_batch=1, max_seats=20,
                      coverage={a: 100}) for a in plan.AREAS]
    p = plan.build_problem(plan.cohort_gaps(_jds()), plan.evidence_weights(_tri()), courses,
                           budget=3000, trainer_hours=None)
    out = solve(p)
    assert out.is_solution and out.total_cost <= 3000
    assert len(out.courses) == 2  # only two courses' fixed costs fit; highest-value areas chosen
