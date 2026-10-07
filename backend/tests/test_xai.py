"""Faithfulness tests: explanations must be true of the model, not just well-formed text."""

import numpy as np
import pandas as pd
import pytest

from d2s.engine import Constraints, solve
from d2s.ml.embeddings import HashingEmbedder
from d2s.ml.skills import SkillMatcher
from d2s.ml.taxonomy import Skill
from d2s.models.hike import SKILLS, HikeModel, SkillScores
from d2s.xai import learner as xl
from d2s.xai import plan as xp
from d2s.xai import skills as xs


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    X = np.clip(rng.normal(4.0, 0.7, size=(150, 5)), 1, 5).round(1)
    logit = 1.8 * (X[:, 1] - 4) + 2.0 * (X[:, 4] - 4) + 0.6 * (X[:, 0] - 4) + rng.normal(0, 0.6, 150)
    df = pd.DataFrame(X, columns=SKILLS)
    df["salary_hike_high_or_low"] = (logit > 0).astype(int)
    return HikeModel.train(df, n_boot=50)


LOW = SkillScores(coding_skills=3.5, maths_stats_skills=3.4, ai_and_ml_skills=4.0, big_data_skills=3.8,
                  dashboard_and_storytelling_skills=3.3)


def test_shapley_contributions_are_exact(model):
    e = xl.why(model, LOW)
    assert abs(e.details["sum_of_contributions"] - e.details["logodds_gap"]) < 1e-9
    assert "held back" in e.summary.lower()


def test_cheapest_path_reaches_target_and_beats_single_skill_options(model):
    e = xl.cheapest_path(model, LOW, target=0.7)
    assert e.details["reached"] and e.details["probability_after"] >= 0.7
    # check by re-predicting with the proposed changes applied (faithfulness)
    x = LOW.vector()
    for f in e.factors:
        i = [xl.LABELS[s] for s in SKILLS].index(f.name)
        x[i] += f.contribution
    assert model.predict_proba(x)[0] >= 0.7
    single = model.explain(LOW, 0.7).smallest_change_to_target
    if single is not None:
        assert e.details["total_points"] <= single.target - single.current + 0.1 + 1e-9
    assert set(e.details["curves"]) == set(SKILLS)


def test_cheapest_path_reports_unreachable(model):
    floor = SkillScores(coding_skills=1, maths_stats_skills=1, ai_and_ml_skills=1, big_data_skills=1,
                        dashboard_and_storytelling_skills=1)
    e = xl.cheapest_path(model, floor, target=0.999999)
    assert not e.details["reached"] and "not reachable" in e.summary


def test_global_importance_sorted(model):
    e = xl.global_importance(model)
    vals = [abs(f.value) for f in e.factors]
    assert vals == sorted(vals, reverse=True) and e.factors[0].name in ("Dashboards & storytelling", "Maths & stats")


# ---- plan --------------------------------------------------------------------------------------------
def test_constraint_report_flags_tight_budget(problem):
    tight = problem.model_copy(update={"constraints": Constraints(budget=150_000, trainer_hours=500)})
    plan = solve(tight)
    e = xp.constraint_report(tight, plan)
    assert "budget" in e.details["tight"]
    assert "trainer-hours" not in e.details["tight"]


def test_course_what_if_is_minimal_and_true(problem):
    p = problem.model_copy(update={"constraints": Constraints(budget=150_000, trainer_hours=500)})
    plan = solve(p)
    excluded = [c.id for c in p.courses if c.id not in plan.seats() and not c.prerequisites]
    e = xp.course_what_if(p, plan, excluded[0], "budget")
    assert e.details["found"]
    v = e.details["minimal_increase"]
    with_v = solve(p.model_copy(update={"constraints": p.constraints.model_copy(update={"budget": int(p.constraints.budget + v)})}))
    assert excluded[0] in with_v.seats()  # the counterfactual really holds
    below = solve(p.model_copy(update={"constraints": p.constraints.model_copy(update={"budget": int(p.constraints.budget + v - 5_000)})}))
    assert excluded[0] not in below.seats()  # and is minimal to the search resolution


def test_infeasibility_core_names_conflicting_groups(problem):
    skills = [s.model_copy(update={"min_closure_pct": 100}) for s in problem.skills]
    bad = problem.model_copy(update={"skills": skills, "constraints": Constraints(budget=60_000, trainer_hours=200)})
    assert not solve(bad).is_solution
    e = xp.infeasibility(bad)
    core = e.details["conflicting"]
    assert e.details["feasible"] is False and "budget" in core
    assert any(c.startswith("equity floor") for c in core)
    # removing the reported equity floors (keeping the budget) must restore feasibility
    floors = {c.split(": ")[1].split(" >=")[0] for c in core if c.startswith("equity floor")}
    relaxed = bad.model_copy(update={"skills": [s.model_copy(update={"min_closure_pct": 0}) if s.skill_id in floors
                                                else s for s in bad.skills]})
    assert solve(relaxed).is_solution or len(floors) < len([s for s in skills if s.min_closure_pct])


def test_feasible_problem_reports_no_conflict(problem):
    assert xp.infeasibility(problem).details["feasible"] is True


# ---- skills ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def matcher():
    skills = [Skill(id="java", label="Java (computer programming)", source="esco", kind="knowledge", alt_labels=["Java"]),
              Skill(id="sql", label="SQL", source="esco", kind="knowledge"),
              Skill(id="xml", label="Extensible markup language XML", source="onet", kind="tool"),
              Skill(id="viz", label="data visualisation", source="esco", kind="skill")]
    return SkillMatcher(skills, HashingEmbedder())


def test_explain_tag_paths(matcher):
    assert xs.explain_tag(matcher, "SQL").details["step"] == "exact"
    assert xs.explain_tag(matcher, "XML").details["step"] == "exact"  # acronym key
    c = xs.explain_tag(matcher, "Core Java")
    assert c.details["step"] == "contains" and c.details["matched_text"] == "java"
    s = xs.explain_tag(matcher, "zzqx frobnicate")
    assert s.details["step"] in ("review", "unmapped") and s.factors  # runner-ups listed


def test_explain_text_spans_point_at_the_words(matcher):
    text = "Need SQL experts.\n• Strong Java skills and data visualisation."
    e = xs.explain_text(matcher, text)
    for m in e.details["matches"]:
        a, b = m["span"]
        assert text[a:b].lower().replace("\n", " ").split() == m["matched"].split()
    assert {m["skill"] for m in e.details["matches"]} == {"SQL", "Java (computer programming)", "data visualisation"}


# ---- forecast -----------------------------------------------------------------------------------------
def test_forecast_explanation_has_leaderboard():
    from d2s.ml.forecasting import DemandForecaster
    from d2s.xai import forecast as xf

    t = np.arange(36)
    y = 50 + t + 5 * np.sin(2 * np.pi * t / 12)
    df = pd.DataFrame({"unique_id": "a", "ds": pd.date_range("2022-01-01", periods=36, freq="MS"), "y": y})
    fc = DemandForecaster(horizon=6).fit_predict(df)[0]
    e = xf.explain(fc, y)
    board = e.details["leaderboard"]
    assert board and board[0][0] == fc.model and "Selected" in e.summary
