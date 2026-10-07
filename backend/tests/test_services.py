import numpy as np
import pandas as pd
import pytest

from d2s.models.hike import SKILLS, HikeModel, SkillScores
from d2s.services import brief, cohort, store


def _jds(n=120, seed=0) -> pd.DataFrame:
    """Synthetic JDS-like data: hike driven by storytelling and maths."""
    rng = np.random.default_rng(seed)
    X = np.clip(rng.normal(4.0, 0.7, size=(n, 5)), 1, 5).round(1)
    logit = 1.8 * (X[:, 1] - 4) + 2.0 * (X[:, 4] - 4) + rng.normal(0, 0.6, n)
    df = pd.DataFrame(X, columns=SKILLS)
    df["salary_hike_high_or_low"] = (logit > 0).astype(int)
    return df


@pytest.fixture(scope="module")
def model():
    return HikeModel.train(_jds(), n_boot=60)


def test_hike_model_learns_true_drivers(model):
    coef = dict(zip(SKILLS, model.coef, strict=True))
    assert coef["dashboard_and_storytelling_skills"] > 0.5 and coef["maths_stats_skills"] > 0.5
    assert abs(coef["coding_skills"]) < coef["maths_stats_skills"]


def test_explanation_contributions_sum_to_logodds(model):
    s = SkillScores(coding_skills=4, maths_stats_skills=3.2, ai_and_ml_skills=4.5, big_data_skills=3.9,
                    dashboard_and_storytelling_skills=3.0)
    pred = model.explain(s)
    logit = np.log(pred.probability / (1 - pred.probability))
    assert abs(sum(c.contribution_logodds for c in pred.contributions) + model.intercept - logit) < 1e-6
    assert pred.probability_low <= pred.probability <= pred.probability_high


def test_counterfactual_unreachable_case_is_reported(model):
    s = SkillScores(coding_skills=4, maths_stats_skills=3.0, ai_and_ml_skills=4, big_data_skills=4,
                    dashboard_and_storytelling_skills=3.0)
    pred = model.explain(s, target_probability=0.7)
    assert pred.smallest_change_to_target is None
    assert any("no single-skill change" in w for w in pred.warnings)


def test_counterfactual_reaches_target_with_smallest_single_change(model):
    s = SkillScores(coding_skills=4, maths_stats_skills=4.0, ai_and_ml_skills=4, big_data_skills=4,
                    dashboard_and_storytelling_skills=3.6)
    pred = model.explain(s, target_probability=0.7)
    assert pred.probability < 0.7
    ch = pred.smallest_change_to_target
    assert ch is not None and ch.probability_after >= 0.7
    # going 0.1 less on that skill must not reach the target (it is the smallest step)
    x = s.vector()
    i = SKILLS.index(ch.skill)
    x[i] = ch.target - 0.1
    assert model.predict_proba(x)[0] < 0.7
    assert all(a.gain >= b.gain for a, b in zip(pred.improvements, pred.improvements[1:]))


def test_model_roundtrip(tmp_path, model):
    p = tmp_path / "m.pkl"
    model.save(p)
    m2 = HikeModel.load(p)
    x = np.array([4, 4, 4, 4, 4.0])
    assert m2.predict_proba(x)[0] == pytest.approx(model.predict_proba(x)[0])


def test_cohort_validation_and_gaps(model):
    df = _jds(30, seed=3).drop(columns="salary_hike_high_or_low").astype(object)  # as read from a CSV upload
    df.loc[2, "coding_skills"] = 7  # out of range
    df.loc[5, "big_data_skills"] = "n/a"  # non-numeric
    prof = cohort.profile(df.rename(columns={"maths_stats_skills": "maths-stats_skills"}), model)
    assert prof.n_learners == 28 and prof.n_rejected == 2
    assert {i.row for i in prof.issues} == {4, 7}
    g = cohort.gaps_frame(prof)
    assert set(g["area"]) == set(SKILLS) and (g["learners_short"] <= 28).all()
    with pytest.raises(ValueError, match="missing required columns"):
        cohort.profile(df.drop(columns="coding_skills"), model)


def test_brief_number_check_flags_invented_numbers():
    n, bad = brief.check_numbers("Run 5 courses for 263 seats, 42% better.", [5, 263])
    assert n == 3 and bad == ["42"]
    n, bad = brief.check_numbers("Cost ₹3.53 lakh and 1,200 seats.", [3.5291, 1200])
    assert bad == []


# ---- end-to-end on the real SAS-derived artifacts (skipped if not built) -------------------------
def _artifacts_ready() -> bool:
    try:
        store.artifact_path("hike_model.pkl")
        store.table("rq1/triangulation_rq1_rq2.csv")
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _artifacts_ready(), reason="artifacts not built")
def test_end_to_end_cohort_plan_evidence_brief():
    from d2s.analysis.traits import load_jds
    from d2s.services import evidence, planner

    m = HikeModel.load(store.artifact_path("hike_model.pkl"))
    jds = load_jds(store.SAS_DIR / "JDS Skill Traits.xlsx").frame
    prof = cohort.profile(jds.drop(columns="salary_hike_high_or_low"), m,
                          store.artifact_table("area_roles.csv"), pd.read_csv(store.CATALOGUE))
    assert prof.n_learners == 139 and all(a.linked_courses for a in prof.areas)
    tri = store.table("rq1/triangulation_rq1_rq2.csv")
    res = planner.optimise(cohort.gaps_frame(prof), tri, store.CATALOGUE, planner.PlanRequest())
    assert res.plan.status.value == "optimal" and res.plan.total_cost <= 400_000
    trails = evidence.build(res, cohort.gaps_frame(prof), tri, store.CATALOGUE)
    assert len(trails) == len(res.plan.courses)
    assert all(t.impact_lost_if_removed_pct > 0 for t in trails)
    b = brief.write(res, trails, prof.n_learners)
    assert b.faithful, b.unverified_numbers
