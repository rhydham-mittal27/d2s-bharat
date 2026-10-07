import io

import pytest
from fastapi.testclient import TestClient

from d2s.services import store


def _ready() -> bool:
    try:
        store.artifact_path("hike_model.pkl")
        store.table("rq1/triangulation_rq1_rq2.csv")
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(not _ready(), reason="analysis outputs / artifacts not built")


@pytest.fixture(scope="module")
def client():
    from d2s.api.main import app

    with TestClient(app) as c:  # runs the lifespan (loads model, starts lookup warm-up)
        yield c


def test_health_and_card(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["load_error"] is None
    card = client.get("/api/model/card").json()
    assert card["n_train"] == 139 and "not_for" in card


def test_market_endpoints(client):
    o = client.get("/api/overview").json()
    assert o["postings_clean"] == 14840 and o["data_role_postings"] == 5138
    d = client.get("/api/market/demand", params={"scope": "data", "top": 5}).json()
    assert len(d) == 5 and d[0]["share"] >= d[-1]["share"]
    assert all(r["q_fdr"] < 0.05 for r in client.get("/api/market/pay").json())
    assert len(client.get("/api/market/triangulation").json()) == 5
    assert client.get("/api/market/demand", params={"scope": "nope"}).status_code == 422


def test_learner_predict_and_validation(client):
    body = {"scores": {"coding_skills": 4.4, "maths_stats_skills": 3.6, "ai_and_ml_skills": 4.6,
                       "big_data_skills": 3.8, "dashboard_and_storytelling_skills": 3.4}}
    r = client.post("/api/learner/predict", json=body).json()
    assert 0 < r["probability"] < 1 and r["improvements"]
    bad = {"scores": {**body["scores"], "coding_skills": 9}}
    assert client.post("/api/learner/predict", json=bad).status_code == 422


def test_cohort_sample_and_upload(client):
    s = client.get("/api/cohort/sample").json()
    assert s["n_learners"] == 139 and len(s["areas"]) == 5
    csv = ("id,coding_skills,maths-stats_skills,ai_and_ml_skills,big_data_skills,dashboard_and_storytelling_skills\n"
           "a,4,3,4,3,3\nb,5,5,5,4,5\nc,9,3,3,3,3\n")
    r = client.post("/api/cohort/profile", files={"file": ("c.csv", io.BytesIO(csv.encode()), "text/csv")}).json()
    assert r["n_learners"] == 2 and r["n_rejected"] == 1
    missing = client.post("/api/cohort/profile", files={"file": ("x.csv", io.BytesIO(b"id,foo\n1,2\n"), "text/csv")})
    assert missing.status_code == 422


def test_plan_full_default_and_custom_gaps(client):
    r = client.post("/api/plan/full", json={}).json()
    plan = r["result"]["plan"]
    assert plan["status"] == "optimal" and plan["total_cost"] <= 400_000
    assert r["brief"]["faithful"] and len(r["evidence"]) == len(plan["courses"])
    assert r["timings_ms"]["total"] < 30_000
    custom = client.post("/api/plan/solve", json={"plan": {"budget": 200000, "trainer_hours": 120},
                                                  "gaps": {"big_data_skills": 0}}).json()
    assert custom["plan"]["total_cost"] <= 200000 and custom["gaps"]["big_data_skills"] == 0
    assert client.post("/api/plan/solve", json={"gaps": {"nope": 3}}).status_code == 422
    assert client.post("/api/plan/solve", json={"plan": {"scheme": "x"}}).status_code == 422


def test_plan_robustness_and_frontier(client):
    rb = client.post("/api/plan/robustness", json={}).json()
    assert len(rb["scenarios"]) == 9 and rb["safe_core"]
    fr = client.post("/api/plan/frontier", params={"levels": 4}, json={}).json()
    costs = [p["cost"] for p in fr]
    assert costs == sorted(costs)


def test_plan_alternatives(client):
    alts = client.post("/api/plan/alternatives", json={}).json()
    assert [a["label"] for a in alts] == ["Plan A", "Plan B", "Plan C"]
    assert alts[0]["impact_pct_of_best"] == 100 and len({frozenset(a["course_names"]) for a in alts}) == 3
    assert all(a["plan"]["total_cost"] <= 400_000 for a in alts)
    cheap = client.post("/api/plan/alternatives", params={"min_diff": 2}, json={"plan": {"budget": 200000}}).json()
    a = set(cheap[0]["course_names"])
    assert all(len(a ^ set(x["course_names"])) >= 2 for x in cheap[1:])


def test_insights(client):
    s = client.get("/api/insights/senior").json()
    assert "Not for screening" in s["note"] and s["n"] == 161
    assert client.get("/api/insights/junior").json()["n"] == 139


def test_xai_endpoints(client):
    learner = {"scores": {"coding_skills": 4.4, "maths_stats_skills": 3.6, "ai_and_ml_skills": 4.6,
                          "big_data_skills": 3.8, "dashboard_and_storytelling_skills": 3.4}}
    g = client.get("/api/xai/model/global").json()
    assert g["question"] == "why" and len(g["factors"]) == 5
    w = client.post("/api/xai/learner/why", json=learner).json()
    assert abs(w["details"]["sum_of_contributions"] - w["details"]["logodds_gap"]) < 1e-9
    wi = client.post("/api/xai/learner/what-if", json={**learner, "target_probability": 0.7}).json()
    assert wi["question"] == "what_if" and "curves" in wi["details"]
    c = client.post("/api/xai/plan/constraints", json={}).json()
    assert c["subject"] == "plan" and c["factors"]
    ci = client.post("/api/xai/plan/course/bigdata/what-if", params={"resource": "trainer_hours"}, json={}).json()
    assert ci["subject"] == "course:bigdata"
    infeasible = {"plan": {"budget": 100000, "min_closure_pct": {"coding_skills": 100, "maths_stats_skills": 100}}}
    f = client.post("/api/xai/plan/feasibility", json=infeasible).json()
    assert f["details"]["feasible"] is False and "budget" in f["details"]["conflicting"]
    assert client.post("/api/xai/plan/feasibility", json={}).json()["details"]["feasible"] is True
    assert client.post("/api/xai/plan/constraints", json={"plan": {"min_closure_pct": {"nope": 5}}}).status_code == 422
