"""The three LangGraph agents with a scripted stand-in for the local model (deterministic, no Ollama)."""

import io
import json
import re

import pytest
from langgraph.types import Command
from tests_support import artifacts_ready

from d2s.agents.copilot import COPILOT
from d2s.agents.extract import parse
from d2s.agents.intake import INTAKE, detect_scale, new_thread, to_1_5
from d2s.agents.stress import STRESS

real = pytest.mark.skipif(not artifacts_ready(), reason="analysis outputs / artifacts not built")

NAMES = {"py_sql": "Python & SQL for Analytics", "sas_prog": "SAS Programming on SAS Viya for Learners",
         "stats": "Applied Statistics & Probability", "ml": "Machine Learning Foundations",
         "dl_nlp": "Deep Learning & NLP", "bigdata": "Big-Data Stack: Hadoop, Spark, Hive (taught as a bundle)",
         "story_tableau": "Data Storytelling with Tableau", "story_powerbi": "Dashboards & Storytelling with Power BI",
         "bootcamp": "Integrated Analytics Bootcamp (capstone)"}


def scripted(answer):
    """A chooser that records what it was asked and returns a fixed answer (or None = model off)."""
    calls = []

    def chooser(system, user, schema):
        calls.append((system, user, schema))
        if answer is None:
            return None
        return answer(schema) if callable(answer) else schema(**answer)

    chooser.calls = calls
    return chooser


# ---- extraction ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("q,intents,scenario", [
    ("What happens if our budget is cut by 30%?", ["what_if"], {"budget_pct": -30}),
    ("A trainer is leaving next term. What changes?", ["what_if"], {"trainer_hours_delta": -60}),
    ("We lose 60 trainer hours, impact?", ["what_if"], {"trainer_hours_delta": -60}),
    ("How sure are you about this plan?", ["plan_confidence"], {}),
    ("Why didn't you pick the deep learning course?", ["why_not_course"], {}),
    ("Is this better than just funding the biggest gap?", ["compare_baselines"], {}),
    ("Give me a couple of other options", ["alternative_plans"], {}),
    ("Budget drops 20% and AI demand jumps, how confident are you?", ["what_if", "plan_confidence"],
     {"budget_pct": -20, "demand_pct": {"ai_and_ml_skills": 50}}),
    ("AI demand falls 30% but big data demand rises 40%", ["what_if"],
     {"demand_pct": {"ai_and_ml_skills": -30, "big_data_skills": 40}}),
    ("What if the statistics course is unavailable?", ["what_if"], {"remove_courses": ["stats"]}),
    ("Management insists we run the bootcamp", ["what_if"], {"require_courses": ["bootcamp"]}),
    ("Course prices go up 15 percent", ["what_if"], {"cost_pct": 15}),
    ("Next batch is 40% bigger", ["what_if"], {"cohort_pct": 40}),
    ("What if we only have 2,50,000?", ["what_if"], {"budget_pct": -38}),
    ("Budget doubles", ["what_if"], {"budget_pct": 100}),
    ("what is the weather", [], {}),
])
def test_extract(q, intents, scenario):
    p = parse(q, NAMES, 400_000, 200)
    assert p.intents == intents and p.scenario == scenario


def test_extract_absolute_plan_and_assumptions():
    p = parse("What's the best plan with 3 lakh and 150 hours?", NAMES, 400_000, 200)
    assert p.intents == ["optimise_plan"] and p.budget == 300_000 and p.trainer_hours == 150 and not p.scenario
    q = parse("A trainer is leaving", NAMES, 400_000, 200)
    assert q.assumptions and "-60" in q.assumptions[0]  # defaults are always reported


# ---- Copilot ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ctx():
    from d2s.agents.tools import AgentContext
    from d2s.api.state import AppState
    from d2s.services import planner, store

    st = AppState()
    st.load()
    return AgentContext(st.sample_gaps, st.tri, store.CATALOGUE, planner.PlanRequest())


def _ask(ctx, q, chooser):
    return COPILOT.invoke({"question": q, "trace": []}, {"configurable": {"ctx": ctx, "chooser": chooser}})


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


@real
def test_copilot_fast_path_never_calls_the_model(ctx):
    model = scripted({"tools": ["none"]})
    out = _ask(ctx, "Budget drops 20% and AI demand jumps. How confident are you?", model)
    a = out["answer"]
    assert a["path"] == "fast" and not model.calls
    assert [s["tool"] for s in a["sections"]] == ["what_if", "plan_confidence"]
    # every number in the answer comes from the tools' own lines
    from_tools = set().union(*(_numbers(" ".join(s["lines"])) for s in a["sections"]), _numbers(" ".join(
        a["assumptions"])), _numbers(" ".join(s["title"] for s in a["sections"])))
    assert _numbers(a["text"]) <= from_tools
    assert [t["node"] for t in out["trace"]] == ["route", "act", "act", "answer"]


@real
def test_copilot_model_path_trusts_only_the_first_choice(ctx):
    model = scripted({"tools": ["optimise_plan", "plan_confidence", "compare_baselines"]})
    out = _ask(ctx, "Can we do better with the money we have?", model)
    assert out["answer"]["path"] == "llm" and len(model.calls) == 1
    assert [s["tool"] for s in out["answer"]["sections"]] == ["optimise_plan"]  # extras unsupported by the question


@real
def test_copilot_without_model_or_off_topic_gives_help(ctx):
    for chooser in (scripted(None), scripted({"tools": ["none"]})):
        a = _ask(ctx, "Tell me a joke", chooser)["answer"]
        assert a["path"] == "none" and not a["sections"] and "I can answer questions" in a["text"]


@real
def test_copilot_asks_for_missing_details(ctx):
    a = _ask(ctx, "Why was it not chosen?", scripted({"tools": ["why_not_course"]}))["answer"]
    assert not a["sections"] and "Which course" in a["text"]


# ---- Stress-test -----------------------------------------------------------------------------------------
@real
@pytest.mark.parametrize("model", ["off", "picks_last"])
def test_stress_test_report(ctx, model):
    def last_id(schema):  # pick the last candidate, which the deterministic fallback would rank lowest
        ids = schema.model_fields["ids"].annotation.__args__[0].__args__
        return schema(ids=[ids[-1]])

    chooser = scripted(None if model == "off" else last_id)
    out = STRESS.invoke({"trace": []}, {"configurable": {"ctx": ctx, "chooser": chooser}})
    r = out["report"]
    assert len(r["probes"]) >= 15 and r["top_risks"] and r["mitigations"]
    assert -80 <= r["breakpoints"]["budget_pct"] <= 0
    assert any("single point of failure" in m for m in r["mitigations"])
    if model == "picks_last":
        assert len(out["follow_ups"]) == 1 and out["chosen"] == [out["candidates"][-1]["id"]]
    else:
        assert 1 <= len(out["follow_ups"]) <= 3
    assert [t["node"] for t in out["trace"]] == ["baseline", "probes", "explore", "follow_ups", "report"]


# ---- Data-Intake -----------------------------------------------------------------------------------------
MESSY = [
    {"Roll No": "R1", "Python Programming (out of 10)": "8", "Stats & Probability": "Good", "AI/ML rating": "4",
     "Hadoop-Spark": "2", "Tableau dashboards": "excellent", "Attendance %": "92"},
    {"Roll No": "R2", "Python Programming (out of 10)": "6", "Stats & Probability": "Average", "AI/ML rating": "3",
     "Hadoop-Spark": "1", "Tableau dashboards": "good", "Attendance %": "85"},
]


def _intake(rows, chooser, profile=lambda recs: {"records": recs}):
    tid = new_thread()
    cfg = {"configurable": {"thread_id": tid, "chooser": chooser, "embedder": None, "profile_fn": profile}}
    out = INTAKE.invoke({"filename": "f.csv", "columns": list(rows[0]), "rows": rows}, cfg)
    return out["__interrupt__"][0].value, cfg


def test_scale_detection_and_rescaling():
    assert detect_scale(["8", "6"], "Python (out of 10)") == "0-10" and to_1_5("8", "0-10") == 4.2
    assert detect_scale(["80", "60"], "Coding %") == "0-100" and to_1_5("50", "0-100") == 3.0
    assert detect_scale(["good", "poor", "excellent"]) == "words" and to_1_5("excellent", "words") == 5
    assert detect_scale(["1", "5", "3"]) == "1-5" and detect_scale(["abc", "x"]) == "unknown"


def test_intake_maps_messy_columns_and_waits_for_approval():
    model = scripted({})  # should not be needed: names are enough
    proposal, cfg = _intake(MESSY, model)
    got = {t["target"]: (t["source"], t["scale"]) for t in proposal["targets"]}
    assert got["coding_skills"] == ("Python Programming (out of 10)", "0-10")
    assert got["maths_stats_skills"] == ("Stats & Probability", "words")
    assert got["id"][0] == "Roll No" and "Attendance %" not in {s for s, _ in got.values()}
    assert not model.calls
    res = INTAKE.invoke(Command(resume={"approve": True, "scales": {"big_data_skills": "0-5"}}), cfg)["result"]
    first = res["profile"]["records"][0]
    assert first == {"id": "R1", "coding_skills": 4.2, "maths_stats_skills": 4, "ai_and_ml_skills": 4.0,
                     "big_data_skills": 2.6, "dashboard_and_storytelling_skills": 5}  # user's edited scale applied


def test_intake_uses_the_model_only_for_leftovers_and_only_real_columns():
    rows = [{"Student": "a", "Q1 score": "3", "Q2 score": "4", "Q3 score": "2", "Q4 score": "5", "Q5 score": "1"}]

    def fill(schema):  # model maps every unknown area to a column it was offered
        opts = {f: schema.model_fields[f].annotation.__args__ for f in schema.model_fields}
        return schema(**{f: o[i] for i, (f, o) in enumerate(opts.items())})

    model = scripted(fill)
    proposal, _ = _intake(rows, model)
    assert len(model.calls) == 1
    srcs = [t["source"] for t in proposal["targets"] if t["target"] != "id"]
    assert all(s in rows[0] for s in srcs) and len(set(srcs)) == 5
    assert all(t["method"] == "model (please check)" for t in proposal["targets"] if t["target"] != "id")


def test_intake_cancel_and_incomplete_mapping():
    _, cfg = _intake(MESSY, scripted(None))
    assert INTAKE.invoke(Command(resume={"approve": False}), cfg)["result"] == {"cancelled": True}
    _, cfg = _intake(MESSY, scripted(None))
    with pytest.raises(ValueError, match="every skill area needs a column"):
        INTAKE.invoke(Command(resume={"approve": True, "mapping": {"coding_skills": None}}), cfg)


# ---- API -------------------------------------------------------------------------------------------------
def _events(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


@real
def test_agent_endpoints():
    from fastapi.testclient import TestClient

    from d2s.api.main import app

    with TestClient(app) as c:
        st = c.get("/api/agents/status").json()
        assert st["agents"] == ["copilot", "stress_test", "intake"] and st["model"]["available"] is False
        ev = _events(c.post("/api/agents/copilot", json={"question": "What if the budget is cut by 30%?"}).text)
        assert [e for e, _ in ev][-1] == "result" and {e for e, _ in ev[:-1]} == {"step"}
        assert ev[-1][1]["path"] == "fast" and ev[-1][1]["sections"][0]["tool"] == "what_if"
        ev = _events(c.post("/api/agents/stress-test", json={}).text)
        assert ev[-1][0] == "result" and ev[-1][1]["mitigations"]

        csv = "Roll No,Coding /5,Maths,AI,Big data,Dashboards\nR1,4,3,4,2,3\nR2,5,5,5,4,5\n"
        r = c.post("/api/agents/intake", files={"file": ("m.csv", io.BytesIO(csv.encode()), "text/csv")}).json()
        tid = r["thread_id"]
        assert {t["target"]: t["source"] for t in r["proposal"]["targets"]}["big_data_skills"] == "Big data"
        assert c.post(f"/api/agents/intake/{tid}", json={"scales": {"coding_skills": "bogus"}}).status_code == 422
        done = c.post(f"/api/agents/intake/{tid}", json={"approve": True}).json()
        assert done["profile"]["n_learners"] == 2
        assert c.post(f"/api/agents/intake/{tid}", json={"approve": True}).status_code == 404  # used up


@real
def test_ties_never_show_up_as_changes(ctx):
    """Equally good plans returned by a multi-threaded solver must not be reported as a change."""
    from d2s.services import whatif

    for sc in ({"budget_pct": 50}, {"demand_pct": {"ai_and_ml_skills": 50}}, {}):
        r = whatif.simulate(ctx.gaps, ctx.tri, ctx.courses, ctx.request, whatif.Scenario(**sc))
        if r.today_still_fits and (r.replanning_gain_pct or 0) <= 0.05:
            assert not r.added and not r.dropped and not r.seat_changes, sc
    runs = [STRESS.invoke({"trace": []}, {"configurable": {"ctx": ctx, "chooser": scripted(None)}})["report"]
            for _ in range(2)]
    assert runs[0]["summary"] == runs[1]["summary"] and runs[0]["breakpoints"] == runs[1]["breakpoints"]


def test_token_meter_totals():
    from d2s.agents.llm import TokenMeter

    m = TokenMeter()
    assert m.summary() == {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": []}
    m.add("Pick", 163, 30, 1600)
    m.add("Choice", 149, 33, 900)
    s = m.summary()
    assert (s["model_calls"], s["input_tokens"], s["output_tokens"], s["total_tokens"]) == (2, 312, 63, 375)
    assert s["calls"][0] == {"purpose": "Pick", "input_tokens": 163, "output_tokens": 30, "total_tokens": 193,
                             "ms": 1600}


@real
def test_results_carry_token_counts():
    from fastapi.testclient import TestClient

    from d2s.api.main import app

    with TestClient(app) as c:  # model off in tests: the fast path must report zero tokens
        ev = _events(c.post("/api/agents/copilot", json={"question": "What if the budget is cut by 30%?"}).text)
        assert ev[-1][1]["tokens"]["model_calls"] == 0 and ev[-1][1]["tokens"]["total_tokens"] == 0
        assert _events(c.post("/api/agents/stress-test", json={}).text)[-1][1]["tokens"]["model_calls"] == 0
