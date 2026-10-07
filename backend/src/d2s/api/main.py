"""D2S Bharat API.  Run:  uv run uvicorn d2s.api.main:app --reload --port 8000

Screens -> endpoints
  Command Center   GET  /api/overview, /api/market/{demand,pay,bundles,triangulation,titles}
                   GET  /api/market/skills/search?q=
  Gap Map          POST /api/learner/predict          (one learner: probability + explanation)
                   POST /api/cohort/profile           (CSV upload, JDS format)
                   GET  /api/cohort/sample            (the SAS JDS cohort)
  What-If          GET  /api/plan/catalogue
                   POST /api/plan/confidence | /plan/baselines | /plan/what-if (+ GET /plan/what-if/presets)
                   POST /api/plan/solve | /plan/alternatives | /plan/robustness | /plan/frontier
  Evidence Trail   POST /api/plan/evidence
  Decision Brief   POST /api/plan/brief               (template brief + number check)
                   POST /api/plan/full                (plan + evidence + brief in one call)
  Insights         GET  /api/insights/junior, /api/insights/senior
  Meta             GET  /api/health, /api/model/card
  Auth (JWT)       POST /api/auth/register | /api/auth/login | /api/auth/token; GET /api/auth/me
                   every other route except /api/health needs `Authorization: Bearer <token>`
  Agents           GET  /api/agents/status; POST /api/agents/copilot | /stress-test (event streams);
                   POST /api/agents/intake (+ /intake/{thread_id} to approve); GET /api/agents/runs
  Database         GET  /api/db/status; /api/courses (CRUD); /api/cohorts; /api/plan/runs
                   (Supabase Postgres, optional: set D2S_DATABASE_URL; vectors in ChromaDB)
"""

import io
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from d2s import __version__
from d2s.agents import llm as agent_llm
from d2s.analysis import plan as planlib
from d2s.api.auth import OwnerId, auth_guard
from d2s.api.auth import router as auth_router
from d2s.api.common import PlanBody, plan_gaps
from d2s.api.db import DB, catalogue_source
from d2s.api.routes_agents import router as agents_router
from d2s.api.routes_db import router as db_router
from d2s.api.state import STATE
from d2s.db import (
    CohortRepository,
    NotFound,
    PlanRunRepository,
    session_scope,
)
from d2s.models.hike import Prediction, SkillScores
from d2s.services import brief as brieflib
from d2s.services import cohort, evidence, insights, market, planner, store, whatif

MAX_UPLOAD_BYTES = 2_000_000
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DB.configure_from_settings()
    STATE.load()
    STATE.warm_lookup()
    threading.Thread(target=agent_llm.warm_up, name="warm-agent-model", daemon=True).start()
    yield
    DB.dispose()


app = FastAPI(title="D2S Bharat API", version=__version__, lifespan=lifespan, dependencies=[Depends(auth_guard)],
              description="Market evidence, learner/cohort gap analysis, CP-SAT training plans, "
                          "evidence trail and decision brief, built on the SAS hackathon data.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Plan-Run-Id"])
app.include_router(auth_router)
app.include_router(db_router)
app.include_router(agents_router)


@app.exception_handler(NotFound)
async def _not_found(_, exc: NotFound):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(store.MissingArtifact)
async def _missing(_, exc: store.MissingArtifact):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def _bad(_, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def _catalogue_frame(source) -> pd.DataFrame:
    return STATE.catalogue if not isinstance(source, list) else planlib.catalogue_frame(source)


def _ready():
    if STATE.load_error:
        raise HTTPException(503, f"server not ready: {STATE.load_error}")
    if STATE.model is None:
        raise HTTPException(503, "server is still loading")


# ---- meta ------------------------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok" if not STATE.load_error and STATE.model else "degraded", "version": __version__,
            "load_error": STATE.load_error, "skill_lookup": "ready" if STATE.lookup else
            ("error: " + STATE.lookup_error if STATE.lookup_error else "warming up"),
            "database": "connected" if DB.enabled else ("error: " + DB.error if DB.error else "not configured"),
            "vector_store": (f"chromadb ({STATE.vectors.count()} skills)" if STATE.vectors is not None else
                             ("error: " + STATE.vector_error if STATE.vector_error else "warming up"))}


@app.get("/api/model/card")
def model_card():
    return json.loads(store.artifact_path("hike_model.json").read_text())


# ---- Command Center ------------------------------------------------------------------------------------
@app.get("/api/overview")
def overview():
    return market.overview()


@app.get("/api/market/demand")
def demand(scope: str = Query("data", pattern="^(data|all)$"), top: int = Query(25, ge=1, le=40)):
    return market.demand(scope, top)


@app.get("/api/market/pay")
def pay(scope: str = Query("data", pattern="^(data|all)$"), significant_only: bool = True):
    return market.pay(scope, significant_only)


@app.get("/api/market/bundles")
def bundles(min_support: float = Query(0.01, ge=0, le=1), min_confidence: float = Query(0.3, ge=0, le=1),
            top: int = Query(20, ge=1, le=100)):
    return market.bundles(min_support, min_confidence, top)


@app.get("/api/market/triangulation")
def triangulation():
    return market.triangulation()


@app.get("/api/market/titles")
def titles():
    return market.ds_titles()


@app.get("/api/market/skills/search", response_model=list[market.SkillHit])
def skill_search(q: str = Query(..., min_length=1, max_length=100), k: int = Query(5, ge=1, le=20)):
    if STATE.lookup is None:
        raise HTTPException(503, STATE.lookup_error or "skill lookup is warming up; retry in a few seconds")
    if STATE.vectors is not None:
        return _vector_search(q.strip(), k)
    return STATE.lookup.search(q, k)  # in-memory fallback when ChromaDB is unavailable


def _vector_search(q: str, k: int) -> list[market.SkillHit]:
    """ChromaDB nearest neighbours; an exact name match always ranks first."""
    vs = STATE.vectors
    matches = vs.query(STATE.lookup.embedder.embed_queries([q])[0], k)
    exact = vs.by_name(q)
    if exact is not None:
        matches = [(exact, 1.0)] + [(m, s) for m, s in matches if m["skill"] != exact["skill"]][: k - 1]
    return [_hit(m, s) for m, s in matches]


def _hit(m: dict, sim: float) -> market.SkillHit:
    has_pay = m.get("pay_odds_ratio") is not None
    return market.SkillHit(
        skill=m["skill"], similarity=float(min(sim, 1.0)), posts_all=int(m["posts_all"]),
        share_all=float(m["share_all"]), posts_data_roles=int(m["posts_data"]),
        share_data_roles=float(m["share_data"]), median_salary_mid_lakh=m.get("median_salary_mid"),
        top_role_family=m.get("top_family"), pay_odds_ratio=m.get("pay_odds_ratio"),
        pay_ci=(m["pay_ci_low"], m["pay_ci_high"]) if has_pay else None, pay_q_fdr=m.get("pay_q_fdr"))


# ---- Gap Map -------------------------------------------------------------------------------------------
class PredictBody(BaseModel):
    scores: SkillScores
    target_probability: float = Field(0.7, gt=0, lt=1)


@app.post("/api/learner/predict", response_model=Prediction)
def predict(body: PredictBody):
    _ready()
    return STATE.model.explain(body.scores, body.target_probability)


@app.get("/api/cohort/sample", response_model=cohort.CohortProfile)
def cohort_sample():
    _ready()
    return cohort.profile(STATE.sample_jds.drop(columns="salary_hike_high_or_low"), STATE.model,
                          STATE.area_roles, _catalogue_frame(catalogue_source()))


@app.post("/api/cohort/profile", response_model=cohort.CohortProfile)
async def cohort_profile(owner: OwnerId, file: UploadFile = File(...)):
    _ready()
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // 1_000_000} MB")
    name = (file.filename or "").lower()
    try:
        df = pd.read_excel(io.BytesIO(raw)) if name.endswith((".xlsx", ".xls")) else \
            pd.read_csv(io.BytesIO(raw), dtype=str)
    except Exception as exc:
        raise HTTPException(422, f"could not read the file: {exc}") from exc
    prof = cohort.profile(df, STATE.model, STATE.area_roles, _catalogue_frame(catalogue_source()))
    if DB.enabled and prof.n_learners:
        g = cohort.gaps_frame(prof)
        gaps = {a: int(n) for a, n in zip(g["area"], g["learners_short"], strict=True)}
        with session_scope(DB.factory) as s:
            row = CohortRepository(s, owner).add(file.filename or "upload", prof.model_dump(mode="json"), gaps)
            prof.cohort_id = row.id
    return prof


# ---- What-If / Evidence / Brief ------------------------------------------------------------------------
_gaps = plan_gaps  # local alias used throughout this module


@app.get("/api/plan/catalogue")
def catalogue():
    _ready()
    return _catalogue_frame(catalogue_source()).fillna("").to_dict("records")


@app.post("/api/plan/solve", response_model=planner.PlanResult)
def plan_solve(body: PlanBody, owner: OwnerId):
    _ready()
    return planner.optimise(_gaps(body, owner), STATE.tri, catalogue_source(), body.plan)


@app.post("/api/plan/robustness")
def plan_robustness(body: PlanBody, owner: OwnerId):
    _ready()
    return planner.robustness(_gaps(body, owner), STATE.tri, catalogue_source(), body.plan)


@app.post("/api/plan/frontier")
def plan_frontier(body: PlanBody, owner: OwnerId, max_budget: int = Query(800_000, ge=50_000, le=10_000_000),
                  levels: int = Query(9, ge=2, le=20)):
    _ready()
    return planner.cost_frontier(_gaps(body, owner), STATE.tri, catalogue_source(), body.plan, max_budget, levels)


@app.post("/api/plan/alternatives")
def plan_alternatives(body: PlanBody, owner: OwnerId, response: Response, k: int = Query(3, ge=1, le=6), min_diff: int = Query(1, ge=1, le=5)):
    """Plan A / B / C: the best plan and the next-best structurally different portfolios."""
    _ready()
    g = _gaps(body, owner)
    alts = planner.alternative_plans(g, STATE.tri, catalogue_source(), body.plan, k, min_diff)
    run_id = _save_run("alternatives", body, g, alts[0]["plan"], {"alternatives": alts}, owner)
    if run_id:
        response.headers["X-Plan-Run-Id"] = run_id
    return alts


@app.post("/api/plan/confidence", response_model=planner.Confidence)
def plan_confidence(body: PlanBody, owner: OwnerId):
    """How robust the recommendation is to how the evidence is read (9 re-optimisations)."""
    _ready()
    return planner.confidence(_gaps(body, owner), STATE.tri, catalogue_source(), body.plan)


@app.post("/api/plan/baselines", response_model=planner.BaselineComparison)
def plan_baselines(body: PlanBody, owner: OwnerId):
    """The optimiser against naive rules (biggest gap first, most in-demand first, cheapest seats first),
    at today's settings and across budgets from 25% to 200% of today's."""
    _ready()
    return planner.baseline_sweep(_gaps(body, owner), STATE.tri, catalogue_source(), body.plan)


class WhatIfBody(PlanBody):
    scenario: whatif.Scenario = whatif.Scenario()


@app.get("/api/plan/what-if/presets")
def what_if_presets():
    return whatif.PRESETS


@app.post("/api/plan/what-if", response_model=whatif.WhatIfResult)
def plan_what_if(body: WhatIfBody, owner: OwnerId):
    """'What happens if...': apply a scenario, re-optimise, and compare with today's plan."""
    _ready()
    return whatif.simulate(_gaps(body, owner), STATE.tri, catalogue_source(), body.plan, body.scenario)


@app.post("/api/plan/evidence", response_model=list[evidence.CourseTrail])
def plan_evidence(body: PlanBody, owner: OwnerId):
    _ready()
    g = _gaps(body, owner)
    cat = catalogue_source()
    res = planner.optimise(g, STATE.tri, cat, body.plan, explain_excluded=False)
    return evidence.build(res, g, STATE.tri, cat)


class FullResponse(BaseModel):
    result: planner.PlanResult
    evidence: list[evidence.CourseTrail]
    brief: brieflib.Brief
    brief_markdown: str
    timings_ms: dict[str, float]
    run_id: str | None = None  # id of the saved plan run (database only)
    baselines: planner.BaselineComparison | None = None  # optimiser vs naive rules, same constraints


def _save_run(kind: str, body: PlanBody, gaps: pd.DataFrame, plan: dict, result: dict,
              owner: str | None) -> str | None:
    """Record the decision in the plan history when the database is on. The id is assigned now and the
    row is written in the background, so the response does not wait for the database (and a failed
    write never fails the request)."""
    if not DB.enabled:
        return None
    run_id = str(uuid.uuid4())
    req = body.plan.model_dump(mode="json")
    g = {a: int(n) for a, n in zip(gaps["area"], gaps["learners_short"], strict=True)}
    cohort_id, label = body.cohort_id, body.label
    DB.submit(lambda s: PlanRunRepository(s, owner).add(kind, req, g, plan, result, cohort_id=cohort_id,
                                                 label=label, run_id=run_id), f"plan run {run_id}")
    return run_id


@app.post("/api/plan/full", response_model=FullResponse)
def plan_full(body: PlanBody, owner: OwnerId):
    return _full(body, save=True, owner=owner)


def _full(body: PlanBody, save: bool, owner: str | None) -> FullResponse:
    _ready()
    t = {}
    t0 = time.perf_counter()
    g = _gaps(body, owner)
    cat = catalogue_source()
    res = planner.optimise(g, STATE.tri, cat, body.plan)
    t["solve_and_explain"] = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    trails = evidence.build(res, g, STATE.tri, cat)
    t["evidence"] = (time.perf_counter() - t1) * 1000
    sample = body.gaps is None and body.cohort_id is None
    b = brieflib.write(res, trails, int(len(STATE.sample_jds)) if sample else None)
    problem = planner._problem(g, STATE.tri, planlib.load_catalogue(cat), body.plan)
    cmp = planner.compare_baselines(problem, res.plan, STATE.tri, res.course_names)
    t["total"] = (time.perf_counter() - t0) * 1000
    out = FullResponse(result=res, evidence=trails, brief=b, brief_markdown=b.markdown(),
                       timings_ms={k: round(v, 1) for k, v in t.items()}, baselines=cmp)
    if save:
        out.run_id = _save_run("full", body, g, res.plan.model_dump(mode="json"),
                               out.model_dump(mode="json", exclude={"run_id"}), owner)
    return out


@app.post("/api/plan/brief", response_model=brieflib.Brief)
def plan_brief(body: PlanBody, owner: OwnerId):
    return _full(body, save=False, owner=owner).brief


# ---- Explainable AI (offline) --------------------------------------------------------------------------
from d2s.xai import learner as xlearner  # noqa: E402
from d2s.xai import plan as xplan  # noqa: E402
from d2s.xai import skills as xskills  # noqa: E402
from d2s.xai.schemas import Explanation  # noqa: E402


@app.get("/api/xai/model/global", response_model=Explanation)
def xai_model_global():
    _ready()
    return xlearner.global_importance(STATE.model)


@app.post("/api/xai/learner/why", response_model=Explanation)
def xai_learner_why(body: PredictBody):
    _ready()
    return xlearner.why(STATE.model, body.scores)


@app.post("/api/xai/learner/what-if", response_model=Explanation)
def xai_learner_what_if(body: PredictBody):
    _ready()
    return xlearner.cheapest_path(STATE.model, body.scores, body.target_probability)


def _problem_and_plan(body: PlanBody, owner: str | None):
    courses = planlib.load_catalogue(catalogue_source())
    problem = planner._problem(_gaps(body, owner), STATE.tri, courses, body.plan)
    return problem, planner.solve(problem, planner.OPTS)


@app.post("/api/xai/plan/constraints", response_model=Explanation)
def xai_plan_constraints(body: PlanBody, owner: OwnerId):
    _ready()
    problem, plan = _problem_and_plan(body, owner)
    if not plan.is_solution:
        return xplan.infeasibility(problem)
    return xplan.constraint_report(problem, plan)


@app.post("/api/xai/plan/course/{course_id}/what-if", response_model=Explanation)
def xai_course_what_if(course_id: str, body: PlanBody, owner: OwnerId,
                       resource: str = Query("budget", pattern="^(budget|trainer_hours|weight)$")):
    _ready()
    problem, plan = _problem_and_plan(body, owner)
    if not plan.is_solution:
        raise HTTPException(422, "the base plan is infeasible; see /api/xai/plan/feasibility")
    return xplan.course_what_if(problem, plan, course_id, resource)


@app.post("/api/xai/plan/feasibility", response_model=Explanation)
def xai_plan_feasibility(body: PlanBody, owner: OwnerId):
    _ready()
    problem = planner._problem(_gaps(body, owner), STATE.tri, planlib.load_catalogue(catalogue_source()), body.plan)
    return xplan.infeasibility(problem)


@app.get("/api/xai/skill/tag", response_model=Explanation)
def xai_skill_tag(tag: str = Query(..., min_length=1, max_length=100)):
    m = _matcher()
    return xskills.explain_tag(m, tag)


class TextBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


@app.post("/api/xai/skill/text", response_model=Explanation)
def xai_skill_text(body: TextBody):
    return xskills.explain_text(_matcher(), body.text)


def _matcher():
    """Tag/text explanations need the full skill matcher (taxonomy + label index); built lazily."""
    if STATE.matcher is None:
        if STATE.lookup is None:
            raise HTTPException(503, "embedding model is warming up; retry in a few seconds")
        with STATE._lock:
            if STATE.matcher is None:
                from d2s.ml.skills import SkillIndex, SkillMatcher
                from d2s.ml.taxonomy import load_esco, load_onet
                from d2s.workers.tasks import index_path

                tax = load_onet().merge(load_esco())
                skills = list(tax.skills.values())
                emb = STATE.lookup.embedder
                idx = SkillIndex.load(store._need(index_path(emb.name, tax.version, "label"), "run_sas_ingestion.py"))
                STATE.matcher = SkillMatcher(skills, emb, index=idx,
                                             semantic_ids={s.id for s in skills if s.source == "esco"})
    return STATE.matcher


# ---- Insights --------------------------------------------------------------------------------------------
@app.get("/api/insights/junior")
def insights_junior():
    return insights.junior()


@app.get("/api/insights/senior")
def insights_senior():
    return insights.senior()
