"""Agent endpoints. Copilot and Stress-test stream their steps as server-sent events (one `step` event per
graph step, then one `result` event); Data-Intake returns a proposal and waits for the user's approval.

    GET  /api/agents/status
    POST /api/agents/copilot              {question, plan, gaps?, cohort_id?}   -> text/event-stream
    POST /api/agents/stress-test          {plan, gaps?, cohort_id?}             -> text/event-stream
    POST /api/agents/intake               multipart file                        -> {thread_id, proposal}
    POST /api/agents/intake/{thread_id}   {approve, mapping?, scales?}          -> {profile, mapping, ...}
    GET  /api/agents/runs                 this user's agent runs (database)
"""

import io
import json
import logging
import time

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from d2s.agents import intake as intake_agent
from d2s.agents.copilot import COPILOT, EXAMPLES
from d2s.agents.llm import TokenMeter, metered, ollama_status
from d2s.agents.stress import STRESS
from d2s.agents.tools import AgentContext
from d2s.analysis import plan as planlib
from d2s.api.auth import OwnerId
from d2s.api.common import PlanBody, plan_gaps
from d2s.api.db import DB, catalogue_source
from d2s.api.state import STATE
from d2s.db import AgentRunRepository, CohortRepository, session_scope
from d2s.services import cohort

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agents", tags=["agents"])
MAX_UPLOAD_BYTES = 2_000_000
_INTAKE_OWNERS: dict[str, str | None] = {}  # thread id -> owner (only the uploader may approve)


def _ready():
    if STATE.load_error or STATE.model is None:
        raise HTTPException(503, f"server not ready: {STATE.load_error or 'still loading'}")


def _ctx(body: PlanBody, owner: str | None) -> AgentContext:
    return AgentContext(gaps=plan_gaps(body, owner), tri=STATE.tri, catalogue=catalogue_source(), request=body.plan)


def _save(agent: str, owner: str | None, inp: dict, out: dict, trace: list, ms: float, path: str | None = None):
    if DB.enabled:
        DB.submit(lambda s: AgentRunRepository(s, owner).add(agent, inp, out, trace, ms, path), f"agent run {agent}")


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _stream(graph, state: dict, config: dict, agent: str, owner, inp: dict, result_key: str, meter: TokenMeter):
    """Run a graph, emitting each step as it finishes; the last event carries the result and token counts."""
    t0 = time.perf_counter()
    trace, final, attributed = [], {}, 0
    try:
        for chunk in graph.stream(state, config, stream_mode="updates"):
            for node, update in chunk.items():
                if not update:
                    continue
                new_calls = meter.calls[attributed:]
                attributed = len(meter.calls)
                for step in update.get("trace", []):
                    if new_calls:  # this node's model calls: their tokens go on its step
                        step = {**step, "tokens": sum(c["total_tokens"] for c in new_calls)}
                        new_calls = []
                    trace.append(step)
                    yield _sse("step", {**step, "node": node, "elapsed_ms": round((time.perf_counter() - t0) * 1000)})
                final.update(update)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        out = {**final.get(result_key, {}), "tokens": meter.summary()}
        yield _sse("result", {**out, "ms": ms})
        _save(agent, owner, inp, out, trace, ms, out.get("path") if isinstance(out, dict) else None)
    except ValueError as exc:
        yield _sse("error", {"detail": str(exc)})
    except Exception:  # never leave the browser waiting on a broken stream
        log.exception("agent %s failed", agent)
        yield _sse("error", {"detail": "the agent failed unexpectedly; see the server log"})


def _sse_response(gen) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---- status --------------------------------------------------------------------------------------------
@router.get("/status")
def status():
    return {"model": ollama_status(), "agents": ["copilot", "stress_test", "intake"], "examples": EXAMPLES}


# ---- Copilot -------------------------------------------------------------------------------------------
class CopilotBody(PlanBody):
    question: str = Field(..., min_length=2, max_length=500)


@router.post("/copilot")
def copilot(body: CopilotBody, owner: OwnerId):
    _ready()
    ctx, meter = _ctx(body, owner), TokenMeter()
    config = {"configurable": {"ctx": ctx, "chooser": metered(meter)}}
    inp = {"question": body.question, "plan": body.plan.model_dump(mode="json")}
    return _sse_response(_stream(COPILOT, {"question": body.question, "trace": []}, config, "copilot", owner, inp,
                                 "answer", meter))


# ---- Stress-test ---------------------------------------------------------------------------------------
@router.post("/stress-test")
def stress_test(body: PlanBody, owner: OwnerId):
    _ready()
    ctx, meter = _ctx(body, owner), TokenMeter()
    config = {"configurable": {"ctx": ctx, "chooser": metered(meter)}}
    inp = {"plan": body.plan.model_dump(mode="json")}
    return _sse_response(_stream(STRESS, {"trace": []}, config, "stress_test", owner, inp, "report", meter))


# ---- Data-Intake ---------------------------------------------------------------------------------------
def _intake_config(thread_id: str, owner: str | None, meter: TokenMeter | None = None) -> dict:
    def profile_fn(records: list[dict]) -> dict:
        cat = catalogue_source()
        frame = STATE.catalogue if not isinstance(cat, list) else planlib.catalogue_frame(cat)
        prof = cohort.profile(pd.DataFrame(records), STATE.model, STATE.area_roles, frame)
        return prof.model_dump(mode="json")

    embedder = STATE.lookup.embedder if STATE.lookup is not None else None
    return {"configurable": {"thread_id": thread_id, "chooser": metered(meter or TokenMeter()), "embedder": embedder,
                             "profile_fn": profile_fn}}


@router.post("/intake")
async def intake_start(owner: OwnerId, file: UploadFile = File(...)):  # noqa: B008
    _ready()
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // 1_000_000} MB")
    name = (file.filename or "upload").lower()
    try:
        df = pd.read_excel(io.BytesIO(raw), dtype=str) if name.endswith((".xlsx", ".xls")) else \
            pd.read_csv(io.BytesIO(raw), dtype=str, sep=None, engine="python")
    except Exception as exc:
        raise HTTPException(422, f"could not read the file: {exc}") from exc
    df = df.dropna(how="all").fillna("")
    if df.empty:
        raise HTTPException(422, "the file has no rows")
    thread_id = intake_agent.new_thread()
    _INTAKE_OWNERS[thread_id] = owner
    t0 = time.perf_counter()
    meter = TokenMeter()
    out = intake_agent.INTAKE.invoke({"filename": file.filename, "columns": [str(c) for c in df.columns],
                                      "rows": df.astype(str).to_dict("records")},
                                     _intake_config(thread_id, owner, meter))
    proposal = out["__interrupt__"][0].value
    ms = round((time.perf_counter() - t0) * 1000, 1)
    _save("intake", owner, {"filename": file.filename, "stage": "proposal"}, {"tokens": meter.summary()},
          proposal.get("trace", []), ms)
    return {"thread_id": thread_id, "proposal": proposal, "tokens": meter.summary(), "ms": ms}


class IntakeDecision(BaseModel):
    approve: bool = True
    mapping: dict[str, str | None] | None = None  # target -> source column (edits to the proposal)
    scales: dict[str, str] | None = None  # target -> scale id


@router.post("/intake/{thread_id}")
def intake_decide(thread_id: str, body: IntakeDecision, owner: OwnerId):
    _ready()
    if thread_id not in _INTAKE_OWNERS or _INTAKE_OWNERS[thread_id] != owner:
        raise HTTPException(404, "no pending intake with this id (it may have expired with a server restart)")
    valid_scales = {*intake_agent.SCALES, "words"}
    if body.scales and any(v not in valid_scales for v in body.scales.values()):
        raise HTTPException(422, f"scales must be one of {sorted(valid_scales)}")
    t0 = time.perf_counter()
    config = _intake_config(thread_id, owner)
    out = intake_agent.INTAKE.invoke(Command(resume=body.model_dump()), config)
    _INTAKE_OWNERS.pop(thread_id, None)
    result = out.get("result", {})
    if result.get("profile") and DB.enabled and result["profile"]["n_learners"]:
        prof = result["profile"]
        gaps = {a["area"]: int(a["learners_short"]) for a in prof["areas"]}
        with session_scope(DB.factory) as s:
            prof["cohort_id"] = CohortRepository(s, owner).add(f"{out.get('filename') or 'intake'} (intake agent)",
                                                               prof, gaps, source="intake").id
    ms = round((time.perf_counter() - t0) * 1000, 1)
    _save("intake", owner, {"filename": out.get("filename"), "decision": body.model_dump()},
          {k: v for k, v in result.items() if k != "profile"} | {"n_learners": (result.get("profile") or {}).get("n_learners")},
          out.get("trace", []), ms)
    return {**result, "tokens": TokenMeter().summary(), "ms": ms}  # approval step: no model calls


# ---- history -------------------------------------------------------------------------------------------
@router.get("/runs")
def runs(owner: OwnerId, limit: int = Query(30, ge=1, le=200), agent: str | None = None):
    if not DB.enabled:
        raise HTTPException(503, "agent history needs the database; set D2S_DATABASE_URL")
    DB.drain()  # include runs still being written in the background
    with session_scope(DB.factory) as s:
        rows = AgentRunRepository(s, owner).list(limit, agent)
        return [{"id": r.id, "agent": r.agent, "path": r.path, "ms": r.ms, "input": r.input,
                 "created_at": r.created_at, "steps": len(r.trace or [])} for r in rows]
