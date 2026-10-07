"""Background jobs wrapping the engine and ML services.

CPU-bound work (CP-SAT, StatsForecast, embedding) runs in ``asyncio.to_thread`` so the worker's
event loop keeps heart-beating, stays cancellable and can publish progress meanwhile.
"""

import asyncio
import threading
from functools import lru_cache
from pathlib import Path

import pandas as pd

from d2s.config import get_settings
from d2s.engine import PlanningProblem, ProgressEvent, SolverOptions
from d2s.engine import analyze as engine_analyze
from d2s.engine import frontier as engine_frontier
from d2s.engine import solve as engine_solve
from d2s.engine import why_not as engine_why_not
from d2s.engine.schemas import Plan
from d2s.workers.registry import TaskContext, TransientError, task


def _options(payload: dict) -> SolverOptions | None:
    return SolverOptions.model_validate(payload["options"]) if payload.get("options") else None


# ---- decision engine ---------------------------------------------------------------

@task("optimize_plan", timeout_s=120, max_tries=1)
async def optimize_plan(ctx: TaskContext, payload: dict) -> dict:
    problem = PlanningProblem.model_validate(payload["problem"])
    r = ctx.reporter
    await r.report("solving", 0.0, f"{len(problem.courses)} courses, {len(problem.skills)} skills")

    def on_progress(ev: ProgressEvent) -> None:
        gap = 1 - ev.objective / ev.best_bound if ev.best_bound else None
        r.threadsafe("solving", None, f"solution #{ev.solutions}", objective=ev.objective,
                     best_bound=ev.best_bound, gap=gap, wall_time_s=ev.wall_time_s)

    plan = await asyncio.to_thread(engine_solve, problem, _options(payload), on_progress)
    return plan.model_dump(mode="json")


@task("sensitivity", timeout_s=300, max_tries=1)
async def sensitivity(ctx: TaskContext, payload: dict) -> dict:
    problem = PlanningProblem.model_validate(payload["problem"])
    opts = _options(payload)
    if payload.get("plan"):
        plan = Plan.model_validate(payload["plan"])
    else:
        await ctx.reporter.report("solving", 0.1, "solving base plan")
        plan = await asyncio.to_thread(engine_solve, problem, opts)
    if not plan.is_solution:
        raise ValueError(f"base plan is {plan.status}: {'; '.join(plan.messages)}")
    await ctx.reporter.report("analyzing", 0.4, "shadow prices + exact marginal values")
    report = await asyncio.to_thread(
        engine_analyze, problem, plan,
        payload.get("budget_delta", 100_000), payload.get("hours_delta", 10),
        payload.get("seats_delta", 10), opts,
    )
    return report.model_dump(mode="json")


@task("pareto_frontier", timeout_s=600, max_tries=1)
async def pareto_frontier(ctx: TaskContext, payload: dict) -> dict:
    problem = PlanningProblem.model_validate(payload["problem"])
    levels = int(payload.get("levels", 8))
    await ctx.reporter.report("sweeping", 0.0, f"{levels} budget levels")
    points = await asyncio.to_thread(
        engine_frontier, problem, levels, payload.get("min_budget"), _options(payload)
    )
    return {"points": [p.model_dump(mode="json") for p in points]}


@task("why_not", timeout_s=120, max_tries=1)
async def why_not(ctx: TaskContext, payload: dict) -> dict:
    problem = PlanningProblem.model_validate(payload["problem"])
    plan = Plan.model_validate(payload["plan"])
    result = await asyncio.to_thread(
        engine_why_not, problem, plan, payload["course_id"], _options(payload)
    )
    return result.model_dump(mode="json")


# ---- ML services ---------------------------------------------------------------------

@task("forecast_demand", timeout_s=600, max_tries=2)
async def forecast_demand(ctx: TaskContext, payload: dict) -> dict:
    from d2s.ml.forecasting import DemandForecaster, demand_weights

    df = pd.DataFrame(payload["records"])
    n = df["unique_id"].nunique() if len(df) else 0
    await ctx.reporter.report("forecasting", 0.0, f"{n} series")
    fc = DemandForecaster(
        horizon=int(payload.get("horizon", 6)),
        level=int(payload.get("level", 80)),
        freq=payload.get("freq", "MS"),
    )
    results = await asyncio.to_thread(fc.fit_predict, df)
    return {
        "forecasts": [r.model_dump(mode="json") for r in results],
        "weights": demand_weights(results, payload.get("weight_band", "point")),
    }


_matcher_lock = threading.Lock()


@lru_cache(maxsize=1)
def _matcher():
    """Load taxonomy + embedding index once per process; cache the index on disk per model."""
    from d2s.ml.embeddings import get_embedder
    from d2s.ml.skills import SkillIndex, SkillMatcher
    from d2s.ml.taxonomy import load_esco, load_onet

    with _matcher_lock:
        taxonomy = load_onet()
        try:
            taxonomy = taxonomy.merge(load_esco())
        except FileNotFoundError:
            pass  # ESCO not downloaded yet: O*NET only
        embedder = get_embedder()
        skills = list(taxonomy.skills.values())
        cache = index_path(embedder.name, taxonomy.version)
        index = None
        if cache.exists():
            index = SkillIndex.load(cache)
            if index.ids != [s.id for s in skills]:
                index = None  # taxonomy changed: rebuild
        if index is None:
            index = SkillIndex.build(skills, embedder)
            index.save(cache)
        return SkillMatcher(skills, embedder, index=index)


def index_path(model_name: str, version: str) -> Path:
    safe = model_name.replace("/", "__")
    return get_settings().dataset_dir / "index" / f"{safe}__{version}.npz"


@task("build_skill_index", timeout_s=1800, max_tries=3)
async def build_skill_index(ctx: TaskContext, payload: dict) -> dict:
    await ctx.reporter.report("loading", 0.0, "taxonomy + embedding model")
    try:
        matcher = await asyncio.to_thread(_matcher)
    except OSError as exc:  # model download / disk hiccup
        raise TransientError(str(exc)) from exc
    return {"skills": len(matcher.index.ids), "model": matcher.embedder.name}


@task("extract_skills", timeout_s=300, max_tries=2)
async def extract_skills(ctx: TaskContext, payload: dict) -> dict:
    texts: list[str] = payload["texts"]
    limit = int(payload.get("limit", 25))
    await ctx.reporter.report("loading", 0.0, "skill matcher")
    matcher = await asyncio.to_thread(_matcher)
    out = []
    for i, text in enumerate(texts):
        matches = await asyncio.to_thread(matcher.extract, text, limit)
        out.append([m.model_dump() for m in matches])
        await ctx.reporter.report("extracting", (i + 1) / len(texts), f"{i + 1}/{len(texts)} texts")
    return {"results": out}
