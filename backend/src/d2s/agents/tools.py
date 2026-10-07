"""Agent tools: thin wrappers over the existing services.

Each tool returns a ToolResult whose ``lines`` are written from the service's output (so every number in
an agent's answer is a number the optimiser or simulator produced), plus ``data`` for the UI.
"""

import time
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel

from d2s.analysis import plan as planlib
from d2s.engine import why_not
from d2s.models.hike import LABELS
from d2s.services import planner, whatif


@dataclass
class AgentContext:
    """Everything a tool needs about the user's current situation (built per request by the API)."""

    gaps: pd.DataFrame
    tri: pd.DataFrame
    catalogue: object  # path or list[Course]
    request: planner.PlanRequest

    def __post_init__(self):
        self.courses = planlib.load_catalogue(self.catalogue)
        self.names = {c.id: c.name for c in self.courses}

    def problem(self, req: planner.PlanRequest | None = None):
        return planner._problem(self.gaps, self.tri, self.courses, req or self.request)


class ToolResult(BaseModel):
    tool: str
    args: dict
    title: str
    lines: list[str]
    data: dict = {}
    ms: float = 0.0
    ok: bool = True


def _lakh(x: float) -> str:
    return f"Rs {x / 1e5:.2f} lakh"


def _timed(fn):
    def wrapper(ctx: AgentContext, **kw) -> ToolResult:
        t = time.perf_counter()
        try:
            r = fn(ctx, **kw)
        except ValueError as exc:
            r = ToolResult(tool=fn.__name__, args=kw, title="Could not run", lines=[str(exc)], ok=False)
        r.ms = round((time.perf_counter() - t) * 1000, 1)
        return r
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ---- tools ----------------------------------------------------------------------------------------------
@_timed
def optimise_plan(ctx: AgentContext, budget: int | None = None, trainer_hours: int | None = None) -> ToolResult:
    """Best training plan for a budget and trainer-hours (defaults: the user's current settings)."""
    req = ctx.request.model_copy(update={k: v for k, v in {"budget": budget, "trainer_hours": trainer_hours}.items()
                                         if v is not None})
    res = planner.optimise(ctx.gaps, ctx.tri, ctx.courses, req, explain_excluded=False)
    p = res.plan
    lines = [(f"Run {len(p.courses)} courses for {p.total_seats} seats at {_lakh(p.total_cost)} of "
              f"{_lakh(req.budget)}, using {p.total_trainer_hours} of {req.trainer_hours} trainer-hours "
              f"(proven optimal).")]
    lines += [f"{ctx.names[c.course_id]}: {c.seats} seats" for c in p.courses]
    closed = [f"{LABELS.get(s.skill_id, s.skill_id)} {s.closure_pct:.0f}%" for s in p.skills]
    lines.append("Gap closed: " + ", ".join(closed) + ".")
    lines.append(f"What limits it: {res.binding.verdict}.")
    return ToolResult(tool="optimise_plan", args={"budget": req.budget, "trainer_hours": req.trainer_hours},
                      title=f"Best plan at {_lakh(req.budget)} and {req.trainer_hours} trainer-hours",
                      lines=lines, data={"plan": p.model_dump(mode="json"), "names": ctx.names})


@_timed
def what_if(ctx: AgentContext, **scenario) -> ToolResult:
    """Apply a change (budget %, trainer-hours, prices %, demand %, cohort %, courses dropped/required) and re-plan."""
    sc = whatif.Scenario(**scenario)
    r = whatif.simulate(ctx.gaps, ctx.tri, ctx.courses, ctx.request, sc)
    return ToolResult(tool="what_if", args=sc.model_dump(exclude_defaults=True), title=r.title, lines=r.narrative,
                      data=r.model_dump(mode="json"), ok=r.after.feasible)


@_timed
def plan_confidence(ctx: AgentContext) -> ToolResult:
    """How robust today's plan is to how the evidence is read (9 re-optimisations)."""
    c = planner.confidence(ctx.gaps, ctx.tri, ctx.courses, ctx.request)
    lines = [f"{c.level.capitalize()} confidence ({c.score:.0%}). {c.explanation}"]
    lines += [f"{x.name}: chosen in {x.chosen_in} of {x.scenarios} readings ({x.level})" for x in c.courses]
    return ToolResult(tool="plan_confidence", args={}, title="How sure is this plan?", lines=lines,
                      data=c.model_dump(mode="json"))


@_timed
def compare_baselines(ctx: AgentContext) -> ToolResult:
    """The optimiser against simple rules (biggest gap first, most in-demand first, cheapest seats first)."""
    problem = ctx.problem()
    plan = planner.solve(problem, planner.OPTS)
    if not plan.is_solution:
        raise ValueError("today's settings have no feasible plan")
    b = planner.compare_baselines(problem, plan, ctx.tri, ctx.names)
    lines = [b.summary] + [f"{r['name']}: {r['impact_pct_of_optimiser']:.1f}% of the optimiser's impact"
                           for r in b.baselines]
    return ToolResult(tool="compare_baselines", args={}, title="Optimiser vs simple rules", lines=lines,
                      data=b.model_dump(mode="json"))


@_timed
def why_not_course(ctx: AgentContext, course_id: str) -> ToolResult:
    """Why a course is not in the plan, and the smallest extra budget that would bring it in."""
    if course_id not in ctx.names:
        raise ValueError(f"unknown course {course_id}")
    name = ctx.names[course_id]
    problem = ctx.problem()
    plan = planner.solve(problem, planner.OPTS)
    if course_id in plan.seats():
        return ToolResult(tool="why_not_course", args={"course_id": course_id}, title=f"{name} is in the plan",
                          lines=[f"{name} is already in the plan with {plan.seats()[course_id]} seats."])
    w = why_not(problem, plan, course_id, planner.OPTS)
    lines = []
    if w.feasible and w.objective_delta_pct is not None:
        lines.append(f"Forcing {name} in would lower the plan's impact by {abs(w.objective_delta_pct):.1f}%: "
                     f"the budget it needs closes more demand-weighted gap elsewhere.")
    else:
        lines.append(f"{name} does not fit: {w.reason}.")
    from d2s.xai.plan import course_what_if

    e = course_what_if(problem, plan, course_id, "budget")
    if e.details and e.details.get("found"):
        lines.append(f"It enters the optimal plan with {_lakh(e.details['minimal_increase'])} more budget.")
    elif e.details:
        lines.append("More budget alone never brings it in; another limit (trainer-hours, prerequisites) blocks it.")
    return ToolResult(tool="why_not_course", args={"course_id": course_id}, title=f"Why not {name}?", lines=lines,
                      data={"why_not": w.model_dump(mode="json", exclude={"alternative_plan"}),
                            "what_it_takes": e.details or {}})


@_timed
def alternative_plans(ctx: AgentContext, k: int = 3) -> ToolResult:
    """Plan A/B/C: the best plan and the next-best plans that use a different set of courses."""
    alts = planner.alternative_plans(ctx.gaps, ctx.tri, ctx.courses, ctx.request, k=max(2, min(k, 5)))
    lines = [f"{a['label']}: {_lakh(a['plan']['total_cost'])}, {a['impact_pct_of_best']:.1f}% impact. {a['tradeoff']}"
             for a in alts]
    return ToolResult(tool="alternative_plans", args={"k": k}, title="Alternative plans", lines=lines,
                      data={"alternatives": alts})


TOOLS = {f.__name__: f for f in (optimise_plan, what_if, plan_confidence, compare_baselines, why_not_course,
                                 alternative_plans)}
TOOL_HELP = {
    "optimise_plan": "the best plan for a budget and trainer-hours",
    "what_if": ("a change: budget cut or increase, a trainer leaving, prices rising, demand shifting, a bigger "
                "batch, a course unavailable or required"),
    "plan_confidence": "how sure, robust or reliable the plan is",
    "compare_baselines": "the optimiser compared with simple rules of thumb",
    "why_not_course": "why a particular course was not chosen",
    "alternative_plans": "other options, Plan B and C",
}
