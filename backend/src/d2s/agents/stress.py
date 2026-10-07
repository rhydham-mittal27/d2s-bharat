"""Agent 2: Stress-test. Finds the conditions under which today's plan breaks, and says what to do.

    baseline -> probes -> explore -> follow_ups -> report

* probes     : a fixed battery of single shocks (budget, trainer-hours, prices, cohort size, losing each
               chosen course, a demand surge in each area) plus the budget and trainer-hour breakpoints
               (largest cut that keeps the same courses), found by bisection over exact re-solves.
* explore    : the local model chooses which combined shocks to test next from candidates built out of
               the worst single shocks (an enum: it cannot invent scenarios). Without the model, the
               three most severe combinations are used.
* follow_ups : runs the chosen combinations.
* report     : ranked risks, breakpoints and mitigations, all written from the simulator's numbers.
"""

import operator
import time
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import Field, create_model

from d2s.agents.llm import Chooser
from d2s.agents.tools import AgentContext
from d2s.models.hike import LABELS
from d2s.services import planner, whatif


class StressState(TypedDict, total=False):
    today: dict
    probes: list[dict]
    breakpoints: dict
    candidates: list[dict]
    chosen: list[str]
    follow_ups: list[dict]
    report: dict
    trace: Annotated[list[dict], operator.add]


def _step(node, detail, **extra):
    return [{"node": node, "detail": detail, "t": time.time(), **extra}]


def _run(ctx: AgentContext, sc: dict) -> dict:
    r = whatif.simulate(ctx.gaps, ctx.tri, ctx.courses, ctx.request, whatif.Scenario(**sc))
    same_weights = not sc.get("demand_pct") and not sc.get("cohort_pct")
    loss = None
    if r.after.feasible and same_weights and r.today.impact:
        loss = round(100 * (r.today.impact - r.after.impact) / r.today.impact, 1)
    return {"scenario": sc, "title": r.title, "feasible": r.after.feasible, "loss_pct": loss if r.after.feasible else 100.0,
            "plan_changes": bool(r.added or r.dropped or r.seat_changes), "added": r.added, "dropped": r.dropped,
            "replanning_gain_pct": r.replanning_gain_pct, "today_still_fits": r.today_still_fits,
            "narrative": r.narrative, "conflicts": r.conflicts}


def _short(r: dict) -> str:
    """'What if budget -30%?' -> 'budget -30%'"""
    return r["title"].removeprefix("What if ").removesuffix("?")


def _breakpoint(ctx: AgentContext, key: str, worst: int, today_courses: set[str], steps: int = 7) -> int | None:
    """Largest cut (as a negative number in `key` units) at which running all of today's courses is still
    optimal. Tested as 'best plan that keeps today's courses' == 'best plan overall', so exact ties
    between equally good plans cannot move the breakpoint."""
    base = ctx.problem()

    def same(v):
        p = whatif._apply(base, whatif.Scenario(**{key: v}))
        free = planner.solve(p, planner.OPTS)
        kept = planner.solve(p.model_copy(update={"forced_in": sorted(set(p.forced_in) | today_courses)}),
                             planner.OPTS)
        return free.is_solution and kept.is_solution and kept.objective >= free.objective - 1e-6 * max(
            1.0, abs(free.objective))

    if same(worst):
        return worst
    lo, hi = 0, worst  # lo keeps the plan, hi breaks it
    for _ in range(steps):
        mid = round((lo + hi) / 2)
        if mid in (lo, hi):
            break
        if same(mid):
            lo = mid
        else:
            hi = mid
    return lo


def build_stress():
    def baseline(state: StressState, config) -> StressState:
        ctx: AgentContext = config["configurable"]["ctx"]
        plan = planner.solve(ctx.problem(), planner.OPTS)
        if not plan.is_solution:
            raise ValueError("today's settings have no feasible plan")
        today = {"courses": plan.seats(), "cost": plan.total_cost, "hours": plan.total_trainer_hours,
                 "impact": plan.objective, "budget": ctx.request.budget, "trainer_hours": ctx.request.trainer_hours}
        return {"today": today, "trace": _step("baseline", f"today: {len(plan.courses)} courses, "
                                                           f"Rs {plan.total_cost:,}, {plan.total_trainer_hours} h")}

    def probes(state: StressState, config) -> StressState:
        ctx: AgentContext = config["configurable"]["ctx"]
        today = state["today"]
        shocks = [{"budget_pct": -10}, {"budget_pct": -20}, {"budget_pct": -30},
                  {"trainer_hours_delta": -30}, {"trainer_hours_delta": -60},
                  {"cost_pct": 10}, {"cost_pct": 20}, {"cohort_pct": 25}]
        shocks += [{"remove_courses": [c]} for c in today["courses"]]
        shocks += [{"demand_pct": {a: 50}} for a in LABELS]
        out = [_run(ctx, sc) for sc in shocks]
        courses = set(today["courses"])
        bp = {"budget_pct": _breakpoint(ctx, "budget_pct", -80, courses)}
        if ctx.request.trainer_hours:
            bp["trainer_hours_delta"] = _breakpoint(ctx, "trainer_hours_delta", -int(ctx.request.trainer_hours * 0.8),
                                                    courses)
        changed = sum(o["plan_changes"] for o in out)
        return {"probes": out, "breakpoints": bp,
                "trace": _step("probes", f"{len(out)} single shocks: plan changes under {changed}; breakpoints "
                                         f"{bp}")}

    def explore(state: StressState, config) -> StressState:
        chooser: Chooser = config["configurable"]["chooser"]
        severe = sorted([p for p in state["probes"] if p["loss_pct"] is not None and p["loss_pct"] > 0],
                        key=lambda p: -p["loss_pct"])[:4]
        cands = []
        for i in range(len(severe)):
            for j in range(i + 1, len(severe)):
                a, b = severe[i]["scenario"], severe[j]["scenario"]
                if set(a) & set(b):  # same lever twice (e.g. two budget cuts): not a combination
                    continue
                merged = {**a, **b}
                if "remove_courses" in a and "remove_courses" in b:
                    merged["remove_courses"] = a["remove_courses"] + b["remove_courses"]
                cands.append({"id": f"c{len(cands) + 1}", "scenario": merged,
                              "hint": severe[i]["loss_pct"] + severe[j]["loss_pct"],
                              "label": f"{_short(severe[i])} + {_short(severe[j])}"})
        cands.append({"id": f"c{len(cands) + 1}", "scenario": {"budget_pct": -20, "cost_pct": 10, "trainer_hours_delta": -30},
                      "hint": 0, "label": "a tight year: budget -20%, prices +10%, -30 trainer-hours"})
        if not cands:
            return {"candidates": [], "chosen": [], "trace": _step("explore", "no severe single shocks to combine")}
        ids = [c["id"] for c in cands]
        Choice = create_model("Choice", ids=(list[Literal[tuple(ids)]], Field(description="up to 3 scenario ids")))
        listing = "\n".join(f"{c['id']}: {c['label']}" for c in cands)
        t = time.perf_counter()
        pick = chooser("You stress-test a training plan. Choose up to 3 combined scenarios most likely to break it, "
                       "favouring plausible ones a college could actually face.", listing, Choice)
        ms = round((time.perf_counter() - t) * 1000)
        if pick and pick.ids:
            chosen, how = list(dict.fromkeys(pick.ids))[:3], "model"
        else:
            chosen, how = [c["id"] for c in sorted(cands, key=lambda c: -c["hint"])[:3]], "most severe (no model)"
        return {"candidates": cands, "chosen": chosen,
                "trace": _step("explore", f"{len(cands)} combinations; chose {', '.join(chosen)} by {how}", ms=ms)}

    def follow_ups(state: StressState, config) -> StressState:
        ctx: AgentContext = config["configurable"]["ctx"]
        by_id = {c["id"]: c for c in state.get("candidates", [])}
        out = [{**_run(ctx, by_id[i]["scenario"]), "label": by_id[i]["label"]} for i in state.get("chosen", [])]
        return {"follow_ups": out, "trace": _step("follow_ups", f"ran {len(out)} combined scenarios")}

    def report(state: StressState, config) -> StressState:
        ctx: AgentContext = config["configurable"]["ctx"]
        today, probes_, bp = state["today"], state["probes"], state["breakpoints"]
        unchanged = [p for p in probes_ if not p["plan_changes"]]
        risks = sorted([p for p in probes_ + state.get("follow_ups", []) if p["loss_pct"]],
                       key=lambda p: -p["loss_pct"])
        lines = [f"Today's plan survives {len(unchanged)} of {len(probes_)} single shocks without any change."]
        b = bp.get("budget_pct")
        if b is not None:
            lines.append(f"Budget: the same courses still run (seat numbers may shrink) down to a {abs(b)}% cut "
                         f"(Rs {round(today['budget'] * (1 + b / 100)):,}); beyond that a course must go.")
        h = bp.get("trainer_hours_delta")
        if h is not None:
            lines.append(f"Trainer-hours: the same courses still run with up to {abs(h)} fewer hours "
                         f"({today['trainer_hours'] + h} h); beyond that a course must go.")
        def risk_line(r: dict) -> str:
            name = r.get("label") or _short(r)
            effect = "no feasible plan" if not r["feasible"] else f"-{r['loss_pct']}% impact"
            return f"{name}: {effect}"

        top = [risk_line(r) for r in risks[:5]]
        demand = [p for p in probes_ if "demand_pct" in p["scenario"]]
        steady = sum(not p["plan_changes"] for p in demand)
        lines.append(f"Demand: a 50% surge in one skill area changes the plan in {len(demand) - steady} of "
                     f"{len(demand)} areas.")
        mitig = []
        for p in probes_:
            if "remove_courses" in p["scenario"] and p["loss_pct"] and p["loss_pct"] >= 15:
                c = p["scenario"]["remove_courses"][0]
                mitig.append(f"{ctx.names.get(c, c)} is a single point of failure (losing it costs "
                             f"{p['loss_pct']}% of impact): line up a backup trainer or vendor.")
        if b is not None and abs(b) < 15:
            mitig.append("Little budget headroom: agree a fallback plan now (see the What-if page) in case of cuts.")
        if h is not None and today["trainer_hours"] and abs(h) < 0.15 * today["trainer_hours"]:
            mitig.append("Trainer capacity is tight: cross-train a second trainer for the busiest course.")
        if any("cost_pct" in p["scenario"] and p["loss_pct"] and p["loss_pct"] >= 5 for p in probes_):
            mitig.append("Price rises hurt: lock in course prices with vendors before committing.")
        if not mitig:
            mitig.append("No single shock causes a large loss; the plan is robust as it stands.")
        return {"report": {"summary": lines, "top_risks": top, "mitigations": mitig, "today": today,
                           "breakpoints": bp, "probes": probes_, "follow_ups": state.get("follow_ups", [])},
                "trace": _step("report", f"{len(top)} ranked risks, {len(mitig)} mitigations")}

    g = StateGraph(StressState)
    for name, fn in (("baseline", baseline), ("probes", probes), ("explore", explore), ("follow_ups", follow_ups),
                     ("report", report)):
        g.add_node(name, fn)
    g.add_edge(START, "baseline")
    g.add_edge("baseline", "probes")
    g.add_edge("probes", "explore")
    g.add_edge("explore", "follow_ups")
    g.add_edge("follow_ups", "report")
    g.add_edge("report", END)
    return g.compile()


STRESS = build_stress()
