"""Agent 1: Planning Copilot. A question in plain English -> the right tools -> an answer built from
their results.

    route ──fast──> act ──> answer
      └──llm──> choose ──> act ──> answer
                   └─────(nothing to do)──> answer

* route  : deterministic intent + argument extraction (d2s.agents.extract). Most questions finish here
           without the language model ("fast path", ~0.5-1 s).
* choose : the local model picks tools from an enum (constrained JSON). Only its first choice is
           trusted unless the question itself supports the others; it never supplies numbers.
* act    : runs the tools (max MAX_CALLS); each result is a trace step.
* answer : written from the tools' own result lines, so every number comes from the optimiser.
"""

import operator
import time
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from d2s.agents.extract import parse
from d2s.agents.llm import Chooser
from d2s.agents.tools import TOOL_HELP, TOOLS, AgentContext

MAX_CALLS = 4
ToolName = Literal["optimise_plan", "what_if", "plan_confidence", "compare_baselines", "why_not_course",
                   "alternative_plans", "none"]


class Pick(BaseModel):
    tools: list[ToolName] = Field(description="1-3 tools that answer the question, most important first")


SYSTEM = ("You route questions for a training-planning tool. Pick the tools that answer the question.\n"
          + "\n".join(f"- {k}: {v}" for k, v in TOOL_HELP.items())
          + "\n- none: the question is not about training plans")

EXAMPLES = [
    "What happens if our budget is cut by 30%?",
    "A trainer is leaving next term. What changes?",
    "How sure are you about this plan?",
    "Why didn't you pick the deep learning course?",
    "Is this better than just funding the biggest gap?",
    "Give me two other options",
    "Budget drops 20% and AI demand jumps. What should we do?",
]


class CopilotState(TypedDict, total=False):
    question: str
    parsed: dict
    path: str  # fast | llm | none
    calls: list[dict]
    notes: list[str]
    results: list[dict]
    answer: dict
    trace: Annotated[list[dict], operator.add]


def _step(node: str, detail: str, **extra) -> list[dict]:
    return [{"node": node, "detail": detail, "t": time.time(), **extra}]


def _calls_for(intents: list[str], p, ctx: AgentContext) -> tuple[list[dict], list[str]]:
    """Turn intents into tool calls with arguments from the parsed question; explain what can't run."""
    calls, notes = [], []
    for intent in intents:
        if intent == "what_if":
            if p.scenario:
                calls.append({"tool": "what_if", "args": p.scenario})
            else:
                notes.append("That sounds like a what-if, but I could not tell what changes. Try e.g. "
                             "\"budget cut by 30%\", \"a trainer leaves\" or \"without the statistics course\".")
        elif intent == "why_not_course":
            chosen = [c for c in p.courses if c in ctx.names]
            if chosen:
                calls += [{"tool": "why_not_course", "args": {"course_id": c}} for c in chosen[:2]]
            else:
                notes.append("Which course? e.g. \"why not deep learning?\"")
        elif intent == "optimise_plan":
            calls.append({"tool": "optimise_plan", "args": {k: v for k, v in
                                                             {"budget": p.budget, "trainer_hours": p.trainer_hours}.items()
                                                             if v is not None}})
        elif intent in TOOLS:
            calls.append({"tool": intent, "args": {}})
    seen, unique = set(), []
    for c in calls:  # no duplicates, bounded
        key = (c["tool"], tuple(sorted(str(c["args"]).split())))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:MAX_CALLS], notes


def build_copilot():
    def route(state: CopilotState, config) -> CopilotState:
        ctx: AgentContext = config["configurable"]["ctx"]
        p = parse(state["question"], ctx.names, ctx.request.budget, ctx.request.trainer_hours)
        calls, notes = _calls_for(p.intents, p, ctx)
        path = "fast" if calls else "llm"
        return {"parsed": p.__dict__, "calls": calls, "notes": notes, "path": path,
                "trace": _step("route", f"understood: {', '.join(p.intents) or 'nothing certain'}"
                               + (f"; {p.scenario}" if p.scenario else ""), path=path)}

    def choose(state: CopilotState, config) -> CopilotState:
        ctx: AgentContext = config["configurable"]["ctx"]
        chooser: Chooser = config["configurable"]["chooser"]
        t = time.perf_counter()
        pick = chooser(SYSTEM, state["question"], Pick)
        ms = round((time.perf_counter() - t) * 1000)
        if pick is None:
            return {"path": "none", "trace": _step("choose", "local model unavailable; no tool chosen", ms=ms)}
        tools = [x for x in pick.tools if x != "none"]
        if not tools:
            return {"path": "none", "trace": _step("choose", "model: not a planning question", ms=ms)}
        from d2s.agents.extract import Parsed

        p = Parsed(**state["parsed"])
        # trust the first choice; keep others only if the question itself supports them
        keep = [tools[0]] + [x for x in tools[1:] if x in p.intents]
        calls, notes = _calls_for(keep, p, ctx)
        return {"calls": calls, "notes": state.get("notes", []) + notes,
                "trace": _step("choose", f"model picked {', '.join(tools)}; using {', '.join(keep)}", ms=ms)}

    def act(state: CopilotState, config) -> CopilotState:
        ctx: AgentContext = config["configurable"]["ctx"]
        results, steps = [], []
        for call in state["calls"]:
            r = TOOLS[call["tool"]](ctx, **call["args"])
            results.append(r.model_dump(mode="json"))
            steps += _step("act", f"{call['tool']}({', '.join(f'{k}={v}' for k, v in call['args'].items())})",
                           tool=call["tool"], ms=r.ms, ok=r.ok)
        return {"results": results, "trace": steps}

    def answer(state: CopilotState, config) -> CopilotState:
        results = state.get("results", [])
        notes = list(state.get("notes", []))
        assumptions = state.get("parsed", {}).get("assumptions", [])
        if not results and not notes:
            notes.append("I can answer questions about this plan: " + "; ".join(TOOL_HELP.values())
                         + ". For example: " + " / ".join(f"\"{e}\"" for e in EXAMPLES[:4]))
        sections = [{"title": r["title"], "lines": r["lines"], "tool": r["tool"], "ok": r["ok"]} for r in results]
        text = "\n\n".join([f"**{s['title']}**\n" + "\n".join(f"- {line}" for line in s["lines"]) for s in sections]
                           + ([("Assumed: " + " ".join(assumptions))] if assumptions else [])
                           + notes)
        return {"answer": {"text": text, "sections": sections, "assumptions": assumptions, "notes": notes,
                           "path": state.get("path", "none")},
                "trace": _step("answer", f"{len(sections)} result(s) written from tool output")}

    g = StateGraph(CopilotState)
    g.add_node("route", route)
    g.add_node("choose", choose)
    g.add_node("act", act)
    g.add_node("answer", answer)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: "act" if s["calls"] else "choose", ["act", "choose"])
    g.add_conditional_edges("choose", lambda s: "act" if s.get("calls") else "answer", ["act", "answer"])
    g.add_edge("act", "answer")
    g.add_edge("answer", END)
    return g.compile()


COPILOT = build_copilot()
