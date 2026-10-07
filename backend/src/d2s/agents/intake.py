"""Agent 3: Data-Intake. Turns a messy cohort file into the 1-5 skill-score format, with a human approving
the mapping before anything is used.

    parse -> map_rules -> map_model -> scales -> review (interrupt: human approves/edits) -> apply

* map_rules : column names matched to the five skill areas (+ learner id) by aliases, then by meaning
              (Sentence Transformers similarity, when the embedder is available).
* map_model : any area still unmapped is offered to the local model as a constrained choice among the
              unused columns (or "none"); it cannot invent a column.
* scales    : detects each column's scale (1-5, 0-5, 0-10, 1-10, 0-100, or words like "good") and
              proposes a linear rescale to 1-5.
* review    : LangGraph interrupt(). The API returns the proposal; the run resumes only when the user
              approves (optionally after editing the mapping or a scale).
* apply     : builds the clean table and profiles the cohort (gaps per area), like a normal upload.
"""

import re
import time
import uuid
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import Field, create_model

from d2s.agents.llm import Chooser
from d2s.models.hike import LABELS, SKILLS

TARGETS = ["id", *SKILLS]
TARGET_LABEL = {"id": "Learner id", **LABELS}
ALIASES = {
    "id": ["id", "student id", "learner id", "roll", "roll no", "enrolment", "enrollment", "reg no", "registration",
           "student", "learner", "name", "email"],
    "coding_skills": ["coding", "code", "programming", "python", "java", "software", "developer"],
    "maths_stats_skills": ["math", "maths", "statistics", "stats", "quant", "quantitative", "probability",
                           "numerical", "analytics math"],
    "ai_and_ml_skills": ["ai", "ml", "machine learning", "artificial intelligence", "deep learning", "ai ml", "aiml"],
    "big_data_skills": ["big data", "bigdata", "hadoop", "spark", "data engineering", "hive"],
    "dashboard_and_storytelling_skills": ["dashboard", "storytelling", "visualisation", "visualization", "viz",
                                          "tableau", "power bi", "powerbi", "reporting", "presentation"],
}
DESCRIPTIONS = {
    "coding_skills": "coding programming skill rating", "maths_stats_skills": "maths and statistics skill rating",
    "ai_and_ml_skills": "AI and machine learning skill rating", "big_data_skills": "big data Hadoop Spark skill rating",
    "dashboard_and_storytelling_skills": "dashboards data visualisation storytelling skill rating",
}
WORDS = {"very poor": 1, "poor": 1, "none": 1, "low": 1, "beginner": 2, "basic": 2, "fair": 2, "average": 3,
         "medium": 3, "moderate": 3, "intermediate": 3, "good": 4, "high": 4, "advanced": 4, "very good": 5,
         "excellent": 5, "expert": 5, "very high": 5}
SCALES = {"1-5": (1, 5), "0-5": (0, 5), "1-10": (1, 10), "0-10": (0, 10), "0-100": (0, 100)}
MAX_ROWS = 5000
EMBED_MIN = 0.55

SAVER = InMemorySaver()  # pending reviews live in this process; a restart discards unapproved uploads


class IntakeState(TypedDict, total=False):
    filename: str
    columns: list[str]
    rows: list[dict]
    mapping: dict  # target -> source column | None
    method: dict  # target -> exact/alias/meaning/model/none
    scales: dict  # target -> scale id | "words" | "unknown"
    warnings: list[str]
    proposal: dict
    decision: dict
    result: dict
    trace: list[dict]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _add_trace(state, node, detail, **extra):
    return (state.get("trace") or []) + [{"node": node, "detail": detail, "t": time.time(), **extra}]


def scale_from_header(header: str) -> str | None:
    h = header.lower()
    if re.search(r"out of 100|/ ?100\b|percent|%", h):
        return "0-100"
    if re.search(r"out of 10\b|/ ?10\b", h):
        return "0-10"
    if re.search(r"out of 5\b|/ ?5\b", h):
        return "1-5"
    return None


def detect_scale(values: list[str], header: str = "") -> str:
    hinted = scale_from_header(header)
    if hinted:
        return hinted
    vals = [str(v).strip().lower() for v in values if str(v).strip() not in ("", "nan", "none")]
    if not vals:
        return "unknown"
    if sum(v in WORDS for v in vals) >= 0.8 * len(vals):
        return "words"
    nums = []
    for v in vals:
        try:
            nums.append(float(v.rstrip("%")))
        except ValueError:
            pass
    if len(nums) < 0.8 * len(vals):
        return "unknown"
    lo, hi = min(nums), max(nums)
    if hi <= 5:
        return "1-5" if lo >= 1 else "0-5"
    if hi <= 10:
        return "1-10" if lo >= 1 else "0-10"
    if hi <= 100:
        return "0-100"
    return "unknown"


def to_1_5(value, scale: str):
    v = str(value).strip().lower()
    if scale == "words":
        return WORDS.get(v)
    try:
        x = float(v.rstrip("%"))
    except ValueError:
        return None
    lo, hi = SCALES.get(scale, (1, 5))
    return round(1 + 4 * (x - lo) / (hi - lo), 2) if hi > lo else None


def build_intake():
    def parse(state: IntakeState, config) -> IntakeState:
        rows = state["rows"][:MAX_ROWS]
        warnings = [] if len(state["rows"]) <= MAX_ROWS else [f"only the first {MAX_ROWS} rows are used"]
        return {"rows": rows, "warnings": warnings,
                "trace": _add_trace(state, "parse", f"{len(state['columns'])} columns, {len(rows)} rows")}

    def map_rules(state: IntakeState, config) -> IntakeState:
        cols = state["columns"]
        normed = {c: _norm(c) for c in cols}
        mapping, method, used = {}, {}, set()
        for target in TARGETS:  # exact canonical names first
            for c in cols:
                if c not in used and normed[c] in (_norm(target), _norm(target.replace("_skills", ""))):
                    mapping[target], method[target] = c, "exact"
                    used.add(c)
                    break
        for target in TARGETS:  # then aliases (whole words)
            if target in mapping:
                continue
            for c in cols:
                if c in used:
                    continue
                # whole-word alias match; a plural "s" is allowed ("Dashboards", "Stats")
                if any(re.search(rf"(?<![a-z]){re.escape(a)}s?(?![a-z])", normed[c]) for a in ALIASES[target]):
                    mapping[target], method[target] = c, "alias"
                    used.add(c)
                    break
        embedder = config["configurable"].get("embedder")
        left = [t for t in SKILLS if t not in mapping]
        free = [c for c in cols if c not in used]
        if embedder is not None and left and free:
            import numpy as np

            tv = embedder.embed_queries([DESCRIPTIONS[t] for t in left])
            cv = embedder.embed_documents([normed[c] for c in free])
            sims = np.asarray(tv) @ np.asarray(cv).T
            for i, t in sorted(enumerate(left), key=lambda x: -sims[x[0]].max()):
                j = int(np.argmax(sims[i]))
                if sims[i, j] >= EMBED_MIN and free[j] not in used:
                    mapping[t], method[t] = free[j], f"meaning ({sims[i, j]:.2f})"
                    used.add(free[j])
        n = sum(1 for t in SKILLS if t in mapping)
        return {"mapping": mapping, "method": method,
                "trace": _add_trace(state, "map_rules", f"{n}/5 skill areas mapped by names and meaning")}

    def map_model(state: IntakeState, config) -> IntakeState:
        mapping, method = dict(state["mapping"]), dict(state["method"])
        left = [t for t in SKILLS if t not in mapping]
        free = [c for c in state["columns"] if c not in mapping.values()]
        if not left or not free:
            return {"trace": _add_trace(state, "map_model", "nothing left for the model")}
        chooser: Chooser = config["configurable"]["chooser"]
        options = tuple(free) + ("none",)
        fields = {t: (Literal[options], Field(description=f"column holding {TARGET_LABEL[t]} scores, or none"))
                  for t in left}
        Schema = create_model("ColumnChoice", **fields)
        sample = {c: [r.get(c) for r in state["rows"][:3]] for c in free}
        t0 = time.perf_counter()
        pick = chooser("Map spreadsheet columns to skill areas. Only choose a column whose name or values clearly "
                       "hold that skill's rating; otherwise answer none.", f"Columns and sample values: {sample}",
                       Schema)
        ms = round((time.perf_counter() - t0) * 1000)
        if pick is None:
            return {"trace": _add_trace(state, "map_model", "local model unavailable; left for the human", ms=ms)}
        took = []
        for t in left:
            c = getattr(pick, t)
            if c != "none" and c not in mapping.values():
                mapping[t], method[t] = c, "model (please check)"
                took.append(f"{TARGET_LABEL[t]} <- {c}")
        return {"mapping": mapping, "method": method,
                "trace": _add_trace(state, "map_model", "model proposed: " + ("; ".join(took) or "nothing"), ms=ms)}

    def scales(state: IntakeState, config) -> IntakeState:
        sc, warnings = {}, list(state.get("warnings", []))
        for t in SKILLS:
            c = state["mapping"].get(t)
            if c:
                sc[t] = detect_scale([r.get(c) for r in state["rows"]], c)
                if sc[t] == "unknown":
                    warnings.append(f"could not tell the scale of '{c}'; choose one")
                elif sc[t] != "1-5":
                    warnings.append(f"'{c}' looks like a {sc[t]} scale; it will be rescaled to 1-5")
        missing = [TARGET_LABEL[t] for t in SKILLS if not state["mapping"].get(t)]
        if missing:
            warnings.append("no column found for: " + ", ".join(missing))
        return {"scales": sc, "warnings": warnings,
                "trace": _add_trace(state, "scales", ", ".join(f"{TARGET_LABEL[t]}: {s}" for t, s in sc.items()))}

    def review(state: IntakeState, config) -> IntakeState:
        proposal = {
            "filename": state.get("filename"),
            "rows": len(state["rows"]),
            "columns": state["columns"],
            "targets": [{"target": t, "label": TARGET_LABEL[t], "source": state["mapping"].get(t),
                         "method": state["method"].get(t, "none"), "scale": state["scales"].get(t),
                         "sample": [r.get(state["mapping"][t]) for r in state["rows"][:3]] if state["mapping"].get(t)
                         else []} for t in TARGETS],
            "warnings": state.get("warnings", []),
            "scale_options": [*SCALES, "words"],
            "trace": state.get("trace", []),
        }
        decision = interrupt(proposal)  # pauses here until the user approves or edits
        return {"proposal": proposal, "decision": decision,
                "trace": _add_trace(state, "review", "approved by user" if decision.get("approve") else "cancelled")}

    def apply(state: IntakeState, config) -> IntakeState:
        d = state["decision"]
        if not d.get("approve"):
            return {"result": {"cancelled": True}}
        mapping = {**state["mapping"], **(d.get("mapping") or {})}
        scale = {**state["scales"], **(d.get("scales") or {})}
        cols = set(state["columns"])
        bad = [v for v in mapping.values() if v and v not in cols]
        missing = [TARGET_LABEL[t] for t in SKILLS if not mapping.get(t)]
        if bad or missing:
            raise ValueError(("unknown columns: " + ", ".join(bad) + ". ") * bool(bad)
                             + ("every skill area needs a column; missing: " + ", ".join(missing)) * bool(missing))
        dropped = 0
        records = []
        for i, r in enumerate(state["rows"]):
            rec = {"id": str(r.get(mapping["id"])) if mapping.get("id") else f"L{i + 1}"}
            for t in SKILLS:
                rec[t] = to_1_5(r.get(mapping[t]), scale.get(t, "1-5"))
            if any(rec[t] is None for t in SKILLS):
                dropped += 1
            records.append(rec)
        profile = config["configurable"]["profile_fn"](records)
        return {"result": {"profile": profile, "mapping": mapping, "scales": scale, "unreadable_rows": dropped},
                "trace": _add_trace(state, "apply", f"{len(records)} rows converted; profile computed")}

    g = StateGraph(IntakeState)
    for name, fn in (("parse", parse), ("map_rules", map_rules), ("map_model", map_model), ("scales", scales),
                     ("review", review), ("apply", apply)):
        g.add_node(name, fn)
    g.add_edge(START, "parse")
    g.add_edge("parse", "map_rules")
    g.add_edge("map_rules", "map_model")
    g.add_edge("map_model", "scales")
    g.add_edge("scales", "review")
    g.add_edge("review", "apply")
    g.add_edge("apply", END)
    return g.compile(checkpointer=SAVER)


INTAKE = build_intake()


def new_thread() -> str:
    return uuid.uuid4().hex
