"""Decision Brief (T0, template-generated) with a faithfulness check.

The brief is assembled from engine and evidence numbers only. `check_numbers` then verifies that
every number appearing in the text can be traced to an allowed value: the same guard will apply
to briefs written later by the team's own SLM (which may paraphrase but must not invent numbers).
"""

import re
from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel

from d2s.analysis.plan import LABEL
from d2s.services.evidence import CourseTrail
from d2s.services.planner import PlanResult

_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


class BriefSection(BaseModel):
    heading: str
    paragraphs: list[str]


class Brief(BaseModel):
    title: str
    sections: list[BriefSection]
    numbers_checked: int
    unverified_numbers: list[str]
    faithful: bool

    def markdown(self) -> str:
        out = [f"# {self.title}"]
        for s in self.sections:
            out.append(f"\n## {s.heading}")
            out.extend(s.paragraphs)
        return "\n\n".join(out)


def _half_up(v: float, d: int) -> str:
    """Round half-up from the decimal representation (70500 / 1e5 -> '0.71', not float's '0.70')."""
    return str(Decimal(repr(v)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def lakh(x: float) -> str:
    return f"₹{_half_up(x / 1e5, 2)} lakh"


def _allowed(values: list[float]) -> set[str]:
    allowed = set()
    for v in values:
        for d in (0, 1, 2):
            for s in (f"{v:.{d}f}", _half_up(v, d)):
                allowed.add(s)
                allowed.add(f"{float(s):,.{d}f}")
    return allowed


def check_numbers(text: str, values: list[float]) -> tuple[int, list[str]]:
    allowed = _allowed(values)
    found = [m.group(0) for m in _NUM.finditer(text)]
    bad = [n for n in found if n.replace(",", "") not in {a.replace(",", "") for a in allowed}]
    return len(found), sorted(set(bad))


def write(result: PlanResult, trails: list[CourseTrail], cohort_size: int | None = None) -> Brief:
    p, b, r = result.plan, result.binding, result.request
    vals: list[float] = [p.total_cost / 1e5, r.budget / 1e5, p.total_trainer_hours, p.total_seats,
                         len(p.courses), b.budget_slack / 1e5, b.step_hours, b.step_budget / 1e5]
    if r.trainer_hours is not None:
        vals.append(r.trainer_hours)
    if cohort_size:
        vals.append(cohort_size)

    def num(v: float) -> float:
        vals.append(v)
        return v

    head = (f"Run {len(p.courses)} courses for {p.total_seats} seats at {lakh(p.total_cost)} of a "
            f"{lakh(r.budget)} budget, using {p.total_trainer_hours} trainer-hours"
            + (f" of {r.trainer_hours}." if r.trainer_hours is not None else "."))
    rec = [head, "The plan is " + ("proven optimal" if p.status.value == "optimal" else "feasible but not proven optimal")
           + " for the stated evidence and cost assumptions."]
    rows = []
    for c in sorted(p.courses, key=lambda x: -x.seats):
        rows.append(f"- {result.course_names[c.course_id]}: {c.seats} seats, {lakh(c.cost)}, {c.trainer_hours} trainer-hours")
    gaps = []
    for s in sorted(p.skills, key=lambda s: -s.closure_pct):
        gaps.append(f"- {LABEL[s.skill_id]}: {num(round(s.closure_pct)):.0f}% of the gap closed "
                    f"({num(round(s.learners_closed)):.0f} of {num(s.learners_short):.0f} learners)")
    why = []
    for t in sorted(trails, key=lambda t: -t.impact_lost_if_removed_pct):
        a = max(t.areas, key=lambda a: a.learners_closed_by_course)
        why.append(f"- {t.name}: removing it loses {num(t.impact_lost_if_removed_pct):.1f}% of total impact. "
                   f"It mainly serves {a.label}, which is in {num(round(a.market_demand_share * 100, 1)):.1f}% of "
                   f"data-role postings, has a market pay odds ratio of {num(round(a.market_pay_or, 2)):.2f} and a "
                   f"junior salary-hike odds ratio of {num(round(a.junior_hike_or, 2)):.2f}.")
    cons = [f"{b.verdict[0].upper()}{b.verdict[1:]}.",
            f"Adding {lakh(b.step_budget)} alone changes impact by {num(round(b.gain_extra_budget)):.0f}; "
            f"adding {b.step_hours} trainer-hours alone by {num(round(b.gain_extra_hours)):.0f}; "
            f"both together by {num(round(b.gain_both)):.0f}."]
    excl = [f"- {w['name']}: {w['reason']}" for w in result.why_not if w.get("objective_delta_pct") is not None]
    for w in result.why_not:
        for k in ("objective_delta", "objective_delta_pct"):
            if w.get(k) is not None:
                vals += [w[k], abs(w[k])]
    caveats = ["Market and outcome evidence are associations from SAS sample data, not causal effects. "
               "Course costs, seats, trainer-hours and coverage are documented assumptions; change them "
               "in the catalogue and re-run."]
    sections = [BriefSection(heading="Recommendation", paragraphs=rec + rows),
                BriefSection(heading="Expected effect on skill gaps", paragraphs=gaps),
                BriefSection(heading="Why these courses", paragraphs=why),
                BriefSection(heading="Constraints and next steps", paragraphs=cons),
                BriefSection(heading="Courses not chosen", paragraphs=excl or ["None."]),
                BriefSection(heading="Caveats", paragraphs=caveats)]
    for c in p.courses:
        vals += [c.seats, c.cost / 1e5, c.trainer_hours]
    brief = Brief(title="Decision Brief: what to teach first", sections=sections, numbers_checked=0,
                  unverified_numbers=[], faithful=True)
    n, bad = check_numbers(brief.markdown(), vals)
    brief.numbers_checked, brief.unverified_numbers, brief.faithful = n, bad, not bad
    return brief
