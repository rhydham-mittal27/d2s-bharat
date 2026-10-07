"""RQ4: turn RQ1-RQ3 evidence into a budget-constrained training plan (OR-Tools CP-SAT engine).

Data-grounded inputs
  * skill gaps  : JDS juniors scoring below the high-hike group's median in each skill area
  * gap values  : evidence score from market demand (RQ1), market pay premium (RQ1, adjusted
                  ordinal logit) and junior salary-hike link (RQ2), each scaled to its maximum
  * uncertainty : the same score at the CI lower / upper bounds, and alternative weightings
Assumption inputs (editable CSV): course catalogue, budget, trainer hours.
"""

import numpy as np
import pandas as pd

from d2s.engine import Constraints, Course, PlanningProblem, SkillGap

AREAS = ["coding_skills", "maths_stats_skills", "ai_and_ml_skills", "big_data_skills",
         "dashboard_and_storytelling_skills"]
LABEL = {"coding_skills": "Coding", "maths_stats_skills": "Maths & stats", "ai_and_ml_skills": "AI & ML",
         "big_data_skills": "Big data", "dashboard_and_storytelling_skills": "Dashboards & storytelling"}

# Weighting schemes over the three evidence components (demand, market pay, junior-hike link).
SCHEMES = {
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "market-led": (0.5, 0.5, 0.0),
    "outcome-led": (0.0, 0.0, 1.0),
}


def cohort_gaps(jds: pd.DataFrame, target_col: str = "salary_hike_high_or_low") -> pd.DataFrame:
    """Juniors below the high-hike group's median score, per skill area (the 'gap' to close)."""
    hi = jds[jds[target_col] == 1]
    rows = []
    for a in AREAS:
        target = float(hi[a].median())
        short = int((jds[a] < target).sum())
        rows.append({"area": a, "target_score": target, "learners_short": short,
                     "share_short": short / len(jds), "cohort_mean": float(jds[a].mean())})
    return pd.DataFrame(rows)


def evidence_weights(tri: pd.DataFrame, scheme: str = "balanced", bound: str = "point") -> pd.DataFrame:
    """Value per learner of closing each area's gap.

    bound: 'point' | 'low' | 'high' -> use point estimates or CI bounds of every component.
    Components are scaled to their maximum across areas (ORs on the log scale), then mixed by
    the scheme's weights. Rescaled so the top area = 100 (only relative values matter).
    """
    t = tri.set_index("area").loc[AREAS]
    col = {"point": ("demand_share", "odds_ratio", "rq2_odds_ratio"),
           "low": ("demand_ci_low", "or_ci_low", "rq2_or_ci_low"),
           "high": ("demand_ci_high", "or_ci_high", "rq2_or_ci_high")}[bound]
    demand = t[col[0]].to_numpy()
    market = np.log(np.clip(t[col[1]].to_numpy(), 1e-9, None)).clip(min=0)  # no value for OR <= 1
    hike = np.log(np.clip(t[col[2]].to_numpy(), 1e-9, None)).clip(min=0)

    def scaled(v):
        return v / v.max() if v.max() > 0 else v

    wd, wm, wh = SCHEMES[scheme]
    score = wd * scaled(demand) + wm * scaled(market) + wh * scaled(hike)
    score = 100 * score / score.max() if score.max() > 0 else score
    return pd.DataFrame({"area": AREAS, "demand_c": scaled(demand), "market_c": scaled(market),
                         "hike_c": scaled(hike), "weight": score})


def load_catalogue(source) -> list[Course]:
    """source: path to the catalogue CSV, or an already-loaded list of courses (e.g. from the database)."""
    if isinstance(source, list):
        return list(source)
    df = pd.read_csv(source).fillna("")
    courses = []
    for r in df.to_dict("records"):
        cov = {a: int(r[f"cov_{a}"]) for a in AREAS if r.get(f"cov_{a}", "") not in ("", 0)}
        courses.append(Course(
            id=r["id"], name=r["name"], fixed_cost=int(r["fixed_cost"]), cost_per_seat=int(r["cost_per_seat"]),
            min_batch=int(r["min_batch"]), max_seats=int(r["max_seats"]), trainer_hours=int(r["trainer_hours"]),
            coverage=cov, prerequisites=[p for p in str(r["prerequisites"]).split(";") if p],
            exclusive_group=r["exclusive_group"] or None, mandatory=str(r["mandatory"]).lower() == "true",
        ))
    return courses


def catalogue_frame(courses: list[Course]) -> pd.DataFrame:
    """Courses in the catalogue CSV's column layout (cov_<area> columns, ';'-joined prerequisites)."""
    rows = []
    for c in courses:
        r = {"id": c.id, "name": c.name, "fixed_cost": c.fixed_cost, "cost_per_seat": c.cost_per_seat,
             "min_batch": c.min_batch, "max_seats": c.max_seats, "trainer_hours": c.trainer_hours,
             "prerequisites": ";".join(c.prerequisites), "exclusive_group": c.exclusive_group or "",
             "mandatory": c.mandatory}
        r.update({f"cov_{a}": c.coverage.get(a) for a in AREAS})
        rows.append(r)
    return pd.DataFrame(rows)


def build_problem(gaps: pd.DataFrame, weights: pd.DataFrame, courses: list[Course], budget: int,
                  trainer_hours: int | None) -> PlanningProblem:
    w = weights.set_index("area")["weight"]
    skills = [SkillGap(skill_id=r.area, learners_short=int(r.learners_short), demand_weight=float(w[r.area]))
              for r in gaps.itertuples()]
    return PlanningProblem(skills=skills, courses=courses,
                           constraints=Constraints(budget=budget, trainer_hours=trainer_hours))
