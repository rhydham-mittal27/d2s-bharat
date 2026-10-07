"""Gap Map: profile an institution's cohort (JDS format) against the high-hike benchmark.

Input : a table with the five JDS skill columns (1-5), optional `id` and optional outcome.
Output: validation report, per-learner hike probability and gaps, per-area gap counts (the RQ4
        optimiser input), and for each area the job roles and courses linked to it.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel

from d2s.models.hike import LABELS, SCALE, SKILLS, HikeModel


class RowIssue(BaseModel):
    row: int
    problem: str


class AreaGap(BaseModel):
    area: str
    label: str
    target_score: float
    learners_short: int
    share_short: float
    mean_shortfall: float  # average points below target among those short
    linked_roles: list[dict]  # role families/titles that demand this area most
    linked_courses: list[dict]  # catalogue courses that cover this area


class LearnerResult(BaseModel):
    id: str
    probability: float
    band: str  # "at risk" | "developing" | "on track"
    gaps: dict[str, float]  # area -> points below target (0 if none)
    top_improvement: str | None


class CohortProfile(BaseModel):
    n_learners: int
    n_rejected: int
    issues: list[RowIssue]
    mean_probability: float
    bands: dict[str, int]
    areas: list[AreaGap]
    learners: list[LearnerResult]
    cohort_id: str | None = None  # set when the cohort was saved to the database


ALIASES = {"maths-stats_skills": "maths_stats_skills"}


def _band(p: float) -> str:
    return "at risk" if p < 0.4 else "developing" if p < 0.7 else "on track"


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[RowIssue]]:
    df = df.rename(columns={c: ALIASES.get(c.strip(), c.strip()) for c in df.columns})
    missing = [s for s in SKILLS if s not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if "id" not in df.columns:
        df = df.assign(id=[f"L{i + 1}" for i in range(len(df))])
    issues, keep = [], []
    for i, row in df.iterrows():
        vals = pd.to_numeric(row[SKILLS], errors="coerce")
        if vals.isna().any():
            issues.append(RowIssue(row=int(i) + 2, problem="non-numeric or missing skill score"))
        elif ((vals < SCALE[0]) | (vals > SCALE[1])).any():
            issues.append(RowIssue(row=int(i) + 2, problem=f"score outside {SCALE[0]:g}-{SCALE[1]:g}"))
        else:
            keep.append(i)
    clean = df.loc[keep].copy()
    clean[SKILLS] = clean[SKILLS].apply(pd.to_numeric)
    clean["id"] = clean["id"].astype(str)
    return clean, issues


def profile(df: pd.DataFrame, model: HikeModel, area_roles: pd.DataFrame | None = None,
            catalogue: pd.DataFrame | None = None, max_learners: int = 500) -> CohortProfile:
    clean, issues = validate(df)
    if clean.empty:
        raise ValueError("no valid learner rows")
    X = clean[SKILLS].to_numpy(dtype=float)
    probs = model.predict_proba(X)
    targets = np.array([model.high_median[s] for s in SKILLS])
    shortfall = np.clip(targets - X, 0, None)

    areas = []
    for j, s in enumerate(SKILLS):
        short = shortfall[:, j] > 0
        roles = []
        if area_roles is not None:
            r = area_roles[area_roles["area"] == s].nlargest(4, "share")
            roles = [{"role": x.role, "share_of_postings": round(float(x.share), 4)} for x in r.itertuples()]
        courses = []
        if catalogue is not None and f"cov_{s}" in catalogue:
            c = catalogue[pd.to_numeric(catalogue[f"cov_{s}"], errors="coerce").fillna(0) > 0]
            courses = [{"id": x["id"], "name": x["name"], "coverage_pct": int(x[f"cov_{s}"])} for _, x in c.iterrows()]
        areas.append(AreaGap(area=s, label=LABELS[s], target_score=float(targets[j]),
                             learners_short=int(short.sum()), share_short=float(short.mean()),
                             mean_shortfall=float(shortfall[short, j].mean()) if short.any() else 0.0,
                             linked_roles=roles, linked_courses=courses))

    learners = []
    for k, (rid, p) in enumerate(zip(clean["id"], probs, strict=True)):
        if k >= max_learners:
            break
        gaps = {s: float(round(shortfall[k, j], 2)) for j, s in enumerate(SKILLS)}
        # largest probability gain from raising one skill to target
        best, best_gain = None, 0.0
        for j, s in enumerate(SKILLS):
            if gaps[s] > 0:
                x2 = X[k].copy()
                x2[j] = targets[j]
                g = float(model.predict_proba(x2)[0] - p)
                if g > best_gain:
                    best, best_gain = LABELS[s], g
        learners.append(LearnerResult(id=rid, probability=float(p), band=_band(float(p)), gaps=gaps,
                                      top_improvement=best))
    bands = pd.Series([_band(float(p)) for p in probs]).value_counts().to_dict()
    return CohortProfile(n_learners=len(clean), n_rejected=len(issues), issues=issues[:50],
                         mean_probability=float(probs.mean()), bands={k: int(v) for k, v in bands.items()},
                         areas=areas, learners=learners)


def gaps_frame(profile_: CohortProfile) -> pd.DataFrame:
    """The cohort's gaps in the shape the RQ4 optimiser expects."""
    return pd.DataFrame([{"area": a.area, "target_score": a.target_score, "learners_short": a.learners_short}
                         for a in profile_.areas])
