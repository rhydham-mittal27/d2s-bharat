"""Request body shared by the planning and agent routes, and the cohort-gap loader behind it."""

import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Field

from d2s.api.db import DB
from d2s.api.state import STATE
from d2s.db import CohortRepository, session_scope
from d2s.services import planner


class PlanBody(BaseModel):
    plan: planner.PlanRequest = planner.PlanRequest()
    gaps: dict[str, int] | None = Field(None, description="area -> learners short; default: SAS JDS cohort")
    cohort_id: str | None = Field(None, description="use a saved cohort's gaps (database); `gaps` overrides areas")
    label: str | None = Field(None, max_length=200, description="name for the saved plan run")


def cohort_gaps(cohort_id: str, owner: str | None) -> dict[str, int]:
    if not DB.enabled:
        raise HTTPException(503, "cohort_id needs the database; set D2S_DATABASE_URL")
    with session_scope(DB.factory) as s:
        return dict(CohortRepository(s, owner).get(cohort_id).gaps)


def plan_gaps(body: PlanBody, owner: str | None) -> pd.DataFrame:
    if body.gaps is None and body.cohort_id is None:
        return STATE.sample_gaps
    overrides = {**(cohort_gaps(body.cohort_id, owner) if body.cohort_id else {}), **(body.gaps or {})}
    base = STATE.sample_gaps.set_index("area")
    unknown = set(overrides) - set(base.index)
    if unknown:
        raise HTTPException(422, f"unknown skill areas: {sorted(unknown)}")
    g = base.copy()
    for a, n in overrides.items():
        if n < 0:
            raise HTTPException(422, f"learners_short must be >= 0 for {a}")
        g.loc[a, "learners_short"] = n
    return g.reset_index()
