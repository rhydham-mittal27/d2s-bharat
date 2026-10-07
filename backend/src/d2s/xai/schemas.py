"""One explanation format shared by every module, so the UI, audit log and (later) the SLM
can treat all explanations the same way."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class Factor(BaseModel):
    """One driver of an outcome."""
    name: str
    value: float | str | None = None
    contribution: float | None = None  # signed effect on the outcome, in `unit`
    unit: str = ""
    source: str = ""  # which computation produced it, e.g. "logit.shapley", "cp-sat.slack"
    note: str = ""


class Explanation(BaseModel):
    subject: str  # e.g. "learner", "plan", "course:ml", "tag:Core Java"
    question: Literal["why", "why_not", "what_if", "how_sure", "is_feasible"]
    summary: str  # one or two plain-language sentences (template-generated)
    factors: list[Factor] = Field(default_factory=list)
    counterfactuals: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: float | None = None
    method: str  # how it was computed, for model cards and audits
    details: dict[str, Any] = Field(default_factory=dict)  # structured data for charts
