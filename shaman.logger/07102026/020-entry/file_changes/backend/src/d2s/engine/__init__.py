"""Decision engine: picks courses and seats that close the most demand-weighted skill gap."""

from d2s.engine.explain import WhyNot, why_not
from d2s.engine.pareto import ParetoPoint, frontier
from d2s.engine.schemas import (
    Constraints,
    Course,
    Plan,
    PlanningProblem,
    ProgressEvent,
    SkillGap,
    SolverOptions,
    SolveStatus,
)
from d2s.engine.sensitivity import SensitivityReport, analyze, lp_shadow_prices
from d2s.engine.solver import solve

__all__ = [
    "Constraints", "Course", "ParetoPoint", "Plan", "PlanningProblem", "ProgressEvent",
    "SensitivityReport", "SkillGap", "SolveStatus", "SolverOptions", "WhyNot",
    "analyze", "frontier", "lp_shadow_prices", "solve", "why_not",
]
