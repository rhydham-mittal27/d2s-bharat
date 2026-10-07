import os
import tempfile

# Tests never touch the real database from backend/.env: env vars beat the .env file, and an empty URL
# means "no database". DB tests create their own in-memory SQLite.
os.environ["D2S_DATABASE_URL"] = ""
# Auth is exercised in test_auth.py; elsewhere it is off. A fixed test secret, never the real one.
os.environ["D2S_AUTH_REQUIRED"] = "false"
os.environ["D2S_JWT_SECRET"] = "test-secret-" + "x" * 40
# ChromaDB in a throwaway folder, never the real dataset/chroma
os.environ["D2S_CHROMA_PATH"] = tempfile.mkdtemp(prefix="d2s-chroma-")
# The local agent model is never called in tests; agents get a scripted chooser instead.
os.environ["D2S_AGENT_LLM_ENABLED"] = "false"

import os

import pytest

os.environ.setdefault("D2S_EMBEDDING_BACKEND", "hashing")
os.environ.setdefault("D2S_QUEUE_BACKEND", "inline")
os.environ.setdefault("D2S_SOLVER_WORKERS", "4")

from d2s.engine import Constraints, Course, PlanningProblem, SkillGap  # noqa: E402


@pytest.fixture
def problem() -> PlanningProblem:
    """Small institution: 4 skills, 6 candidate courses, one prerequisite chain, one exclusive pair."""
    return PlanningProblem(
        skills=[
            SkillGap(skill_id="python", learners_short=60, demand_weight=9.0),
            SkillGap(skill_id="sql", learners_short=40, demand_weight=6.0),
            SkillGap(skill_id="ml", learners_short=30, demand_weight=10.0),
            SkillGap(skill_id="excel", learners_short=50, demand_weight=2.0),
        ],
        courses=[
            Course(id="py101", fixed_cost=50_000, cost_per_seat=1_000, min_batch=10, max_seats=60,
                   trainer_hours=40, coverage={"python": 100}),
            Course(id="sql101", fixed_cost=30_000, cost_per_seat=800, min_batch=10, max_seats=40,
                   trainer_hours=30, coverage={"sql": 100}),
            Course(id="ml201", fixed_cost=80_000, cost_per_seat=2_000, min_batch=10, max_seats=30,
                   trainer_hours=60, coverage={"ml": 100, "python": 20}, prerequisites=["py101"]),
            Course(id="excel_basic", fixed_cost=10_000, cost_per_seat=300, min_batch=10, max_seats=50,
                   trainer_hours=20, coverage={"excel": 100}, exclusive_group="spreadsheets"),
            Course(id="excel_adv", fixed_cost=25_000, cost_per_seat=600, min_batch=10, max_seats=50,
                   trainer_hours=30, coverage={"excel": 100, "sql": 20}, exclusive_group="spreadsheets"),
            Course(id="da_bootcamp", fixed_cost=120_000, cost_per_seat=1_500, min_batch=15, max_seats=40,
                   trainer_hours=80, coverage={"python": 50, "sql": 60, "ml": 30}),
        ],
        constraints=Constraints(budget=400_000, trainer_hours=200),
    )
