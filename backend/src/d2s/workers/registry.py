"""Task registry shared by every queue backend.

A task is ``async def fn(ctx: TaskContext, payload: dict) -> dict``. Its payload and result
must be JSON-serialisable so the same task runs inline or on an arq worker unchanged.
"""

import hashlib
import json
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from d2s.workers.progress import Reporter


class TransientError(Exception):
    """Raise for failures worth retrying (network, Redis, model download). Everything else fails fast."""


@dataclass
class TaskContext:
    job_id: str
    reporter: Reporter
    attempt: int = 1
    extras: dict[str, Any] = field(default_factory=dict)


TaskFn = Callable[[TaskContext, dict], Awaitable[dict]]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    fn: TaskFn
    timeout_s: int
    max_tries: int


TASKS: dict[str, TaskSpec] = {}


def task(name: str, timeout_s: int = 300, max_tries: int = 3):
    def register(fn: TaskFn) -> TaskFn:
        if name in TASKS:
            raise ValueError(f"task {name} registered twice")
        TASKS[name] = TaskSpec(name, fn, timeout_s, max_tries)
        return fn

    return register


def job_id_for(task_name: str, payload: dict) -> str:
    """Idempotent id: the same task + payload maps to the same job (no double runs on re-click)."""
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return f"{task_name}:{digest[:20]}"


def backoff_s(attempt: int, base: float = 2.0, cap: float = 60.0) -> float:
    """Exponential backoff with full jitter."""
    return random.uniform(0, min(cap, base * 2 ** (attempt - 1)))
