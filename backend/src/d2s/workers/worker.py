"""arq worker entrypoint:  ``uv run arq d2s.workers.worker.WorkerSettings``

Each registered task is wrapped so it gets a TaskContext, publishes progress to Redis,
and turns TransientError into arq.Retry with exponential backoff.
"""

from arq import Retry, cron, func
from arq.connections import RedisSettings

from d2s.config import get_settings
from d2s.workers import tasks as _tasks  # noqa: F401  (registers tasks)
from d2s.workers.progress import RedisProgressSink
from d2s.workers.queue import run_task
from d2s.workers.registry import TASKS, TaskSpec, TransientError, backoff_s


def _wrap(spec: TaskSpec):
    async def handler(ctx: dict, payload: dict) -> dict:
        try:
            return await run_task(spec.name, payload, ctx["job_id"], ctx["progress"], ctx["job_try"])
        except TransientError as exc:
            if ctx["job_try"] >= spec.max_tries:
                raise
            raise Retry(defer=backoff_s(ctx["job_try"])) from exc

    handler.__qualname__ = handler.__name__ = spec.name
    # run_task enforces spec.timeout_s itself; give arq a little slack on top.
    return func(handler, name=spec.name, timeout=spec.timeout_s + 30, max_tries=spec.max_tries)


async def startup(ctx: dict) -> None:
    ctx["progress"] = RedisProgressSink(ctx["redis"], ttl_s=get_settings().job_result_ttl_s)


async def nightly_skill_index(ctx: dict) -> None:
    """Re-embed the taxonomy if it changed (no-op when the cached index is current)."""
    await run_task("build_skill_index", {}, "cron:build_skill_index", ctx["progress"])


class WorkerSettings:
    functions = [_wrap(spec) for spec in TASKS.values()]
    cron_jobs = [cron(nightly_skill_index, hour=2, minute=0, run_at_startup=False)]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    keep_result = get_settings().job_result_ttl_s
    max_jobs = 4  # CP-SAT already uses multiple threads per job
    health_check_interval = 30
