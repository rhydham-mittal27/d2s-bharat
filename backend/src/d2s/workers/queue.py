"""JobQueue protocol and its backends. App code depends on the protocol, never on arq directly,
so the queue library can be swapped (arq is in maintenance-only mode; SAQ/Taskiq are drop-ins).
"""

import asyncio
from enum import StrEnum
from typing import Any, Protocol

from d2s.config import get_settings
from d2s.workers import tasks as _tasks  # noqa: F401  (registers tasks)
from d2s.workers.progress import MemoryProgressSink, ProgressSink, Reporter
from d2s.workers.registry import TASKS, TaskContext, TransientError, backoff_s, job_id_for


class JobStatus(StrEnum):
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    NOT_FOUND = "not_found"


class JobFailed(Exception):
    pass


class JobQueue(Protocol):
    progress: ProgressSink

    async def enqueue(self, task: str, payload: dict, job_id: str | None = None) -> str: ...
    async def status(self, job_id: str) -> JobStatus: ...
    async def result(self, job_id: str, timeout_s: float | None = None) -> dict: ...
    async def close(self) -> None: ...


async def run_task(name: str, payload: dict, job_id: str, sink: ProgressSink, attempt: int = 1) -> dict:
    """Execute a registered task with timeout + start/end progress events (shared by all backends)."""
    spec = TASKS[name]
    reporter = Reporter(sink, job_id)
    await reporter.report("running", 0.0, f"{name} (attempt {attempt})")
    try:
        result = await asyncio.wait_for(spec.fn(TaskContext(job_id, reporter, attempt), payload),
                                        timeout=spec.timeout_s)
    except TransientError:
        raise
    except TimeoutError:
        await reporter.report("failed", None, f"timed out after {spec.timeout_s}s")
        raise
    except Exception as exc:
        await reporter.report("failed", None, f"{type(exc).__name__}: {exc}")
        raise
    await reporter.report("complete", 1.0, "done")
    return result


class InlineQueue:
    """Runs jobs as asyncio tasks in this process. Same semantics as arq, minus durability."""

    def __init__(self, progress: MemoryProgressSink | None = None):
        self.progress = progress or MemoryProgressSink()
        self._jobs: dict[str, asyncio.Task] = {}

    async def enqueue(self, task: str, payload: dict, job_id: str | None = None) -> str:
        if task not in TASKS:
            raise KeyError(f"unknown task {task}")
        jid = job_id or job_id_for(task, payload)
        existing = self._jobs.get(jid)
        if existing and not (existing.done() and existing.exception()):
            return jid  # idempotent: same job already queued/running/done
        await self.progress.publish(_queued(jid))
        self._jobs[jid] = asyncio.create_task(self._run(task, payload, jid))
        return jid

    async def _run(self, task: str, payload: dict, jid: str) -> dict:
        spec = TASKS[task]
        for attempt in range(1, spec.max_tries + 1):
            try:
                return await run_task(task, payload, jid, self.progress, attempt)
            except TransientError as exc:
                if attempt == spec.max_tries:
                    await Reporter(self.progress, jid).report("failed", None, f"gave up: {exc}")
                    raise
                await asyncio.sleep(backoff_s(attempt))
        raise AssertionError("unreachable")

    async def status(self, job_id: str) -> JobStatus:
        t = self._jobs.get(job_id)
        if t is None:
            return JobStatus.NOT_FOUND
        if not t.done():
            latest = await self.progress.latest(job_id)
            return JobStatus.QUEUED if latest and latest.stage == "queued" else JobStatus.IN_PROGRESS
        return JobStatus.FAILED if t.exception() else JobStatus.COMPLETE

    async def result(self, job_id: str, timeout_s: float | None = None) -> dict:
        t = self._jobs.get(job_id)
        if t is None:
            raise KeyError(job_id)
        try:
            return await asyncio.wait_for(asyncio.shield(t), timeout_s)
        except TimeoutError:
            raise
        except Exception as exc:
            raise JobFailed(str(exc)) from exc

    async def close(self) -> None:
        for t in self._jobs.values():
            t.cancel()
        await asyncio.gather(*self._jobs.values(), return_exceptions=True)


class ArqQueue:
    """Redis-backed durable queue; jobs execute on `arq d2s.workers.worker.WorkerSettings`."""

    def __init__(self, pool, progress: ProgressSink):
        self._pool = pool
        self.progress = progress

    @classmethod
    async def connect(cls, redis_url: str | None = None) -> "ArqQueue":
        from arq import create_pool
        from arq.connections import RedisSettings

        from d2s.workers.progress import RedisProgressSink

        s = get_settings()
        pool = await create_pool(RedisSettings.from_dsn(redis_url or s.redis_url))
        return cls(pool, RedisProgressSink(pool, ttl_s=s.job_result_ttl_s))

    async def enqueue(self, task: str, payload: dict, job_id: str | None = None) -> str:
        if task not in TASKS:
            raise KeyError(f"unknown task {task}")
        jid = job_id or job_id_for(task, payload)
        job = await self._pool.enqueue_job(task, payload, _job_id=jid)
        if job is not None:  # None = arq already has this job id (idempotent)
            await self.progress.publish(_queued(jid))
        return jid

    async def status(self, job_id: str) -> JobStatus:
        from arq.jobs import Job
        from arq.jobs import JobStatus as ArqStatus

        job = Job(job_id, self._pool)
        st = await job.status()
        if st is ArqStatus.complete:
            info = await job.result_info()
            return JobStatus.COMPLETE if info and info.success else JobStatus.FAILED
        return {
            ArqStatus.deferred: JobStatus.QUEUED,
            ArqStatus.queued: JobStatus.QUEUED,
            ArqStatus.in_progress: JobStatus.IN_PROGRESS,
        }.get(st, JobStatus.NOT_FOUND)

    async def result(self, job_id: str, timeout_s: float | None = None) -> dict:
        from arq.jobs import Job

        try:
            return await Job(job_id, self._pool).result(timeout=timeout_s)
        except TimeoutError:
            raise
        except Exception as exc:
            raise JobFailed(str(exc)) from exc

    async def close(self) -> None:
        await self._pool.aclose()


def _queued(jid: str):
    from d2s.workers.progress import JobProgress

    return JobProgress(job_id=jid, stage="queued", pct=0.0, message="queued")


async def create_queue() -> JobQueue:
    if get_settings().queue_backend == "arq":
        return await ArqQueue.connect()
    return InlineQueue()
