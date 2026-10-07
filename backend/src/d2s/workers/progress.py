"""Job progress events: published by tasks, streamed to the UI (SSE) by the API layer.

Two sinks share one protocol:
* ``MemoryProgressSink``: in-process (dev/tests, InlineQueue)
* ``RedisProgressSink``: pub/sub channel ``d2s:progress:{job_id}`` + latest-state key
"""

import asyncio
import json
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any, Protocol

from pydantic import BaseModel, Field

TERMINAL_STAGES = {"complete", "failed"}


class JobProgress(BaseModel):
    job_id: str
    stage: str  # queued | running | <task-specific> | complete | failed
    pct: float | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)

    @property
    def terminal(self) -> bool:
        return self.stage in TERMINAL_STAGES


class ProgressSink(Protocol):
    async def publish(self, event: JobProgress) -> None: ...
    async def latest(self, job_id: str) -> JobProgress | None: ...
    def subscribe(self, job_id: str) -> AsyncIterator[JobProgress]: ...


class MemoryProgressSink:
    def __init__(self) -> None:
        self._latest: dict[str, JobProgress] = {}
        self._history: dict[str, list[JobProgress]] = defaultdict(list)
        self._subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)

    async def publish(self, event: JobProgress) -> None:
        self._latest[event.job_id] = event
        self._history[event.job_id].append(event)
        for q in self._subscribers[event.job_id]:
            q.put_nowait(event)

    async def latest(self, job_id: str) -> JobProgress | None:
        return self._latest.get(job_id)

    def history(self, job_id: str) -> list[JobProgress]:
        return list(self._history[job_id])

    async def subscribe(self, job_id: str) -> AsyncIterator[JobProgress]:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[job_id].append(q)
        try:
            if (last := self._latest.get(job_id)) is not None:
                yield last
                if last.terminal:
                    return
            while True:
                event = await q.get()
                yield event
                if event.terminal:
                    return
        finally:
            self._subscribers[job_id].remove(q)


class RedisProgressSink:
    def __init__(self, redis, ttl_s: int = 86_400, prefix: str = "d2s:progress:"):
        self._r = redis
        self._ttl = ttl_s
        self._prefix = prefix

    async def publish(self, event: JobProgress) -> None:
        key = self._prefix + event.job_id
        payload = event.model_dump_json()
        await self._r.set(key + ":latest", payload, ex=self._ttl)
        await self._r.publish(key, payload)

    async def latest(self, job_id: str) -> JobProgress | None:
        raw = await self._r.get(self._prefix + job_id + ":latest")
        return JobProgress.model_validate_json(raw) if raw else None

    async def subscribe(self, job_id: str) -> AsyncIterator[JobProgress]:
        pubsub = self._r.pubsub()
        await pubsub.subscribe(self._prefix + job_id)
        try:
            if (last := await self.latest(job_id)) is not None:
                yield last
                if last.terminal:
                    return
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                event = JobProgress.model_validate(json.loads(msg["data"]))
                yield event
                if event.terminal:
                    return
        finally:
            await pubsub.unsubscribe()
            await pubsub.aclose()


class Reporter:
    """Bound to one job. ``report`` is async; ``threadsafe`` is for code running in worker threads."""

    def __init__(self, sink: ProgressSink, job_id: str, loop: asyncio.AbstractEventLoop | None = None):
        self.sink = sink
        self.job_id = job_id
        self._loop = loop or asyncio.get_running_loop()

    async def report(self, stage: str, pct: float | None = None, message: str = "", **data) -> None:
        await self.sink.publish(
            JobProgress(job_id=self.job_id, stage=stage, pct=pct, message=message, data=data)
        )

    def threadsafe(self, stage: str, pct: float | None = None, message: str = "", **data) -> None:
        asyncio.run_coroutine_threadsafe(self.report(stage, pct, message, **data), self._loop)
