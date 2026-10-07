import asyncio

import pytest

from d2s.workers.queue import InlineQueue, JobFailed, JobStatus
from d2s.workers.registry import TASKS, TaskContext, TransientError, job_id_for, task


def test_job_id_is_idempotent():
    assert job_id_for("t", {"a": 1, "b": 2}) == job_id_for("t", {"b": 2, "a": 1})
    assert job_id_for("t", {"a": 1}) != job_id_for("t", {"a": 2})


def test_all_core_tasks_registered():
    assert {"optimize_plan", "sensitivity", "pareto_frontier", "why_not",
            "forecast_demand", "build_skill_index", "extract_skills"} <= set(TASKS)


async def test_optimize_plan_job_streams_progress(problem):
    q = InlineQueue()
    payload = {"problem": problem.model_dump(mode="json")}
    jid = await q.enqueue("optimize_plan", payload)
    events = [e async for e in q.progress.subscribe(jid)]
    result = await q.result(jid, timeout_s=60)

    assert result["status"] == "optimal"
    assert await q.status(jid) is JobStatus.COMPLETE
    stages = [e.stage for e in events]
    assert stages[-1] == "complete"
    history = [e.stage for e in q.progress.history(jid)]
    assert history[0] == "queued" and "solving" in history
    assert any("objective" in e.data for e in q.progress.history(jid))
    # Re-enqueueing the same payload doesn't start a second job.
    assert await q.enqueue("optimize_plan", payload) == jid
    await q.close()


async def test_forecast_job_returns_weights():
    q = InlineQueue()
    records = [{"unique_id": "sql", "ds": f"2024-{m:02d}-01", "y": 10 + m} for m in range(1, 13)]
    records += [{"unique_id": "sql", "ds": f"2025-{m:02d}-01", "y": 22 + m} for m in range(1, 13)]
    jid = await q.enqueue("forecast_demand", {"records": records, "horizon": 3})
    result = await q.result(jid, timeout_s=120)
    assert result["weights"]["sql"] > 0
    assert len(result["forecasts"][0]["points"]) == 3
    await q.close()


async def test_failed_job_reports_failure():
    q = InlineQueue()
    jid = await q.enqueue("why_not", {"problem": {"skills": [], "courses": [], "constraints": {"budget": 1}},
                                      "plan": {"status": "optimal"}, "course_id": "nope"})
    with pytest.raises(JobFailed):
        await q.result(jid, timeout_s=30)
    assert await q.status(jid) is JobStatus.FAILED
    assert (await q.progress.latest(jid)).stage == "failed"
    await q.close()


async def test_transient_errors_are_retried(monkeypatch):
    monkeypatch.setattr("d2s.workers.queue.backoff_s", lambda attempt: 0)
    calls = {"n": 0}

    @task("_flaky_test_task", timeout_s=5, max_tries=3)
    async def flaky(ctx: TaskContext, payload: dict) -> dict:
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientError("redis blip")
        return {"attempt": ctx.attempt}

    try:
        q = InlineQueue()
        jid = await q.enqueue("_flaky_test_task", {})
        assert await q.result(jid, timeout_s=10) == {"attempt": 3}
        await q.close()
    finally:
        TASKS.pop("_flaky_test_task", None)


async def test_unknown_task_rejected():
    with pytest.raises(KeyError):
        await InlineQueue().enqueue("does_not_exist", {})


async def test_status_not_found():
    assert await InlineQueue().status("missing") is JobStatus.NOT_FOUND


async def test_reporter_threadsafe_from_worker_thread():
    from d2s.workers.progress import MemoryProgressSink, Reporter

    sink = MemoryProgressSink()
    r = Reporter(sink, "j1")
    await asyncio.to_thread(r.threadsafe, "solving", 0.5, "from thread")
    await asyncio.sleep(0.05)
    assert (await sink.latest("j1")).message == "from thread"
