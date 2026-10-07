# d2s backend: core services

Decision engine (OR-Tools CP-SAT + GLOP), ML services (Sentence-Transformers, StatsForecast) and
background workers (arq + Redis). Design rationale: [`../docs/core-services.md`](../docs/core-services.md).

## Setup

```bash
cd backend
uv sync --all-extras          # Python 3.13, see .python-version
uv run pytest                 # fast suite (no Redis, no model download)
uv run pytest -m slow         # tests that download the embedding model
```

## Layout

| Package | What it does | Heavy deps |
|---|---|---|
| `d2s.engine` | `solve`, `analyze` (shadow prices + exact marginals), `frontier` (Pareto), `why_not` | ortools |
| `d2s.ml.embeddings` | `Embedder` protocol; `SentenceTransformerEmbedder` (prod), `HashingEmbedder` (tests) | sentence-transformers |
| `d2s.ml.taxonomy` | O\*NET 30.3 + ESCO v1.2 CSV loaders | pandas |
| `d2s.ml.skills` | `SkillMatcher`: hybrid semantic + lexical extraction with evidence | numpy |
| `d2s.ml.forecasting` | `DemandForecaster`: pattern routing, backtest selection, intervals | statsforecast |
| `d2s.workers` | task registry, `InlineQueue` / `ArqQueue`, progress sinks, arq `WorkerSettings` | arq, redis |

## Quick use

```python
from d2s.engine import PlanningProblem, solve, analyze, frontier, why_not

plan = solve(problem)                       # Plan: status, courses, seats, skill closure, gap
report = analyze(problem, plan)             # ₹ marginal value of budget / trainer hours
curve = frontier(problem, levels=8)         # cost-vs-impact Pareto points
reason = why_not(problem, plan, "ml201")    # why a course was left out
```

```python
from d2s.workers.queue import create_queue

q = await create_queue()                    # inline by default, arq when D2S_QUEUE_BACKEND=arq
jid = await q.enqueue("optimize_plan", {"problem": problem.model_dump(mode="json")})
async for event in q.progress.subscribe(jid):   # queued → running → solving… → complete
    print(event.stage, event.data)
plan = await q.result(jid)
```

## Running workers with Redis

```bash
# Redis on Windows: Memurai, or WSL:  sudo apt install redis-server && redis-server
set D2S_QUEUE_BACKEND=arq
uv run arq d2s.workers.worker.WorkerSettings
```

## Configuration (`D2S_*` env vars or `.env`)

| Variable | Default | Notes |
|---|---|---|
| `D2S_QUEUE_BACKEND` | `inline` | `inline` \| `arq` |
| `D2S_REDIS_URL` | `redis://localhost:6379/0` | |
| `D2S_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | multilingual: `intfloat/multilingual-e5-small` |
| `D2S_EMBEDDING_BACKEND` | `torch` | `onnx` / `openvino` for 2–3× CPU speed, `hashing` for tests |
| `D2S_SOLVER_TIME_LIMIT_S` | `10` | per solve |
| `D2S_SOLVER_WORKERS` | `8` | CP-SAT portfolio threads |
| `D2S_SOLVER_GAP_LIMIT` | `0.005` | stop at 0.5% optimality gap |
| `D2S_DATASET_DIR` | `../dataset` | O\*NET / ESCO files and the cached embedding index |
