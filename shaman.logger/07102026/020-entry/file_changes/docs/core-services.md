# Core Services: Workers, ML Services, Decision Engine

> Design for the three backend cores from [features.md](../features.md) §0.
> Code: [`backend/src/d2s/`](../backend/src/d2s). Research snapshot: October 2026.

```
  ┌──────────┐        ┌────────────────┐        ┌──────────────────┐
  │ Workers  │        │  ML services   │        │ Decision engine  │
  │ (arq +   │        │ Sentence-Trans │        │ OR-Tools CP-SAT  │
  │  Redis)  │        │ StatsForecast  │        │ + GLOP (duals)   │
  └────┬─────┘        └───────┬────────┘        └────────┬─────────┘
```

## Design principles

1. **Pure cores, thin adapters.** `engine/` and `ml/` are plain Python with Pydantic inputs and outputs. They do no I/O, use no Redis, and know nothing about FastAPI. `workers/` is the only layer that knows about queues.
2. **Every core is swappable behind a Protocol:** `Embedder`, `JobQueue`, `ProgressSink`. Tests use in-memory implementations, so the suite runs without Redis, model downloads or a network.
3. **The optimizer decides; nothing else does.** ML produces *inputs* (demand, gaps) and the engine produces *decisions*. The LLM (later) only explains.
4. **Integer-exact, reproducible solves:** money in whole rupees and weights scaled to integers, with a fixed seed option. The same input always produces the same plan, which the audit trail requires.

```
backend/src/d2s/
├── config.py               # pydantic-settings (env: D2S_*)
├── engine/                 # ── Decision engine ──
│   ├── schemas.py          #   Course, SkillGap, Constraints → Plan
│   ├── model.py            #   CP-SAT model builder
│   ├── solver.py           #   solve(): params, hints, progress callback
│   ├── sensitivity.py      #   GLOP LP-relaxation duals + exact finite-difference
│   ├── pareto.py           #   cost-vs-impact frontier (ε-constraint sweep)
│   └── explain.py          #   "why not course X?" (forced re-solve)
├── ml/                     # ── ML services ──
│   ├── embeddings.py       #   Embedder protocol, SentenceTransformer + Hashing impls
│   ├── taxonomy.py         #   O*NET 30.3 (+ ESCO CSV) loaders → Skill/Occupation records
│   ├── skills.py           #   SkillMatcher: hybrid semantic + lexical extraction
│   └── forecasting.py      #   DemandForecaster: smooth/intermittent routing, intervals, backtest
└── workers/                # ── Workers ──
    ├── progress.py         #   ProgressSink protocol (memory / Redis pub-sub)
    ├── queue.py            #   JobQueue protocol: InlineQueue (dev/test), ArqQueue (prod)
    ├── tasks.py            #   job functions wrapping engine/ml, run in threads
    └── worker.py           #   arq WorkerSettings (retries, timeouts, cron)
```

---

## 1. Decision Engine (OR-Tools CP-SAT + GLOP)

### 1.1 Model

**Inputs**
- `SkillGap(skill_id, learners_short, demand_weight)`: how many learners lack the skill, and how much the market values it (from forecasting).
- `Course(id, fixed_cost, cost_per_seat, min_batch, max_seats, trainer_hours, coverage{skill: %}, prerequisites[], exclusive_group, mandatory)`.
- `Constraints(budget, trainer_hours, max_total_seats, min_skill_closure{skill: %})`.

**Variables**
- `run[c] ∈ {0,1}` and `seats[c] ∈ [0, max_seats]`, linked by `min_batch·run ≤ seats ≤ max_seats·run`.
- `closed[s] = min(learners_short[s]·100, Σ_c coverage[c,s]·seats[c])`, using `AddMinEquality`. The cap gives **saturation**: training beyond the gap is worth nothing, so the solver spreads money across skills instead of over-serving one.

**Constraints**
- Budget: `Σ fixed·run + cost_per_seat·seats ≤ budget`
- Trainer capacity: `Σ trainer_hours·run ≤ trainer_hours`
- **Prerequisites** (precedence-constrained knapsack): `run[c] ≤ run[p]` for each prerequisite `p` of `c`
- Exclusive groups: `Σ run[c in group] ≤ 1`. Mandatory courses: `run[c] = 1`.
- Equity floor: `closed[s] ≥ pct·learners_short[s]` for protected skills or segments.

**Objective:** maximize `Σ_s weight_s · closed[s]`, with weights scaled to integers. CP-SAT needs integer coefficients, and scaling keeps solves exact and reproducible.

### 1.2 Solver practice (from CP-SAT Primer / OR-Tools docs)
| Practice | Why |
|---|---|
| `max_time_in_seconds` + `relative_gap_limit` (default 0.5%) | Bounded latency for the what-if UI; report the remaining gap honestly. |
| `num_workers` = 8 (configurable) | CP-SAT's portfolio search is much stronger with ≥ 8 workers; very low worker counts can fail to prove optimality. |
| **Solution hints** from the previous plan | What-if edits are small changes, so warm-starting cuts solve time substantially. |
| `CpSolverSolutionCallback` → progress events | Streams each improving solution (objective, bound, time) to the UI over SSE. |
| Status mapping (`OPTIMAL / FEASIBLE / INFEASIBLE / MODEL_INVALID / UNKNOWN`) | Never present a FEASIBLE plan as optimal; always show the gap. |
| Pre-validation (unknown prerequisites, cycles, mandatory over budget) | Gives human-readable errors instead of a bare `INFEASIBLE`. |

### 1.3 Sensitivity ("what's the next rupee worth?")
CP-SAT is an integer solver, so it has no duals. We use two methods together:
1. **LP relaxation in GLOP** (`pywraplp`, continuous `run` and `seats`) gives the **shadow price** of the budget, trainer-hours and per-course seat-cap constraints. It's instant, but approximate (an upper bound on the true integer marginal value).
2. **Exact finite difference:** re-solve CP-SAT with `budget + Δ` (and `hours + Δ`), warm-started. It costs two extra solves but gives the true integer marginal value. The UI shows both: "₹1 lakh more ≈ +X weighted-learners (LP), +Y exact".

### 1.4 Pareto frontier (cost vs. impact)
CP-SAT has no native multi-objective mode, so we use the **ε-constraint method**: sweep the budget over N levels, maximize impact at each level, then minimize the cost needed to reach that impact (a lexicographic second solve with the impact fixed). Dominated points are dropped. The result is a cost → impact curve the user can click.

### 1.5 Explain-why-not
For any course left out of the plan, re-solve with `run[c] = 1` and report the objective delta, or "infeasible because prerequisite P / exclusive with Q / exceeds budget". This feeds the auditable evidence trail.

---

## 2. ML Services

### 2.1 Embeddings: `Embedder` protocol
- **Default model: `BAAI/bge-small-en-v1.5`** (33M params, 384-dim, strong English retrieval, fast on CPU). It's configurable via `D2S_EMBEDDING_MODEL`.
  - Multilingual upgrade (Hindi and regional languages): `intfloat/multilingual-e5-small`, or `BAAI/bge-m3` (100+ languages, 8K context, dense + sparse).
  - Strongest model under 500M: `google/embeddinggemma-300m` (gated licence; < 200 MB RAM quantized).
- **Sentence-Transformers v5 APIs:** `encode_query()` for job-post sentences and `encode_document()` for taxonomy labels. These apply the model's asymmetric prompts automatically.
- **CPU speed-up:** `backend="onnx"` (v5.1 adds ONNX/OpenVINO backends, 2–3× faster, with optional int8 quantization). Set with `D2S_EMBEDDING_BACKEND`.
- Embeddings are L2-normalized so cosine similarity is a dot product, matching pgvector `<=>` / `<#>` later.
- `HashingEmbedder`: a deterministic character-n-gram embedder with no model download, used in tests and offline dev.
- The model is loaded **once per worker process** (in arq's `on_startup` context) and requests are batched.

### 2.2 Taxonomy (O\*NET 30.3 now, ESCO when downloaded)
O\*NET 30.3 reorganised skills, so the loaders use the **new files**:
| File | Use |
|---|---|
| `Occupation Data.txt` | occupation code, title, description |
| `Essential Skills.txt`, `Transferable Skills.txt`, `Knowledge.txt` | element → occupation with **Importance (IM)** and **Level (LV)** scales |
| `Software Skills.txt` | concrete tools with **Hot Technology** / **In Demand** flags, our best demand prior |
| `Emerging Tasks.txt` | new or changed tasks, a seed for emerging-skill detection |
| `Content Model Reference.txt` | element descriptions (better embedding text than bare names) |

The ESCO loader reads `skills_en.csv`, `occupations_en.csv` and `occupationSkillRelations_en.csv` (pinned v1.2.x) once the files are in `dataset/esco/`.

### 2.3 Skill extraction: `SkillMatcher` (hybrid)
1. Split the text into sentences and bullet clauses.
2. **Semantic:** embed the clauses and take the top-k skills by cosine.
3. **Lexical:** exact or word-boundary label matches (catches "Python", "AutoCAD" that vectors can blur).
4. **Fuse** with Reciprocal Rank Fusion (`Σ 1/(60 + rank)`), apply a similarity threshold, and keep the best span as **evidence** for the audit trail.
This uses the in-memory numpy index now; the same interface moves to pgvector HNSW + `pg_search` BM25 later.

### 2.4 Demand forecasting: `DemandForecaster` (Nixtla StatsForecast)
Skill-demand series, which are monthly mention counts per skill, are often **sparse**. One model doesn't fit all, so:
1. **Classify each series** with Syntetos–Boylan: ADI (average inter-demand interval) and CV² of the non-zero sizes.
   - *smooth / erratic* (ADI < 1.32) → **AutoETS** and **AutoARIMA**, with native prediction intervals.
   - *intermittent / lumpy* → **CrostonOptimized**, **TSB**, **IMAPA**, **ADIDA**. These give point forecasts only, so we add **`ConformalIntervals`** for calibrated bands.
   - *too short* (< 2·h points) → **HistoricAverage / Naive** baseline.
2. **Backtest-based model selection:** `cross_validation` (rolling origin), choosing the model with the lowest MAE per series.
3. Output a `point`, `lo`, `hi` (default 80% level) per skill and month. The **`lo` band** feeds the engine's robust mode, and `point` feeds `demand_weight`.

---

## 3. Workers (arq + Redis)

### 3.1 Why arq, and its caveat
arq is async-native (fits FastAPI), small, and has retries, cron and result storage. **Caveat:** upstream says it's in *maintenance-only mode* (it still gets releases; 0.28.0 shipped April 2026). So all code depends on our **`JobQueue` protocol**, never on arq directly. If needed we can swap to **SAQ** or **Taskiq** (both async and Redis-backed) by adding one adapter.

### 3.2 Queue backends
| Backend | When | Notes |
|---|---|---|
| `InlineQueue` | dev / tests / no Redis | Runs the job in-process (thread), same API, same progress events |
| `ArqQueue` | production | `create_pool` → `enqueue_job(_job_id=…)`; idempotent job IDs |
Selected by `D2S_QUEUE_BACKEND=inline|arq`.

### 3.3 Job design
- **CPU-bound work (CP-SAT, StatsForecast, embedding) runs in `asyncio.to_thread`**, so the worker's event loop keeps heart-beating and stays cancellable.
- **Retries:** arq `Retry(defer=…)` with exponential backoff + jitter, only for *transient* errors (Redis/network). Model errors such as infeasible input fail fast and are never retried.
- **Timeouts:** per-job `job_timeout` (optimization 120 s, forecast 600 s, taxonomy embedding 1800 s).
- **Idempotency:** job ID = hash of the input payload, so a duplicate click doesn't run twice. `keep_result` is 1 day so the UI can fetch it.
- **Progress:** jobs publish `{job_id, stage, pct, message, data}` through a `ProgressSink`. The Redis sink uses pub/sub (`d2s:progress:{job_id}`) plus a latest-state key, which FastAPI streams to the browser over SSE.
- **Cron:** a quarterly taxonomy refresh and nightly forecast recomputation (`arq.cron`).
- **Windows dev:** Redis isn't native. Use **Memurai** (native Windows Redis), WSL `redis-server`, or just `inline` mode.

---

## Sources
- CP-SAT: [CP-SAT Primer: parameters](https://d-krupke.github.io/cpsat-primer/parameters.html) · [OR-Tools solver docs](https://github.com/google/or-tools/blob/stable/ortools/sat/docs/solver.md) · [num_workers optimality issue #3662](https://github.com/google/or-tools/issues/3662) · [Solution hints issue #3750](https://github.com/google/or-tools/issues/3750) · [Recipes for OR-Tools](https://xiang.es/posts/cp-sat/) · [Callbacks & workers discussion](https://groups.google.com/g/or-tools-discuss/c/YFuka1Ip4I0)
- LP duals / sensitivity: [MathOpt reference](https://developers.google.com/optimization/service/reference/rpc/google.research.optimization.v1/mathopt) · [OR-Tools LP example](https://code.morphllm.com/google/or-tools/raw/branch/stable/examples/python/linear_programming.py)
- Multi-objective: [SAUGMECON for CP (CP 2026)](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2026.14) · [Precedence-constrained knapsack](https://optimization-online.org/tag/precedence-constrained-knapsack-problem/)
- Sentence-Transformers: [v5.0.0 release](https://github.com/huggingface/sentence-transformers/releases/tag/v5.0.0) · [v5.1.0 ONNX/OpenVINO](https://github.com/huggingface/sentence-transformers/releases/tag/v5.1.0) · [Speeding up inference](https://www.sbert.net/docs/multi_vector_encoder/usage/efficiency.html)
- Embedding models: [Best embedding models 2026 (Prem)](https://blog.premai.io/best-embedding-models-for-rag-2026-ranked-by-mteb-score-cost-and-self-hosting) · [Zilliz 2026 benchmark](https://www.zilliz.com/blog/choose-embedding-model-rag-2026) · [EmbeddingGemma](https://deepmind.google/models/gemma/embeddinggemma/)
- StatsForecast: [Intermittent data tutorial](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/IntermittentData) · [CrostonOptimized](https://nixtlaverse.nixtla.io/statsforecast/docs/models/crostonoptimized.html) · [Conformal prediction](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/ConformalPrediction) · [AutoETS](https://nixtlaverse.nixtla.io/statsforecast/docs/models/autoets.html) · [Cross-validation](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/crossvalidation.html.md)
- arq: [arq docs](https://github.com/samuelcolvin/arq/blob/main/docs/index.rst) · [FastAPI + arq retries](https://davidmuraya.com/blog/fastapi-arq-retries/) · [Celery vs ARQ vs RQ 2026](https://projectsupply.in/blog/celery-vs-arq-vs-rq-python-task-queue-comparison-2026) · [Lightweight queues for LLM apps](https://dangquan1402.github.io/llm-engineering-notes/2026/04/02/lightweight-task-queues-for-llm-apps.html)
- Redis on Windows: [Memurai](https://redis.io/tutorials/howtos/how-to-run-redis-on-windows-natively-with-memurai.md)
