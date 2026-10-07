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
3. **The optimizer decides; nothing else does.** ML produces *inputs* (demand, gaps) and the engine produces *decisions*. Our own self-hosted SLM (later) only explains; no third-party LLM API is used.
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
- **Cron:** a nightly skill-index check at 02:00 (re-embeds only if the taxonomy changed). Nightly forecast recomputation will be added once demand data is ingested.
- **Windows dev:** Redis isn't native. Use **Memurai** (native Windows Redis), WSL `redis-server`, or just `inline` mode.

---

# Appendix

## A. Decision Engine

### A.1 Definitions
| Term | Meaning in D2S |
|---|---|
| **Skill gap** (`SkillGap`) | A skill the cohort lacks. `learners_short` = how many learners lack it; `demand_weight` = how much the market wants it (from forecasts). |
| **Course / intervention** (`Course`) | A training option the institution could run, with its costs, seat limits, trainer hours, the skills it teaches (`coverage`) and its prerequisites. |
| **Coverage %** | How much of one learner's gap in a skill a single seat closes. `100` = fully closes it, `20` = partial exposure (e.g. an ML course that also teaches some Python). |
| **Plan** | The engine's output: which courses run, how many seats each, the cost, and how much of each skill gap closes. |
| **Impact / objective** | `Σ demand_weight × learners_closed`: "demand-weighted learners trained". Higher is better. |
| **Saturation** | A skill's value stops at its gap: training 80 people for a 60-person gap counts as 60. This is modelled with `closed = min(gap, supply)`. |
| **Constraint** | A hard rule the plan can never break: budget, trainer hours, total seats, prerequisites, exclusive groups, mandatory courses, equity floors. |
| **Prerequisite** | Course B can only run if course A also runs (`run_B ≤ run_A`). |
| **Exclusive group** | At most one course from the group runs (e.g. "Excel Basic" *or* "Excel Advanced"). |
| **Equity floor** (`min_closure_pct`) | A guaranteed minimum share of a skill's gap that must close, regardless of its demand weight. |
| **CP-SAT** | Google OR-Tools' integer solver (constraint programming + SAT + LP techniques). It finds the best whole-number plan. |
| **GLOP** | Google's linear-programming solver. We use it on the *relaxed* (fractional) model to get shadow prices. |
| **LP relaxation** | The same model with "run a course" allowed to be fractional (0.4 of a course). Easier to solve, gives an optimistic bound. |
| **Shadow price / dual value** | How much the objective improves per one extra unit of a limited resource (₹1 of budget, 1 trainer hour). |
| **Finite difference** | Re-solving with the resource actually increased (e.g. +₹1 lakh) and measuring the real change in impact. |
| **Optimality gap** | `(best_bound − objective) / best_bound`: how far the plan *could* be from perfect. 0% means proven optimal. |
| **Best bound** | The solver's proof that no plan can score higher than this. |
| **Solution hint / warm start** | Giving the solver a previous plan as a starting guess so it finds good plans faster. |
| **Pareto frontier** | The set of plans where you can't get more impact without spending more money. Every other plan is "dominated". |
| **ε-constraint method** | A way to build the frontier: cap cost at ε, maximize impact, repeat for several ε values. |
| **Lexicographic solve** | Two-stage solve: first maximize impact, then, keeping that impact, minimize cost. Removes wasted spending. |
| **Why-not** | Re-solving with an excluded course forced in, to show what including it would cost. |

### A.2 How it works
```
PlanningProblem ─► validate() ─► build_model() ─► CP-SAT solve ─► Plan
   (skills,          │ duplicate ids          │ run[c] ∈ {0,1}             │ status, gap, bound
    courses,         │ unknown prereqs        │ seats[c] ∈ [0, max]        │ chosen courses + seats
    constraints,     │ prereq cycles          │ closed[s] = min(gap, Σcov·seats)
    forced in/out)   │ mandatory > budget     │ maximize Σ w·closed        │ skill closure %
                     └─► INVALID + message    └─► progress callback ─────► SSE to UI
```
1. **Validate first.** Mistakes like a prerequisite cycle or mandatory courses exceeding the budget become readable messages instead of a bare "INFEASIBLE".
2. **Build the integer model.** Money is in whole rupees and coverage in whole percent. Demand weights are scaled so the largest is 1000. Everything stays integer, so the same input always gives the same plan.
3. **Solve** with a time limit (10 s), 8 parallel workers, a 0.5% gap tolerance and a fixed seed. Each improving solution is pushed to the progress callback.
4. **Extract the plan.** The objective is recomputed in real units from the chosen seats, so rounding in the weights never leaks into reported numbers.
5. **Optional analyses** on top of a plan:
   - `analyze()`: GLOP shadow prices plus exact re-solves with +₹1 lakh budget, +10 trainer hours and +10 seats.
   - `frontier()`: 8 budget levels, each solved lexicographically, with dominated points removed.
   - `why_not(course)`: a forced re-solve plus static checks (excluded prerequisite, exclusive rival, too expensive).

### A.3 Q&A
**Q: Why CP-SAT and not a plain LP or a greedy "best value per rupee" ranking?**
Courses are all-or-nothing (you can't run 0.4 of a course), have minimum batch sizes, prerequisites and either-or choices. Greedy ranking ignores these interactions and can pick a course whose prerequisite then breaks the budget. An LP can return fractional courses. CP-SAT handles all of these exactly and proves how good the answer is.

**Q: What does "optimal" actually guarantee?**
That no plan satisfying *the given constraints* scores higher on *the given weights*. It does not guarantee that the weights (forecasts) are right. That's why we also offer the pessimistic `lo` band as weights, and sensitivity analysis.

**Q: Why are there two sensitivity numbers (LP estimate vs. exact)?**
The LP shadow price is instant but optimistic, because it assumes you can buy fractional courses. The exact number re-solves the real integer problem, so it reflects real lumps: one more ₹1 lakh might add nothing if no whole course fits, while ₹1.2 lakh might unlock a whole course. Show the exact number to decision-makers, and use the LP number to rank where to look.

**Q: Why does the solver sometimes return FEASIBLE instead of OPTIMAL?**
It hit the time limit or the 0.5% gap tolerance before finishing the proof. The plan is still valid, and the `gap` field says how close to perfect it is guaranteed to be. Increase `time_limit_s` or lower `gap_limit` if you need a proof.

**Q: What happens when no plan is possible?**
Two cases. *Invalid* input (e.g. mandatory courses cost more than the budget) is caught before solving with a specific message. *Infeasible* combinations (e.g. 100% equity floors on every skill with a small budget) return `INFEASIBLE` with advice on which constraints to relax.

**Q: Why cap value at the gap (saturation)?**
Without it, the solver would pour the whole budget into the single highest-weight skill and train far more people than lack it. Saturation forces it to spread spending across skills, which is what a sensible planner does.

**Q: How fast is it?**
Typical institutional problems (tens of courses, tens of skills) solve to proven optimality in well under a second. The warm-start hint makes what-if slider changes faster still. Large regional problems use the time limit and report the gap.

**Q: Can two users get different plans for the same input?**
No. A fixed random seed and integer data make solves reproducible. That reproducibility is the basis of the audit trail.

---

## B. ML Services

### B.1 Definitions
| Term | Meaning in D2S |
|---|---|
| **Taxonomy** | The official list of skills and occupations we map everything onto. For us that's O\*NET 30.3 now, plus ESCO v1.2 once downloaded. |
| **O\*NET element** | An O\*NET skill, knowledge area or ability, with a stable ID like `2.A.1.a` (Reading Comprehension). |
| **Importance (IM) / Level (LV)** | O\*NET ratings per occupation: how important a skill is (1–5) and what level is needed (0–7). |
| **Hot Technology / In Demand** | O\*NET flags on software tools that employers frequently request. A ready-made demand signal. |
| **ESCO URI** | ESCO's permanent identifier for a skill or occupation, e.g. `http://data.europa.eu/esco/skill/…`. |
| **Embedding** | A list of numbers (384 for bge-small) representing a text's meaning. Similar meanings give nearby vectors. |
| **Cosine similarity** | How aligned two embeddings are, from −1 to 1. Vectors are normalized, so it's a simple dot product. |
| **Query vs. document encoding** | Sentence-Transformers v5 encodes short noisy inputs (job-post sentences) and canonical entries (skill descriptions) slightly differently, which improves matching. |
| **Skill index** (`SkillIndex`) | All taxonomy skills embedded once and saved to disk (`dataset/index/*.npz`), tagged with the model name and taxonomy version. |
| **Lexical match** | Exact word match of a skill label in the text ("SQL", "AutoCAD"). |
| **Hybrid search** | Semantic + lexical results combined, catching both paraphrases and exact tool names. |
| **RRF (Reciprocal Rank Fusion)** | Merges two ranked lists by summing `1/(60 + rank)` from each. It doesn't need the scores to be comparable. |
| **Evidence** | The exact clause of the job post that triggered a skill match, kept for the audit trail. |
| **Time series** | One skill's monthly demand count, e.g. Python mentions per month. |
| **Horizon** | How many months ahead we forecast (default 6). |
| **ADI** (average demand interval) | Average number of periods between non-zero demands. High ADI = sparse. |
| **CV²** | Squared coefficient of variation of the non-zero demand sizes. High CV² = sizes vary a lot. |
| **Demand pattern** | Syntetos–Boylan class: *smooth* (regular), *erratic* (regular timing, variable size), *intermittent* (sparse, steady size), *lumpy* (sparse and variable). We add *short* and *zero*. |
| **AutoETS / AutoARIMA** | Classical models for regular series that select their own trend/seasonality settings. |
| **Croston / SBA / IMAPA / ADIDA** | Models built for sparse series: they forecast "how often" and "how much" separately, or aggregate first. |
| **Prediction interval** | A range (`lo`–`hi`) the real value should fall in with a given probability (80% by default). |
| **Conformal interval** | A prediction interval built from the model's own past errors in backtests; works for any model, with no distribution assumptions. |
| **Backtest / cross-validation** | Pretend it's an earlier date, forecast, compare with what really happened, repeat. Used to pick the best model per skill. |
| **MAE** | Mean absolute error: the average size of forecast misses. Lower is better. |
| **Demand weight** | Total forecast demand over the horizon. This becomes `SkillGap.demand_weight` in the engine. |

### B.2 How it works
**Taxonomy → index (once, then cached):**
```
O*NET text files ──► load_onet() ──► Skills (essential, transferable, knowledge, tools)
ESCO CSV (optional) ─► load_esco() ─┘        + Occupations + Occupation↔Skill links (IM/LV)
                                      │
                                      ▼
                           embed "label. description" ──► SkillIndex (.npz, keyed by model+version)
```
**Skill extraction (per job post):**
```
text ─► split into clauses ─┬─► embed clauses ─► cosine vs index ─► top-3/clause ≥ 0.6 ─┐
                            └─► 1–5 word n-grams ─► exact label lookup ──────────────────┤
                                                                                        ▼
                                                     RRF fuse ─► SkillMatch(score, similarity,
                                                                 lexical?, evidence clause)
```
**Forecasting (per batch of skills):**
```
(unique_id, ds, y) ─► fill missing months with 0 ─► classify each series (ADI, CV²)
   ├─ zero            ─► zero forecast
   ├─ short           ─► HistoricAverage / Naive
   ├─ smooth/erratic  ─► AutoETS / AutoARIMA / SeasonalNaive   (native intervals)
   └─ intermittent/lumpy ─► Croston / SBA / IMAPA / ADIDA       (conformal intervals)
            ▼
   rolling-origin backtest ─► lowest-MAE model per skill ─► point, lo, hi per month (clipped ≥ 0)
            ▼
   demand_weights(point | lo | hi) ─► engine SkillGap.demand_weight
```

### B.3 Q&A
**Q: Why not use an LLM to extract skills?**
Embeddings cost far less per document, run on a CPU, are deterministic, and always return IDs from the official taxonomy. Generative models can invent skill names and are slower. Our own SLM could later act as a reviewer for low-confidence matches, without any external API.

**Q: Why combine semantic and lexical matching?**
Each misses what the other catches. Vectors understand "built dashboards for leadership" ≈ data visualization, but can blur "PostgreSQL" vs "MySQL". Exact matching nails tool names but misses paraphrases. RRF merges both rankings without needing to calibrate their scores.

**Q: Why bge-small-en-v1.5 as the default model?**
It's small (33M parameters), fast on a laptop CPU, strong at English retrieval, and freely licensed. For Hindi and regional languages, switch `D2S_EMBEDDING_MODEL` to `intfloat/multilingual-e5-small` or `BAAI/bge-m3`. The index rebuilds automatically because it's keyed by model name.

**Q: What is the HashingEmbedder for? Is it used in production?**
No. It matches by spelling, not meaning. It exists so tests and offline development run without downloading a model. Production uses Sentence-Transformers.

**Q: What does the 0.6 similarity threshold mean, and can it change?**
Matches below 0.6 cosine are dropped as too weak. The right value depends on the model, so it's a constructor parameter. Tune it on a small hand-labelled set of job posts.

**Q: Why classify series before forecasting?**
Most skills appear in only some months. Models like ETS assume regular data and badly over- or under-forecast sparse series. Croston-family models are designed for sparse demand. Routing each series to the right family is the biggest accuracy win for little effort.

**Q: Why are the intervals important to the engine?**
Planning on the average forecast is risky if demand disappoints. Using the `lo` band as weights gives a cautious plan that still pays off in a weak market. Comparing plans made with `point` and `lo` shows how robust a recommendation is.

**Q: Why fill missing months with zero?**
In job-posting data, "no postings this month" really is zero demand, not missing data. Leaving gaps would make the series look denser than it is.

**Q: How much history is needed?**
At least `max(horizon × (backtest_windows + 1), 12)` months, which is 18 by default, for full model selection. Shorter series still get a simple average forecast, flagged as `short`.

---

## C. Workers

### C.1 Definitions
| Term | Meaning in D2S |
|---|---|
| **Job** | One background run of a task with a payload, e.g. "optimize this problem". |
| **Task** | A registered async function `fn(ctx, payload) -> dict`. There are 7: `optimize_plan`, `sensitivity`, `pareto_frontier`, `why_not`, `forecast_demand`, `build_skill_index`, `extract_skills`. |
| **Payload / result** | Plain JSON in and out, so the same task runs inline or on a remote arq worker. |
| **Task registry** (`TASKS`) | The dictionary every queue backend reads to know which tasks exist, with their timeouts and retry counts. |
| **Queue backend** | Where jobs wait and run: `InlineQueue` (same process; dev and tests) or `ArqQueue` (Redis; production). |
| **JobQueue protocol** | The interface the app uses (`enqueue`, `status`, `result`, `close`), so the queue library can be swapped without touching app code. |
| **arq** | A small async Python job queue built on Redis. In maintenance-only mode upstream, so it's kept behind the protocol. |
| **Redis** | An in-memory data store used as arq's queue, result store and progress pub/sub. |
| **Worker** | A separate process (`arq d2s.workers.worker.WorkerSettings`) that pulls jobs from Redis and runs them. |
| **Idempotent job ID** | `task:<hash of payload>`. The same request maps to the same job, so double-clicks don't run work twice. |
| **TransientError** | A failure worth retrying (network blip, model download hiccup). All other errors fail immediately. |
| **Exponential backoff with jitter** | The wait before retry k is random in `[0, min(60 s, 2·2^(k−1) s)]`, which avoids retry storms. |
| **Timeout** | The maximum runtime per task (optimize 120 s, forecast 600 s, index build 1800 s). |
| **Progress event** (`JobProgress`) | `{job_id, stage, pct, message, data, ts}`. Stages run `queued → running → task-specific → complete \| failed`. |
| **Progress sink** | Where events go: memory (inline) or Redis pub/sub channel `d2s:progress:{job_id}` plus a `:latest` key. |
| **Reporter** | A per-job helper with `report()` (async) and `threadsafe()` (callable from solver threads). |
| **`asyncio.to_thread`** | Runs CPU-heavy code in a thread so the worker's event loop stays responsive (heartbeats, progress, cancellation). |
| **Cron job** | A scheduled task. Currently a nightly 02:00 skill-index check. |

### C.2 How it works
```
API / caller                      Queue backend                     Task (in worker)
────────────                      ─────────────                     ────────────────
enqueue("optimize_plan", p) ──►  job_id = task:hash(p)
                                  already exists? → return same id
                                  publish  queued
                                  ── run ──────────────────────────► run_task():
                                                                      publish running
                                                                      wait_for(fn, timeout)
                                                                        └ to_thread(solve, on_progress)
                                                                            each solution → Reporter.threadsafe
                                                                              → publish solving {objective, gap}
                                                                      publish complete / failed
subscribe(job_id) ◄── events ──── memory queue / Redis pub/sub ◄─────┘
result(job_id)    ◄── JSON ────── task result (kept 24 h in Redis)
```
- **Retries:** if a task raises `TransientError`, InlineQueue sleeps with backoff and retries; on arq the wrapper raises `arq.Retry(defer=backoff)`. After `max_tries` the job fails. Engine tasks use `max_tries=1`, because a deterministic solve fails the same way every time.
- **Concurrency:** an arq worker runs up to 4 jobs at once (`max_jobs=4`), because each CP-SAT solve already uses several threads.
- **Model loading:** the skill matcher (taxonomy + embedding model + index) loads once per worker process and is reused by every extraction job.

### C.3 Q&A
**Q: Why use a job queue at all instead of solving inside the API request?**
Pareto sweeps, forecasts over thousands of skills and index builds take seconds to minutes. Running them in the web server would block it and time out browsers. Jobs return an ID immediately, and the UI streams progress.

**Q: Do I need Redis to develop?**
No. The default `D2S_QUEUE_BACKEND=inline` runs jobs in-process with the same API and progress events. Redis (Memurai on Windows, or WSL) is only needed for `arq` mode, where jobs survive restarts and run on separate machines.

**Q: arq is "maintenance-only". Is that a risk?**
A small one. It's stable and still released (0.28.0, April 2026). Because the app only talks to the `JobQueue` protocol, moving to SAQ or Taskiq means writing one adapter class. No task code changes.

**Q: What if the same request is submitted twice?**
It gets the same job ID (a hash of task + payload) and shares the first job's progress and result. A job that *failed* can be resubmitted and runs again.

**Q: Why aren't optimization jobs retried?**
A solve is deterministic: an infeasible or invalid problem fails identically on every try. Retrying wastes worker time. Only flaky external failures (`TransientError`) are retried.

**Q: How does the UI show live solver progress?**
CP-SAT calls our callback on each better plan from its own threads. `Reporter.threadsafe()` hands the event to the event loop, which publishes it to the sink (Redis pub/sub in production). The FastAPI layer (next step) will relay the `subscribe()` stream to the browser over SSE.

**Q: What happens if a worker crashes mid-job?**
With arq, jobs stay in Redis until they succeed or fail ("pessimistic execution"), so another worker picks them up after a restart. With InlineQueue, jobs are lost when the process exits. That's acceptable for development only.

**Q: How do I add a new background task?**
Write `async def my_task(ctx, payload) -> dict` in `workers/tasks.py` and decorate it with `@task("my_task", timeout_s=…, max_tries=…)`. It's automatically available to both queue backends and the arq worker.

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
