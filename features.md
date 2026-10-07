# D2S Bharat — Feature Specification

> Software design for D2S Bharat (see [README.md](README.md)).
> **Stack:** Next.js (frontend) + Python / FastAPI (backend) + PostgreSQL / pgvector.
> Research snapshot: October 2026. Versions below are current as of that date. Recheck them before you pin.

Each feature is tagged with a delivery tier:

- **`[MVP]`**: needed for the hackathon demo or first pilot
- **`[V1]`**: first paid institutional deployment
- **`[V2+]`**: scale and advanced features

---

## 0. Architecture at a Glance

```
┌──────────────────────────── Next.js 16 (App Router, React 19.2) ────────────────────────────┐
│ Command Center │ Gap Map │ What-If Simulator │ Decision Brief │ Evidence Trail │ Admin       │
│ Server Components + "use cache" │ Server Actions │ TanStack Query │ SSE streaming             │
└───────────────▲──────────────────────────────── typed client (generated from OpenAPI) ──────┘
                │ REST + SSE
┌───────────────┴──────────────────────────── FastAPI (Python 3.13) ──────────────────────────┐
│ /ingest  /skills  /demand  /gaps  /optimize  /scenarios  /briefs  /mcp                       │
│ Pydantic v2 schemas │ async SQLAlchemy 2 │ SSE / JSONL streaming endpoints                   │
└──────┬──────────────────────┬──────────────────────────┬────────────────────────────────────┘
       │                      │                          │
  ┌────▼─────┐        ┌───────▼────────┐        ┌────────▼─────────┐
  │ Workers  │        │  ML services   │        │ Decision engine  │
  │ (arq +   │        │ Sentence-Trans │        │ OR-Tools CP-SAT  │
  │  Redis)  │        │ StatsForecast  │        │ + GLOP (duals)   │
  └────┬─────┘        └───────┬────────┘        └────────┬─────────┘
       └──────────────────────┼──────────────────────────┘
                     ┌────────▼─────────────────────────────┐
                     │ PostgreSQL 17 + pgvector 0.8         │
                     │ (+ pg_search BM25 for hybrid search) │
                     └──────────────────────────────────────┘
                     Own SLM (explain only, self-hosted llama.cpp) ──► OTel traces
```

---

## 1. Data Ingestion & Labour-Market Intelligence (Python)

### 1.1 Chosen taxonomy sources: ESCO + O\*NET ✅ (decided)

ESCO and O\*NET are the **official skill/occupation backbone** of D2S Bharat. Both are free and usable today, with no partnership needed.

| | **ESCO** (European Commission) | **O\*NET** (US Dept. of Labor) |
|---|---|---|
| **Coverage** | ~3,000 occupations, ~14,000 skills, essential/optional skill links | ~900 O\*NET-SOC occupations with skills, knowledge, abilities, tasks, technologies, outlook |
| **Strengths for us** | Fine-grained skill list, stable URIs, **28 languages** | Rich importance/level ratings, **"Hot Technology"** flags, Bright Outlook, crosswalks (SOC, CIP education codes) |
| **API** | `https://ec.europa.eu/esco/api`: `search`, `resource/occupation`, `resource/skill`; no key, no signup, CORS-friendly JSON | O\*NET Web Services: free developer account at `services.onetcenter.org`, credentials in minutes; JSON/XML |
| **Bulk data** | CSV + RDF downloads from the ESCO portal; **ESCO Local API** (self-hosted) for production load | Full **O\*NET database download** (refreshed quarterly) |
| **License** | EC reuse terms (check the portal before commercial launch) | Creative Commons Attribution, so attribution is required in the UI and docs |

**How we use them:**
- **Bulk-load first, API second.** Load the ESCO CSV and O\*NET database files into Postgres at setup. Use the live APIs only for incremental lookups and freshness checks. This avoids rate limits and keeps everything offline-capable and fast. `[MVP]`
- **Pin the ESCO version.** Always pass `selectedVersion` (for example, `v1.2.x`), because the API default is the outdated v1.0.9 and results differ noticeably between versions. Store the version on every skill row. `[MVP]`
- **Unified skill ID.** Our `skill` table has its own UUID, with `esco_uri` and `onet_element_id` columns plus a `skill_crosswalk` table. ESCO is the **primary** skill vocabulary (finer-grained). O\*NET enriches it with importance/level scores, hot-technology flags and occupation outlook. `[MVP]`
- **Crosswalk ESCO ↔ O\*NET** by embedding similarity between skill labels and descriptions, then human review of low-confidence pairs in the Data Admin queue. `[MVP]` auto-match, `[V1]` review UI
- **Pre-computed embeddings** for every ESCO and O\*NET skill and occupation label + description, stored in pgvector (HNSW). These power skill extraction (§2). `[MVP]`
- **Quarterly sync job** (arq cron): diff the new ESCO release or O\*NET database against the stored version, upsert changes, and re-embed only changed rows. `[V1]`
- **Attribution footer** in the dashboard and briefs ("Includes data from ESCO and O\*NET"). `[MVP]`

### 1.2 Other data sources
| Source | What we get | Access | Tier |
|---|---|---|---|
| **Job descriptions** (CSV upload, institution partners, job boards) | Raw demand signal, extracted into ESCO/O\*NET skills | Upload / connector | `[MVP]` |
| **Institutional data** | Courses, cohorts, assessments, budgets, seats, trainers | CSV/XLSX upload, then ERP/LMS API | `[MVP]` |
| **NSQF / National Qualifications Register (NQR)** | 8,500+ NSQF-aligned qualifications, Qualification Packs → NOS → job roles | No public API; scrape or download in bulk. Map onto the ESCO/O\*NET backbone | `[V2+]` |
| **National Career Service (NCS)** | Live Indian vacancies | Needs partnership or data-sharing | `[V2+]` |
| **PLFS (MoSPI)** | LFPR, WPR, unemployment by state, sector and education | Public reports and unit data | `[V2+]` |

### 1.3 Ingestion pipeline features
- **Pluggable connector interface:** every source implements `fetch() → normalize() → upsert()`, so adding a source never touches core code. `[MVP]`
- **Scheduled refresh** with arq cron jobs (for example, nightly ESCO/O\*NET diff, hourly vacancy pull). `[V1]`
- **Raw → staged → curated zones:** keep raw payloads (JSONB / Parquet) so you can rerun extraction when models improve. `[V1]`
- **Polars + DuckDB** for bulk transforms on large job-post dumps, which can be 5–20× faster than pandas on large joins. Load the results into Postgres afterwards. `[V1]`
- **Data-quality gates:** dedupe job posts (MinHash / embedding similarity), drop spam, flag stale postings. `[V1]`
- **Provenance on every row:** source, fetch time and version, so the audit trail works. `[MVP]`

---

## 2. Skill Extraction & Normalization (NLP)

- **Embedding-based ESCO matching** `[MVP]`: split each job description into sentences, embed them with Sentence Transformers, and match each sentence to the nearest ESCO skill by cosine similarity, keeping only matches above a threshold. This is the approach the `esco-skill-extractor` package uses, so we can start from it.
- **Domain-tuned models** `[V1]`: swap the generic encoder for job-domain models, for example JobBERT-style title encoders or contrastive skill-extraction bi-encoders (~110M params). Recent research shows these come close to LLM-based extraction at a fraction of the cost.
- **Two-stage retrieve → rerank** `[V1]`: pgvector ANN gets the top 50 candidate skills, then a cross-encoder reranks them. This raises precision on ambiguous phrases.
- **Hybrid lexical + semantic search** `[V1]`: BM25 (ParadeDB `pg_search`) plus pgvector, merged with **Reciprocal Rank Fusion**. Exact matches ("Kubernetes", "AutoCAD") come from BM25 and paraphrases come from vectors.
- **Job-title normalization** `[V1]`: map messy Indian job titles ("Sr. Exec – Data Ops") to ESCO occupations, enriched with the linked O\*NET-SOC profile.
- **Crosswalk ESCO ↔ O\*NET** `[MVP]` (see §1.1): one unified skill ID. NSQF job roles can be mapped onto this backbone later `[V2+]` to make it fully "Bharat-native".
- **Emerging-skill detection** `[V2+]`: cluster high-similarity phrases that match no existing taxonomy skill well, track how often they appear over time, and alert when a cluster grows fast (for example, "agentic AI workflows"). An analyst approves new skills before they enter the taxonomy.
- **Multilingual input** `[V2+]`: translate Hindi and regional-language job posts with the **Bhashini** APIs (22 scheduled languages, free for developers), or use multilingual embedding models directly.

---

## 3. Data Layer (PostgreSQL + pgvector)

- **Core entities:** Skill, Occupation, Job, Location, Course, Cohort, Institution, DemandSignal, Constraint, Intervention, Scenario, DecisionBrief. `[MVP]`
- **pgvector 0.8 features to use:**
  - **HNSW indexes** for fast approximate search over skill and job embeddings. `[MVP]`
  - **Iterative index scans:** keep filtered vector queries correct (for example, "similar skills *in the IT sector*") without returning too few rows. `[MVP]`
  - **`halfvec`:** half-precision storage that halves memory and allows indexing up to 4,000 dimensions. `[V1]`
  - **Binary quantization + re-rank** for very large job-post tables. `[V2+]`
- **Async SQLAlchemy 2 + Alembic migrations**, typed models shared with Pydantic schemas. `[MVP]`
- **Time-partitioned `demand_signal` table** (monthly partitions) for fast time-series queries. `[V1]`
- **Row-Level Security (RLS)** per institution for multi-tenant isolation. `[V1]`
- **Materialized views** for dashboard rollups (demand by skill × region × month). `[V1]`

---

## 4. Demand Forecasting (ML)

- **Baseline models** `[MVP]`: statsmodels ETS / ARIMA per skill time series.
- **Fast batch forecasting with Nixtla StatsForecast** `[V1]`: AutoARIMA, AutoETS, Theta and CES, accelerated with numba. It is several times faster than statsmodels, which matters when forecasting thousands of skill × region series.
- **Hierarchical reconciliation (HierarchicalForecast)** `[V1]`: forecasts at the skill → skill-group → sector and district → state → national levels add up consistently (MinTrace, BottomUp, TopDown).
- **Calibrated uncertainty: conformal prediction intervals** `[V1]`: StatsForecast `ConformalIntervals` (or MAPIE) give confidence bands with coverage guarantees. The optimizer can then plan for pessimistic as well as expected demand.
- **Model selection by backtesting** `[V1]`: cross-validate over time and pick the best model per series automatically.
- **Leading indicators** `[V2+]`: add exogenous signals (posting velocity, Google Trends, PLFS sector growth) using scikit-learn gradient boosting with lag features.
- **Forecast drift monitoring** `[V2+]`: alert when actual demand falls outside the predicted band.

---

## 5. Skill-Gap Intelligence

- **Cohort skill profile** `[MVP]`: built from course mappings (which skills each course teaches), assessments and self-reports.
- **Demand-weighted gap score** `[MVP]`:
  `gap(skill) = forecast_demand(skill, region) × (target_proficiency − cohort_proficiency)`
- **Gap → occupation → course linkage** `[MVP]`: for each gap, show the occupations it unlocks and the courses or NSQF qualifications that close it.
- **Skill adjacency / transferability** `[V1]`: use embedding similarity and co-occurrence in job posts to find skills that are "one hop" from what the cohort already has. These are cheaper to teach.
- **Skill knowledge graph** `[V2+]`: skills, occupations and courses as a graph, used for career-path and prerequisite-chain queries.
- **Segment drill-down** `[V1]`: gaps by department, batch, gender and region (aggregated, privacy-safe).

---

## 6. Core Decision Engine (OR-Tools): the differentiator

### 6.1 Optimization model `[MVP]`
- **Solver:** Google OR-Tools **CP-SAT**, which handles integer and boolean decisions plus logical constraints.
- **Decision variables:** run course *c*? How many seats in course *c*? Which cohort goes to which course?
- **Objective:** maximize Σ (demand-weighted skill-gap closure) + λ · (market-opportunity coverage).
- **Hard constraints:**
  - Total budget
  - Seats per course and trainer capacity / hours
  - **Prerequisites:** `run[c2] ⇒ run[c1]`, or a student's eligibility depends on earlier courses
  - Institutional rules (minimum batch size, mandatory courses, mutually exclusive courses)
- **Solution hints & time limits:** return the best plan found within N seconds, with a reported optimality gap.

### 6.2 Advanced optimization features
- **Shadow prices / marginal value** `[V1]`: solve the LP relaxation with **GLOP** to get dual values, which answer questions like *"₹1 lakh more budget would add +X% gap closure"* and *"10 more seats in Data Analytics is worth Y"*. This helps institutions decide where to put money next.
- **Multi-objective trade-offs / Pareto frontier** `[V1]`: CP-SAT has no built-in Pareto support, so we build one with ε-constraint sweeps or lexicographic solves. The output is a cost-vs-impact curve the user can pick a point on.
- **Robust optimization** `[V1]`: see §6.3.
- **Fairness / equity constraints** `[V1]`: minimum coverage for under-represented groups and regions. This matters for CSR and government buyers.
- **Multi-period planning** `[V2+]`: semester-by-semester rollout with budget carry-over and prerequisite sequencing over time.
- **Explain-why-not** `[V1]`: for any course that was not chosen, re-solve with it forced in and report the cost (for example, "−4.2% gap closure").

### 6.3 Uncertainty-aware (scenario-robust) planning `[V1]`
Forecasts are **scenarios, not single numbers**. The forecaster already emits `lo / point / hi` bands per skill. These become scenarios *k* ∈ {P10, P50, P90}, each with a probability *p_k*.

- **Market absorption per scenario:** training beyond what the market can absorb is wasted. Per skill and scenario, the value counted is `min(learners trained, cohort gap, market absorption_k)`. A low-demand scenario therefore penalizes over-training, not just under-training.
- **Objective options** (selectable; each is a small CP-SAT extension with one value variable *V_k* per scenario):
  | Mode | Objective | Meaning for the institution |
  |---|---|---|
  | Expected | max Σ p_k·V_k | best on average |
  | Worst-case (max-min) | max min_k V_k | safest if demand disappoints |
  | **Blended** (default) | max α·E[V] + (1−α)·min_k V_k | average performance with downside protection; α is a slider |
  | Min-max regret | min max_k (V*_k − V_k) | "never more than X% worse than the best plan for any scenario" (needs one solve per scenario for V*_k) |
- **Shortfall guardrails:** for critical skills, require closure ≥ floor *even in P10*.
- **Outputs:** the plan's coverage under each scenario, the worst-case shortfall, regret, and **plan stability** (which courses survive in every scenario: the "safe core" of the plan).
- **Stress test:** *"If demand is 30% below forecast, which plan is still safe?"* Scale scenario demand by any factor and re-solve. This is exposed in the What-If Simulator (§7).

### 6.4 Decision backtesting `[V1]`
*"Did our decisions actually work?"* is answered with a **rolling-origin backtest of whole decisions**, not just forecasts:
1. Pick a historical cutoff *T* (e.g. end of 2024). Using **only data up to T**, forecast demand, compute gaps and generate the D2S plan.
2. Generate **baseline plans** with the same budget and constraints:
   - *Top-gap-first:* fund courses for the largest skill gaps in order.
   - *Most-popular-course:* by past enrolment or posting volume.
   - *Best-value-per-rupee (greedy):* the strongest simple heuristic.
   - *Random* (sanity floor) and *Oracle* (planned with actual future demand, i.e. the achievable ceiling).
3. Score every plan against **what actually happened** after *T*: demand coverage (trained learners matched to real demand ÷ real demand), oversupply (trained beyond demand), cost per covered learner, and % of oracle achieved.
4. Repeat for several cutoffs and report mean ± spread.

**Integrity rules:** no data after *T* may leak into the D2S or baseline plans. Every published figure must come from a real backtest run with its configuration stored. Synthetic data is used only to test the code and is always labelled as synthetic. **Data needed:** ≥ 2–3 years of time-stamped job postings (or NCS/PLFS series) for the regions concerned.

### 6.5 Multi-institution resource pooling `[V2+]`
Joint optimization across institutions *i* in a cluster (district, university system, CSR programme):
- **Variables:** course *c* run at host *h*, with `seats[c, h, i]` allocated to learners from institution *i*. A shared trainer pool (`assign[t, c, h]`), and shared lab capacity per host.
- **Constraints:** each trainer's hours across all campuses (plus travel time); host lab and room capacity; an **access matrix** for which institutions' learners can reach which host (distance, transport, online/hybrid mode); per-institution budgets plus a **cost-sharing rule** (proportional to seats used).
- **Fairness:** per-institution closure floors, so pooling never benefits one college at another's expense.
- **Outputs:** *"Run ML201 at College B with 12 seats for College A"*, *"Share trainer T between A and B on Tue/Thu"*, plus the **pooling gain**: the pooled plan vs. each institution planning alone (sum of separate solves). Pooling is proposed only when that gain is positive.

---

## 7. What-If Decision Simulator

- **Interactive sliders** for budget, seats, trainers and course toggles. The plan re-optimizes live. `[MVP]`
- **Streaming solver progress** over **SSE** (FastAPI supports SSE and JSONL streaming natively): the UI shows the improving solution while CP-SAT searches. `[V1]`
- **Scenario save / compare / fork:** side-by-side diff of up to 3 scenarios (chosen courses, coverage, cost). `[MVP]`
- **Pareto chart:** click a point on the frontier to load that plan. `[V1]`
- **Sensitivity view** using shadow prices: "what's the next best rupee to spend?" `[V1]`
- **Shareable scenario links** (URL state) for committee review. `[V1]`

---

## 8. Explainability Layer: own SLM, no external LLM API (SLM ≠ Decision Maker)

**Decision (2026-10-07):** we do **not** call any third-party LLM API. Decision briefs are written by **our own small language model (SLM)**, self-hosted. No institutional or student data leaves our infrastructure, and there is no per-token bill.

> "Train our own SLM" here means **fine-tuning an open small base model** (≈1–4B parameters) on our task, not pre-training from scratch. Pre-training needs billions of tokens and large GPU clusters. Fine-tuning a small model on a few thousand examples fits on one consumer GPU.

### 8.1 Three-tier brief generation
| Tier | What | When |
|---|---|---|
| **T0 Template brief** `[MVP]` | Deterministic Python templates (Jinja) render the plan: numbers inserted directly from engine output. 100% factual, a little robotic. | Hackathon demo; also the permanent **fallback** if the SLM fails validation. |
| **T1 Fine-tuned SLM** `[V1]` | A 1–4B open model (candidates: Qwen3, Gemma 3, Phi-4-mini, Llama 3.2 3B, SmolLM3), fine-tuned with LoRA/QLoRA on *(plan JSON → brief)* pairs. Rewrites the facts into fluent, institution-friendly language. | After we have training data from T0 + human edits. |
| **T2 Distilled / domain SLM** `[V2+]` | Larger training set (real reviewed briefs, Hindi/regional variants), possibly a smaller distilled model for CPU-only deployment. | Scale. |

### 8.2 Training pipeline
1. **Synthetic data:** generate thousands of varied `PlanningProblem`s → solve with the engine → render T0 briefs → paraphrase and enrich them (human writers or a one-off offline teacher model; never at runtime) → **human review**.
2. **Fine-tune** with **Unsloth** (QLoRA: an 8B model fits in ~6 GB VRAM; 1–4B models need much less) or Hugging Face TRL/PEFT.
3. **Evaluate** on held-out plans: number-faithfulness (every figure matches the engine), coverage (every chosen course mentioned), readability, and human preference vs. T0.
4. **Export** to **GGUF** (Q4_K_M) and serve with **llama.cpp**, which runs on CPU (≈30–70 tokens/s for 1–4B models), or with **vLLM** on a GPU at higher volume.
5. **Version** each model (data hash + base model + LoRA config) for the audit trail.

### 8.3 Guarantees kept from the original design
- **Input = engine output only** `[MVP]`: the SLM sees the plan, evidence, assumptions and metrics, never raw student data, and never decides anything.
- **Constrained decoding** `[V1]`: llama.cpp / vLLM grammar-constrained JSON (llguidance / xgrammar) so briefs always parse into typed sections: summary, recommendations, risks, assumptions.
- **Grounding & anti-hallucination checks** `[MVP]`: every number in the brief must match an engine metric. If any check fails, **fall back to the T0 template brief**. A well-formatted output is not necessarily a correct one.
- **Citations back to evidence** `[V1]`: every claim links to its trail, `Market Signal → Skill Gap → Constraint → Optimization → Intervention`.
- **Streaming briefs to the UI** `[V1]`: token stream from the SLM server → FastAPI SSE → Next.js.
- **Multilingual briefs** `[V2+]`: translate with Bhashini, or fine-tune Hindi/regional variants.
- **Export to PDF / DOCX** for committee meetings. `[V1]`
- **"Ask the plan" Q&A** `[V2+]`: the SLM may only *call* engine endpoints (`get_gap`, `run_scenario`, `explain_why_not`). It never invents numbers.

### 8.4 Explainable AI (XAI) across every module
Full design: [docs/xai.md](docs/xai.md). Every recommendation answers **Why? / Why not? / What if?** using the models' own internals (glass-box), aligned with MeitY's *"Understandable by Design"* principle (India AI Governance Guidelines, Nov 2025).
- **Engine:** binding-constraint report, per-course contribution + leave-one-out, shadow prices ✅, why-not ✅, counterfactual thresholds ("₹40K more and ml201 gets chosen"), minimal conflict set when infeasible (CP-SAT assumptions), robustness across forecast bands.
- **Forecasts:** pattern rationale (ADI/CV²), backtest leaderboard, MSTL trend/seasonality decomposition, plain-language intervals; SHAP only once feature-based ML exists `[V2+]`.
- **Skill extraction:** evidence clause ✅, highlighted spans, runner-up skills, calibrated confidence.
- **Gaps & briefs:** gap-score waterfall, data provenance, sentence-level citations, faithfulness report.
- **Cross-cutting:** one `Explanation` schema, "Why?" button on every number, Evidence Trail, audit log, model cards.

---

## 9. Frontend (Next.js 16)

### 9.1 Framework features to use
- **App Router + React Server Components:** heavy data pages render on the server with zero client JS for tables. `[MVP]`
- **Cache Components / `"use cache"` directive:** explicit, opt-in caching (Next.js 16 replaced the earlier implicit caching and PPR with this). Cache taxonomy and forecast pages, and keep scenario pages dynamic. `[MVP]`
- **`cacheLife` profiles + `revalidateTag()` / `updateTag()`:** invalidate dashboard caches precisely when a new forecast run or ingest finishes. `[V1]`
- **Server Actions** for forms (uploads, scenario save) with progressive enhancement. `[MVP]`
- **Turbopack** (default bundler in 16) for fast dev and builds. `[MVP]`
- **React 19.2:** `<Activity />` keeps tabs of the scenario-compare view mounted, View Transitions animate moves between Gap Map and drill-down, and `useEffectEvent` helps with slider handlers. `[V1]`
- **`instrumentation.ts`** for OpenTelemetry. `[V1]`

### 9.2 Screens
| Screen | Key visuals | Tier |
|---|---|---|
| **Command Center** | KPI cards (gap-closure %, budget used, market coverage), top-10 gaps, alerts | `[MVP]` |
| **Gap Map** | Skill × cohort heatmap; India choropleth of demand by state/district | `[MVP]` heatmap, `[V1]` map |
| **Demand Explorer** | Forecast lines with conformal bands; emerging-skill radar | `[MVP]` |
| **What-If Simulator** | Sliders, live re-optimization, Pareto curve, scenario diff | `[MVP]` |
| **Decision Brief** | Streamed narrative, inline citations, export | `[MVP]` |
| **Evidence Trail** | Lineage graph Signal → Gap → Constraint → Optimization → Intervention | `[V1]` |
| **Data Admin** | Upload wizard, connector status, data-quality report, taxonomy review queue | `[MVP]` upload, `[V1]` rest |

### 9.3 UI libraries
- **shadcn/ui + Tailwind**: component system.
- **Recharts**: standard charts (safe default). **Apache ECharts**: heatmaps, **Sankey** (budget → courses → skills → occupations), large datasets.
- **React Flow (`@xyflow/react`)**: Evidence Trail lineage graph and prerequisite graph.
- **deck.gl / MapLibre**: India district-level choropleth. `[V1]`
- **TanStack Table**: large virtualized skill and course tables.
- **TanStack Query**: client-side data fetching and cache for interactive views.

### 9.4 Type-safe API contract
- FastAPI generates OpenAPI, and **`@hey-api/openapi-ts`** generates a typed TS SDK, **Zod schemas** and **TanStack Query hooks**. There are no hand-written API types, and backend changes break the frontend build instead of production. `[MVP]`

---

## 10. Backend (Python / FastAPI)

- **FastAPI (0.142.x) on Starlette 1.0**, Python 3.13, managed with **uv**. `[MVP]`
- **Pydantic v2** models as the single schema source (API, validation, SLM brief schema). `[MVP]`
- **Native SSE / JSONL streaming endpoints** for solver progress and brief streaming. `[V1]`
- **Background jobs with arq + Redis** (async-native, fits FastAPI): ingestion, embedding, forecasting, long optimization runs, with job status polling. Use FastAPI `BackgroundTasks` only for small in-process work. `[MVP]`
- **Modular routers:** `ingest`, `taxonomy`, `demand`, `gaps`, `optimize`, `scenarios`, `briefs`, `admin`. `[MVP]`
- **Model serving:** load Sentence Transformers once per worker, and batch embedding requests. Use ONNX/quantized models for cheap CPU inference. `[V1]`
- **MCP server (`fastapi-mcp` / FastMCP)** `[V2+]`: expose engine endpoints as MCP tools, so institutions' own AI assistants can query D2S Bharat safely with the same auth and validation.
- **Rate limiting, idempotency keys** on optimize and ingest endpoints. `[V1]`

---

## 11. Security, Multi-Tenancy & Compliance

- **Auth: Better Auth** (TypeScript, self-hosted, your own Postgres) with the **organizations** plugin (institution = organization), roles, invitations, 2FA and passkeys. FastAPI verifies the session JWT. `[V1]` (`[MVP]`: simple email login)
- **RBAC roles:** Admin, Planner, Viewer, Auditor, CSR-Funder (read-only aggregate). `[V1]`
- **Tenant isolation** via Postgres RLS plus an org-scoped API layer. `[V1]`
- **DPDP Act 2023 alignment:** minimize student PII, aggregate before analytics, consent records, data-retention policy, audit log of who saw what. `[V1]`
- **Immutable audit log** of every scenario run and brief (inputs hash → outputs), so recommendations can be reproduced and audited. `[V1]`
- **SSO (SAML/OIDC)** for universities and government. `[V2+]`

---

## 12. Observability & Quality

- **OpenTelemetry** across Next.js (`instrumentation.ts`) and FastAPI, with one trace from the click through the API, optimizer and SLM. `[V1]`
- **SLM monitoring** (self-hosted; e.g. Langfuse self-hosted or plain OTel + Grafana): brief latency, template-fallback rate, faithfulness-check failures, model version per brief. `[V1]`
- **ML monitoring:** extraction precision on a labelled sample, forecast MAPE, solver time and optimality gap dashboards. `[V1]`
- **Testing:** pytest + Hypothesis for optimizer invariants (for example, "plan never exceeds budget"), Playwright E2E for core flows. `[MVP]`

---

## 13. Integrations & Ecosystem `[V2+]`

- **Public REST API + webhooks** ("new emerging skill detected", "forecast updated").
- **LMS/ERP connectors** (Moodle, SAP, university ERPs) to sync courses and cohorts automatically.
- **CSV/Excel round-trip** for planners who live in spreadsheets. `[MVP]`
- **Benchmarking network:** anonymized cross-institution comparisons (premium analytics tier).
- **Regional / national roll-up dashboards** for state skill missions.

---

## 14. Suggested Delivery Plan

| Phase | Scope |
|---|---|
| **Hackathon MVP** | Bulk-load ESCO (pinned version) + O\*NET database → crosswalk + embeddings → CSV job-post upload → skill extraction → statsmodels forecast → gap score → CP-SAT plan → what-if sliders → template decision brief (T0). Single tenant. |
| **Pilot (V1)** | Scenario-robust planning (P10/P50/P90), decision backtesting vs baselines, quarterly ESCO/O\*NET sync, crosswalk review UI, hybrid search, StatsForecast + conformal bands, shadow prices, Pareto view, Evidence Trail, Better Auth orgs + RLS, observability. |
| **Scale (V2+)** | Multi-institution resource pooling, NSQF/NQR mapping, NCS vacancies, PLFS signals, emerging-skill detection, hierarchical regional forecasts, robust/multi-period optimization, Bhashini multilingual, MCP server, benchmarking network. |

---

## Sources

**Frontend**
- [Next.js 16.3 release](https://nextjs.org/blog/next-16-3) · [Turbopack in 16.3](https://nextjs.org/blog/next-16-3-turbopack) · [Next.js 16: what's new (LogRocket)](https://blog.logrocket.com/next-js-16-whats-new/) · [Next.js 16 features (Strapi)](https://strapi.io/blog/next-js-16-features) · [Next.js 16 (MakerKit)](https://makerkit.dev/blog/tutorials/nextjs-16)
- [AI SDK UI (v6)](https://v6.ai-sdk.dev/docs/ai-sdk-ui) · [Vercel AI data stream protocol with Python](https://ai.pydantic.dev/ui/vercel-ai) · [AI SDK Python streaming template](https://vercel.com/templates/python/ai-sdk-python-streaming)
- [Best React chart libraries 2026](https://www.fusioncharts.com/blog/best-react-chart-library/) · [React chart libraries (Querio)](https://querio.ai/articles/best-react-chart-libraries-data-visualization)
- [FastAPI: generate clients](https://fastapi.tiangolo.com/ja/advanced/generate-clients/) · [openapi-ts](https://pypi.org/project/openapi-ts)
- [Better Auth vs Clerk vs NextAuth (MakerKit)](https://makerkit.dev/blog/tutorials/better-auth-vs-clerk) · [Next.js auth comparison 2026](https://www.iloveblogs.blog/post/nextjs-authentication-comparison-2026)

**Backend & data**
- [FastAPI release notes](https://fastapi.tiangolo.com/release-notes/) · [fastapi on PyPI](https://pypi.org/project/fastapi/) · [FastAPI updates (Releasebot)](https://releasebot.io/updates/tiangolo/fastapi)
- [pgvector 0.8.0 (Nile)](https://www.thenile.dev/blog/pgvector-080) · [Scaling pgvector (ClickHouse)](https://clickhouse.com/resources/engineering/scale-vector-search-postgres) · [Hybrid vector search (Crunchy Data)](https://www.crunchydata.com/blog/hybrid-vector-search)
- [Hybrid search in PostgreSQL (ParadeDB)](https://paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual)
- [Python data stack 2026](https://awabrao.lovable.app/blog/python-data-stack-2026) · [DuckDB newsletter July 2026](https://motherduck.com/blog/duckdb-ecosystem-newsletter-july-2026/)
- [Background jobs patterns (arq/Celery)](https://skills.sh/yonatangross/orchestkit/background-jobs) · [FastAPI BackgroundTasks vs Celery](https://levelup.gitconnected.com/fastapi-background-tasks-vs-celery-which-is-right-for-your-application-dff0a7216e55)
- [FastAPI MCP server guide](https://fast.io/resources/fastapi-mcp-server-guide/) · [FastMCP + Pydantic AI](https://gofastmcp.com/integrations/pydantic-ai) · [FastMCP vs fastapi-mcp 2026](https://mcp.directory/blog/fastmcp-vs-fastapi-mcp-vs-python-sdk-2026)

**ML, NLP & optimization**
- [esco-skill-extractor](https://pypi.org/project/esco-skill-extractor) · [Skill extraction research (arXiv 2601.09119)](https://arxiv.org/html/2601.09119) · [arXiv 2505.24640](https://arxiv.org/html/2505.24640v1) · [TechWolf skill-extraction dataset](https://huggingface.co/datasets/TechWolf/skill-extraction-techwolf/blob/main/README.md)
- [Ontology-aligned embeddings for labour-market analytics](https://arxiv.org/pdf/2509.04942) · [JAMES job-title normalization](https://arxiv.org/abs/2202.10739v2) · [Andela emerging roles research](https://www.andela.com/research/emergent-roles)
- [StatsForecast](https://nixtlaverse.nixtla.io/statsforecast) · [StatsForecast conformal prediction](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/ConformalPrediction) · [HierarchicalForecast paper](https://www.paperswithcode.com/paper/hierarchicalforecast-a-python-benchmarking) · [MAPIE paper](https://arxiv.org/pdf/2207.12274)
- [OR-Tools GitHub](https://github.com/google/or-tools) · [CP-SAT Primer](https://www.sourcepulse.org/projects/2335957) · [OR-Tools agent skill notes](https://skills.cat/skills/tondevrel/scientific-agent-skills/ortools) · [Sensitivity analysis / shadow prices](https://lumix.readthedocs.io/en/latest/user-guide/analysis/sensitivity.html)

**SLM & observability**
- [Best lightweight language models 2026 (Prem)](https://blog.premai.io/best-lightweight-language-models-worth-running) · [Small local LLMs](https://www.promptquorum.com/local-llms/small-local-llm-models) · [Best small LLM for local deployment](https://www.ertas.ai/best/best-small-llm-for-local-deployment)
- [Fine-tune & self-host with Unsloth](https://pinggy.io/blog/finetune_and_selfhost_llms_locally_with_unsloth/) · [Fine-tune locally guide 2026](https://toolhalla.ai/blog/fine-tune-llm-locally-guide-2026)
- [llguidance](https://github.com/guidance-ai/llguidance) · [Structured output with local LLMs](https://insiderllm.com/guides/structured-output-local-llms/) · [vLLM structured outputs](https://vllm.hexdocs.pm/structured_outputs.html)
- [Langfuse OpenTelemetry](https://langfuse.com/integrations/native/opentelemetry.md)

**Indian data & taxonomies**
- [ESCO web service API](https://esco.ec.europa.eu/en/use-esco/use-esco-services-api/esco-web-service-api) · [ESCO API guide](https://jobspipe.dev/blog/esco-api) · [O\*NET API guide](https://jobspipe.dev/blog/onet-api.md)
- [National Career Service (Wikipedia)](https://en.wikipedia.org/wiki/National_Career_Service_(India)) · [NCS at centre of India's job market](https://www.tice.news/governance-policy/national-career-service-portal-digital-jobs-india-12230145) · [Skill India employment data](https://www.ibef.org/blogs/how-skill-india-measures-success-through-skill-and-employment-data)
- [National Qualifications Register (NCVET)](https://ncvet.gov.in/?p=2524) · [NQR about](https://nqr.gov.in/aboutus) · [NQR overview](https://nationalskillsnetwork.in/national-qualifications-register-nqr-a-unified-platform-for-skill-based-qualifications/)
- [Bhashini (Vikaspedia)](https://en.vikaspedia.in/viewcontent/e-governance/digital-india/bhashini) · [Bhashini overview](https://anantamias.com/bhashini-bhasha-interface-for-india/?pdf=1)
