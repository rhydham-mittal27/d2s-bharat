# Explainable AI (XAI) in D2S Bharat

> Every number and recommendation D2S shows must answer **"why?"**, **"why not?"** and **"what would change it?"**
> Companion to [core-services.md](core-services.md) and [features.md](../features.md) §8. Research snapshot: October 2026.

## 1. Why XAI matters here

- **Decision-makers must defend the plan.** A dean or CSR committee can't approve "the AI said so". They need the drivers, the trade-offs and the evidence.
- **Policy alignment.** India's **AI Governance Guidelines** (MeitY, Nov 2025) include the *"Understandable by Design"* principle: disclose how AI systems make decisions, their data sources and their behaviour. D2S builds this in rather than adding it later.
- **Trust and correction.** When an explanation looks wrong (bad skill match, odd forecast), users can spot and fix the input. That makes the system better over time.

## 2. Design principles

1. **Glass-box first.** Our core components are *inherently interpretable*: an explicit optimization model, classical forecasting models and similarity search. Explanations come from their real internals (constraints, duals, model components, matched text). We don't approximate a black box after the fact.
2. **Faithful over pretty.** Every explanation is computed from the same data and model that produced the decision, and can be re-run. The SLM may *phrase* an explanation but never *create* one.
3. **Three questions everywhere:** *Why?* (drivers), *Why not?* (the alternative and its cost), *What if?* (the smallest change that flips the outcome).
4. **One explanation format.** All modules emit the same `Explanation` object, so the UI, audit log and SLM treat them uniformly.

```python
class Factor(BaseModel):       # one driver of the outcome
    name: str                  # "budget", "demand_weight[python]", "pattern=intermittent"
    value: float | str
    contribution: float | None # share of the outcome attributable to this factor
    source: str                # "engine.dual", "forecast.mstl", "matcher.lexical", ...

class Explanation(BaseModel):
    subject: str               # "plan:123/course:ml201", "forecast:sql", "match:post42/skill:s:sql"
    question: Literal["why", "why_not", "what_if"]
    summary: str               # one plain-language sentence (template-generated)
    factors: list[Factor]
    evidence: list[str]        # ids/spans: job-post clauses, data rows, constraint names
    counterfactuals: list[str] # "with ₹40,000 more budget, ml201 would be chosen"
    confidence: float | None
    method: str                # how it was computed (for model cards / audits)
```

## 3. XAI per module

### 3.1 Decision Engine (optimization XAI)

| Question | Technique | Status |
|---|---|---|
| Why this plan? *Which limits shaped it?* | **Binding-constraint report:** slack of budget, trainer hours, seats, per-course max seats, equity floors. A constraint with zero slack is "binding" and is what's holding the plan back. | new `[MVP]` |
| How much did each course contribute? | **Contribution breakdown:** each chosen course's share of every skill's closed learners (proportional to its coverage × seats), plus a **leave-one-out** re-solve: impact lost if the course is dropped *and the rest re-optimized*. | new `[MVP]` |
| What is one more rupee / hour worth? | **Shadow prices** (GLOP duals) + **exact finite differences** | ✅ built (`engine/sensitivity.py`) |
| Why not course X? | **Forced re-solve:** impact delta of including X, plus static blockers (prerequisite excluded, exclusive rival, too expensive) | ✅ built (`engine/explain.py`) |
| What would make X chosen? | **Counterfactual threshold search:** binary-search the smallest budget increase, or the smallest demand-weight increase of X's skills, at which the optimal plan includes X. Recent research shows exact integer-program counterfactuals are computationally hard (Σ₂ᵖ-complete), so we search **one parameter at a time**, which is tractable and easy to read. | new `[V1]` |
| Why is no plan possible? | **Minimal conflict set:** every constraint group (budget, hours, seats, each equity floor, each mandatory/forced course) gets an *assumption literal*. When CP-SAT reports INFEASIBLE, `sufficient_assumptions_for_infeasibility()` names a small set that conflicts: *"the budget + the 100% ML floor + mandatory Excel can't all hold"*. | new `[MVP]` |
| How do cost and impact trade off? | **Pareto frontier** | ✅ built (`engine/pareto.py`) |
| How safe is this plan if demand is off? | **Robustness check:** re-solve with forecast `lo` and `hi` weights and report **plan stability** (overlap of chosen courses) and impact range. A course that survives all three is a robust recommendation. | new `[V1]` |

### 3.2 Demand forecasting XAI

| Question | Technique | Status |
|---|---|---|
| Why this model? | **Pattern rationale:** show ADI and CV² with the Syntetos–Boylan class ("sparse: demand in 1 of every 3.1 months"), plus the **backtest leaderboard** (MAE of every candidate, winner highlighted). | pattern ✅, leaderboard new `[MVP]` |
| What drives the forecast? | **MSTL decomposition** (StatsForecast): split history into *trend + seasonality + remainder*, e.g. "rising +4 postings/month; peaks every July (placement season)". | new `[V1]` |
| How sure is it? | **Prediction intervals** (native or conformal), turned into plain language: "80% chance between 120 and 180 postings". | ✅ built |
| Is there enough data? | **Data-sufficiency flags:** short history, all-zero, recent structural break, gap-filled months count. | partial ✅ (`short`, `zero`) |
| Which outside signals matter? *(once exogenous features exist)* | **SHAP** on the gradient-boosting model with lag/exogenous features (posting velocity, PLFS growth, Hot Technology flags). Only needed when we add feature-based ML. | `[V2+]` |

### 3.3 Skill extraction XAI

| Question | Technique | Status |
|---|---|---|
| Why was this skill detected? | **Evidence clause** + match type (lexical / semantic / both) + similarity vs. threshold | ✅ built (`SkillMatch.evidence`, `.lexical`, `.similarity`) |
| Which words triggered it? | **Span highlighting:** exact label span for lexical hits. For semantic hits, **token alignment** (MaxSimE-style: match each clause token to its most similar skill-description token by contextual embedding) to highlight the words that drove the match. | lexical span new `[MVP]`, token alignment `[V1]` |
| What else could it have been? | **Runner-up skills:** the next 2 candidates with their similarity, so reviewers can correct a near-miss in one click. | new `[MVP]` |
| How reliable is this match? | **Calibrated confidence:** map similarity to precision measured on a labelled sample (isotonic calibration). | `[V1]` |

### 3.4 Skill-gap XAI
- **Gap score decomposition:** `gap = forecast demand × (target − current proficiency)`, shown as a waterfall: *demand 140 × shortfall 0.45 = 63*. `[MVP]`
- **Provenance:** which job posts, which taxonomy version (`onet-30.3`, `esco-1.2.1`), and which cohort assessment each number came from. `[MVP]`

### 3.5 SLM brief XAI
- **Sentence-level citations:** each sentence carries the `Explanation.subject` IDs it verbalizes. Clicking it opens the underlying explanation. `[V1]`
- **Faithfulness report:** numbers checked against the engine, plus whether the brief came from the SLM or the template fallback, and why. `[MVP]`
- The SLM receives `Explanation` objects as input, so it **narrates computed explanations and never invents reasons**.

## 4. Cross-cutting

| Item | What | Tier |
|---|---|---|
| **Evidence Trail UI** | Graph view `Market Signal → Skill Gap → Constraint → Optimization → Intervention`; every node opens its `Explanation`. | `[V1]` |
| **"Why?" button** | On every course, number and chart: shows `summary`, factor waterfall, evidence, counterfactuals. | `[MVP]` |
| **Audit log** | Each plan stores its inputs hash, solver parameters, model versions and generated explanations. Re-running reproduces them exactly (fixed seed, integer data). | `[V1]` |
| **Model cards** | One page per model (embedding model, forecaster, SLM): intended use, training data, metrics, limits, version. Needed for transparency reports. | `[V1]` |
| **Explanation quality checks** | Automated tests: explanations are *faithful* (re-running the counterfactual really flips the decision), *stable* (same input gives the same explanation) and *complete* (every chosen course has a contribution entry). | `[MVP]` |

## 5. Implementation plan (backend)
```
backend/src/d2s/xai/
├── schemas.py        # Factor, Explanation
├── engine.py         # binding_constraints, contributions (+ leave-one-out), infeasibility_core,
│                     #   counterfactual_budget, counterfactual_weight, robustness
├── forecasting.py    # pattern_rationale, backtest_leaderboard, mstl_decomposition
├── skills.py         # lexical_span, runner_ups (token alignment in V1)
└── narrate.py        # template sentences for Explanation.summary (T0; SLM narrates later)
```
New worker task: `explain` (`{kind, subject, payload}` → `Explanation`), so heavy explanations (leave-one-out, counterfactual search, robustness) run in the background with progress like other jobs.

## Sources
- India AI Governance Guidelines (MeitY, Nov 2025): [IASGyan summary](https://www.iasgyan.in/daily-current-affairs/meity-released-india-ai-governance-guidelines-roadmap) · [JSA Law analysis](https://www.jsalaw.com/wp-content/uploads/2025/11/JSA-Prism-InfoTech-Nov-2025-India-AI-Governance.final_.pdf) · [CASRAI guide](https://casrai.org/guides/india-ai-governance-guidelines)
- Counterfactual explanations for optimization: [Counterfactual Explanations for Integer Optimization Problems (arXiv 2510.17624)](https://arxiv.org/html/2510.17624v2) · [Counterfactual explanations / inverse CP (CP 2021)](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2021.35)
- Infeasibility cores with CP-SAT assumptions: [CPMpy: solving with assumptions / unsat cores](https://cpmpy.readthedocs.io/en/io/unsat_core_extraction.html) · [OR-Tools CP-SAT response API](https://ocaml.org/p/ortools/9.15.0-1/doc/ortools/Ortools/Sat/Response/index.html)
- Forecast decomposition: [StatsForecast MSTL](https://nixtlaverse.nixtla.io/statsforecast/docs/models/multipleseasonaltrend.html) · [Multiple seasonalities tutorial](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/MultipleSeasonalities) · [MSTL paper](https://arxiv.org/abs/2107.13462)
- Explaining embedding similarity: [Interpretable text embeddings & similarity explanation primer](https://arxiv.org/pdf/2502.14862v2) · [MaxSimE](https://lamarr-institute.org/publication/maxsime-explaining-transformer-based-semantic-similarity-via-contextualized-best-matching-token-pairs/) · [Explaining text similarity in transformers](https://arxiv.org/html/2405.06604v1)
