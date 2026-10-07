# D2S Bharat

**From labour-market demand to institutional action.**

D2S Bharat (Demand-to-Skills) is an AI-powered workforce decision engine for institutions. It answers one question:

> *Given what the market needs and what we can afford, what should we teach next?*

It takes in market demand, finds the skill gaps, weighs the training options, and recommends the best intervention:

```
Market Demand → Skill Gaps → Training Choices → Optimized Intervention
```

**Built for:** universities • skilling providers • CSR workforce programs

> Built for the **Build for Bharat** hackathon by **Team Code5urge**.
> **Status:** concept / proposal stage. This README describes the planned system as presented in the pitch deck.

---

## The Problem: The Market-Skills Gap

1. **Demand keeps changing.** New roles and technologies keep creating new skill requirements. Meanwhile, institutions often rely on old curricula and static skill assessments.
2. **Skills ≠ market demand.** Institutions may know what their students are learning now. They lack a systematic way to tell which skills are becoming critical in the job market.
3. **Training capacity is limited.** Budgets, seats, trainers and prerequisites impose hard limits. Institutions can't teach everything.

Most existing tools stop at *"What skills are missing?"* D2S Bharat goes further and recommends what to do about it.

---

## The Solution

| Pillar | What it does |
|---|---|
| **1. Demand Intelligence** | Collects job, occupation and emerging-skill signals. Extracts and normalizes skills using ESCO / O\*NET. Forecasts upcoming skill-demand trends. |
| **2. Skill-Gap Intelligence** | Compares market demand with the skills a student cohort already has. Identifies and prioritizes high-impact gaps. Links each gap to relevant occupations and courses. |
| **3. Constraint-Aware Optimization** | Picks the training interventions with the highest potential impact, within real-world limits: budget, training seats, prerequisites, trainer capacity and institutional constraints. |
| **4. Explainable Decisions** | Turns the optimized result into a human-readable decision brief that the institution can act on. |

**Key design principle:** *The optimization engine makes the decision. Our own self-hosted small language model (SLM) only explains and summarizes the resulting plan.* (SLM ≠ Decision Maker). No third-party LLM API is used, so no institutional data leaves our servers.

---

## Architecture

```
┌──────────────────────────┐
│ DATA SOURCES             │  ESCO, O*NET, National Career Service (NCS), Skill India Digital,
│                          │  job descriptions, institutional courses, cohort skills, budgets & capacity
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ INGESTION & NORMALIZATION│  Data parsing → Skill extraction → Sentence Transformers
│                          │  → Semantic matching → ESCO/O*NET skill mapping
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ DATA LAYER               │  PostgreSQL + pgvector
│                          │  Skill, Occupation, Job, Location, Course, Cohort, Institution,
│                          │  Demand Signal, Constraint, Intervention
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ ANALYTICS & ML ENGINE    │  scikit-learn + statsmodels for demand forecasting
│                          │  → demand-weighted cohort skill-gap analysis
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ CORE DECISION ENGINE     │  Constrained optimization with Google OR-Tools
│                          │  Objective: maximize skill-gap closure + market-opportunity coverage
│                          │  subject to budget, seats, prerequisites, trainer capacity
│                          │  → prioritized courses, programs, resource allocation, projected coverage
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ APPLICATION LAYER        │  FastAPI: ingestion, analytics, forecasting, gap analysis,
│                          │  optimization, what-if scenarios, decision-brief APIs
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ EXPLAINABILITY LAYER     │  Own fine-tuned SLM (self-hosted) receives ONLY optimization results,
│                          │  evidence, assumptions & metrics → human-readable Decision Brief
│                          │  (numbers checked against the engine; template fallback)
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ FRONTEND                 │  React / Next.js dashboard: Command Center, Gap Map,
│                          │  What-If Simulator, Decision Brief
└────────────┬─────────────┘
             ▼
   INSTITUTIONAL ACTION:  Priority Skill Gaps → Recommended Courses
                          → Budget/Capacity Allocation → Expected Market Opportunity Coverage
```

### Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React / Next.js |
| Backend / API | FastAPI |
| Database | PostgreSQL + pgvector |
| Skill extraction & semantic matching | Sentence Transformers |
| Demand forecasting | scikit-learn, statsmodels |
| Optimization | Google OR-Tools |
| Explanation layer | Own SLM: open 1–4B base model fine-tuned with LoRA (Unsloth), served with llama.cpp |
| Skill taxonomies & data | ESCO, O\*NET, Indian labour-market signals (NCS, Skill India Digital) |

---

## What Makes It Different

1. **Decision, not just diagnosis.** Other platforms identify missing skills. D2S Bharat decides what an institution should teach next.
2. **Constraint-aware AI.** Recommendations account for budget, seats, prerequisites and capacity, so they can actually be carried out.
3. **Demand-weighted skill intelligence.** Labour-market demand is combined with cohort-level gaps, so skills are ranked by their potential market impact.
4. **Optimization before generative AI.** OR-Tools picks the intervention mathematically. Our own SLM only explains the result, which reduces reliance on subjective AI-generated advice.
5. **What-If Decision Simulator.** Institutions can change the budget, seats or course choices and immediately compare the effect on skill-gap closure and market coverage.
6. **Evidence-backed and auditable.** Every recommendation can be traced back through
   `Market Signal → Skill Gap → Constraint → Optimization → Intervention`.
7. **Explainable AI (XAI) by design.** Every number answers *Why? Why not? What if?*: binding constraints, each course's contribution, counterfactuals ("₹40K more and this course gets chosen"), forecast decomposition and the exact job-post text behind each skill. This aligns with MeitY's "Understandable by Design" principle. See [docs/xai.md](docs/xai.md).

---

## Scalability

The same AI and optimization engine is designed to grow from a single institution to national scale:

- **Institutional scale:** start with one institution, then expand across its departments, cohorts and courses.
- **Regional scale:** combine labour-market signals across cities, states and regions to plan skills and training for each region.
- **Ecosystem scale:** connect universities, skilling providers and CSR programs, and keep adding skills, occupations, courses and datasets.

**Vision:** Institution → Region → Multi-Institution Ecosystem → National Workforce Intelligence

---

## Impact

**For institutions**
- Curriculum that matches market demand more closely
- Better use of training budgets and available seats
- Emerging skill gaps spotted sooner
- Curriculum and program planning backed by evidence

**For learners and the workforce**
- More relevant, market-ready skills
- Better access to high-demand training
- Less mismatch between education and employment
- Stronger regional workforce readiness

---

## Cost & Viability

**Estimated MVP infrastructure cost: ₹3K–₹11K / month**

| Component | Est. monthly cost |
|---|---|
| Cloud + database | ₹2K–₹5K |
| Embeddings / semantic search | ₹1K–₹3K |
| Self-hosted SLM explanation layer (CPU inference; no per-token fees) | ₹0–₹3K (extra RAM/CPU) |
| Core ML + optimization | Open-source |

Plus a **one-time** fine-tuning cost of a few rented GPU-hours per model version.

**Why it's economically viable:**
- No expensive foundation-model pre-training: we fine-tune a small open model
- Built on open-source ML and optimization tools
- No third-party LLM API bills; the SLM is used only in the final explanation layer
- The modular architecture keeps infrastructure costs under control
- Costs scale with actual institutional usage

### Business Model

- **SaaS for institutions:** annual subscription based on institution size and usage
- **Enterprise / government:** customized workforce-intelligence deployments and large regional planning contracts
- **Analytics & decision support:** premium forecasting, scenario analysis and benchmarking
- **Custom integrations:** connecting to existing university and skill-development systems

**Commercialization path:**
MVP → Institutional Pilot → SaaS Deployment → Multi-Institution Platform → Regional / National Scale

---

## Long-Term Vision

D2S Bharat aims to become the **decision-intelligence layer that connects labour-market demand, education and workforce planning.**

> *"Build once. Scale across institutions, regions and workforce ecosystems."*

```
Better Data → Better Training Decisions → Better Skill Alignment → Stronger Workforce Readiness
```

---

## Team Code5urge

- Shaman Sharma
- Anurag Prajapati
- Amrita Raman
- Rhydham Mittal
