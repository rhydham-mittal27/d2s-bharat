# RQ4: what should an institution teach first, within its budget?

Prescriptive analytics: the RQ1–RQ3 evidence feeds an integer optimisation model (Google OR-Tools
CP-SAT) that chooses **which courses to run and for how many learners**, under budget, trainer-hour,
seat, batch-size, prerequisite and either/or constraints. Code: `backend/src/d2s/analysis/plan.py`,
`backend/src/d2s/engine/`, `backend/scripts/run_rq4.py`. Tables: `dataset/processed/rq4/`.

## Inputs: what is data and what is assumed

| Input | Source | Values |
|---|---|---|
| **Skill gaps** | **Data (JDS)**: juniors scoring below the high-hike group's median in each area, with the 139 JDS juniors as the illustrative cohort | Maths & stats 82 · Coding 77 · AI & ML 71 · Big data 65 · Dashboards & storytelling 58 |
| **Value of closing a gap** | **Data (RQ1 + RQ2)**: equal-weighted mix of market demand share, log market pay odds ratio (adjusted) and log RQ2 junior-hike odds ratio, each scaled to its maximum | Maths & stats 100 · Coding 99.7 · AI & ML 77.3 · Dashboards 62.9 · Big data 59.4 |
| **Uncertainty** | **Data**: the same score at the CI lower and upper bounds, under 3 weighting schemes (balanced / market-led / outcome-led), giving 9 scenarios | see robustness |
| **Courses, costs, seats, hours, prerequisites** | **Assumption**: `course_catalogue_assumptions.csv` (editable; each row states its cost basis, e.g. trainer ₹1,500/h, SAS Viya for Learners free for academic use) | 9 candidate courses |
| **Budget / trainer capacity** | **Assumption** (scenario) | ₹4.0 lakh, 200 trainer-hours |

Personality traits (RQ3) are **not** modelled as trainable courses: personality is relatively
stable, so those findings inform mentoring, role allocation and assessment, not curriculum.

## The optimal plan (₹4.0 L, 200 trainer-hours): proven optimal

| Course | Seats | Cost | Trainer-hours |
|---|---|---|---|
| Python & SQL for Analytics | 35 | ₹0.88 L | 40 |
| SAS Programming on SAS Viya for Learners | 60 | ₹0.45 L | 30 |
| Applied Statistics & Probability | 60 | ₹0.90 L | 40 |
| Machine Learning Foundations | 27 | ₹1.13 L | 48 |
| Dashboards & Storytelling with Power BI | 60 | ₹0.64 L | 28 |
| **Total** | **242** | **₹3.995 L** | **186 / 200** |

**Gap closed:** Coding 100% · Maths & stats 94% · Dashboards & storytelling 93% · AI & ML 38% ·
Big data 0%.

### Why some courses were left out (forced re-solve)
| Course not chosen | Impact if forced in | Reason |
|---|---|---|
| Deep Learning & NLP | −35.4% | expensive per seat (GPU) and needs ML first; the money closes more gap in foundations |
| Integrated Bootcamp | −17.7% | high fixed cost, only partial coverage of each area |
| Big-Data Stack | −6.6% | big data has the lowest evidence score; competes for the same budget |
| Tableau storytelling | −3.8% | either/or with Power BI, which closes nearly the same gap more cheaply |

## Sensitivity: what would change the decision?
- **Budget is the binding constraint at ₹4 L.** +₹1 lakh raises impact from 20,903 to 23,391
  (**+11.9%**). The LP shadow price (0.0235 per ₹) agrees with the exact re-solve (0.0249 per ₹).
- **Trainer-hours are not binding at ₹4 L** (186 of 200 used; marginal value 0).
- **Beyond about ₹4.5 L, trainer capacity becomes the bottleneck, not money.** With 200 hours, even
  ₹8 L buys no more impact (budget marginal value = 0; each extra trainer-hour = +59 impact).
  With **400 trainer-hours**, Big Data and Deep Learning & NLP join the plan and impact rises to
  27,158 at ₹7.7 L (AI & ML gap 96% closed, big data 62%). **Implication: expand faculty, or share
  trainers across institutions, before raising the budget.**
- **Diminishing returns:** the Pareto frontier rises steeply to about ₹2.2 L (foundations), then
  flattens.

## Robustness across 9 evidence scenarios

| Course | Chosen in | Seats range |
|---|---|---|
| **Applied Statistics & Probability** | **9 / 9** | 60 (always the maximum) |
| **SAS Programming (VFL)** | **9 / 9** | 60 |
| **Python & SQL** | **9 / 9** | 15–35 |
| **Machine Learning Foundations** | **9 / 9** | 27–50 |
| Dashboards & Storytelling (Power BI) | 7 / 9 | dropped only in the market-led low/point scenarios |
| DL & NLP, Big Data, Tableau, Bootcamp | 0 / 9 | at ₹4 L / 200 h |

**Safe core = Statistics + SAS + Python & SQL + ML foundations**: recommended whichever way the
evidence is weighted or bounded. Dashboards & storytelling is included unless the institution
weights *only* what job postings advertise. RQ2 shows it is the second-strongest junior-hike
signal, so we recommend keeping it.

## Recommendation for an institution (illustrative cohort)
1. **Run the safe core now:** statistics (60 seats), SAS on Viya for Learners (60), Python & SQL
   (35), ML foundations (27+).
2. **Add Power BI storytelling (60 seats):** strongly linked to junior salary hikes, under-advertised
   by the market.
3. **To go further, add trainer capacity before budget:** at 400 trainer-hours, add the Big-Data
   bundle and DL/NLP.
4. **Use RQ3 for people decisions:** openness and conscientiousness are associated with senior
   success, so use them in mentoring, project allocation and structured assessment, not as courses.

## Limitations
- Costs, seats and trainer-hours are **assumptions**; edit the catalogue CSV and rerun. The SAS
  course is favoured partly because Viya for Learners has zero per-seat cost.
- "Gap" uses the JDS cohort as a stand-in for an institution's own assessment data; target = the
  high-hike group's median (5.0 for four areas, 3.8 for big data).
- Coverage % (how much of a learner's gap one seat closes) is an assumption, not measured.
- The objective values learners equally within an area and ignores timing (single-term plan).
  Multi-period sequencing and multi-institution pooling are designed but not run here.

## Figures
![](01_inputs.png)
![](02_plan.png)
![](03_pareto.png)
![](04_robustness.png)
