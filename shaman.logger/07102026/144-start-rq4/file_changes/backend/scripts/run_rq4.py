"""RQ4: evidence-based, budget-constrained training plan (CP-SAT) with sensitivity and robustness.

Needs: run_rq1.py (triangulation) and run_rq2_rq3.py outputs; course catalogue CSV (assumptions).
Usage: uv run python scripts/run_rq4.py [--budget 400000] [--trainer-hours 200]
Outputs: dataset/processed/rq4/* ; reports/rq4/*.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from d2s.analysis import plan  # noqa: E402
from d2s.analysis.traits import load_jds  # noqa: E402
from d2s.config import get_settings  # noqa: E402
from d2s.engine import SolverOptions, analyze, frontier, solve, why_not  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
SAS = ROOT / "hackathon" / "SAS Data Problem Statement and Instructions Hackathon"
S = get_settings()
OUT = S.dataset_dir / "processed" / "rq4"
FIG = ROOT / "reports" / "rq4"
CATALOGUE = FIG / "course_catalogue_assumptions.csv"
OPTS = SolverOptions(time_limit_s=20, workers=8, gap_limit=0.0)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 9})


def lakh(x):
    return f"₹{x / 1e5:.2f} L"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_inputs(gaps, w):
    g = gaps.set_index("area").loc[plan.AREAS]
    labels = [plan.LABEL[a] for a in plan.AREAS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    a.barh(labels, g["learners_short"], color="#ef4444", alpha=0.8)
    for i, (n, t) in enumerate(zip(g["learners_short"], g["target_score"], strict=True)):
        a.text(n, i, f" {n} (target {t:.1f})", va="center", fontsize=8)
    a.set_title("Gap: juniors below the high-hike group's median (of 139)")
    ww = w.set_index("area").loc[plan.AREAS]
    left = np.zeros(len(ww))
    for comp, c, lab in (("demand_c", "#2563eb", "market demand"), ("market_c", "#ea580c", "market pay premium"),
                         ("hike_c", "#16a34a", "junior-hike link")):
        v = ww[comp].to_numpy() / 3 * 100 / (ww[["demand_c", "market_c", "hike_c"]].sum(axis=1).max() / 3)
        b.barh(labels, v, left=left, color=c, label=lab)
        left += v
    b.set_title("Value per learner (balanced evidence score, top = 100)")
    b.legend(fontsize=7, loc="lower right")
    return save(fig, "01_inputs.png")


def fig_plan(plan_obj, problem, budget):
    names = {c.id: c.name for c in problem.courses}
    cs = plan_obj.courses
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    a.barh([names[c.course_id][:42] for c in cs][::-1], [c.cost for c in cs][::-1], color="#2563eb")
    for i, c in enumerate(cs[::-1]):
        a.text(c.cost, i, f" {c.seats} seats · {lakh(c.cost)} · {c.trainer_hours} h", va="center", fontsize=8)
    a.set_title(f"Optimal plan: {lakh(plan_obj.total_cost)} of {lakh(budget)}, "
                f"{plan_obj.total_trainer_hours} trainer-hours ({plan_obj.status.value})")
    a.set_xlabel("cost (₹)")
    sk = {s.skill_id: s for s in plan_obj.skills}
    labels = [plan.LABEL[a_] for a_ in plan.AREAS]
    closed = [sk[a_].learners_closed for a_ in plan.AREAS]
    short = [sk[a_].learners_short for a_ in plan.AREAS]
    y = np.arange(len(labels))
    b.barh(y, short, color="#e2e8f0", label="learners with the gap")
    b.barh(y, closed, color="#16a34a", label="gap closed by the plan")
    for i, (c, s) in enumerate(zip(closed, short, strict=True)):
        b.text(s, i, f" {100 * c / s:.0f}%", va="center", fontsize=8)
    b.set_yticks(y, labels)
    b.legend(fontsize=7, loc="lower right")
    b.set_title("Gap closure by skill area")
    return save(fig, "02_plan.png")


def fig_pareto(points, budget):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xs = [p.cost / 1e5 for p in points]
    ys = [p.impact for p in points]
    ax.plot(xs, ys, "-o", color="#2563eb")
    for p in points:
        ax.annotate(f"{len(p.plan.courses)} courses", (p.cost / 1e5, p.impact), fontsize=7,
                    xytext=(4, -10), textcoords="offset points")
    ax.axvline(budget / 1e5, color="k", ls="--", lw=0.8)
    ax.text(budget / 1e5, min(ys), " chosen budget", fontsize=8)
    ax.set_xlabel("cost (₹ lakh)")
    ax.set_ylabel("impact (evidence-weighted learners trained)")
    ax.set_title("Cost-impact Pareto frontier: what each extra lakh buys")
    return save(fig, "03_pareto.png")


def fig_robustness(matrix):
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(matrix) + 1.8))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="Greens", vmin=0, vmax=matrix.to_numpy().max())
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)), [i[:40] for i in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.iat[i, j]
            ax.text(j, i, int(v) if v else "–", ha="center", va="center", fontsize=8,
                    color="white" if v > matrix.to_numpy().max() * 0.6 else "black")
    ax.set_title("Robustness: seats per course across 9 evidence scenarios (scheme × CI bound)")
    fig.colorbar(im, ax=ax, label="seats")
    return save(fig, "04_robustness.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=400_000)
    ap.add_argument("--trainer-hours", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    jds = load_jds(SAS / "JDS Skill Traits.xlsx").frame
    tri = pd.read_csv(S.dataset_dir / "processed" / "rq1" / "triangulation_rq1_rq2.csv")
    courses = plan.load_catalogue(CATALOGUE)
    gaps = plan.cohort_gaps(jds)
    w = plan.evidence_weights(tri, "balanced", "point")
    gaps.to_csv(OUT / "gaps.csv", index=False)
    w.to_csv(OUT / "weights_balanced.csv", index=False)
    problem = plan.build_problem(gaps, w, courses, args.budget, args.trainer_hours)

    best = solve(problem, OPTS)
    print("PLAN", best.status, "objective", round(best.objective, 1), "cost", best.total_cost,
          "hours", best.total_trainer_hours, best.messages)
    for c in best.courses:
        print("  ", c.course_id, c.seats, c.cost, c.trainer_hours)
    pd.DataFrame([c.model_dump() for c in best.courses]).to_csv(OUT / "plan_courses.csv", index=False)
    pd.DataFrame([s.model_dump() for s in best.skills]).to_csv(OUT / "plan_skills.csv", index=False)

    sens = analyze(problem, best, budget_delta=100_000, hours_delta=20, options=OPTS)
    marg = pd.DataFrame([m.model_dump() for m in sens.marginals])
    marg.to_csv(OUT / "sensitivity.csv", index=False)
    print(marg.round(4).to_string(index=False))
    print("LP shadow prices:", sens.shadow_prices.model_dump(exclude={"course_reduced_costs"}))

    points = frontier(problem.model_copy(update={"constraints": problem.constraints.model_copy(
        update={"budget": 800_000})}), levels=10, min_budget=100_000, options=OPTS)
    pareto = pd.DataFrame([{"budget_cap": p.budget_cap, "cost": p.cost, "impact": p.impact,
                            "courses": "+".join(c.course_id for c in p.plan.courses)} for p in points])
    pareto.to_csv(OUT / "pareto.csv", index=False)
    print(pareto.to_string(index=False))

    excluded = [c.id for c in courses if c.id not in best.seats()]
    wn = pd.DataFrame([why_not(problem, best, cid, OPTS).model_dump(exclude={"alternative_plan"}) for cid in excluded])
    wn.to_csv(OUT / "why_not.csv", index=False)
    print(wn.to_string(index=False))

    scen = {}
    for scheme in plan.SCHEMES:
        for bound in ("low", "point", "high"):
            ww = plan.evidence_weights(tri, scheme, bound)
            p = solve(plan.build_problem(gaps, ww, courses, args.budget, args.trainer_hours), OPTS)
            scen[f"{scheme} / {bound}"] = p.seats()
    names = {c.id: c.name for c in courses}
    matrix = pd.DataFrame(scen).reindex([c.id for c in courses]).fillna(0).astype(int)
    matrix.index = [names[i] for i in matrix.index]
    matrix.to_csv(OUT / "robustness_matrix.csv")
    core = [i for i in matrix.index if (matrix.loc[i] > 0).all()]
    never = [i for i in matrix.index if (matrix.loc[i] == 0).all()]
    print(matrix.to_string())
    print("SAFE CORE:", core, "\nNEVER:", never)

    figs = [fig_inputs(gaps, w), fig_plan(best, problem, args.budget), fig_pareto(points, args.budget),
            fig_robustness(matrix)]
    summary = {
        "budget": args.budget, "trainer_hours": args.trainer_hours, "status": best.status.value,
        "objective": best.objective, "total_cost": best.total_cost, "total_trainer_hours": best.total_trainer_hours,
        "total_seats": best.total_seats, "courses": best.seats(),
        "closure_pct": {s.skill_id: round(s.closure_pct, 1) for s in best.skills},
        "marginals": marg.to_dict("records"), "safe_core": core, "never_chosen": never, "figures": figs,
        "lp_budget_shadow_per_lakh": sens.shadow_prices.budget_per_rupee * 1e5,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("DONE")


if __name__ == "__main__":
    main()
