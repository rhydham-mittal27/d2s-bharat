"""Train/test evaluation of the AI/ML layer, with charts.

Usage : uv run python scripts/evaluate_aiml.py
Output: reports/aiml/*.png + metrics.json (+ README written separately from these numbers)

1. Hike model (deployed): 200 stratified 75/25 train/test splits, out-of-fold predictions,
   comparison with baseline / tree / random forest / gradient boosting, calibration, learning curve.
2. Cohort profiler: out-of-fold probabilities vs actual outcome.
3. Forecaster: AirPassengers 12-month holdout; 60 intermittent series holdout + interval coverage.
4. Optimiser: random small instances vs brute force; solve-time scaling.
5. Skill matcher: TechWolf labelled sentences (external validation) + lookup latency.
"""

import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from d2s.analysis.traits import load_jds  # noqa: E402
from d2s.models.hike import LABELS, SKILLS, TARGET, HikeModel, SkillScores  # noqa: E402
from d2s.services import store  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "aiml"
SEED = 42
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 9})
C = {"L2 logistic (deployed)": "#2563eb", "Decision tree (d≤3)": "#ea580c", "Random forest": "#16a34a",
     "Gradient boosting": "#9333ea", "Majority baseline": "#94a3b8"}
M: dict = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name, flush=True)


def models():
    logit = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 20), cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="neg_log_loss", max_iter=5000))
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "L2 logistic (deployed)": logit,
        "Decision tree (d≤3)": DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "Gradient boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                            min_samples_leaf=10, random_state=SEED),
    }


# ---- 1. hike model --------------------------------------------------------------------------------
def hike_model():
    print("1. hike model", flush=True)
    jds = load_jds(store.SAS_DIR / "JDS Skill Traits.xlsx").frame
    X, y = jds[SKILLS].to_numpy(float), jds[TARGET].to_numpy(int)

    # 200 random stratified 75/25 train/test splits
    sss = StratifiedShuffleSplit(n_splits=200, test_size=0.25, random_state=SEED)
    scores = {k: {"auc": [], "acc": [], "brier": []} for k in models()}
    for tr, te in sss.split(X, y):
        for name, mdl in models().items():
            m = clone(mdl).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            scores[name]["auc"].append(roc_auc_score(y[te], p) if name != "Majority baseline" else 0.5)
            scores[name]["acc"].append(accuracy_score(y[te], (p >= 0.5).astype(int)))
            scores[name]["brier"].append(brier_score_loss(y[te], p))
    M["hike_holdout_200_splits"] = {k: {m: {"mean": float(np.mean(v)), "p2.5": float(np.percentile(v, 2.5)),
                                            "p97.5": float(np.percentile(v, 97.5))} for m, v in d.items()}
                                    for k, d in scores.items()}

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    names = list(scores)
    for ax, key, lab in zip(axes, ("auc", "acc", "brier"), ("test ROC AUC", "test accuracy", "test Brier score (lower = better)"),
                            strict=True):
        ax.boxplot([scores[n][key] for n in names], vert=False, widths=0.6, patch_artist=True,
                   boxprops={"facecolor": "#e2e8f0"}, medianprops={"color": "k"})
        ax.set_yticks(range(1, len(names) + 1), names if key == "auc" else [""] * len(names))
        ax.set_title(lab)
    fig.suptitle("Hike model vs alternatives on 200 random stratified 75/25 train/test splits (JDS, n = 139)",
                 fontweight="bold")
    save(fig, "01_hike_model_comparison.png")

    # out-of-fold predictions (repeated CV, averaged) for ROC / PR / calibration / confusion
    oof = {}
    for name in ("L2 logistic (deployed)", "Decision tree (d≤3)", "Random forest", "Gradient boosting"):
        ps = []
        for r in range(10):
            cv = StratifiedKFold(5, shuffle=True, random_state=SEED + r)
            ps.append(cross_val_predict(clone(models()[name]), X, y, cv=cv, method="predict_proba")[:, 1])
        oof[name] = np.mean(ps, axis=0)
    p = oof["L2 logistic (deployed)"]
    M["hike_oof"] = {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
                     "brier": float(brier_score_loss(y, p)), "accuracy_at_0.5": float(accuracy_score(y, p >= 0.5)),
                     "prevalence": float(y.mean())}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name, pp in oof.items():
        fpr, tpr, _ = roc_curve(y, pp)
        axes[0].plot(fpr, tpr, color=C[name], lw=2.2 if "logistic" in name else 1.3,
                     label=f"{name} (AUC {roc_auc_score(y, pp):.3f})")
        pr, rc, _ = precision_recall_curve(y, pp)
        axes[1].plot(rc, pr, color=C[name], lw=2.2 if "logistic" in name else 1.3,
                     label=f"{name} (AP {average_precision_score(y, pp):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC (out-of-fold)")
    axes[0].legend(fontsize=7)
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall (out-of-fold)")
    axes[1].legend(fontsize=7)
    cm = confusion_matrix(y, (p >= 0.5).astype(int))
    axes[2].imshow(cm, cmap="Blues")
    for i, j in itertools.product(range(2), range(2)):
        axes[2].text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    axes[2].set_xticks([0, 1], ["pred low", "pred high"])
    axes[2].set_yticks([0, 1], ["actual low", "actual high"])
    axes[2].set_title(f"Deployed model, threshold 0.5\naccuracy {accuracy_score(y, p >= 0.5):.3f}")
    M["hike_confusion"] = cm.tolist()
    save(fig, "02_hike_roc_pr_confusion.png")

    # calibration + probability separation
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("L2 logistic (deployed)", "Random forest", "Gradient boosting"):
        fr, mp = calibration_curve(y, oof[name], n_bins=6, strategy="quantile")
        a.plot(mp, fr, "o-", color=C[name], label=f"{name} (Brier {brier_score_loss(y, oof[name]):.3f})")
    a.plot([0, 1], [0, 1], "k--", lw=0.8)
    a.set_xlabel("predicted probability")
    a.set_ylabel("observed share with high hike")
    a.set_title("Calibration (out-of-fold, 6 quantile bins)")
    a.legend(fontsize=7)
    bins = np.linspace(0, 1, 21)
    b.hist(p[y == 0], bins=bins, alpha=0.6, color="#ef4444", label="actual low hike")
    b.hist(p[y == 1], bins=bins, alpha=0.6, color="#16a34a", label="actual high hike")
    for t, lab in ((0.4, "at risk | developing"), (0.7, "developing | on track")):
        b.axvline(t, color="k", ls="--", lw=0.8)
        b.text(t, b.get_ylim()[1] * 0.95, " " + lab, fontsize=7, rotation=90, va="top")
    b.set_xlabel("out-of-fold predicted probability of a high hike")
    b.set_title("Cohort profiler: separation and risk bands")
    b.legend(fontsize=7)
    bands = pd.cut(p, [-0.01, 0.4, 0.7, 1.0], labels=["at risk", "developing", "on track"])
    M["profiler_bands_vs_actual"] = pd.crosstab(bands, y).rename(columns={0: "actual_low", 1: "actual_high"}).to_dict()
    save(fig, "03_hike_calibration_profiler.png")

    # learning curve
    sizes = [30, 45, 60, 75, 90, 104]
    lc = {n: [] for n in sizes}
    for n in sizes:
        for tr, te in StratifiedShuffleSplit(n_splits=60, test_size=35, random_state=SEED).split(X, y):
            sub = np.random.default_rng(n + tr[0]).choice(tr, n, replace=False)
            if len(np.unique(y[sub])) < 2:
                continue
            m = clone(models()["L2 logistic (deployed)"]).fit(X[sub], y[sub])
            lc[n].append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    mu = [np.mean(lc[n]) for n in sizes]
    lo = [np.percentile(lc[n], 10) for n in sizes]
    hi = [np.percentile(lc[n], 90) for n in sizes]
    ax.fill_between(sizes, lo, hi, color="#2563eb", alpha=0.15, label="10th-90th percentile")
    ax.plot(sizes, mu, "o-", color="#2563eb", label="mean test AUC")
    ax.set_xlabel("training juniors")
    ax.set_ylabel("test AUC (35 held out)")
    ax.set_ylim(0.5, 1)
    ax.set_title("Learning curve: performance plateaus by ~75 juniors")
    ax.legend(fontsize=8)
    M["hike_learning_curve"] = {str(n): float(np.mean(lc[n])) for n in sizes}
    save(fig, "04_hike_learning_curve.png")

    # deployed artifact: coefficients + an example explanation
    hm = HikeModel.load(store.artifact_path("hike_model.pkl"))
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8))
    order = np.argsort(hm.coef)
    lo_c, hi_c = np.percentile(hm.boot_coef, [5, 95], axis=0)
    a.errorbar(hm.coef[order], range(5), xerr=[hm.coef[order] - lo_c[order], hi_c[order] - hm.coef[order]],
               fmt="o", color="#2563eb", capsize=3)
    a.axvline(0, color="k", lw=0.8)
    a.set_yticks(range(5), [LABELS[SKILLS[i]] for i in order])
    a.set_xlabel("log-odds per +1 SD (90% bootstrap interval)")
    a.set_title(f"Deployed model coefficients (C = {hm.C:.2f}, {len(hm.boot_coef)} bootstrap refits)")
    ex = SkillScores(coding_skills=4.4, maths_stats_skills=3.6, ai_and_ml_skills=4.6, big_data_skills=3.8,
                     dashboard_and_storytelling_skills=3.4)
    pr = hm.explain(ex)
    cs = pr.contributions
    b.barh([c.label for c in cs], [c.contribution_logodds for c in cs],
           color=["#16a34a" if c.contribution_logodds > 0 else "#ef4444" for c in cs])
    b.axvline(0, color="k", lw=0.8)
    best = pr.improvements[0]
    b.set_xlabel("contribution to log-odds vs an average junior")
    b.set_title(f"Example learner: P(high hike) = {pr.probability:.2f} [{pr.probability_low:.2f}, {pr.probability_high:.2f}]\n"
                f"best lever: {best.label} {best.current:.1f} → {best.target:.1f} gives {best.probability_after:.2f}")
    M["example_explanation"] = {"probability": pr.probability, "interval": [pr.probability_low, pr.probability_high],
                                "best_lever": best.model_dump(),
                                "smallest_change_to_70pct": pr.smallest_change_to_target.model_dump()
                                if pr.smallest_change_to_target else None}
    save(fig, "05_hike_coefficients_and_explanation.png")


# ---- 3. forecaster ------------------------------------------------------------------------------------
def forecaster():
    print("3. forecaster", flush=True)
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, SeasonalNaive
    from statsforecast.utils import AirPassengersDF

    from d2s.ml.forecasting import DemandForecaster

    df = AirPassengersDF.copy()
    df["unique_id"] = "air"
    h = 12
    tr, te = df.iloc[:-h], df.iloc[-h:]
    o = DemandForecaster(horizon=h, freq="ME", level=80).fit_predict(tr)[0]
    pt = np.array([p.point for p in o.points])
    lo, hi = np.array([p.lo for p in o.points]), np.array([p.hi for p in o.points])
    sf = StatsForecast(models=[AutoETS(season_length=12), AutoARIMA(season_length=12), SeasonalNaive(season_length=12)],
                       freq="ME")
    f = sf.forecast(df=tr, h=h)
    errs = {m: float(np.mean(np.abs(f[m].to_numpy() - te["y"].to_numpy()))) for m in ("AutoETS", "AutoARIMA", "SeasonalNaive")}
    errs["Combination"] = float(np.mean(np.abs((f["AutoETS"] + f["AutoARIMA"]).to_numpy() / 2 - te["y"].to_numpy())))
    errs[f"D2S selected ({o.model})"] = float(np.mean(np.abs(pt - te["y"].to_numpy())))
    cov = float(np.mean((te["y"].to_numpy() >= lo) & (te["y"].to_numpy() <= hi)))
    M["forecast_airpassengers"] = {"selected": o.model, "test_mae": errs, "interval80_coverage": cov}

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    a.plot(tr["ds"].iloc[-48:], tr["y"].iloc[-48:], color="#334155", label="history")
    a.plot(te["ds"], te["y"], "k-", lw=2, label="actual (held out)")
    a.plot(te["ds"], pt, "-", color="#2563eb", lw=2, label=f"forecast ({o.model})")
    a.fill_between(te["ds"], lo, hi, color="#2563eb", alpha=0.15, label=f"80% interval (covers {cov:.0%})")
    a.set_title("AirPassengers: 12-month holdout")
    a.legend(fontsize=7)
    names = list(errs)
    b.barh(names, [errs[n] for n in names], color=["#94a3b8"] * 4 + ["#2563eb"])
    for i, n in enumerate(names):
        b.text(errs[n], i, f" {errs[n]:.1f}", va="center", fontsize=8)
    b.set_xlabel("test MAE (passengers, thousands)")
    b.set_title("Test error by model")
    save(fig, "06_forecast_airpassengers.png")

    # intermittent holdout: coverage of conformal intervals across 60 series x 6 months
    rng = np.random.default_rng(7)
    frames, fut = [], {}
    n_hist, hh = 48, 6
    for i in range(60):
        d = np.where(rng.random(n_hist + hh) < 0.3, rng.poisson(4, n_hist + hh) + 1, 0)
        ds = pd.date_range("2021-01-01", periods=n_hist + hh, freq="MS")
        frames.append(pd.DataFrame({"unique_id": f"s{i}", "ds": ds[:n_hist], "y": d[:n_hist]}))
        fut[f"s{i}"] = d[n_hist:]
    t0 = time.perf_counter()
    out = DemandForecaster(horizon=hh, level=80).fit_predict(pd.concat(frames))
    runtime = time.perf_counter() - t0
    inside, rate_err, models_used = [], [], {}
    for r in out:
        actual = fut[r.unique_id]
        inside += [(p.lo <= a <= p.hi) for p, a in zip(r.points, actual, strict=True)]
        rate_err.append(abs(r.total() / hh - actual.mean()))
        models_used[r.model] = models_used.get(r.model, 0) + 1
    M["forecast_intermittent"] = {"series": len(out), "pointwise_interval_coverage": float(np.mean(inside)),
                                  "mean_abs_rate_error": float(np.mean(rate_err)), "models_selected": models_used,
                                  "runtime_s": runtime}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4))
    a.bar(list(models_used), list(models_used.values()), color="#7c3aed")
    a.set_title("Model selected per intermittent series (60)")
    a.tick_params(axis="x", rotation=20)
    b.hist(rate_err, bins=15, color="#64748b")
    b.set_xlabel("|forecast − actual| mean monthly demand")
    b.set_title(f"Rate error (mean {np.mean(rate_err):.2f}); 80% interval coverage {np.mean(inside):.0%}")
    save(fig, "07_forecast_intermittent.png")


# ---- 4. optimiser ---------------------------------------------------------------------------------------
def optimiser():
    print("4. optimiser", flush=True)
    from d2s.engine import Constraints, Course, PlanningProblem, SkillGap, SolverOptions, solve

    rng = np.random.default_rng(SEED)

    def brute(p):
        best = 0.0
        for seats in itertools.product(*[range(c.max_seats + 1) for c in p.courses]):
            alloc = dict(zip([c.id for c in p.courses], seats, strict=True))
            cost, ok = 0, True
            for c in p.courses:
                s = alloc[c.id]
                if s:
                    if s < c.min_batch or any(alloc[q] == 0 for q in c.prerequisites):
                        ok = False
                    cost += c.fixed_cost + c.cost_per_seat * s
            if not ok or cost > p.constraints.budget:
                continue
            v = sum(k.demand_weight * min(k.learners_short, sum(c.coverage.get(k.skill_id, 0) * alloc[c.id] for c in p.courses) / 100)
                    for k in p.skills)
            best = max(best, v)
        return best

    agree = 0
    n_inst = 40
    for i in range(n_inst):
        skills = [SkillGap(skill_id=f"k{j}", learners_short=int(rng.integers(2, 6)), demand_weight=float(rng.uniform(1, 10)))
                  for j in range(3)]
        courses = []
        for c in range(3):
            cov = {f"k{j}": int(rng.choice([0, 50, 100])) for j in range(3)}
            courses.append(Course(id=f"c{c}", fixed_cost=int(rng.integers(5, 20)), cost_per_seat=int(rng.integers(1, 4)),
                                  min_batch=int(rng.integers(1, 3)), max_seats=4, coverage={k: v for k, v in cov.items() if v},
                                  prerequisites=["c0"] if c == 2 and rng.random() < 0.5 else []))
        p = PlanningProblem(skills=skills, courses=courses, constraints=Constraints(budget=int(rng.integers(15, 45))))
        r = solve(p, SolverOptions(workers=4, gap_limit=0.0))
        agree += abs(r.objective - brute(p)) < 1e-6
    M["optimiser_vs_bruteforce"] = {"instances": n_inst, "identical_optimum": agree}

    sizes = [(5, 10), (10, 20), (20, 40), (30, 80), (50, 150), (80, 300)]
    times = []
    for n_sk, n_c in sizes:
        ts = []
        for rep in range(3):
            skills = [SkillGap(skill_id=f"k{j}", learners_short=int(rng.integers(20, 120)),
                               demand_weight=float(rng.uniform(1, 100))) for j in range(n_sk)]
            courses = []
            for c in range(n_c):
                ks = rng.choice(n_sk, size=int(rng.integers(1, 4)), replace=False)
                courses.append(Course(id=f"c{c}", fixed_cost=int(rng.integers(20_000, 150_000)),
                                      cost_per_seat=int(rng.integers(0, 3000)), min_batch=15, max_seats=60,
                                      trainer_hours=int(rng.integers(20, 90)),
                                      coverage={f"k{k}": int(rng.choice([40, 60, 100])) for k in ks},
                                      prerequisites=[f"c{int(rng.integers(0, c))}"] if c > 3 and rng.random() < 0.2 else [],
                                      exclusive_group=f"g{c // 5}" if rng.random() < 0.2 else None))
            p = PlanningProblem(skills=skills, courses=courses,
                                constraints=Constraints(budget=n_c * 40_000, trainer_hours=n_c * 20))
            t0 = time.perf_counter()
            r = solve(p, SolverOptions(time_limit_s=30, workers=8, gap_limit=0.0))
            ts.append((time.perf_counter() - t0, r.status.value))
        times.append({"skills": n_sk, "courses": n_c, "median_s": float(np.median([t for t, _ in ts])),
                      "runs_s": [t for t, _ in ts], "statuses": [s for _, s in ts]})
    M["optimiser_scaling"] = times
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, t in enumerate(times):
        ax.scatter([i] * len(t["runs_s"]), t["runs_s"], color="#16a34a", alpha=0.6, s=25)
        ax.plot([i - 0.25, i + 0.25], [t["median_s"]] * 2, color="k", lw=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(times)), [f"{t['skills']} skills\n{t['courses']} courses" for t in times], fontsize=8)
    n_opt = sum(s == "optimal" for t in times for s in t["statuses"])
    n_all = sum(len(t["statuses"]) for t in times)
    ax.set_ylabel("solve time (s, log scale)")
    ax.set_title(f"CP-SAT: {n_opt}/{n_all} runs proven optimal; brute-force agreement {agree}/{n_inst}\n"
                 "dots = runs, bar = median; difficulty depends on constraint tightness, not only size",
                 fontsize=9)
    save(fig, "08_optimiser_scaling.png")


# ---- 5. skill matcher -------------------------------------------------------------------------------------
def matcher():
    print("5. skill matcher", flush=True)
    from d2s.ml.embeddings import get_embedder
    from d2s.services.market import SkillLookup

    v = pd.read_csv(store.processed() / "ingest" / "eval_variants__BAAI__bge-small-en-v1.5.csv")
    v = v[v["query"] == "no-prompt"]
    sweep = pd.read_csv(store.processed() / "ingest" / "eval_threshold_sweep.csv")
    e = get_embedder()
    look = SkillLookup(e)
    lat = []
    for q in ["pyspark", "power bi dashboards", "statistics", "deep learning", "sas", "hadoop", "excel vba",
              "machine learning engineer", "tableau", "nlp"] * 3:
        t0 = time.perf_counter()
        look.search(q, 5)
        lat.append((time.perf_counter() - t0) * 1000)
    M["matcher"] = {"techwolf_variants": v.round(4).to_dict("records"),
                    "lookup_latency_ms": {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95))}}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.6))
    x = np.arange(4)
    for i, (_, r) in enumerate(v.iterrows()):
        a.bar(x + (i - 1) * 0.27, [100 * r["hit@1"], 100 * r["hit@5"], 100 * r["hit@10"], 100 * r["mrr"]], 0.27,
              label=r["doc_text"], color=["#94a3b8", "#2563eb", "#16a34a"][i % 3])
    a.set_xticks(x, ["hit@1", "hit@5", "hit@10", "MRR×100"])
    a.set_ylabel("%")
    a.set_title("Skill matching on TechWolf labelled sentences (n = 578)")
    a.legend(fontsize=7, title="what is embedded", title_fontsize=7)
    b.plot(sweep["threshold"], 100 * sweep["precision@1"], "-o", ms=3, color="#16a34a", label="precision@1")
    b.plot(sweep["threshold"], 100 * sweep["coverage"], "-o", ms=3, color="#64748b", label="coverage")
    b.set_xlabel("similarity threshold")
    b.set_title(f"Threshold trade-off; lookup latency median {np.median(lat):.0f} ms")
    b.legend(fontsize=8)
    save(fig, "09_skill_matcher.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]  # e.g. `forecaster` to rerun one section and merge into metrics.json
    path = OUT / "metrics.json"
    if only and path.exists():
        M.update(json.loads(path.read_text(encoding="utf-8")))
    t0 = time.perf_counter()
    for name, fn in (("hike", hike_model), ("forecaster", forecaster), ("optimiser", optimiser), ("matcher", matcher)):
        if not only or name in only:
            fn()
    M["runtime_s"] = time.perf_counter() - t0
    path.write_text(json.dumps(M, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in M.items() if k != "matcher"}, indent=1, default=str)[:6000])
    print("DONE", OUT)


if __name__ == "__main__":
    main()
