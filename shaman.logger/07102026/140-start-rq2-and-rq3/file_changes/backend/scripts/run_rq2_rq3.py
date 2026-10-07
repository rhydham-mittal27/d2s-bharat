"""RQ2 (JDS: junior skills -> salary hike) and RQ3 (SDS: senior Big Five traits -> success).

Usage: uv run python scripts/run_rq2_rq3.py
Outputs: dataset/processed/rq23/*.csv + summary.json ; reports/rq2_rq3/*.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

from d2s.analysis import traits  # noqa: E402
from d2s.config import get_settings  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
SAS = ROOT / "hackathon" / "SAS Data Problem Statement and Instructions Hackathon"
DATA_OUT = get_settings().dataset_dir / "processed" / "rq23"
FIG = ROOT / "reports" / "rq2_rq3"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 9})
HI, LO = "#16a34a", "#ef4444"


def pretty(f: str) -> str:
    return f.replace("_skills", "").replace("_", " ").replace("and", "&")


def fig_distributions(ds, key, labels):
    n = len(ds.features)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n, 3.4))
    for ax, f in zip(axes, ds.features, strict=True):
        data = [ds.frame.loc[ds.y == 0, f], ds.frame.loc[ds.y == 1, f]]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], (LO, HI), strict=True):
            patch.set_facecolor(c)
            patch.set_alpha(0.35)
        rng = np.random.default_rng(0)
        for i, d in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=6, color=(LO, HI)[i - 1], alpha=0.6)
        ax.set_xticks([1, 2], labels)
        ax.set_title(pretty(f), fontsize=8.5)
    fig.suptitle(f"{ds.name}: feature distributions by outcome", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"{key}_1_distributions.png", bbox_inches="tight")
    plt.close(fig)


def fig_effects(uni, key, title):
    u = uni.sort_values("rank_biserial")
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(u) + 1.2))
    y = np.arange(len(u))
    colors = [HI if r > 0 else LO for r in u["rank_biserial"]]
    ax.errorbar(u["rank_biserial"], y, xerr=[u["rank_biserial"] - u["rb_ci_low"], u["rb_ci_high"] - u["rank_biserial"]],
                fmt="none", ecolor="#94a3b8", capsize=3)
    ax.scatter(u["rank_biserial"], y, color=colors, s=50, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    labels = [f"{pretty(f)}{' *' if s else ''}" for f, s in zip(u["feature"], u["significant_holm_0.05"], strict=True)]
    ax.set_yticks(y, labels)
    ax.set_xlabel("rank-biserial effect size (high vs low outcome), 95% bootstrap CI")
    ax.set_title(f"{title}\n* = significant after Holm correction (p < 0.05)")
    fig.tight_layout()
    fig.savefig(FIG / f"{key}_2_effect_sizes.png", bbox_inches="tight")
    plt.close(fig)


def fig_corr(corr, key, title):
    labels = [pretty(c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{title}: Spearman correlations")
    fig.tight_layout()
    fig.savefig(FIG / f"{key}_3_correlations.png", bbox_inches="tight")
    plt.close(fig)


def fig_models(comp, perm, key, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, metric, label in ((axes[0], "roc_auc", "ROC AUC"), (axes[1], "balanced_accuracy", "balanced accuracy")):
        x = np.arange(len(comp))
        m = comp[f"{metric}_mean"]
        err = [m - comp[f"{metric}_p2.5"], comp[f"{metric}_p97.5"] - m]
        ax.bar(x, m, color=["#94a3b8", "#2563eb", "#ea580c"], yerr=err, capsize=4)
        for xi, v in zip(x, m, strict=True):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_xticks(x, comp["model"], rotation=12)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"{label}: repeated 5-fold CV (20×)")
    fig.suptitle(f"{title}: model vs majority baseline (permutation test p = {perm['p_value']:.3f})",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"{key}_4_models.png", bbox_inches="tight")
    plt.close(fig)


def fig_odds(orat, key, title):
    o = orat.sort_values("odds_ratio_per_sd")
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(o) + 1.2))
    y = np.arange(len(o))
    ax.errorbar(o["odds_ratio_per_sd"], y, xerr=[o["odds_ratio_per_sd"] - o["or_ci_low"], o["or_ci_high"] - o["odds_ratio_per_sd"]],
                fmt="none", ecolor="#94a3b8", capsize=3)
    ax.scatter(o["odds_ratio_per_sd"], y, s=50, zorder=3,
               color=[HI if v > 1 else LO for v in o["odds_ratio_per_sd"]])
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y, [pretty(f) for f in o["feature"]])
    ax.set_xlabel("odds ratio per +1 SD (L2 logistic regression), 95% bootstrap CI, log scale")
    ax.set_title(f"{title}\n(other features held constant; C = {orat.attrs['C']:.3g})")
    fig.tight_layout()
    fig.savefig(FIG / f"{key}_5_odds_ratios.png", bbox_inches="tight")
    plt.close(fig)


def fig_tree(tree, ds, key, title, class_names):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    plot_tree(tree, feature_names=[pretty(f) for f in ds.features], class_names=class_names,
              filled=True, rounded=True, fontsize=7, impurity=False, proportion=False, ax=ax)
    ax.set_title(f"{title}: depth-3 decision tree (min 10 per leaf)")
    fig.tight_layout()
    fig.savefig(FIG / f"{key}_6_tree.png", bbox_inches="tight")
    plt.close(fig)


def run(ds, key, title, class_labels):
    print(f"== {ds.name}", flush=True)
    a = traits.audit(ds)
    uni = traits.univariate(ds)
    corr = traits.correlations(ds)
    v = traits.vif(ds)
    comp, _ = traits.model_comparison(ds)
    perm = traits.permutation_significance(ds)
    orat = traits.odds_ratios(ds)
    tree, rules, timp = traits.tree_explanation(ds)

    for name, df in (("univariate", uni), ("correlations", corr), ("vif", v), ("models", comp),
                     ("odds_ratios", orat), ("tree_importance", timp)):
        df.to_csv(DATA_OUT / f"{key}_{name}.csv", index=(name == "correlations"))
    (DATA_OUT / f"{key}_tree_rules.txt").write_text(rules, encoding="utf-8")

    fig_distributions(ds, key, class_labels)
    fig_effects(uni, key, title)
    fig_corr(corr, key, title)
    fig_models(comp, perm, key, title)
    fig_odds(orat, key, title)
    fig_tree(tree, ds, key, title, class_labels)

    print(json.dumps(a, indent=1, default=str))
    print(uni.round(3).to_string(index=False))
    print(comp.round(3).to_string(index=False))
    print("permutation:", perm)
    print(orat.round(3).to_string(index=False))
    print(timp.round(3).to_string(index=False))
    print(rules)
    return {"audit": a, "permutation_test": perm, "logit_C": orat.attrs["C"],
            "max_vif": float(v["vif"].max())}


def main():
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    summary = {
        "rq2_jds": run(traits.load_jds(SAS / "JDS Skill Traits.xlsx"), "rq2_jds",
                       "RQ2 junior skills → salary hike", ["low hike", "high hike"]),
        "rq3_sds": run(traits.load_sds(SAS / "SDS Personality Traits.xlsx"), "rq3_sds",
                       "RQ3 senior traits → success", ["low success", "high success"]),
    }
    (DATA_OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("DONE", DATA_OUT, FIG)


if __name__ == "__main__":
    main()
