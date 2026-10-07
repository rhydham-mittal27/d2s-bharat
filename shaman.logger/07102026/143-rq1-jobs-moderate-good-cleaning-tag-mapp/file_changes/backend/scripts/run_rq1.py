"""RQ1 (strengthened): demand and pay for skills and roles, SAS job data only.

Needs: scripts/run_sas_ingestion.py outputs (tag_mapping.csv, text_matches_raw.csv.gz) and
       scripts/run_rq2_rq3.py outputs (for the RQ1 x RQ2 triangulation).
Usage: uv run python scripts/run_rq1.py
Outputs: dataset/processed/rq1/* ; reports/rq1/*.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from d2s.analysis import jobs  # noqa: E402
from d2s.config import get_settings  # noqa: E402
from d2s.ingest import parsing  # noqa: E402
from d2s.ml.embeddings import SentenceTransformerEmbedder  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
SAS = ROOT / "hackathon" / "SAS Data Problem Statement and Instructions Hackathon"
S = get_settings()
IN_SAS = S.dataset_dir / "processed" / "sas"
IN_RQ23 = S.dataset_dir / "processed" / "rq23"
OUT = S.dataset_dir / "processed" / "rq1"
FIG = ROOT / "reports" / "rq1"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 9})
AREA_LABEL = {"big_data_skills": "Big data", "maths_stats_skills": "Maths & stats", "coding_skills": "Coding",
              "ai_and_ml_skills": "AI & ML", "dashboard_and_storytelling_skills": "Dashboards & storytelling"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_families(posts):
    fam = posts["family"].value_counts()
    med = posts.groupby("family")["sal_mid_lakh"].median().reindex(fam.index)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    colors = ["#2563eb" if f in jobs.DATA_FAMILIES else "#94a3b8" for f in fam.index]
    a.barh(fam.index[::-1], fam.values[::-1], color=colors[::-1])
    for i, v in enumerate(fam.values[::-1]):
        a.text(v, i, f" {v:,}", va="center", fontsize=8)
    a.set_title("Postings by role family (blue = data roles)")
    b.barh(med.index[::-1], med.values[::-1], color=colors[::-1])
    for i, v in enumerate(med.values[::-1]):
        b.text(v, i, f" {v:.1f} L", va="center", fontsize=8)
    b.set_title("Median salary-band midpoint (lakh/yr)")
    return save(fig, "01_role_families.png")


def fig_canonical(canon):
    uses = canon.groupby("canonical_source")["posts"].sum()
    order = ["esco_onet", "merged_esco_onet", "merged_tag", "tag"]
    labels = {"esco_onet": "mapped to ESCO/O*NET", "merged_esco_onet": "near-duplicate of a mapped tag",
              "merged_tag": "near-duplicate of a data-derived skill", "tag": "own data-derived skill"}
    uses = uses.reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    left = 0
    cols = ["#16a34a", "#86efac", "#93c5fd", "#cbd5e1"]
    for k, c in zip(order, cols, strict=True):
        v = 100 * uses[k] / uses.sum()
        ax.barh([0], [v], left=left, color=c, label=f"{labels[k]} ({v:.1f}%)")
        if v > 4:
            ax.text(left + v / 2, 0, f"{v:.0f}%", ha="center", va="center", fontsize=8)
        left += v
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of recruiter tag uses")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2)
    ax.set_title("Every tag now resolves to a canonical skill")
    return save(fig, "02_canonical_coverage.png")


def fig_demand(dem, title, name):
    d = dem.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    y = np.arange(len(d))
    ax.barh(y, 100 * d["share"], color="#2563eb", alpha=0.85)
    ax.errorbar(100 * d["share"], y, xerr=[100 * (d["share"] - d["share_ci_low"]), 100 * (d["share_ci_high"] - d["share"])],
                fmt="none", ecolor="k", capsize=2, lw=0.8)
    ax.set_yticks(y, [s[:40] for s in d["skill"]])
    ax.set_xlabel("% of postings (Wilson 95% CI)")
    ax.set_title(title)
    return save(fig, name)


def fig_pay(adj, title, name):
    a = adj[adj["q_fdr"] < 0.05].copy()
    a = pd.concat([a.head(12), a.tail(8)]).drop_duplicates("skill").sort_values("odds_ratio")
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(a) + 1.5))
    y = np.arange(len(a))
    ax.errorbar(a["odds_ratio"], y, xerr=[a["odds_ratio"] - a["or_ci_low"], a["or_ci_high"] - a["odds_ratio"]],
                fmt="none", ecolor="#94a3b8", capsize=3)
    ax.scatter(a["odds_ratio"], y, s=40, zorder=3, color=["#16a34a" if v > 1 else "#ef4444" for v in a["odds_ratio"]])
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(y, [s[:40] for s in a["skill"]])
    ax.set_xlabel("odds ratio of a higher salary band (ordinal logit, 95% CI, log scale)")
    ax.set_title(title)
    return save(fig, name)


def fig_rules(rules):
    r = rules[(rules["support"] >= 0.01) & (rules["confidence"] >= 0.3)].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(r))
    ax.barh(y, r["lift"], color="#7c3aed", alpha=0.8)
    for i, (c, s) in enumerate(zip(r["confidence"], r["support"], strict=True)):
        ax.text(r["lift"].iloc[i], i, f"  conf {c:.0%}, support {s:.1%}", va="center", fontsize=7)
    ax.set_yticks(y, [f"{a[:24]} → {b[:24]}" for a, b in zip(r["antecedent"], r["consequent"], strict=True)])
    ax.set_xlabel("lift (how much more often than chance the two skills co-occur)")
    ax.set_title("Skill bundles: strongest co-occurrence rules (support ≥ 1%, confidence ≥ 30%)")
    return save(fig, "06_skill_bundles.png")


def fig_triangulation(tri):
    t = tri.sort_values("rq2_odds_ratio")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    y = np.arange(len(t))
    labels = [AREA_LABEL[a] for a in t["area"]]
    axes[0].barh(y, 100 * t["demand_share"], color="#2563eb")
    axes[0].set_title("Market demand\n(% of data-role postings)")
    for ax, col, lo, hi, ttl, c in (
        (axes[1], "odds_ratio", "or_ci_low", "or_ci_high", "Market pay premium\n(OR of higher salary band)", "#ea580c"),
        (axes[2], "rq2_odds_ratio", "rq2_or_ci_low", "rq2_or_ci_high", "Junior salary-hike link\n(RQ2 OR per +1 SD)", "#16a34a"),
    ):
        ax.errorbar(t[col], y, xerr=[t[col] - t[lo], t[hi] - t[col]], fmt="o", color=c, capsize=3)
        ax.axvline(1, color="k", lw=0.8)
        ax.set_title(ttl)
    axes[0].set_yticks(y, labels)
    fig.suptitle("Triangulation: the five JDS skill areas in the job market (RQ1) vs junior outcomes (RQ2)",
                 fontweight="bold")
    return save(fig, "07_triangulation.png")


def fig_ds_jobs(ds, st):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    a.scatter(ds["min_experience"], ds["avg_lakh"], s=np.clip(ds["num_of_jobs"], 5, 300) / 3, alpha=0.4, color="#2563eb")
    a.set_xlabel("minimum experience (years)")
    a.set_ylabel("average salary (lakh/yr)")
    a.set_title(f"Salary vs experience (Spearman ρ = {st['spearman_minexp_avgsalary']:.2f})")
    top = pd.Series(st["top_companies"]).iloc[::-1]
    b.barh(top.index, top.values, color="#64748b")
    b.set_title(f"Top 10 companies = {100 * st['top10_company_share_of_openings']:.0f}% of openings")
    b.set_xlabel("openings (num_of_jobs)")
    return save(fig, "08_ds_jobs.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    posts, _ = parsing.load_sas_analytics(SAS / "Analytics Jobs.csv")
    posts["family"] = posts["title"].map(jobs.role_family)
    tag_table = pd.read_csv(IN_SAS / "tag_mapping.csv", keep_default_na=False)
    tag_table["key"] = tag_table["key"].astype(str)
    text = pd.read_csv(IN_SAS / "text_matches_raw.csv.gz")
    print("posts", len(posts), posts["family"].value_counts().to_dict(), flush=True)

    embedder = SentenceTransformerEmbedder(S.embedding_model, S.embedding_backend, S.embedding_batch_size)
    canon = jobs.canonical_tags(tag_table, embedder)
    canon.to_csv(OUT / "canonical_tags.csv", index=False)
    ps = jobs.post_skill_table(posts, canon, text)
    ps.to_csv(OUT / "post_skills_canonical.csv.gz", index=False)
    uses = canon.groupby("canonical_source")["posts"].sum()
    print("canonical sources (tag uses):", (100 * uses / uses.sum()).round(1).to_dict(), flush=True)

    data_posts = posts[posts["family"].isin(jobs.DATA_FAMILIES)]
    data_ps = ps[ps["post_id"].isin(data_posts["post_id"])]
    dem_all = jobs.demand(ps, posts)
    dem_data = jobs.demand(data_ps, data_posts)
    dem_all.to_csv(OUT / "demand_all.csv", index=False)
    dem_data.to_csv(OUT / "demand_data_roles.csv", index=False)
    jobs.demand_by_family(ps, posts).to_csv(OUT / "demand_by_family.csv", index=False)

    top_all = dem_all["skill"].head(40).tolist()
    unadj = jobs.pay_unadjusted(posts, ps, top_all)
    unadj.to_csv(OUT / "pay_unadjusted_all.csv", index=False)
    adj_all, info_all = jobs.pay_adjusted(posts, ps, top_all)
    adj_all.to_csv(OUT / "pay_adjusted_all.csv", index=False)
    top_data = dem_data["skill"].head(30).tolist()
    adj_data, info_data = jobs.pay_adjusted(data_posts, data_ps, top_data)
    adj_data.to_csv(OUT / "pay_adjusted_data_roles.csv", index=False)
    print("pay model all:", info_all, "\npay model data roles:", info_data, flush=True)

    rules = jobs.association_rules(data_ps, data_posts)
    rules.to_csv(OUT / "association_rules_data_roles.csv", index=False)

    area, info_area = jobs.area_market_table(data_posts, data_ps)
    rq2 = pd.read_csv(IN_RQ23 / "rq2_jds_odds_ratios.csv").rename(columns={
        "feature": "area", "odds_ratio_per_sd": "rq2_odds_ratio", "or_ci_low": "rq2_or_ci_low", "or_ci_high": "rq2_or_ci_high"})
    uni = pd.read_csv(IN_RQ23 / "rq2_jds_univariate.csv").rename(columns={"feature": "area", "rank_biserial": "rq2_rank_biserial"})
    tri = area.merge(rq2[["area", "rq2_odds_ratio", "rq2_or_ci_low", "rq2_or_ci_high"]], on="area").merge(
        uni[["area", "rq2_rank_biserial"]], on="area")
    tri.to_csv(OUT / "triangulation_rq1_rq2.csv", index=False)
    broad, _ = jobs.area_market_table(data_posts, data_ps, jobs.JDS_AREAS_BROAD)
    broad.to_csv(OUT / "triangulation_sensitivity_broad_dashboard.csv", index=False)
    print("SENSITIVITY broad dashboard definition:\n", broad.round(3).to_string(index=False))

    ds = parsing.load_sas_ds_jobs(SAS / "DataScience Jobs.csv")
    st = jobs.ds_jobs_stats(ds)

    figs = [fig_families(posts), fig_canonical(canon),
            fig_demand(dem_data, f"Most demanded skills in DATA roles ({len(data_posts):,} postings)", "03_demand_data_roles.png"),
            fig_pay(adj_all, "Skills linked to higher salary bands, all postings\n(adjusted for experience, role family, multi-city; FDR q < 0.05)",
                    "04_pay_premium_all.png"),
            fig_pay(adj_data, "Skills linked to higher salary bands, DATA roles only\n(adjusted; FDR q < 0.05)",
                    "05_pay_premium_data_roles.png"),
            fig_rules(rules), fig_triangulation(tri), fig_ds_jobs(ds, st)]

    summary = {
        "posts": len(posts), "data_role_posts": len(data_posts),
        "families": posts["family"].value_counts().to_dict(),
        "family_median_salary": posts.groupby("family")["sal_mid_lakh"].median().round(2).to_dict(),
        "canonical_tag_uses_pct": (100 * uses / uses.sum()).round(2).to_dict(),
        "distinct_canonical_skills": int(canon["canonical"].nunique()),
        "pay_model_all": info_all, "pay_model_data_roles": info_data, "area_model": info_area,
        "ds_jobs": st, "figures": figs,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option("display.width", 200)
    print(dem_data.head(15).round(3).to_string(index=False))
    print(adj_all.round(3).to_string(index=False))
    print(adj_data.round(3).to_string(index=False))
    print(rules[(rules.support >= 0.01) & (rules.confidence >= 0.3)].head(15).round(3).to_string(index=False))
    print(tri.round(3).to_string(index=False))
    print(json.dumps(st, indent=1, default=str))
    print("DONE")


if __name__ == "__main__":
    main()
