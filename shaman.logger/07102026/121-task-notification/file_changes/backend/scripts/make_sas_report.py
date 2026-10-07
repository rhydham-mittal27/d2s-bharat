"""Charts + report for the SAS-only ingestion run (dataset/processed/sas -> reports/sas_ingestion).

Usage: uv run python scripts/make_sas_report.py      Every number is computed from the run outputs.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from d2s.config import get_settings  # noqa: E402

IN = get_settings().dataset_dir / "processed" / "sas"
OUT = Path(__file__).resolve().parents[2] / "reports" / "sas_ingestion"
C = {"esco": "#2563eb", "onet": "#ea580c", "exact": "#16a34a", "contains": "#86efac",
     "semantic": "#2563eb", "review": "#f59e0b", "unmapped": "#cbd5e1"}
TAG_ORDER = ["exact", "contains", "semantic", "review", "unmapped"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 9})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    return name


def fig_cleaning(s, posts):
    c = s["cleaning"]
    actions = [
        ("exact duplicate rows removed", c["exact_duplicate_rows"]),
        ("job_type spelling variants merged", c["job_type_variants_merged"]),
        ("multi-city rows split", c["multi_city_rows"]),
        ("truncated tags repaired ('Rest...' → 'Rest')", c["tags_truncated_fixed"]),
        ("empty / '...' tags dropped", c["tags_dropped_empty_or_ellipsis"]),
        ("case-duplicate tags dropped", c["tags_case_duplicates_dropped"]),
        ("descriptions truncated at source", c["descriptions_truncated"]),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    lbl, val = zip(*actions, strict=True)
    a.barh(lbl[::-1], val[::-1], color="#334155")
    for i, v in enumerate(val[::-1]):
        a.text(v, i, f" {v:,}", va="center", fontsize=8)
    a.set_title(f"Cleaning actions (Analytics Jobs: {c['raw_rows']:,} → {c['kept_rows']:,} rows)")
    miss = {"job_description": c["missing_description"], "job_type": c["missing_job_type"],
            "key_skills": c["missing_key_skills"]}
    pct = {k: 100 * v / c["kept_rows"] for k, v in miss.items()}
    b.bar(pct.keys(), pct.values(), color=["#f59e0b", "#ef4444", "#16a34a"])
    for i, (k, v) in enumerate(pct.items()):
        b.text(i, v, f"{v:.1f}%\n({miss[k]:,})", ha="center", va="bottom", fontsize=8)
    b.set_ylim(0, 100)
    b.set_title("Missing values after de-duplication")
    b.set_ylabel("% of rows")
    return save(fig, "01_cleaning.png")


def fig_profile(posts):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4))
    order = ["0to3", "3to6", "6to10", "10to15", "15to25", "25to50"]
    vc = posts["salary"].value_counts().reindex(order).fillna(0)
    a.bar([o.replace("to", "–") for o in order], vc.values, color="#2563eb")
    a.set_xlabel("salary band (lakh ₹ / yr)")
    a.set_title("Salary bands (Analytics Jobs)")
    b.hist(posts["exp_min"].dropna(), bins=range(0, 21), color="#64748b")
    b.set_xlabel("minimum years of experience required")
    b.set_title("Experience requirements")
    return save(fig, "02_salary_experience.png")


def fig_tags(tags, s):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
    order = TAG_ORDER
    uses = tags.groupby("method")["posts"].sum().reindex(order).fillna(0)
    distinct = tags["method"].value_counts().reindex(order).fillna(0)
    x = np.arange(2)
    bottom = np.zeros(2)
    for m in order:
        vals = np.array([100 * distinct[m] / distinct.sum(), 100 * uses[m] / uses.sum()])
        a.bar(x, vals, bottom=bottom, color=C[m], label=m.replace("_", " "))
        for xi, (v, b0) in enumerate(zip(vals, bottom, strict=True)):
            if v > 5:
                a.text(xi, b0 + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8)
        bottom += vals
    a.set_xticks(x, [f"distinct tags\n({int(distinct.sum()):,})", f"tag uses in posts\n({int(uses.sum()):,})"])
    a.set_ylabel("%")
    a.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.45, 1))
    a.set_title("Recruiter key_skills → ESCO/O*NET")
    sem = tags[tags["method"].isin(["semantic", "review", "unmapped"])]
    b.hist(sem["similarity"], bins=40, color="#2563eb", alpha=0.8)
    for t, lab in ((0.78, "review"), (0.85, "accept"), (0.90, "accept 1-word")):
        b.axvline(t, color="k", ls="--", lw=1)
        b.text(t, b.get_ylim()[1] * 0.92, f" {lab}", fontsize=8)
    b.set_xlabel("similarity of tag ↔ nearest skill (non-exact tags)")
    b.set_title("Nearest-skill confidence for non-exact tags")
    return save(fig, "03_tag_mapping.png")


def fig_calibration(cal, th):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(cal["threshold"], 100 * cal["precision"], "-o", ms=3, label="precision vs. recruiter tags", color="#16a34a")
    ax.plot(cal["threshold"], 100 * cal["recall"], "-o", ms=3, label="recall of recruiter tags", color="#64748b")
    ax.plot(cal["threshold"], 100 * cal["f1"], "-", lw=2.5, label="F1", color="#2563eb")
    ax.axvline(th, color="k", ls="--", lw=1)
    ax.text(th, ax.get_ylim()[1] * 0.95, f" chosen {th}", fontsize=8)
    ax.set_xlabel("semantic similarity threshold")
    ax.set_ylabel("%")
    ax.legend(fontsize=8)
    ax.set_title("Threshold calibrated inside the SAS data (key_skills as weak labels)")
    return save(fig, "04_threshold_calibration.png")


def fig_top_skills(prof, n_posts):
    top = prof.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.barh(top["label"].str.slice(0, 42), 100 * top["share_of_posts"],
            color=[C.get(t, "#999") for t in top["taxonomy"]])
    ax.set_xlabel("% of postings")
    ax.set_title(f"Most demanded skills (key_skills + text, {n_posts:,} SAS postings)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in ("esco", "onet")],
              labels=["ESCO skill", "O*NET skill/tool"], loc="lower right")
    return save(fig, "05_top_skills.png")


def fig_salary(prof, overall):
    p = prof[prof["posts"] >= 60].copy()
    sel = pd.concat([p.nlargest(12, "salary_median_lakh"), p.nsmallest(8, "salary_median_lakh")])
    sel = sel.drop_duplicates("skill_id").sort_values("salary_median_lakh")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = np.arange(len(sel))
    ax.hlines(y, overall, sel["salary_median_lakh"], color="#cbd5e1", lw=2)
    ax.scatter(sel["salary_median_lakh"], y, s=20 + sel["posts"] / sel["posts"].max() * 300,
               c=np.where(sel["salary_median_lakh"] >= overall, "#16a34a", "#ef4444"), zorder=3)
    ax.axvline(overall, color="k", ls="--", lw=1)
    ax.text(overall, len(sel) - 0.3, f" all postings: {overall:.1f} L", fontsize=8)
    ax.set_yticks(y, [f"{lbl[:36]} (n={n})" for lbl, n in zip(sel["label"], sel["posts"], strict=True)])
    ax.set_xlabel("median salary-band midpoint (lakh ₹ / yr)")
    ax.set_title("Skills associated with higher / lower pay (skills in ≥ 60 postings)\n"
                 "association only, not causation")
    return save(fig, "06_skill_salary.png")


def fig_city(sc):
    piv = sc.pivot_table(index="label", columns="city", values="posts", aggfunc="sum").fillna(0)
    share = piv / piv.sum(axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(share.to_numpy(), aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(share.columns)), share.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(share.index)), [s[:36] for s in share.index])
    fig.colorbar(im, ax=ax, label="share of the city's top-skill mentions")
    ax.set_title("Skill mix by city (top 15 skills × top 10 cities)")
    return save(fig, "07_skill_city.png")


def fig_ds_titles(bt):
    bt = bt.head(12).iloc[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    a.barh(bt.index, bt["jobs"], color="#2563eb")
    for i, v in enumerate(bt["jobs"]):
        a.text(v, i, f" {int(v):,}", va="center", fontsize=8)
    a.set_title("Job openings by title (DataScience Jobs)")
    a.set_xlabel("num_of_jobs (sum over companies)")
    y = np.arange(len(bt))
    b.hlines(y, bt["min_lakh_p25"], bt["max_lakh_p75"], color="#93c5fd", lw=6)
    b.scatter(bt["avg_lakh_weighted"], y, color="#1e3a8a", zorder=3, label="job-weighted average")
    b.set_xlabel("salary, lakh ₹ / yr (bar: P25 of min → P75 of max)")
    b.set_title("Salary range by title")
    b.legend(fontsize=8, loc="lower right")
    return save(fig, "08_ds_titles_salary.png")


def fig_titles(tt):
    show = tt.head(16)
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(show) + 1))
    ax.axis("off")
    cell = [[t[:40], o[:52], f"{s:.2f}"] for t, o, s in zip(show["title"], show["occupation_1"], show["similarity_1"], strict=True)]
    tbl = ax.table(cellText=cell, colLabels=["job title (SAS data)", "nearest ESCO / O*NET occupation", "sim"],
                   loc="center", cellLoc="left", colWidths=[0.36, 0.56, 0.08])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.25)
    ax.set_title("Job titles → standard occupations")
    return save(fig, "09_title_occupations.png")


def readme(s, figs, prof, overall):
    c, t, ps = s["cleaning"], s["tags"], s["post_skills"]
    cal = s["calibration_at_threshold"]
    dm = t["distinct_by_method"]
    uses = t["tag_uses_by_method"]
    tot_uses = sum(uses.values())
    hi = prof[prof["posts"] >= 60].nlargest(5, "salary_median_lakh")
    lines = [
        "# SAS hackathon data: ingestion & normalization report",
        "",
        "**Data used: only the SAS-provided files** (`Analytics Jobs.csv`, `DataScience Jobs.csv`). "
        "ESCO v1.2.1 + O\\*NET 30.3 are used purely as the skill/occupation *vocabulary*. "
        "Generated by `backend/scripts/make_sas_report.py`; every number is computed from the run outputs.",
        "",
        "Pipeline: **Data parsing → Skill extraction → Sentence Transformers (`" + s["model"] + "`) → "
        "Semantic matching → ESCO/O\\*NET skill mapping**",
        "",
        "## 1. Data preparation",
        "| Step | Count |",
        "|---|---|",
        f"| Raw rows → kept after exact-duplicate removal | {c['raw_rows']:,} → **{c['kept_rows']:,}** "
        f"({c['exact_duplicate_rows']:,} duplicates) |",
        f"| Missing `job_description` / `job_type` / `key_skills` | {c['missing_description']:,} / "
        f"{c['missing_job_type']:,} / {c['missing_key_skills']:,} |",
        f"| `job_type` spelling variants merged (Analytics/analytics/ANALYTICS/analytic…) | {c['job_type_variants_merged']:,} |",
        f"| Multi-city rows split into a location list | {c['multi_city_rows']:,} |",
        f"| Descriptions truncated at source (end in `...`) | {c['descriptions_truncated']:,} |",
        f"| Tags: total / truncated & repaired / empty-or-`...` dropped / case-duplicates dropped | "
        f"{c['tags_total']:,} / {c['tags_truncated_fixed']:,} / {c['tags_dropped_empty_or_ellipsis']:,} / "
        f"{c['tags_case_duplicates_dropped']:,} |",
        "| Experience `\"5-10 yrs\"` → `exp_min`, `exp_max`; salary `\"10to15\"` → lakh min/max/mid | all rows parsed |",
        f"| DataScience Jobs: salaries `\"7.8L\"` → numeric lakh | {s['ds_jobs']['rows']:,} rows, "
        f"{s['ds_jobs']['companies']:,} companies, {s['ds_jobs']['jobs_total']:,} openings |",
        "",
        "## 2. Skill mapping results",
        f"- **Recruiter key_skills → ESCO/O\\*NET:** {t['distinct']:,} distinct tags: exact {dm.get('exact', 0):,}, "
        f"contains-a-known-skill {dm.get('contains', 0):,}, semantic (≥0.85, one-word ≥0.90) {dm.get('semantic', 0):,}, "
        f"review queue (0.78–0.85) {dm.get('review', 0):,}, unmapped {dm.get('unmapped', 0):,}. Weighted by usage, "
        f"**{100 * sum(uses.get(m, 0) for m in ('exact', 'contains', 'semantic')) / tot_uses:.1f}% of tag uses are "
        f"mapped**; {100 * uses.get('review', 0) / tot_uses:.1f}% await review.",
        f"- **Text extraction threshold calibrated in-dataset: {s['threshold']}** (max F1 vs recruiter tags: "
        f"precision {100 * cal['precision']:.1f}%, recall {100 * cal['recall']:.1f}%).",
        f"- **Final table:** {ps['rows']:,} (posting, skill) pairs across {ps['posts_with_skill']:,} postings; "
        f"by signal {ps['by_signal']}.",
        f"- Overall median salary-band midpoint: **{overall:.1f} lakh**. Highest-paying skills (≥ 60 postings): "
        + ", ".join(f"{r.label} ({r.salary_median_lakh:.1f} L)" for r in hi.itertuples()) + ".",
        "",
        "## 3. Limitations",
        "- Descriptions are **truncated (~100 chars)** in the source, so text adds little beyond key_skills; "
        "key_skills is the primary signal.",
        "- Recruiter tags are **weak labels** (incomplete, noisy): calibration precision is a lower bound.",
        "- Salary is a **band** (midpoint used); skill–salary links are associations, not causal effects.",
        f"- `job_type` is {100 * c['missing_job_type'] / c['kept_rows']:.0f}% missing, so it cannot be used for segmentation.",
        "",
        "## Figures",
        *[f"![{f}]({f})" for f in figs],
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.loads((IN / "summary.json").read_text())
    posts = pd.read_csv(IN / "posts_clean.csv.gz")
    tags = pd.read_csv(IN / "tag_mapping.csv")
    cal = pd.read_csv(IN / "threshold_calibration.csv")
    prof = pd.read_csv(IN / "skill_profile.csv")
    sc = pd.read_csv(IN / "skill_city.csv")
    bt = pd.read_csv(IN / "ds_jobs_by_title.csv", index_col=0)
    tt = pd.read_csv(IN / "title_occupations.csv")
    overall = float(posts["sal_mid_lakh"].median())
    figs = [
        fig_cleaning(s, posts), fig_profile(posts), fig_tags(tags, s), fig_calibration(cal, s["threshold"]),
        fig_top_skills(prof, s["cleaning"]["kept_rows"]), fig_salary(prof, overall), fig_city(sc),
        fig_ds_titles(bt), fig_titles(tt),
    ]
    readme(s, figs, prof, overall)
    print("wrote", figs, "+ README.md")


if __name__ == "__main__":
    main()
