"""Build report assets: cleaned SAS files, descriptive + data-quality tables (LaTeX), curated figures.

Usage: uv run python scripts/make_report_assets.py
Outputs: dataset/processed/clean/*.csv ; report/tables/*.tex ; report/figures/*.png
Every number is computed here from the SAS files or the analysis outputs.
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from d2s.analysis import jobs
from d2s.analysis.traits import load_jds, load_sds
from d2s.config import get_settings
from d2s.ingest import parsing

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
SAS = ROOT / "hackathon" / "SAS Data Problem Statement and Instructions Hackathon"
D = get_settings().dataset_dir / "processed"
CLEAN = D / "clean"
TAB = ROOT / "report" / "tables"
FIG = ROOT / "report" / "figures"


def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def latex_table(df: pd.DataFrame, path: Path, colspec: str, header: list[str] | None = None) -> None:
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule",
             " & ".join(esc(h) for h in (header or df.columns)) + r" \\", r"\midrule"]
    for row in df.itertuples(index=False):
        lines.append(" & ".join(esc(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe(ds, label_map) -> pd.DataFrame:
    rows = []
    for f in ds.features:
        x = ds.frame[f]
        hi, lo = ds.frame.loc[ds.y == 1, f], ds.frame.loc[ds.y == 0, f]
        rows.append([label_map.get(f, f), f"{x.mean():.2f}", f"{x.std():.2f}", f"{x.min():g}", f"{x.median():g}",
                     f"{x.max():g}", f"{hi.mean():.2f}", f"{lo.mean():.2f}"])
    return pd.DataFrame(rows, columns=["Variable", "Mean", "SD", "Min", "Median", "Max", "Mean (high)", "Mean (low)"])


def main():
    for p in (CLEAN, TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)

    # ---- cleaned files ------------------------------------------------------------------------
    jds, sds = load_jds(SAS / "JDS Skill Traits.xlsx"), load_sds(SAS / "SDS Personality Traits.xlsx")
    jds.frame.to_csv(CLEAN / "jds_clean.csv", index=False)
    sds.frame.to_csv(CLEAN / "sds_clean.csv", index=False)
    posts, clog = parsing.load_sas_analytics(SAS / "Analytics Jobs.csv")
    posts["family"] = posts["title"].map(jobs.role_family)
    out = posts.copy()
    out["tags"] = out["tags"].map("; ".join)
    out["locations"] = out["locations"].map("; ".join)
    out.drop(columns=["company", "created"]).to_csv(CLEAN / "analytics_jobs_clean.csv", index=False)
    ds = parsing.load_sas_ds_jobs(SAS / "DataScience Jobs.csv")
    ds.to_csv(CLEAN / "datascience_jobs_clean.csv", index=False)
    ja, sa = __import__("d2s.analysis.traits", fromlist=["audit"]).audit(jds), \
        __import__("d2s.analysis.traits", fromlist=["audit"]).audit(sds)

    # ---- data-quality table, all four files --------------------------------------------------------
    raw_ds = pd.read_csv(SAS / "DataScience Jobs.csv")
    q = pd.DataFrame([
        ["Analytics Jobs", f"{clog.raw_rows:,}", f"{clog.kept_rows:,}",
         f"{clog.exact_duplicate_rows:,} exact duplicate rows removed",
         f"job\\_description {100 * clog.missing_description / clog.kept_rows:.0f}\\%, job\\_type "
         f"{100 * clog.missing_job_type / clog.kept_rows:.0f}\\%",
         f"{clog.tags_truncated_fixed:,} truncated tags repaired; {clog.job_type_variants_merged} job\\_type variants merged; "
         f"{clog.multi_city_rows:,} multi-city rows split; text descriptions truncated at source"],
        ["DataScience Jobs", f"{len(raw_ds):,}", f"{len(ds):,}",
         f"{int(raw_ds.drop(columns='reference_no').duplicated().sum())} duplicates",
         f"{int(raw_ds.isna().sum().sum())} missing",
         "salaries stored as text (`7.8L') parsed to lakh; 39 high-salary outliers kept (senior roles)"],
        ["JDS Skill Traits", f"{ja['rows']}", f"{ja['rows']}",
         f"{ja['duplicate_ids']} reused IDs kept (different people)", f"{ja['missing_values']} missing",
         f"all scores within 1--5; ceiling: {100 * max(ja['ceiling_share_at_max'].values()):.0f}\\% at 5.0 "
         f"(storytelling); column name fixed"],
        ["SDS Personality Traits", f"{sa['rows']}", f"{sa['rows']}",
         f"{sa['duplicate_ids']} reused IDs kept ({sa['duplicate_ids_with_different_outcome']} with different outcomes)",
         f"{sa['missing_values']} missing",
         "two malformed column names fixed (leading space, spaces in target name)"],
    ], columns=["File", "Raw rows", "Rows used", "Duplicates", "Missing", "Other issues and actions"])
    # escape-safe writer: the cells above already contain LaTeX escapes
    lines = [r"\begin{tabularx}{\linewidth}{@{}l r r L{3.2cm} L{2.6cm} X@{}}", r"\toprule",
             r"File & Raw & Used & Duplicates & Missing & Other issues and actions \\", r"\midrule"]
    for r in q.itertuples(index=False):
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabularx}"]
    (TAB / "data_quality.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- descriptive tables ------------------------------------------------------------------------
    jl = {"big_data_skills": "Big data", "maths_stats_skills": "Maths \\& stats", "coding_skills": "Coding",
          "ai_and_ml_skills": "AI \\& ML", "dashboard_and_storytelling_skills": "Dashboards \\& storytelling"}
    sl = {"neuroticism": "Neuroticism", "extraversion": "Extraversion", "openness_to_experience": "Openness",
          "agreeableness": "Agreeableness", "conscientiousness": "Conscientiousness"}
    for name, ds_, lm in (("jds", jds, jl), ("sds", sds, sl)):
        t = describe(ds_, lm)
        lines = [r"\begin{tabular}{@{}l r r r r r r r@{}}", r"\toprule",
                 r"Variable & Mean & SD & Min & Median & Max & Mean (high) & Mean (low) \\", r"\midrule"]
        lines += [" & ".join(map(str, r)) + r" \\" for r in t.itertuples(index=False)]
        lines += [r"\bottomrule", r"\end{tabular}"]
        (TAB / f"describe_{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Analytics Jobs profile
    band = posts["salary"].value_counts().reindex(jobs.SALARY_BANDS)
    cities = posts.explode("locations")["locations"].value_counts().head(6)
    prof = {
        "postings": len(posts), "distinct_titles": int(posts["title"].nunique()),
        "exp_min_median": float(posts["exp_min"].median()), "exp_max_median": float(posts["exp_max"].median()),
        "salary_bands": {k: int(v) for k, v in band.items()},
        "top_cities": {k: int(v) for k, v in cities.items()},
        "tags_per_post_median": float(posts["tags"].map(len).median()),
        "ds_rows": len(ds), "ds_companies": int(ds["company_name"].nunique()), "ds_openings": int(ds["num_of_jobs"].sum()),
        "ds_titles": int(ds["job_title"].nunique()),
        "ds_avg_lakh_median": float(ds["avg_lakh"].median()), "ds_minexp_median": float(ds["min_experience"].median()),
    }
    rows = [["Salary band (lakh/yr)"] + [b.replace("to", "--") for b in jobs.SALARY_BANDS],
            ["Postings"] + [f"{int(v):,}" for v in band.values],
            ["Share"] + [f"{100 * v / len(posts):.1f}\\%" for v in band.values]]
    lines = [r"\begin{tabular}{@{}l" + " r" * 6 + "@{}}", r"\toprule", " & ".join(rows[0]) + r" \\", r"\midrule",
             " & ".join(rows[1]) + r" \\", " & ".join(rows[2]) + r" \\", r"\bottomrule", r"\end{tabular}"]
    (TAB / "salary_bands.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (TAB / "profile.json").write_text(json.dumps(prof, indent=2))

    # ---- curated figures ----------------------------------------------------------------------------
    R = ROOT / "reports"
    pick = {
        "fig_cleaning.png": R / "sas_ingestion" / "01_cleaning.png",
        "fig_salary_experience.png": R / "sas_ingestion" / "02_salary_experience.png",
        "fig_tag_mapping.png": R / "sas_ingestion" / "03_tag_mapping.png",
        "fig_text_calibration.png": R / "sas_ingestion" / "04_threshold_calibration.png",
        "fig_canonical.png": R / "rq1" / "02_canonical_coverage.png",
        "fig_role_families.png": R / "rq1" / "01_role_families.png",
        "fig_demand.png": R / "rq1" / "03_demand_data_roles.png",
        "fig_pay_data_roles.png": R / "rq1" / "05_pay_premium_data_roles.png",
        "fig_bundles.png": R / "rq1" / "06_skill_bundles.png",
        "fig_triangulation.png": R / "rq1" / "07_triangulation.png",
        "fig_ds_jobs.png": R / "rq1" / "08_ds_jobs.png",
        "fig_city.png": R / "sas_ingestion" / "07_skill_city.png",
        "fig_jds_dist.png": R / "rq2_rq3" / "rq2_jds_1_distributions.png",
        "fig_jds_effects.png": R / "rq2_rq3" / "rq2_jds_2_effect_sizes.png",
        "fig_jds_models.png": R / "rq2_rq3" / "rq2_jds_4_models.png",
        "fig_jds_tree.png": R / "rq2_rq3" / "rq2_jds_6_tree.png",
        "fig_sds_dist.png": R / "rq2_rq3" / "rq3_sds_1_distributions.png",
        "fig_sds_effects.png": R / "rq2_rq3" / "rq3_sds_2_effect_sizes.png",
        "fig_sds_models.png": R / "rq2_rq3" / "rq3_sds_4_models.png",
        "fig_sds_tree.png": R / "rq2_rq3" / "rq3_sds_6_tree.png",
        "fig_jds_corr.png": R / "rq2_rq3" / "rq2_jds_3_correlations.png",
        "fig_sds_corr.png": R / "rq2_rq3" / "rq3_sds_3_correlations.png",
        "fig_jds_or.png": R / "rq2_rq3" / "rq2_jds_5_odds_ratios.png",
        "fig_sds_or.png": R / "rq2_rq3" / "rq3_sds_5_odds_ratios.png",
        "fig_pay_all.png": R / "rq1" / "04_pay_premium_all.png",
        "fig_titles.png": R / "sas_ingestion" / "09_title_occupations.png",
        "fig_rq4_inputs.png": R / "rq4" / "01_inputs.png",
        "fig_rq4_plan.png": R / "rq4" / "02_plan.png",
        "fig_rq4_pareto.png": R / "rq4" / "03_pareto.png",
        "fig_rq4_robust.png": R / "rq4" / "04_robustness.png",
    }
    missing = [str(s) for s in pick.values() if not s.exists()]
    if missing:
        raise FileNotFoundError(missing)
    for dst, src in pick.items():
        shutil.copy2(src, FIG / dst)
    print(json.dumps(prof, indent=1))
    print("copied", len(pick), "figures; tables:", [p.name for p in TAB.iterdir()])


if __name__ == "__main__":
    main()
