"""Train and persist the artifacts the application services load at start-up.

Needs: run_sas_ingestion.py, run_rq1.py, run_rq2_rq3.py outputs.
Usage: uv run python scripts/build_artifacts.py
Outputs (dataset/artifacts/):
  hike_model.pkl     junior salary-hike model + bootstrap ensemble (Gap Map)
  hike_model.json    model card: training data, C, coefficients, CV metrics
  skill_stats.csv    per canonical skill: demand (all / data roles), median salary, top family
  area_roles.csv     per JDS skill area: share of postings in each role family that require it
"""

import json
import sys

import numpy as np
import pandas as pd

from d2s.analysis import jobs
from d2s.analysis.traits import load_jds
from d2s.ingest import parsing
from d2s.models.hike import LABELS, SKILLS, HikeModel
from d2s.services import store

sys.stdout.reconfigure(encoding="utf-8")


def main():
    out = store.artifacts()
    out.mkdir(parents=True, exist_ok=True)

    # 1. hike model ------------------------------------------------------------------------------
    jds = load_jds(store.SAS_DIR / "JDS Skill Traits.xlsx").frame
    m = HikeModel.train(jds)
    cv = store.table("rq23/rq2_jds_models.csv", "run_rq2_rq3.py").set_index("model")
    m.cv_auc = float(cv.loc["L2 logistic regression", "roc_auc_mean"])
    m.save(out / "hike_model.pkl")
    card = {
        "name": "junior salary-hike model", "training_data": "SAS JDS Skill Traits (139 junior data scientists)",
        "algorithm": "standardised L2 logistic regression, C tuned by 5-fold CV on log-loss",
        "C": m.C, "n_train": m.n_train, "bootstrap_refits": len(m.boot_coef),
        "coefficients_per_sd": {LABELS[s]: round(float(c), 4) for s, c in zip(SKILLS, m.coef, strict=True)},
        "intercept": round(m.intercept, 4),
        "cv_auc_repeated_5fold": round(m.cv_auc, 4),
        "high_hike_group_median": {LABELS[s]: v for s, v in m.high_median.items()},
        "intended_use": "estimate a junior's likelihood of a high salary hike from five skill scores and show which "
                        "skill improvements matter most; association-based, not causal; sample data",
        "not_for": "hiring or pay decisions about individuals",
    }
    (out / "hike_model.json").write_text(json.dumps(card, indent=2))
    print("hike model:", card["coefficients_per_sd"], "AUC", card["cv_auc_repeated_5fold"])

    # 2. skill statistics -----------------------------------------------------------------------------
    posts, _ = parsing.load_sas_analytics(store.SAS_DIR / "Analytics Jobs.csv")
    posts["family"] = posts["title"].map(jobs.role_family)
    ps = store.table("rq1/post_skills_canonical.csv.gz").merge(
        posts[["post_id", "family", "sal_mid_lakh"]], on="post_id")
    n_all = posts["post_id"].nunique()
    data_ids = set(posts.loc[posts["family"].isin(jobs.DATA_FAMILIES), "post_id"])
    n_data = len(data_ids)
    g = ps.groupby("skill")
    stats = pd.DataFrame({
        "posts_all": g["post_id"].nunique(),
        "posts_data": g.apply(lambda d: d["post_id"].isin(data_ids).sum(), include_groups=False),
        "median_salary_mid": g["sal_mid_lakh"].median(),
        "top_family": g["family"].agg(lambda s: s.value_counts().index[0]),
    }).reset_index()
    stats["share_all"] = stats["posts_all"] / n_all
    stats["share_data"] = stats["posts_data"] / n_data
    stats.sort_values("posts_all", ascending=False).to_csv(out / "skill_stats.csv", index=False)
    print("skill_stats:", len(stats), "skills")
    from d2s.ml.embeddings import get_embedder
    from d2s.services.market import SkillLookup

    lookup = SkillLookup(get_embedder())  # writes skill_lookup.npz (embedding cache)
    print("skill_lookup cache:", len(lookup.stats), "skills")

    # 3. JDS area -> role families --------------------------------------------------------------------
    flags = jobs.jds_area_flags(ps[["post_id", "skill"]], posts)
    fam = posts.set_index("post_id")["family"]
    rows = []
    for area in SKILLS:
        sub = flags[area]
        for f, ids in fam.groupby(fam):
            share = float(sub.reindex(ids.index).mean())
            rows.append({"area": area, "role": f, "share": share, "family_postings": len(ids)})
    ar = pd.DataFrame(rows)
    ar = ar[ar["role"] != "Other"]
    ar.to_csv(out / "area_roles.csv", index=False)
    print(ar.sort_values(["area", "share"], ascending=[True, False]).groupby("area").head(2).round(3).to_string(index=False))
    print("DONE", out)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
