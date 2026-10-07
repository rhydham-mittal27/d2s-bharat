"""Command Center: market evidence from the SAS job data (RQ1), plus semantic skill lookup.

All tables are precomputed by run_rq1.py / build_artifacts.py; lookup embeds the query with the
same sentence-transformer used for skill normalisation and searches the canonical skill index.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel

from d2s.ml.embeddings import Embedder
from d2s.services import store


class SkillHit(BaseModel):
    skill: str
    similarity: float
    posts_all: int
    share_all: float
    posts_data_roles: int
    share_data_roles: float
    median_salary_mid_lakh: float | None
    top_role_family: str | None
    pay_odds_ratio: float | None  # adjusted ordinal-logit OR, if the skill is among those modelled
    pay_ci: tuple[float, float] | None
    pay_q_fdr: float | None


def overview() -> dict:
    rq1 = store.summary("rq1/summary.json", "run_rq1.py")
    sas = store.summary("sas/summary.json", "run_sas_ingestion.py")
    dem = store.table("rq1/demand_data_roles.csv")
    return {
        "postings_raw": sas["cleaning"]["raw_rows"],
        "postings_clean": rq1["posts"],
        "data_role_postings": rq1["data_role_posts"],
        "distinct_canonical_skills": rq1["distinct_canonical_skills"],
        "tag_uses_mapped_to_taxonomy_pct": round(rq1["canonical_tag_uses_pct"]["esco_onet"]
                                                 + rq1["canonical_tag_uses_pct"].get("merged_esco_onet", 0), 1),
        "openings_ds_jobs": sas["ds_jobs"]["jobs_total"],
        "companies_ds_jobs": sas["ds_jobs"]["companies"],
        "experience_or_per_year": round(rq1["pay_model_data_roles"]["exp_min_or_per_year"], 2),
        "top_demanded_data_skills": dem.head(5)[["skill", "share"]].round(4).to_dict("records"),
        "families": rq1["families"],
        "family_median_salary_lakh": rq1["family_median_salary"],
    }


def demand(scope: str = "data", top: int = 25) -> list[dict]:
    t = store.table("rq1/demand_data_roles.csv" if scope == "data" else "rq1/demand_all.csv")
    return t.head(top).round(4).to_dict("records")


def pay(scope: str = "data", significant_only: bool = True) -> list[dict]:
    t = store.table("rq1/pay_adjusted_data_roles.csv" if scope == "data" else "rq1/pay_adjusted_all.csv")
    if significant_only:
        t = t[t["q_fdr"] < 0.05]
    return t.round(4).to_dict("records")


def bundles(min_support: float = 0.01, min_confidence: float = 0.3, top: int = 20) -> list[dict]:
    r = store.table("rq1/association_rules_data_roles.csv")
    r = r[(r["support"] >= min_support) & (r["confidence"] >= min_confidence)]
    return r.head(top).round(4).to_dict("records")


def triangulation() -> list[dict]:
    return store.table("rq1/triangulation_rq1_rq2.csv").round(4).to_dict("records")


def ds_titles() -> list[dict]:
    t = store.table("sas/ds_jobs_by_title.csv", "run_sas_ingestion.py")
    return t.rename(columns={t.columns[0]: "job_title"}).round(2).to_dict("records")


class SkillLookup:
    """Free-text skill -> nearest canonical skills with their market evidence."""

    def __init__(self, embedder: Embedder, min_posts: int = 3):
        stats = store.artifact_table("skill_stats.csv")
        self.stats = stats[stats["posts_all"] >= min_posts].reset_index(drop=True)
        self.embedder = embedder
        names = self.stats["skill"].astype(str).tolist()
        cache = store.artifacts() / "skill_lookup.npz"
        self.matrix = None
        if cache.exists():  # built by build_artifacts.py; avoids ~75 s of embedding at start-up
            z = np.load(cache, allow_pickle=False)
            if str(z["model_name"]) == embedder.name and list(z["names"]) == names:
                self.matrix = z["matrix"]
        if self.matrix is None:
            self.matrix = embedder.embed_documents(names)
            np.savez_compressed(cache, names=np.array(names), matrix=self.matrix,
                                model_name=np.array(embedder.name))
        pay_all = store.table("rq1/pay_adjusted_all.csv").set_index("skill")
        self.pay = pay_all

    def search(self, query: str, k: int = 5) -> list[SkillHit]:
        q = query.strip()
        if not q:
            return []
        sims = self.embedder.embed_queries([q])[0] @ self.matrix.T
        exact = self.stats.index[self.stats["skill"].str.lower() == q.lower()]
        if len(exact):
            sims[exact] = np.maximum(sims[exact], 1.0)
        top = np.argsort(-sims)[:k]
        hits = []
        for i in top:
            r = self.stats.iloc[i]
            p = self.pay.loc[r["skill"]] if r["skill"] in self.pay.index else None
            hits.append(SkillHit(
                skill=r["skill"], similarity=float(min(sims[i], 1.0)), posts_all=int(r["posts_all"]),
                share_all=float(r["share_all"]), posts_data_roles=int(r["posts_data"]),
                share_data_roles=float(r["share_data"]),
                median_salary_mid_lakh=None if pd.isna(r["median_salary_mid"]) else float(r["median_salary_mid"]),
                top_role_family=r["top_family"] if isinstance(r["top_family"], str) else None,
                pay_odds_ratio=float(p["odds_ratio"]) if p is not None else None,
                pay_ci=(float(p["or_ci_low"]), float(p["or_ci_high"])) if p is not None else None,
                pay_q_fdr=float(p["q_fdr"]) if p is not None else None,
            ))
        return hits
