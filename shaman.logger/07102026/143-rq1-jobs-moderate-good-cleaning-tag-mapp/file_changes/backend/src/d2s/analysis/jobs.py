"""RQ1: which skills and roles are demanded, and which are associated with higher pay?

Builds on the SAS ingestion outputs (dataset/processed/sas). Strengthens the first pass by:
  * role families from transparent title rules (so data roles can be analysed on their own)
  * canonical skills for *every* recruiter tag: ESCO/O*NET label where confidently mapped,
    otherwise a data-derived label that merges near-duplicate tag spellings
  * demand shares with Wilson 95% CIs
  * pay: rank tests with Benjamini-Hochberg FDR, and an ordinal logistic model of the six
    salary bands that controls for experience and role family (adjusted pay premium)
  * skill co-occurrence rules (support / confidence / lift)
  * triangulation with RQ2: the five JDS skill areas measured in the job market
"""

import re

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from d2s.ml.skills import _norm

SALARY_BANDS = ["0to3", "3to6", "6to10", "10to15", "15to25", "25to50"]

# Ordered rules: first match wins. Patterns run on the lower-cased title.
ROLE_RULES: list[tuple[str, str]] = [
    ("Data Science / ML", r"data scien|machine learning|\bml\b|deep learning|\bai\b|artificial intel|\bnlp\b|"
                          r"computer vision|statistician|\bquant|predictive|modell?er|modell?ing|\bmle\b"),
    ("Data Engineering", r"data engineer|big data|hadoop|spark|\betl\b|data architect|data warehous|databricks|"
                         r"kafka|informatica|data platform|\bdba\b|database|data model"),
    ("Analytics / BI", r"data analy|analytics|\bbi\b|business intelligence|power ?bi|tableau|\bmis\b|reporting|"
                       r"insight|\bsas\b|statistic|research analyst"),
    ("Business Analysis / Product", r"business analy|\bba\b|product manager|product owner|product analyst|"
                                    r"functional consultant|business consultant"),
    ("Marketing / SEO / Content", r"\bseo\b|\bsem\b|\bsmo\b|digital marketing|marketing|content|social media|\bppc\b|"
                                  r"brand|copywrit|campaign"),
    ("Finance / Risk", r"financ|account|credit|risk|audit|\btax|banking|investment|trading|equity|\bca\b|"
                       r"treasury|actuar|valuation|fp&a"),
    ("Software / IT", r"develop|software|engineer|java|python|\.net|\bsap\b|android|\bios\b|web|full ?stack|"
                      r"\bqa\b|test|devops|cloud|architect|programmer|\bit\b|technical|salesforce|oracle|"
                      r"splunk|maximo|angular|\bmule\b|\besb\b|opentext|bot framework|application lead"),
    ("Sales / Ops / HR / Support", r"sales|business development|operations|\bhr\b|recruit|customer|support|"
                                   r"logistic|supply chain|store|data entry|home base|admin|executive assistant|"
                                   r"telecall|back office|bpo|process associate|voice|fresher|procurement|purchase"),
    ("Management / Consulting", r"project|program|delivery|consult|strateg|advisor|director|\bhead\b|\bcoo\b|"
                                r"\bceo\b|chief|entrepreneur|\bmanager\b|business research"),
    ("Analytics / BI", r"\banalyst\b"),  # generic "Analyst" falls back to analytics
]

# Corrections from the manual spot-check of semantic tag mappings: these tags keep their own
# (data-derived) label instead of the wrong ESCO match, e.g. "Analytical" -> "analytical chemistry".
REVIEWED_TAG_OVERRIDES = frozenset({"analytical", "business development", "double click", "outsourcing"})
DATA_FAMILIES = ["Data Science / ML", "Data Engineering", "Analytics / BI", "Business Analysis / Product"]

# The five JDS skill areas (RQ2), measured on canonical skill labels in postings.
JDS_AREAS: dict[str, str] = {
    "big_data_skills": r"hadoop|spark|big data|\bhive\b|kafka|\bpig\b|nosql|hbase|mapreduce|databricks|"
                       r"data lake|flume|sqoop|impala",
    "maths_stats_skills": r"statistic|mathemat|probabilit|regression|econometric|hypothesis|quantitative|"
                          r"forecast|predictive model|time series|operations research",
    "coding_skills": r"python|^r$|\bsql\b|\bsas\b|java \(computer|programming|c\+\+|\bvba\b|scala|"
                     r"computer programming|\bperl\b|matlab",
    "ai_and_ml_skills": r"machine learning|deep learning|artificial intel|neural|natural language|\bnlp\b|"
                        r"computer vision|tensorflow|keras|pytorch|data mining|\bai\b",
    # Narrow: visualisation & dashboard tools / storytelling. "reporting" and "MIS" are excluded here
    # because in postings they mostly mark clerical reporting jobs; see JDS_AREAS_BROAD.
    "dashboard_and_storytelling_skills": r"tableau|power ?bi|business intelligence|visuali|dashboard|qlik|"
                                        r"storytell|data presentation|spotfire",
}
# Sensitivity variant matching the JDS definition literally ("visualization, reporting and storytelling").
JDS_AREAS_BROAD = {**JDS_AREAS, "dashboard_and_storytelling_skills":
                   JDS_AREAS["dashboard_and_storytelling_skills"] + r"|report|\bmis\b"}


def role_family(title: str) -> str:
    t = str(title).lower()
    for family, pattern in ROLE_RULES:
        if re.search(pattern, t):
            return family
    return "Other"


# ---- canonical skills --------------------------------------------------------------------------

def canonical_tags(tag_table: pd.DataFrame, embedder, merge_sim: float = 0.92) -> pd.DataFrame:
    """key -> canonical skill label for EVERY distinct tag.

    1. confidently mapped tags (exact / contains / semantic) keep their ESCO/O*NET label;
    2. every other tag joins the most frequent *more popular* tag whose spelling it nearly
       duplicates (embedding similarity >= merge_sim between the tags themselves);
    3. otherwise it stays its own data-derived skill, labelled with its most common spelling.
    """
    t = tag_table.copy()
    mapped = t["method"].isin(["exact", "contains", "semantic"]) & ~t["key"].isin(REVIEWED_TAG_OVERRIDES)
    t["canonical"] = np.where(mapped, t["nearest_label"], None)
    t["canonical_source"] = np.where(mapped, "esco_onet", None)

    order = t.sort_values("posts", ascending=False).reset_index(drop=True)
    vecs = embedder.embed_documents(order["tag"].tolist())
    canon = order["canonical"].tolist()
    source = order["canonical_source"].tolist()
    for i in range(len(order)):
        if isinstance(canon[i], str) and canon[i]:  # already mapped (unmapped rows are NaN/None)
            continue
        if i:
            sims = vecs[:i] @ vecs[i]
            j = int(np.argmax(sims))
            if sims[j] >= merge_sim:
                canon[i] = canon[j]
                source[i] = "merged_" + ("esco_onet" if "esco_onet" in source[j] else "tag")
                continue
        canon[i] = order.at[i, "tag"].strip()
        source[i] = "tag"
    order["canonical"] = canon
    order["canonical_source"] = source
    return order


def post_skill_table(posts: pd.DataFrame, canon: pd.DataFrame, text_matches: pd.DataFrame) -> pd.DataFrame:
    """(post_id, skill) from recruiter tags (canonicalised) + exact-name matches in title/description."""
    ex = posts[["post_id", "tags"]].explode("tags").dropna(subset=["tags"])
    ex["key"] = ex["tags"].map(_norm)
    tag_rows = ex.merge(canon[["key", "canonical", "canonical_source"]], on="key")[
        ["post_id", "canonical", "canonical_source"]].rename(columns={"canonical": "skill"})
    tag_rows["signal"] = "key_skills"
    txt = text_matches[text_matches["method"] != "semantic"][["post_id", "label"]].rename(columns={"label": "skill"})
    txt["canonical_source"] = "esco_onet"
    txt["signal"] = "text"
    both = pd.concat([tag_rows, txt], ignore_index=True)
    return both.drop_duplicates(["post_id", "skill"])


# ---- demand -----------------------------------------------------------------------------------------

def demand(post_skills: pd.DataFrame, posts: pd.DataFrame, top: int = 40) -> pd.DataFrame:
    n = posts["post_id"].nunique()
    c = post_skills.groupby("skill")["post_id"].nunique().nlargest(top)
    lo, hi = proportion_confint(c.to_numpy(), n, method="wilson")
    return pd.DataFrame({"skill": c.index, "posts": c.to_numpy(), "share": c.to_numpy() / n,
                         "share_ci_low": lo, "share_ci_high": hi})


def demand_by_family(post_skills: pd.DataFrame, posts: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    m = post_skills.merge(posts[["post_id", "family"]], on="post_id")
    sizes = posts.groupby("family")["post_id"].nunique()
    g = m.groupby(["family", "skill"])["post_id"].nunique().rename("posts").reset_index()
    g["share_in_family"] = g["posts"] / g["family"].map(sizes)
    return g.sort_values(["family", "posts"], ascending=[True, False]).groupby("family").head(top_n)


# ---- pay ---------------------------------------------------------------------------------------------

def _design(posts: pd.DataFrame, post_skills: pd.DataFrame, skills: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=posts["post_id"])
    has = post_skills[post_skills["skill"].isin(skills)]
    for s in skills:
        X[s] = X.index.isin(has.loc[has["skill"] == s, "post_id"]).astype(float)
    return X


def pay_unadjusted(posts: pd.DataFrame, post_skills: pd.DataFrame, skills: list[str]) -> pd.DataFrame:
    """Rank-biserial (Cliff's delta) of salary midpoint, posts with vs without the skill; BH-FDR."""
    sal = posts.set_index("post_id")["sal_mid_lakh"]
    X = _design(posts, post_skills, skills)
    rows = []
    for s in skills:
        with_, without = sal[X[s] == 1], sal[X[s] == 0]
        u = stats.mannwhitneyu(with_, without, alternative="two-sided")
        rows.append({"skill": s, "posts": len(with_), "median_with": with_.median(),
                     "median_without": without.median(),
                     "cliffs_delta": 2 * u.statistic / (len(with_) * len(without)) - 1, "p": u.pvalue})
    out = pd.DataFrame(rows)
    out["q_fdr"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.sort_values("cliffs_delta", ascending=False).reset_index(drop=True)


def pay_adjusted(posts: pd.DataFrame, post_skills: pd.DataFrame, skills: list[str],
                 families: bool = True) -> tuple[pd.DataFrame, dict]:
    """Ordinal logit of the 6 salary bands on skill dummies + experience (+ role family).

    exp(coef) = odds ratio of being in a *higher* salary band when the posting lists the skill,
    holding experience (and role family) constant.
    """
    df = posts.set_index("post_id")
    y = pd.Categorical(df["salary"], categories=SALARY_BANDS, ordered=True)
    X = _design(posts, post_skills, skills)
    X["exp_min_years"] = df["exp_min"].to_numpy()
    X["multi_city"] = (df["locations"].map(len) > 1).astype(float).to_numpy() if "locations" in df else 0.0
    if families:
        fam = pd.get_dummies(df["family"], prefix="fam", dtype=float)
        # one reference category: "Other" when present, else the first family (dummies would sum to 1)
        ref = "fam_Other" if "fam_Other" in fam.columns else fam.columns[0]
        X = pd.concat([X, fam.drop(columns=[ref])], axis=1)
    keep = ~pd.isna(y)
    X = X.loc[keep]
    X = X.loc[:, X.std() > 0]
    res = OrderedModel(np.asarray(y.codes)[keep], X, distr="logit").fit(method="bfgs", maxiter=2000, disp=False)
    ci = res.conf_int()
    rows = []
    for s in skills:
        if s not in res.params.index:
            continue
        rows.append({"skill": s, "odds_ratio": np.exp(res.params[s]), "or_ci_low": np.exp(ci.loc[s, 0]),
                     "or_ci_high": np.exp(ci.loc[s, 1]), "p": res.pvalues[s]})
    out = pd.DataFrame(rows)
    out["q_fdr"] = multipletests(out["p"], method="fdr_bh")[1]
    info = {"n": int(keep.sum()), "exp_min_or_per_year": float(np.exp(res.params["exp_min_years"])),
            "pseudo_r2": float(res.prsquared), "converged": bool(res.mle_retvals.get("converged", True))}
    return out.sort_values("odds_ratio", ascending=False).reset_index(drop=True), info


# ---- co-occurrence ------------------------------------------------------------------------------

def association_rules(post_skills: pd.DataFrame, posts: pd.DataFrame, top: int = 60,
                      min_support: float = 0.005) -> pd.DataFrame:
    n = posts["post_id"].nunique()
    skills = post_skills.groupby("skill")["post_id"].nunique().nlargest(top).index
    X = _design(posts, post_skills, list(skills)).to_numpy(dtype=bool)
    support = X.mean(axis=0)
    co = (X.T.astype(np.int32) @ X.astype(np.int32)) / n
    rows = []
    for i, a in enumerate(skills):
        for j, b in enumerate(skills):
            if i == j or co[i, j] < min_support:
                continue
            conf = co[i, j] / support[i]
            rows.append({"antecedent": a, "consequent": b, "support": co[i, j], "confidence": conf,
                         "lift": conf / support[j]})
    return pd.DataFrame(rows).sort_values(["lift", "support"], ascending=False).reset_index(drop=True)


# ---- triangulation with RQ2 ------------------------------------------------------------------------

def jds_area_flags(post_skills: pd.DataFrame, posts: pd.DataFrame, areas: dict | None = None) -> pd.DataFrame:
    lower = post_skills.assign(s=post_skills["skill"].str.lower())
    flags = pd.DataFrame(index=posts["post_id"])
    for area, pat in (areas or JDS_AREAS).items():
        ids = lower.loc[lower["s"].str.contains(pat, regex=True), "post_id"].unique()
        flags[area] = flags.index.isin(ids).astype(float)
    return flags


def area_market_table(posts: pd.DataFrame, post_skills: pd.DataFrame,
                      areas: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Per JDS area, in data-role postings: demand share and adjusted pay odds ratio."""
    flags = jds_area_flags(post_skills, posts, areas)
    pseudo = flags.stack().reset_index()
    pseudo.columns = ["post_id", "skill", "v"]
    pseudo = pseudo[pseudo["v"] == 1][["post_id", "skill"]]
    pay, info = pay_adjusted(posts, pseudo, list(JDS_AREAS), families=True)
    n = len(flags)
    share = flags.mean()
    lo, hi = proportion_confint((share * n).round().to_numpy(), n, method="wilson")
    dem = pd.DataFrame({"area": share.index, "demand_share": share.to_numpy(), "demand_ci_low": lo,
                        "demand_ci_high": hi})
    return dem.merge(pay.rename(columns={"skill": "area"}), on="area"), info


# ---- DataScience Jobs (company-level) ---------------------------------------------------------------

def ds_jobs_stats(ds: pd.DataFrame) -> dict:
    rho, p = stats.spearmanr(ds["min_experience"], ds["avg_lakh"])
    by_co = ds.groupby("company_name")["num_of_jobs"].sum().sort_values(ascending=False)
    shares = by_co / by_co.sum()
    q1, q3 = ds["avg_lakh"].quantile([0.25, 0.75])
    out_hi = ds[ds["avg_lakh"] > q3 + 1.5 * (q3 - q1)]
    weighted = (ds["avg_lakh"] * ds["num_of_jobs"]).sum() / ds["num_of_jobs"].sum()
    return {
        "spearman_minexp_avgsalary": float(rho), "spearman_p": float(p),
        "top10_company_share_of_openings": float(shares.head(10).sum()),
        "hhi_openings": float((shares ** 2).sum()),
        "top_companies": {k: int(v) for k, v in by_co.head(10).items()},
        "salary_outliers_iqr": len(out_hi),
        "salary_outlier_examples": out_hi.nlargest(5, "avg_lakh")[["company_name", "job_title", "avg_lakh",
                                                                   "num_of_jobs"]].to_dict("records"),
        "job_weighted_avg_lakh": float(weighted),
        "row_avg_lakh_median": float(ds["avg_lakh"].median()),
    }
