"""RQ2 / RQ3: which measured skills (JDS) or Big Five traits (SDS) are associated with a binary
outcome (salary hike high/low; success high/low)?

Small samples (139 / 161), so the design is deliberately conservative:
  * data audit (ranges, duplicates, outliers) before any modelling
  * per-feature Mann-Whitney U with rank-biserial effect size + bootstrap CI, Holm-corrected
  * L2-regularised logistic regression (C tuned by inner CV) and a depth<=3 decision tree,
    scored by repeated stratified 5-fold CV against a majority-class baseline
  * permutation test of the cross-validated AUC (is the model better than chance?)
  * bootstrap CIs for standardised odds ratios; permutation importance for the tree
Everything is seeded and reproducible.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    permutation_test_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

SEED = 42


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    features: list[str]
    target: str
    scale: tuple[float, float] | None  # expected value range, for the audit
    audit: dict = field(default_factory=dict)

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.features]

    @property
    def y(self) -> pd.Series:
        return self.frame[self.target]


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace(" ", "").replace("-", "_") for c in df.columns]
    return df


def load_jds(path) -> Dataset:
    raw = pd.read_excel(path)
    df = _clean_columns(raw)
    feats = ["big_data_skills", "maths_stats_skills", "coding_skills", "ai_and_ml_skills",
             "dashboard_and_storytelling_skills"]
    return Dataset("JDS (junior skills -> salary hike)", df, feats, "salary_hike_high_or_low", (1, 5),
                   audit={"renamed_columns": {a: b for a, b in zip(raw.columns, df.columns, strict=True) if a != b}})


def load_sds(path) -> Dataset:
    raw = pd.read_excel(path)
    df = _clean_columns(raw).rename(columns={"success_classification_high_low": "success_high_low"})
    feats = ["neuroticism", "extraversion", "openness_to_experience", "agreeableness", "conscientiousness"]
    renamed = {a: b for a, b in zip(raw.columns, df.columns, strict=True) if a != b}
    return Dataset("SDS (senior traits -> success)", df, feats, "success_high_low", None,
                   audit={"renamed_columns": renamed})


# ---- 1. data audit ------------------------------------------------------------------------------

def audit(ds: Dataset) -> dict:
    df = ds.frame
    dup_ids = df[df["id"].duplicated(keep=False)]
    a = {
        "rows": len(df),
        "missing_values": int(df.isna().sum().sum()),
        "fully_identical_rows": int(df.duplicated().sum()),
        "duplicate_id_rows": len(dup_ids),
        "duplicate_ids": int(dup_ids["id"].nunique()),
        "duplicate_ids_with_different_outcome": int(dup_ids.groupby("id")[ds.target].nunique().gt(1).sum()),
        "class_counts": {int(k): int(v) for k, v in ds.y.value_counts().sort_index().items()},
        "majority_baseline_accuracy": float(ds.y.value_counts(normalize=True).max()),
    }
    out_of_range = {}
    outliers = {}
    for f in ds.features:
        x = df[f]
        if ds.scale:
            out_of_range[f] = int(((x < ds.scale[0]) | (x > ds.scale[1])).sum())
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[f] = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
    a["out_of_range"] = out_of_range
    a["iqr_outliers"] = outliers
    a["ceiling_share_at_max"] = (
        {f: float((df[f] == ds.scale[1]).mean()) for f in ds.features} if ds.scale else {}
    )
    ds.audit.update(a)
    return ds.audit


# ---- 2. descriptives & univariate tests ----------------------------------------------------------

def _rank_biserial(x1: np.ndarray, x0: np.ndarray) -> float:
    u = stats.mannwhitneyu(x1, x0, alternative="two-sided").statistic
    return 2 * u / (len(x1) * len(x0)) - 1  # +1: high group always higher


def univariate(ds: Dataset, n_boot: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    hi, lo = ds.frame[ds.y == 1], ds.frame[ds.y == 0]
    for f in ds.features:
        x1, x0 = hi[f].to_numpy(), lo[f].to_numpy()
        mw = stats.mannwhitneyu(x1, x0, alternative="two-sided")
        r = _rank_biserial(x1, x0)
        boots = [
            _rank_biserial(rng.choice(x1, len(x1)), rng.choice(x0, len(x0))) for _ in range(n_boot)
        ]
        pooled = np.sqrt(((len(x1) - 1) * x1.var(ddof=1) + (len(x0) - 1) * x0.var(ddof=1)) / (len(x1) + len(x0) - 2))
        rows.append({
            "feature": f,
            "mean_high": x1.mean(), "mean_low": x0.mean(),
            "median_high": np.median(x1), "median_low": np.median(x0),
            "mean_diff": x1.mean() - x0.mean(),
            "cohens_d": (x1.mean() - x0.mean()) / pooled if pooled else 0.0,
            "rank_biserial": r,
            "rb_ci_low": float(np.percentile(boots, 2.5)), "rb_ci_high": float(np.percentile(boots, 97.5)),
            "mannwhitney_p": mw.pvalue,
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["mannwhitney_p"], method="holm")[1]
    out["significant_holm_0.05"] = out["p_holm"] < 0.05
    return out.sort_values("rank_biserial", key=np.abs, ascending=False).reset_index(drop=True)


def correlations(ds: Dataset) -> pd.DataFrame:
    return ds.frame[ds.features + [ds.target]].corr(method="spearman")


def vif(ds: Dataset) -> pd.DataFrame:
    X = StandardScaler().fit_transform(ds.X)
    X = np.column_stack([np.ones(len(X)), X])
    return pd.DataFrame({
        "feature": ds.features,
        "vif": [variance_inflation_factor(X, i + 1) for i in range(len(ds.features))],
    })


# ---- 3. models -------------------------------------------------------------------------------------

def _logit():
    # C is tuned on log-loss, not AUC: AUC ignores coefficient scale, so it happily picks extreme
    # shrinkage and the odds ratios collapse towards 1 (seen in the first run: SDS OR 1.26/SD
    # despite Cohen's d 1.85). Log-loss keeps the coefficients calibrated and interpretable.
    return make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=np.logspace(-3, 2, 20), cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                             scoring="neg_log_loss", max_iter=5000),
    )


def _tree():
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=SEED)


def model_comparison(ds: Dataset, repeats: int = 20) -> tuple[pd.DataFrame, dict]:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=SEED)
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "roc_auc": "roc_auc"}
    models = {
        "majority baseline": DummyClassifier(strategy="most_frequent"),
        "L2 logistic regression": _logit(),
        "decision tree (depth<=3)": _tree(),
    }
    rows, raw = [], {}
    for name, m in models.items():
        res = cross_validate(m, ds.X, ds.y, cv=cv, scoring=scoring, n_jobs=-1)
        raw[name] = res
        row = {"model": name}
        for k in scoring:
            v = res[f"test_{k}"]
            # mean over folds; 95% interval across the 5x`repeats` fold scores
            row[f"{k}_mean"] = v.mean()
            row[f"{k}_p2.5"] = np.percentile(v, 2.5)
            row[f"{k}_p97.5"] = np.percentile(v, 97.5)
        rows.append(row)
    return pd.DataFrame(rows), raw


def permutation_significance(ds: Dataset, n_permutations: int = 500) -> dict:
    """Is the logistic model's CV AUC better than with shuffled labels?"""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    simple = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    score, perm, p = permutation_test_score(simple, ds.X, ds.y, scoring="roc_auc", cv=cv,
                                            n_permutations=n_permutations, random_state=SEED, n_jobs=-1)
    return {"cv_auc": float(score), "null_auc_mean": float(perm.mean()),
            "null_auc_p95": float(np.percentile(perm, 95)), "p_value": float(p)}


def odds_ratios(ds: Dataset, n_boot: int = 1000) -> pd.DataFrame:
    """Odds ratio per +1 SD of each feature (L2 logit, C chosen on full data), bootstrap 95% CI."""
    full = _logit().fit(ds.X, ds.y)
    C = float(full[-1].C_[0])
    coef = full[-1].coef_[0]
    rng = np.random.default_rng(SEED)
    boot = []
    X, y = ds.X.to_numpy(), ds.y.to_numpy()
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[idx], y[idx])
        boot.append(m[-1].coef_[0])
    boot = np.array(boot)
    out = pd.DataFrame({
        "feature": ds.features,
        "coef_std": coef,
        "odds_ratio_per_sd": np.exp(coef),
        "or_ci_low": np.exp(np.percentile(boot, 2.5, axis=0)),
        "or_ci_high": np.exp(np.percentile(boot, 97.5, axis=0)),
        "share_boot_positive": (boot > 0).mean(axis=0),
    })
    out["ci_excludes_1"] = (out["or_ci_low"] > 1) | (out["or_ci_high"] < 1)
    out.attrs["C"] = C
    return out.sort_values("coef_std", key=np.abs, ascending=False).reset_index(drop=True)


def tree_explanation(ds: Dataset) -> tuple[DecisionTreeClassifier, str, pd.DataFrame]:
    tree = _tree().fit(ds.X, ds.y)
    rules = export_text(tree, feature_names=ds.features, decimals=2, show_weights=True)
    # importance on held-out folds (permutation, AUC drop), averaged over CV splits
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    imps = []
    for tr, te in cv.split(ds.X, ds.y):
        m = _tree().fit(ds.X.iloc[tr], ds.y.iloc[tr])
        r = permutation_importance(m, ds.X.iloc[te], ds.y.iloc[te], scoring="roc_auc",
                                   n_repeats=30, random_state=SEED)
        imps.append(r.importances_mean)
    imp = pd.DataFrame({"feature": ds.features, "perm_importance_auc_drop": np.mean(imps, axis=0),
                        "gini_importance": tree.feature_importances_})
    return tree, rules, imp.sort_values("perm_importance_auc_drop", ascending=False).reset_index(drop=True)
