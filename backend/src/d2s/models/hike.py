"""Junior salary-hike model (RQ2) for the Gap Map: predict, explain, and suggest what to improve.

* Model     : standardised L2 logistic regression, C tuned on log-loss (same as the RQ2 analysis).
* Uncertainty: a bootstrap ensemble of refits gives a 90% range for each predicted probability.
* Explanation: each skill's contribution to the log-odds, relative to the average junior
               (coef x standardised distance from the cohort mean), so contributions sum exactly
               to the prediction's log-odds minus the baseline log-odds.
* Counterfactuals: the smallest single-skill raise (in 0.1 steps, up to the scale max) that brings
               the probability to a target, and the probability gain from raising each skill to
               the high-hike group's median.
Scores outside the model's training range are clipped and flagged, never silently extrapolated.
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

SKILLS = ["coding_skills", "maths_stats_skills", "ai_and_ml_skills", "big_data_skills",
          "dashboard_and_storytelling_skills"]
LABELS = {"coding_skills": "Coding", "maths_stats_skills": "Maths & stats", "ai_and_ml_skills": "AI & ML",
          "big_data_skills": "Big data", "dashboard_and_storytelling_skills": "Dashboards & storytelling"}
TARGET = "salary_hike_high_or_low"
SCALE = (1.0, 5.0)
SEED = 42


class SkillScores(BaseModel):
    coding_skills: float = Field(ge=SCALE[0], le=SCALE[1])
    maths_stats_skills: float = Field(ge=SCALE[0], le=SCALE[1])
    ai_and_ml_skills: float = Field(ge=SCALE[0], le=SCALE[1])
    big_data_skills: float = Field(ge=SCALE[0], le=SCALE[1])
    dashboard_and_storytelling_skills: float = Field(ge=SCALE[0], le=SCALE[1])

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, s) for s in SKILLS], dtype=float)


class Contribution(BaseModel):
    skill: str
    label: str
    score: float
    cohort_mean: float
    contribution_logodds: float  # + pushes towards a high hike


class Improvement(BaseModel):
    skill: str
    label: str
    current: float
    target: float
    probability_after: float
    gain: float


class Prediction(BaseModel):
    probability: float
    probability_low: float  # 5th percentile across the bootstrap ensemble
    probability_high: float  # 95th percentile
    baseline_probability: float  # probability for an average junior
    contributions: list[Contribution]
    improvements: list[Improvement]  # raise each skill to the high-hike median, best first
    smallest_change_to_target: Improvement | None
    target_probability: float
    warnings: list[str] = []


@dataclass
class HikeModel:
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    C: float
    boot_coef: np.ndarray
    boot_intercept: np.ndarray
    high_median: dict[str, float]
    train_min: np.ndarray
    train_max: np.ndarray
    n_train: int
    cv_auc: float | None = None
    meta: dict = field(default_factory=dict)

    # ---- training & persistence ------------------------------------------------------------
    @classmethod
    def train(cls, jds: pd.DataFrame, n_boot: int = 300) -> "HikeModel":
        X = jds[SKILLS].to_numpy(dtype=float)
        y = jds[TARGET].to_numpy(dtype=int)
        mean, std = X.mean(axis=0), X.std(axis=0, ddof=0)
        Z = (X - mean) / std
        cv = LogisticRegressionCV(Cs=np.logspace(-3, 2, 20), cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                  scoring="neg_log_loss", max_iter=5000).fit(Z, y)
        C = float(cv.C_[0])
        full = LogisticRegression(C=C, max_iter=5000).fit(Z, y)
        rng = np.random.default_rng(SEED)
        bc, bi = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            m = LogisticRegression(C=C, max_iter=5000).fit(Z[idx], y[idx])
            bc.append(m.coef_[0])
            bi.append(m.intercept_[0])
        hi = jds[jds[TARGET] == 1]
        return cls(mean=mean, std=std, coef=full.coef_[0], intercept=float(full.intercept_[0]), C=C,
                   boot_coef=np.array(bc), boot_intercept=np.array(bi),
                   high_median={s: float(hi[s].median()) for s in SKILLS},
                   train_min=X.min(axis=0), train_max=X.max(axis=0), n_train=len(y))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "HikeModel":
        obj = pickle.loads(path.read_bytes())
        if not isinstance(obj, HikeModel):
            raise TypeError(f"{path} is not a HikeModel artifact")
        return obj

    # ---- inference ------------------------------------------------------------------------------
    def _z(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def _p(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self._z(X) @ self.coef + self.intercept)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Vectorised probability for an (n, 5) array in SKILLS order."""
        return self._p(np.atleast_2d(np.asarray(X, dtype=float)))

    def _interval(self, x: np.ndarray) -> tuple[float, float]:
        logits = self._z(x[None, :]) @ self.boot_coef.T + self.boot_intercept
        ps = 1 / (1 + np.exp(-logits.ravel()))
        return float(np.percentile(ps, 5)), float(np.percentile(ps, 95))

    def explain(self, scores: SkillScores, target_probability: float = 0.7) -> Prediction:
        x = scores.vector()
        warnings = []
        lo, hi = self.train_min, self.train_max
        outside = [LABELS[s] for s, v, a, b in zip(SKILLS, x, lo, hi, strict=True) if v < a or v > b]
        if outside:
            warnings.append(f"outside the training range for {', '.join(outside)}: values clipped")
            x = np.clip(x, lo, hi)
        p = float(self._p(x[None, :])[0])
        p_lo, p_hi = self._interval(x)
        base = float(1 / (1 + np.exp(-self.intercept)))  # all skills at the cohort mean
        z = self._z(x)
        contribs = [Contribution(skill=s, label=LABELS[s], score=float(x[i]), cohort_mean=float(self.mean[i]),
                                 contribution_logodds=float(self.coef[i] * z[i])) for i, s in enumerate(SKILLS)]
        contribs.sort(key=lambda c: c.contribution_logodds)

        improvements = []
        for i, s in enumerate(SKILLS):
            tgt = max(self.high_median[s], x[i])
            if tgt <= x[i]:
                continue
            x2 = x.copy()
            x2[i] = tgt
            p2 = float(self._p(x2[None, :])[0])
            improvements.append(Improvement(skill=s, label=LABELS[s], current=float(x[i]), target=float(tgt),
                                            probability_after=p2, gain=p2 - p))
        improvements.sort(key=lambda m: -m.gain)

        smallest = None
        if p < target_probability:
            best = None
            for i, s in enumerate(SKILLS):
                if self.coef[i] <= 0:
                    continue
                for v in np.round(np.arange(x[i] + 0.1, SCALE[1] + 1e-9, 0.1), 1):
                    x2 = x.copy()
                    x2[i] = v
                    p2 = float(self._p(x2[None, :])[0])
                    if p2 >= target_probability:
                        step = v - x[i]
                        if best is None or step < best[0]:
                            best = (step, Improvement(skill=s, label=LABELS[s], current=float(x[i]), target=float(v),
                                                      probability_after=p2, gain=p2 - p))
                        break
            smallest = best[1] if best else None
            if smallest is None:
                warnings.append(f"no single-skill change reaches {target_probability:.0%}; improve several skills")
        return Prediction(probability=p, probability_low=p_lo, probability_high=p_hi, baseline_probability=base,
                          contributions=contribs, improvements=improvements, smallest_change_to_target=smallest,
                          target_probability=target_probability, warnings=warnings)
