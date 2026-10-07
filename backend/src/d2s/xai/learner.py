"""Explaining the junior salary-hike model.

* why      : exact Shapley values. For a linear model on the log-odds scale with the cohort mean as
             the reference, the Shapley value of a skill is coef_k x (x_k - mean_k) / sd_k; they sum
             exactly to the prediction's log-odds minus the reference log-odds. No sampling needed.
* what-if  : (a) the probability curve for each skill with the others fixed;
             (b) the cheapest combined change to reach a target probability. The log-odds is linear,
             so raising skills in order of log-odds gained per scale point, each up to the scale max,
             minimises the total points needed (a fractional-knapsack argument). Steps of 0.1.
* global   : coefficients per SD and per scale point, with bootstrap 90% intervals and sign stability.
"""

import numpy as np

from d2s.models.hike import LABELS, SCALE, SKILLS, HikeModel, SkillScores
from d2s.xai.schemas import Explanation, Factor


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def why(model: HikeModel, scores: SkillScores) -> Explanation:
    pred = model.explain(scores)
    base = pred.baseline_probability
    factors = [Factor(name=c.label, value=c.score, contribution=round(c.contribution_logodds, 4), unit="log-odds",
                      source="logit.shapley", note=f"cohort mean {c.cohort_mean:.2f}")
               for c in sorted(pred.contributions, key=lambda c: -abs(c.contribution_logodds))]
    up = [f for f in factors if (f.contribution or 0) > 0.05]
    down = [f for f in factors if (f.contribution or 0) < -0.05]
    parts = [f"Estimated chance of a high salary hike: {pred.probability:.0%} "
             f"(range {pred.probability_low:.0%}-{pred.probability_high:.0%}), "
             f"against {base:.0%} for an average junior."]
    if down:
        parts.append("Held back most by " + ", ".join(f"{f.name} ({f.value:.1f})" for f in down[:2]) + ".")
    if up:
        parts.append("Helped most by " + ", ".join(f"{f.name} ({f.value:.1f})" for f in up[:2]) + ".")
    total = sum(c.contribution_logodds for c in pred.contributions)  # unrounded: exact identity
    return Explanation(
        subject="learner", question="why", summary=" ".join(parts), factors=factors,
        confidence=round(1 - (pred.probability_high - pred.probability_low), 3),
        method="exact Shapley values of a standardised logistic model (reference = cohort mean); "
               "range = 5th-95th percentile over bootstrap refits",
        details={"probability": pred.probability, "probability_low": pred.probability_low,
                 "probability_high": pred.probability_high, "baseline_probability": base,
                 "sum_of_contributions": total, "logodds_gap": _logit(pred.probability) - _logit(base),
                 "warnings": pred.warnings},
    )


def curves(model: HikeModel, scores: SkillScores, step: float = 0.25) -> dict[str, list[dict]]:
    x = scores.vector()
    grid = np.round(np.arange(SCALE[0], SCALE[1] + 1e-9, step), 2)
    out = {}
    for i, s in enumerate(SKILLS):
        X = np.repeat(x[None, :], len(grid), axis=0)
        X[:, i] = grid
        out[s] = [{"score": float(g), "probability": float(p)} for g, p in zip(grid, model.predict_proba(X), strict=True)]
    return out


def cheapest_path(model: HikeModel, scores: SkillScores, target: float = 0.7, step: float = 0.1) -> Explanation:
    x = scores.vector()
    p0 = float(model.predict_proba(x)[0])
    per_point = model.coef / model.std  # log-odds gained per +1 scale point
    order = [i for i in np.argsort(-per_point) if per_point[i] > 0]
    changes, x2 = [], x.copy()
    if p0 < target:
        need = _logit(target) - _logit(p0)
        for i in order:
            room = SCALE[1] - x2[i]
            if room <= 0 or need <= 0:
                continue
            raise_by = min(room, need / per_point[i])
            raise_by = float(np.ceil(raise_by / step - 1e-9) * step)
            raise_by = min(raise_by, room)
            x2[i] += raise_by
            need -= raise_by * per_point[i]
            changes.append((i, raise_by))
    p1 = float(model.predict_proba(x2)[0])
    reached = p1 >= target
    factors = [Factor(name=LABELS[SKILLS[i]], value=f"{x[i]:.1f} -> {x[i] + d:.1f}", contribution=round(d, 2),
                      unit="scale points", source="logit.greedy-l1",
                      note=f"{per_point[i]:.2f} log-odds per point") for i, d in changes]
    if p0 >= target:
        summary = f"Already at {p0:.0%}, above the {target:.0%} target; no change needed."
    elif reached:
        summary = (f"Cheapest combined change from {p0:.0%} to {p1:.0%} (target {target:.0%}): "
                   + "; ".join(f"{f.name} {f.value}" for f in factors) + f" ({sum(d for _, d in changes):.1f} points in total).")
    else:
        summary = (f"Even raising every helpful skill to {SCALE[1]:g} only reaches {p1:.0%}; "
                   f"the {target:.0%} target is not reachable through these five skills.")
    return Explanation(
        subject="learner", question="what_if", summary=summary, factors=factors,
        counterfactuals=[f"{f.name}: {f.value}" for f in factors],
        method="greedy allocation by log-odds per scale point (optimal for a linear logit with box bounds), 0.1 steps",
        details={"probability_now": p0, "probability_after": p1, "target": target, "reached": reached,
                 "total_points": float(sum(d for _, d in changes)), "curves": curves(model, scores)},
    )


def global_importance(model: HikeModel) -> Explanation:
    lo, hi = np.percentile(model.boot_coef, [5, 95], axis=0)
    stable = (np.sign(model.boot_coef) == np.sign(model.coef)).mean(axis=0)
    factors = []
    for i in np.argsort(-np.abs(model.coef)):
        s = SKILLS[i]
        factors.append(Factor(name=LABELS[s], value=round(float(model.coef[i]), 4), contribution=round(float(np.exp(model.coef[i])), 3),
                              unit="odds ratio per +1 SD", source="logit.coefficients",
                              note=f"90% CI [{lo[i]:.2f}, {hi[i]:.2f}] log-odds; "
                                   f"{model.coef[i] / model.std[i]:.2f} log-odds per scale point; "
                                   f"same sign in {stable[i]:.0%} of bootstrap refits"))
    top = factors[:2]
    return Explanation(
        subject="hike_model", question="why",
        summary=f"Across all juniors, {top[0].name} and {top[1].name} matter most: one standard deviation more "
                f"multiplies the odds of a high hike by {top[0].contribution:.2f} and {top[1].contribution:.2f}. "
                f"All five skills push in the same (positive) direction.",
        factors=factors, method=f"standardised L2 logistic regression (C = {model.C:.2f}); bootstrap n = {len(model.boot_coef)}",
        details={"cv_auc": model.cv_auc, "n_train": model.n_train},
    )
