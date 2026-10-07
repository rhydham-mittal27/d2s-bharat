import numpy as np
import pandas as pd

from d2s.analysis import traits


def _synthetic(n=150, seed=0) -> traits.Dataset:
    """Outcome driven by f1 only; f2 is noise."""
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    y = (f1 + rng.normal(scale=0.7, size=n) > 0).astype(int)
    df = pd.DataFrame({"id": np.arange(n), "f1": f1, "f2": f2, "y": y})
    return traits.Dataset("synthetic", df, ["f1", "f2"], "y", None)


def test_univariate_detects_true_signal_and_not_noise():
    uni = traits.univariate(_synthetic(), n_boot=300).set_index("feature")
    assert uni.loc["f1", "significant_holm_0.05"]
    assert not uni.loc["f2", "significant_holm_0.05"]
    assert uni.loc["f1", "rb_ci_low"] > 0


def test_models_beat_baseline_and_odds_ratio_direction():
    ds = _synthetic()
    comp, _ = traits.model_comparison(ds, repeats=3)
    comp = comp.set_index("model")
    assert comp.loc["L2 logistic regression", "roc_auc_mean"] > 0.8
    assert abs(comp.loc["majority baseline", "roc_auc_mean"] - 0.5) < 1e-9
    orat = traits.odds_ratios(ds, n_boot=200).set_index("feature")
    assert orat.loc["f1", "or_ci_low"] > 1  # real effect
    assert orat.loc["f2", "or_ci_low"] < 1 < orat.loc["f2", "or_ci_high"]  # noise CI spans 1


def test_audit_flags_duplicate_ids():
    ds = _synthetic(n=20)
    ds.frame.loc[1, "id"] = 0
    a = traits.audit(ds)
    assert a["duplicate_ids"] == 1 and a["duplicate_id_rows"] == 2
