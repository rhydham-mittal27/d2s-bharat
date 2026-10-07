"""Forecasting under realistic conditions (the SAS data has no dates, so we use real and
realistic series to validate the module itself, not to produce project results).

* Real series : AirPassengers (monthly, trend + seasonality), shipped with StatsForecast.
* Intermittent: 60 Croston-type series (random arrivals, random sizes), known generating process.
* Messy input : duplicate dates, gaps, negatives, string ids, end-of-month dates, many series.
"""

import time

import numpy as np
import pandas as pd
import pytest

from d2s.ml.forecasting import DemandForecaster, Pattern


def _mase(y, f, insample, m=12):
    scale = np.mean(np.abs(insample[m:] - insample[:-m]))
    return float(np.mean(np.abs(y - f)) / scale)


def test_air_passengers_beats_seasonal_naive_and_intervals_cover():
    from statsforecast.utils import AirPassengersDF

    df = AirPassengersDF.copy()
    df["unique_id"] = "air"
    h = 12
    train, test = df.iloc[:-h], df.iloc[-h:]
    fc = DemandForecaster(horizon=h, freq="ME", level=80, backtest_windows=2)
    out = fc.fit_predict(train)[0]
    assert out.pattern in (Pattern.SMOOTH, Pattern.ERRATIC)
    point = np.array([p.point for p in out.points])
    lo, hi = np.array([p.lo for p in out.points]), np.array([p.hi for p in out.points])
    y = test["y"].to_numpy()
    snaive = train["y"].to_numpy()[-12:]  # seasonal naive forecast
    ins = train["y"].to_numpy()
    assert _mase(y, point, ins) < _mase(y, snaive, ins), "selected model should beat seasonal naive"
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    assert coverage >= 0.5, f"80% interval covered only {coverage:.0%} of 12 months"


def test_intermittent_series_routed_and_conformal_coverage():
    rng = np.random.default_rng(7)
    frames, futures = [], {}
    n_hist, h = 48, 6
    for i in range(60):
        demand = np.where(rng.random(n_hist + h) < 0.3, rng.poisson(4, n_hist + h) + 1, 0)
        ds = pd.date_range("2021-01-01", periods=n_hist + h, freq="MS")
        frames.append(pd.DataFrame({"unique_id": f"s{i}", "ds": ds[:n_hist], "y": demand[:n_hist]}))
        futures[f"s{i}"] = demand[n_hist:]
    fc = DemandForecaster(horizon=h, level=80, backtest_windows=2)
    out = fc.fit_predict(pd.concat(frames))
    assert sum(r.pattern in (Pattern.INTERMITTENT, Pattern.LUMPY) for r in out) >= 50
    # Croston-type point forecasts estimate the mean demand rate: compare totals, not single months
    rate_err = np.mean([abs(r.total() / h - futures[r.unique_id].mean()) for r in out])
    assert rate_err < 1.5
    inside = [p.lo <= a <= p.hi for r in out for p, a in zip(r.points, futures[r.unique_id], strict=True)]
    assert np.mean(inside) >= 0.72, f"80% intervals covered only {np.mean(inside):.0%}"
    for r in out:
        assert all(0 <= p.lo <= p.point <= p.hi for p in r.points)


def test_messy_input_is_handled():
    df = pd.DataFrame({
        "unique_id": [1, 1, 1, 1, 2, 2, "3", "3"],
        "ds": ["2024-01-31", "2024-01-31", "2024-03-31", "2024-04-30", "2024-01-01", "2024-02-01",
               "2024-01-01", "2024-01-01"],
        "y": [5, 3, -2, 7, 0, 0, 4, 1],
    })
    fc = DemandForecaster(horizon=3, freq="ME")
    prepared = fc._prepare(df)
    one = prepared[prepared["unique_id"] == "1"]["y"].tolist()
    assert one == [8.0, 0.0, 0.0, 7.0]  # duplicates summed, gap filled, negative clipped to 0
    assert set(prepared["unique_id"]) == {"1", "2", "3"}


@pytest.mark.slow
def test_scales_to_many_series():
    rng = np.random.default_rng(1)
    n = 500
    ds = pd.date_range("2022-01-01", periods=30, freq="MS")
    df = pd.DataFrame({"unique_id": np.repeat([f"k{i}" for i in range(n)], 30), "ds": np.tile(ds, n),
                       "y": rng.poisson(10, n * 30)})
    t = time.perf_counter()
    out = DemandForecaster(horizon=6).fit_predict(df)
    assert len(out) == n
    assert time.perf_counter() - t < 300
