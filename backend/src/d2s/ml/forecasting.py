"""Skill-demand forecasting with Nixtla StatsForecast.

Monthly skill-mention counts are often sparse, so each series is routed by its demand
pattern (Syntetos-Boylan classification) to a suitable model family:

    smooth / erratic       -> AutoETS, AutoARIMA, SeasonalNaive  (native prediction intervals)
    intermittent / lumpy   -> CrostonOptimized, CrostonSBA, IMAPA, ADIDA  (+ conformal intervals)
    short history          -> HistoricAverage, Naive
    all zeros              -> zero forecast

Within a family the model with the lowest rolling-origin backtest MAE wins, per series.
Input/Output use the Nixtla long format: unique_id, ds, y.
"""

from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel

ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


class Pattern(StrEnum):
    SMOOTH = "smooth"
    ERRATIC = "erratic"
    INTERMITTENT = "intermittent"
    LUMPY = "lumpy"
    SHORT = "short"
    ZERO = "zero"


class ForecastPoint(BaseModel):
    ds: pd.Timestamp
    point: float
    lo: float
    hi: float

    model_config = {"arbitrary_types_allowed": True}


class SeriesForecast(BaseModel):
    unique_id: str
    pattern: Pattern
    model: str
    backtest_mae: float | None
    points: list[ForecastPoint]
    candidates_mae: dict[str, float] = {}  # every candidate's backtest MAE (for the explanation leaderboard)
    backtest_windows: int | None = None

    def total(self, which: str = "point") -> float:
        return float(sum(getattr(p, which) for p in self.points))


def classify(y: np.ndarray, min_history: int) -> Pattern:
    y = np.asarray(y, dtype=float)
    nonzero = y[y > 0]
    if len(nonzero) == 0:
        return Pattern.ZERO
    if len(y) < min_history:
        return Pattern.SHORT
    adi = len(y) / len(nonzero)
    cv2 = (nonzero.std() / nonzero.mean()) ** 2 if len(nonzero) > 1 else 0.0
    if adi < ADI_CUTOFF:
        return Pattern.SMOOTH if cv2 < CV2_CUTOFF else Pattern.ERRATIC
    return Pattern.INTERMITTENT if cv2 < CV2_CUTOFF else Pattern.LUMPY


class DemandForecaster:
    def __init__(
        self,
        horizon: int = 6,
        freq: str = "MS",
        level: int = 80,
        season_length: int = 12,
        backtest_windows: int = 2,
        min_history: int | None = None,
        n_jobs: int = -1,
        max_backtest_windows: int = 4,
    ):
        self.h = horizon
        self.freq = freq
        self.level = level
        self.season_length = season_length
        self.windows = backtest_windows
        self.min_history = min_history or max(horizon * (backtest_windows + 1), 12)
        self.n_jobs = n_jobs
        self.max_windows = max_backtest_windows

    # ---- model families -------------------------------------------------------------
    def _models(self, pattern: Pattern, min_len: int) -> tuple[list, bool]:
        """Return (models, has_native_intervals)."""
        from statsforecast.models import (
            ADIDA,
            IMAPA,
            AutoARIMA,
            AutoETS,
            CrostonOptimized,
            CrostonSBA,
            HistoricAverage,
            Naive,
            SeasonalNaive,
        )
        from statsforecast.utils import ConformalIntervals

        seasonal = min_len >= 2 * self.season_length
        m = self.season_length if seasonal else 1
        if pattern in (Pattern.SMOOTH, Pattern.ERRATIC):
            models = [AutoETS(season_length=m), AutoARIMA(season_length=m)]
            if seasonal:
                models.append(SeasonalNaive(season_length=m))
            return models, True
        if pattern in (Pattern.INTERMITTENT, Pattern.LUMPY):
            ci = ConformalIntervals(h=self.h, n_windows=self._windows(min_len))
            return [
                CrostonOptimized(prediction_intervals=ci),
                CrostonSBA(prediction_intervals=ci),
                IMAPA(prediction_intervals=ci),
                ADIDA(prediction_intervals=ci),
            ], True
        return [HistoricAverage(), Naive()], True

    # ---- public API -----------------------------------------------------------------
    def fit_predict(self, df: pd.DataFrame) -> list[SeriesForecast]:
        """df: columns unique_id, ds (datetime), y (non-negative counts)."""
        df = self._prepare(df)
        lengths = df.groupby("unique_id").size()
        patterns = {
            uid: classify(g["y"].to_numpy(), self.min_history) for uid, g in df.groupby("unique_id")
        }
        results: list[SeriesForecast] = []
        for pattern in Pattern:
            ids = [uid for uid, p in patterns.items() if p is pattern]
            if not ids:
                continue
            part = df[df["unique_id"].isin(ids)]
            if pattern is Pattern.ZERO:
                results.extend(self._zeros(part))
            else:
                results.extend(self._fit_family(part, pattern, int(lengths[ids].min())))
        return sorted(results, key=lambda r: r.unique_id)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = {"unique_id", "ds", "y"} - set(df.columns)
        if missing:
            raise ValueError(f"forecast input missing columns {sorted(missing)}")
        df = df[["unique_id", "ds", "y"]].copy()
        df["unique_id"] = df["unique_id"].astype(str)
        df["ds"] = snap_to_freq(pd.to_datetime(df["ds"]), self.freq)
        df["y"] = df["y"].astype(float).clip(lower=0)
        # Fill gaps with zeros: a month with no postings is zero demand, not missing data.
        filled = []
        for uid, g in df.groupby("unique_id"):
            g = g.groupby("ds", as_index=False)["y"].sum()
            idx = pd.date_range(g["ds"].min(), g["ds"].max(), freq=self.freq)
            g = g.set_index("ds").reindex(idx, fill_value=0.0).rename_axis("ds").reset_index()
            g["unique_id"] = uid
            filled.append(g)
        return pd.concat(filled, ignore_index=True)[["unique_id", "ds", "y"]]

    def _zeros(self, part: pd.DataFrame) -> list[SeriesForecast]:
        out = []
        for uid, g in part.groupby("unique_id"):
            future = pd.date_range(g["ds"].max(), periods=self.h + 1, freq=self.freq)[1:]
            out.append(SeriesForecast(
                unique_id=uid, pattern=Pattern.ZERO, model="Zero", backtest_mae=None,
                points=[ForecastPoint(ds=d, point=0.0, lo=0.0, hi=0.0) for d in future],
            ))
        return out

    def _windows(self, min_len: int) -> int:
        """More backtest windows when history allows: 2 windows proved too few to pick a model
        reliably (AirPassengers: SeasonalNaive won on 2 windows, AutoARIMA on 4 and on the test year)."""
        min_train = max(2 * self.season_length, self.h)
        possible = max(0, (min_len - min_train) // self.h)
        return int(min(self.max_windows, max(self.windows, possible)))

    def _fit_family(self, part: pd.DataFrame, pattern: Pattern, min_len: int) -> list[SeriesForecast]:
        from statsforecast import StatsForecast

        models, _ = self._models(pattern, min_len)
        # process pools only pay off for large batches (60 series: 1.5 s single-core vs 19 s pooled)
        n_series = part["unique_id"].nunique()
        sf = StatsForecast(models=models, freq=self.freq, n_jobs=self.n_jobs if n_series >= 100 else 1)
        names = [repr(m) for m in models]
        combine = pattern in (Pattern.SMOOTH, Pattern.ERRATIC) and {"AutoETS", "AutoARIMA"} <= set(names)

        best_model, maes, all_scores = self._select(sf, part, names, pattern, min_len, combine)
        fc = sf.forecast(df=part, h=self.h, level=[self.level])
        if combine:
            fc = _add_combination(fc, self.level)

        # For intermittent series, if >10% of past periods were zero then the lower 10th percentile
        # of next-period demand is 0. Conformal bounds sat above 0 and missed every zero month
        # (coverage 59-68% for a nominal 80%); flooring at 0 restored 80% on the holdout test.
        zero_floor = set()
        if pattern in (Pattern.INTERMITTENT, Pattern.LUMPY):
            zs = part.groupby("unique_id")["y"].apply(lambda v: float((v == 0).mean()))
            zero_floor = set(zs[zs > (100 - self.level) / 200].index)
        out = []
        for uid, g in fc.groupby("unique_id"):
            name = best_model.get(uid, names[0])
            lo_col, hi_col = f"{name}-lo-{self.level}", f"{name}-hi-{self.level}"
            pts = []
            for r in g.to_dict("records"):
                point = max(0.0, float(r[name]))
                lo = max(0.0, float(r[lo_col])) if lo_col in r and pd.notna(r[lo_col]) else point
                hi = max(point, float(r[hi_col])) if hi_col in r and pd.notna(r[hi_col]) else point
                if uid in zero_floor:
                    lo = 0.0
                pts.append(ForecastPoint(ds=r["ds"], point=point, lo=min(lo, point), hi=hi))
            out.append(SeriesForecast(unique_id=str(uid), pattern=pattern, model=name,
                                      backtest_mae=maes.get(uid), points=pts,
                                      candidates_mae=all_scores.get(str(uid), {}),
                                      backtest_windows=None if pattern is Pattern.SHORT else self._windows(min_len)))
        return out

    def _select(
        self, sf, part: pd.DataFrame, names: list[str], pattern: Pattern, min_len: int, combine: bool
    ) -> tuple[dict[str, str], dict[str, float], dict[str, dict[str, float]]]:
        if pattern is Pattern.SHORT:
            return {}, {}, {}
        cv = sf.cross_validation(df=part, h=self.h, n_windows=self._windows(min_len), step_size=self.h)
        if combine:
            cv = _add_combination(cv, None)
            names = [*names, COMBO]
        best, maes, every = {}, {}, {}
        for uid, g in cv.groupby("unique_id"):
            scores = {n: float(np.mean(np.abs(g[n] - g["y"]))) for n in names if n in g}
            scores = {n: s for n, s in scores.items() if np.isfinite(s)}
            if scores:
                name = min(scores, key=scores.get)
                best[str(uid)], maes[str(uid)] = name, scores[name]
                every[str(uid)] = scores
        return best, maes, every


COMBO = "Combination(AutoETS,AutoARIMA)"


def _add_combination(frame: pd.DataFrame, level: int | None) -> pd.DataFrame:
    """Equal-weight forecast combination of AutoETS and AutoARIMA (a robust default in the
    forecasting literature; best on the AirPassengers test year). Interval bounds are the mean of
    the two models' bounds, an approximation stated in the model card."""
    frame = frame.copy()
    frame[COMBO] = (frame["AutoETS"] + frame["AutoARIMA"]) / 2
    if level is not None:
        for side in ("lo", "hi"):
            a, b = f"AutoETS-{side}-{level}", f"AutoARIMA-{side}-{level}"
            if a in frame and b in frame:
                frame[f"{COMBO}-{side}-{level}"] = (frame[a] + frame[b]) / 2
    return frame


def snap_to_freq(ds: pd.Series, freq: str) -> pd.Series:
    """Move every date onto the frequency's anchor (e.g. 2024-01-15 -> 2024-01-01 for 'MS',
    -> 2024-01-31 for 'ME'). Without this, gap-filling with date_range silently dropped
    observations whose dates were off-anchor (found by the messy-input test)."""
    from pandas.tseries.frequencies import to_offset

    off = to_offset(freq)
    code = off.name
    base = code[:-1] if len(code) > 1 and code[-1] in "ES" and code[:-1] in {"M", "Q", "Y", "BM", "BQ", "BY"} else code
    periods = ds.dt.to_period(base)
    if code.endswith("E"):
        return periods.dt.to_timestamp(how="end").dt.normalize()
    return periods.dt.to_timestamp(how="start")


def demand_weights(forecasts: list[SeriesForecast], which: str = "point") -> dict[str, float]:
    """Total forecast demand over the horizon per series; 'lo' gives the pessimistic band."""
    if which not in ("point", "lo", "hi"):
        raise ValueError("which must be point, lo or hi")
    return {f.unique_id: f.total(which) for f in forecasts}
