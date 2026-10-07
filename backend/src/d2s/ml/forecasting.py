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
        n_jobs: int = 1,
    ):
        self.h = horizon
        self.freq = freq
        self.level = level
        self.season_length = season_length
        self.windows = backtest_windows
        self.min_history = min_history or max(horizon * (backtest_windows + 1), 12)
        self.n_jobs = n_jobs

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
            ci = ConformalIntervals(h=self.h, n_windows=self.windows)
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
        df["ds"] = pd.to_datetime(df["ds"])
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

    def _fit_family(self, part: pd.DataFrame, pattern: Pattern, min_len: int) -> list[SeriesForecast]:
        from statsforecast import StatsForecast

        models, _ = self._models(pattern, min_len)
        sf = StatsForecast(models=models, freq=self.freq, n_jobs=self.n_jobs)
        names = [repr(m) for m in models]

        best_model, maes = self._select(sf, part, names, pattern)
        fc = sf.forecast(df=part, h=self.h, level=[self.level])

        out = []
        for uid, g in fc.groupby("unique_id"):
            name = best_model.get(uid, names[0])
            lo_col, hi_col = f"{name}-lo-{self.level}", f"{name}-hi-{self.level}"
            pts = []
            for r in g.to_dict("records"):
                point = max(0.0, float(r[name]))
                lo = max(0.0, float(r[lo_col])) if lo_col in r and pd.notna(r[lo_col]) else point
                hi = max(point, float(r[hi_col])) if hi_col in r and pd.notna(r[hi_col]) else point
                pts.append(ForecastPoint(ds=r["ds"], point=point, lo=min(lo, point), hi=hi))
            out.append(SeriesForecast(unique_id=str(uid), pattern=pattern, model=name,
                                      backtest_mae=maes.get(uid), points=pts))
        return out

    def _select(
        self, sf, part: pd.DataFrame, names: list[str], pattern: Pattern
    ) -> tuple[dict[str, str], dict[str, float]]:
        if pattern is Pattern.SHORT:
            return {}, {}
        cv = sf.cross_validation(df=part, h=self.h, n_windows=self.windows, step_size=self.h)
        best, maes = {}, {}
        for uid, g in cv.groupby("unique_id"):
            scores = {n: float(np.mean(np.abs(g[n] - g["y"]))) for n in names if n in g}
            scores = {n: s for n, s in scores.items() if np.isfinite(s)}
            if scores:
                name = min(scores, key=scores.get)
                best[str(uid)], maes[str(uid)] = name, scores[name]
        return best, maes


def demand_weights(forecasts: list[SeriesForecast], which: str = "point") -> dict[str, float]:
    """Total forecast demand over the horizon per series; 'lo' gives the pessimistic band."""
    if which not in ("point", "lo", "hi"):
        raise ValueError("which must be point, lo or hi")
    return {f.unique_id: f.total(which) for f in forecasts}
