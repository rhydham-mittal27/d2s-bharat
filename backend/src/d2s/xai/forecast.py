"""Explaining a demand forecast: why this model family, why this model, how sure.

(Forecasting is dormant on the SAS data, which has no dates; this explains forecasts for any dated
series the module is given.)
"""

import numpy as np

from d2s.ml.forecasting import ADI_CUTOFF, CV2_CUTOFF, Pattern, SeriesForecast
from d2s.xai.schemas import Explanation, Factor

FAMILY = {
    Pattern.SMOOTH: "regular demand, so trend/seasonal models (ETS, ARIMA) were compared",
    Pattern.ERRATIC: "regular timing but variable size, so trend/seasonal models were compared",
    Pattern.INTERMITTENT: "sparse demand with steady sizes, so intermittent-demand models (Croston family) were compared",
    Pattern.LUMPY: "sparse and variable demand, so intermittent-demand models were compared",
    Pattern.SHORT: "too little history to backtest, so a simple average/naive forecast is used",
    Pattern.ZERO: "no demand in the history, so the forecast is zero",
}


def explain(forecast: SeriesForecast, history: np.ndarray) -> Explanation:
    y = np.asarray(history, dtype=float)
    nz = y[y > 0]
    adi = len(y) / len(nz) if len(nz) else float("inf")
    cv2 = float((nz.std() / nz.mean()) ** 2) if len(nz) > 1 else 0.0
    board = sorted(forecast.candidates_mae.items(), key=lambda kv: kv[1])
    factors = [Factor(name="ADI (avg periods between demands)", value=round(adi, 2), source="pattern",
                      note=f"cut-off {ADI_CUTOFF}"),
               Factor(name="CV² of non-zero sizes", value=round(cv2, 3), source="pattern", note=f"cut-off {CV2_CUTOFF}")]
    factors += [Factor(name=m, value=round(e, 3), unit="backtest MAE", source="backtest",
                       note="selected" if m == forecast.model else "") for m, e in board]
    width = np.mean([p.hi - p.lo for p in forecast.points]) if forecast.points else 0.0
    level = np.mean([p.point for p in forecast.points]) if forecast.points else 0.0
    lead = ""
    if len(board) > 1:
        lead = (f" It beat the runner-up {board[1][0]} by {board[1][1] - board[0][1]:.2f} MAE "
                f"over {forecast.backtest_windows} rolling backtest windows.")
    summary = (f"Pattern '{forecast.pattern.value}' (ADI {adi:.2f}, CV² {cv2:.2f}): {FAMILY[forecast.pattern]}. "
               f"Selected {forecast.model}.{lead} The 80% interval is on average {width:.1f} wide around a "
               f"mean forecast of {level:.1f}.")
    return Explanation(subject=f"series:{forecast.unique_id}", question="why", summary=summary, factors=factors,
                       method="Syntetos-Boylan classification + rolling-origin backtest leaderboard",
                       details={"adi": adi, "cv2": cv2, "leaderboard": board, "pattern": forecast.pattern.value})
