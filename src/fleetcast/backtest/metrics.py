"""Forecast accuracy metrics.

**WAPE leads** — it is the right metric for intermittent, zero-heavy demand
(MAPE explodes when actuals are near zero). MAE/RMSE are reported everywhere.
MAPE is computed everywhere for completeness (over y>0 rows) but is only
*meaningful* on high-volume zones, so the eval gate reads ``mape_high_volume``.

    WAPE = sum|y - yhat| / sum|y|
    MAE  = mean|y - yhat|
    RMSE = sqrt(mean((y - yhat)^2))
    MAPE = mean(|y - yhat| / y)   over rows with y > 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y)))
    return float(np.sum(np.abs(y - yhat)) / denom) if denom > _EPS else float("nan")


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """MAPE over rows with strictly positive actuals (else undefined)."""
    mask = y > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])))


def core_metrics(
    df: pd.DataFrame, y_col: str = "y_true", yhat_col: str = "y_pred"
) -> dict[str, float]:
    """WAPE/MAE/RMSE/MAPE for one (model[, segment]) slice of predictions."""
    y = df[y_col].to_numpy(dtype=float)
    yhat = df[yhat_col].to_numpy(dtype=float)
    return {
        "wape": wape(y, yhat),
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "mape": mape(y, yhat),
        "n": len(df),
    }


def high_volume_zones(panel: pd.DataFrame, quantile: float) -> list[int]:
    """The top ``(1 - quantile)`` fraction of zones by total demand.

    Rank-based (take the top-k zones) rather than thresholding on the quantile
    value, so ties or odd zone counts can't over-select.
    """
    totals = panel.groupby("zone_id")["demand"].sum().sort_values(ascending=False)
    k = max(1, round((1.0 - quantile) * len(totals)))
    return [int(z) for z in totals.index[:k]]


def summarize_model(
    preds: pd.DataFrame,
    high_vol: list[int],
) -> dict:
    """Pooled, per-fold, and per-zone metrics for one model's predictions."""
    pooled = core_metrics(preds)
    pooled["mape_high_volume"] = mape(
        preds.loc[preds["zone_id"].isin(high_vol), "y_true"].to_numpy(float),
        preds.loc[preds["zone_id"].isin(high_vol), "y_pred"].to_numpy(float),
    )
    per_fold = [
        {"fold": int(f), **{k: core_metrics(g)[k] for k in ("wape", "mae", "rmse")}}
        for f, g in preds.groupby("fold")
    ]
    per_zone = {
        int(z): {k: core_metrics(g)[k] for k in ("wape", "mae", "rmse", "mape")}
        for z, g in preds.groupby("zone_id")
    }
    return {"pooled": pooled, "per_fold": per_fold, "per_zone": per_zone}
