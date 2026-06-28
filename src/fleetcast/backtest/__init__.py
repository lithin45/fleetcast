"""FleetCast backtest package — rolling-origin folds, metrics, baselines, harness."""

from __future__ import annotations

from .folds import Fold, forecast_slice, make_folds, train_slice
from .forecasters import ArimaForecaster, SeasonalNaiveForecaster, default_baselines
from .run import BacktestResult, run_backtest, run_baselines

__all__ = [
    "ArimaForecaster",
    "BacktestResult",
    "Fold",
    "SeasonalNaiveForecaster",
    "default_baselines",
    "forecast_slice",
    "make_folds",
    "run_backtest",
    "run_baselines",
    "train_slice",
]
