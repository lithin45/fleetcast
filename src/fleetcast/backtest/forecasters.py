"""Baseline forecasters behind a common interface.

A ``Forecaster`` takes the full feature panel and a fold and returns predictions
for the fold's test window as ``(zone_id, hour, y_pred)``. It may use only
information available at the origin — the harness extracts ``y_true`` separately,
so a forecaster never sees the test targets it is being scored on.

The same interface is reused by the LightGBM model (Phase 4) and the conformal
wrapper (Phase 5), so every model is evaluated on identical folds.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
import pandas as pd

from ..config import Config
from ..logging import get_logger
from .folds import Fold, forecast_slice, train_slice

log = get_logger(__name__)


class Forecaster(Protocol):
    name: str

    def predict_fold(self, panel: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        """Return columns (zone_id, hour, y_pred) for the fold's test window."""
        ...


class SeasonalNaiveForecaster:
    """Same hour, one week ago. Predicts ``demand[t-168h]`` — which, for any
    horizon <= 168h, is always observed before the origin (no leakage). This is
    the primary baseline the model must beat by >= 20% WAPE."""

    name = "seasonal_naive"

    def predict_fold(self, panel: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        test = forecast_slice(panel, fold)
        pred = test["lag_168h"].copy()
        if pred.isna().any():
            # Rare warm-up fallback: the zone's mean demand over the train window.
            zone_mean = train_slice(panel, fold).groupby("zone_id")["demand"].mean()
            pred = pred.fillna(test["zone_id"].map(zone_mean)).fillna(0.0)
        return pd.DataFrame(
            {
                "zone_id": test["zone_id"].to_numpy(),
                "hour": test["hour"].to_numpy(),
                "y_pred": np.clip(pred.to_numpy(dtype=float), 0.0, None),
            }
        )


def _fit_forecast_arima(task: tuple) -> np.ndarray | None:
    """Module-level worker (picklable, so it parallelizes): fit SARIMAX on one
    zone's demand series and forecast ``steps`` ahead. Returns a non-negative
    forecast array, or ``None`` to signal the caller should fall back."""
    series, order, seasonal_order, maxiter, steps = task
    if series is None:
        return None
    import warnings

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=maxiter)
            yhat = np.clip(np.asarray(res.forecast(steps=steps), dtype=float), 0.0, None)
        return yhat if np.all(np.isfinite(yhat)) else None
    except Exception:  # robustness: never crash the backtest
        return None


class ArimaForecaster:
    """Classical ARIMA baseline (statsmodels SARIMAX, per zone, daily seasonality).

    A **daily-batch** baseline shown for context: the model is refit per zone once
    every ``refit_every_hours`` (a realistic retrain cadence) on a trailing window
    of demand, and that fit forecasts ahead for the rest of the day. Forecasts for
    the whole retrain period are computed once and cached, so the 336 hourly folds
    cost ~14 refits, not 336. Per-zone fits run in a process pool (``n_jobs``); any
    fit failure falls back to seasonal-naive for that zone, so the baseline never
    crashes the backtest. (The headline gate compares the one-step-ahead LightGBM
    to the one-step-valid seasonal-naive; ARIMA is the classical reference point.)
    """

    name = "arima"

    def __init__(self, cfg: Config):
        arima = cfg.baselines.get("arima", {})
        self.order = tuple(arima.get("order", [1, 1, 1]))
        self.seasonal_order = tuple(arima.get("seasonal_order", [1, 0, 1, 24]))
        self.train_window = int(arima.get("train_window_hours", 504))
        self.maxiter = int(arima.get("maxiter", 50))
        self.n_jobs = int(arima.get("n_jobs", -1))
        self._refit_every = int(cfg.backtest.get("refit_every_hours", 24))
        self._seasonal_naive = SeasonalNaiveForecaster()
        self._executor = None  # one pool reused across folds (workers import once)
        self._fit_origin = None  # last retrain origin
        self._cache: dict[tuple[int, pd.Timestamp], float] = {}  # (zone, hour) -> forecast

    def _workers(self, n_zones: int) -> int:
        n = self.n_jobs
        if n < 0:
            n = max(1, (os.cpu_count() or 2) - 1)
        # A process pool only pays off for enough independent fits.
        return n if (n > 1 and n_zones > 4) else 1

    def _run(self, tasks: list[tuple], n_zones: int) -> list[np.ndarray | None]:
        workers = self._workers(n_zones)
        if workers <= 1:
            return [_fit_forecast_arima(t) for t in tasks]
        try:
            if self._executor is None:
                from concurrent.futures import ProcessPoolExecutor

                # Pin each worker to single-threaded BLAS BEFORE the pool is spawned
                # (children inherit the env). Otherwise N workers x multi-threaded
                # numpy/scipy oversubscribe the cores and the wall-time barely drops.
                for var in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    os.environ.setdefault(var, "1")
                # Reused across all folds so each worker imports statsmodels ONCE
                # (per-fold pools re-import every time and dominate the runtime).
                self._executor = ProcessPoolExecutor(max_workers=workers)
            return list(self._executor.map(_fit_forecast_arima, tasks))
        except Exception as exc:  # broken pool -> drop it and run serially
            log.warning("ARIMA parallel pool failed (%s); running serially", exc)
            self.close()
            return [_fit_forecast_arima(t) for t in tasks]

    def close(self) -> None:
        """Shut down the worker pool (call after the backtest run)."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _refit_period(self, panel: pd.DataFrame, fold: Fold) -> None:
        """Refit per zone on data < origin and cache forecasts for the whole period."""
        train = train_slice(panel, fold)
        min_obs = max(2 * self.seasonal_order[3], 48)
        steps = self._refit_every
        zones = sorted(int(z) for z in train["zone_id"].unique())

        tasks: list[tuple] = []
        for z in zones:
            series = (
                train[train["zone_id"] == z].sort_values("hour")["demand"].astype(float).to_numpy()
            )
            if len(series) > self.train_window:
                series = series[-self.train_window :]
            eligible = series if len(series) >= min_obs else None
            tasks.append((eligible, self.order, self.seasonal_order, self.maxiter, steps))

        results = self._run(tasks, n_zones=len(zones))
        self._cache = {}
        for z, yhat in zip(zones, results, strict=True):
            if yhat is None:
                continue
            for step, val in enumerate(yhat):
                self._cache[(z, fold.test_start + pd.Timedelta(hours=step))] = float(val)

    def predict_fold(self, panel: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        if self._fit_origin is None or (fold.train_end - self._fit_origin) >= pd.Timedelta(
            hours=self._refit_every
        ):
            self._refit_period(panel, fold)
            self._fit_origin = fold.train_end

        test = forecast_slice(panel, fold)
        fallback = self._seasonal_naive.predict_fold(panel, fold).set_index(["zone_id", "hour"])
        out_zone, out_hour, out_pred = [], [], []
        for r in test.itertuples(index=False):
            z, h = int(r.zone_id), r.hour
            pred = self._cache.get((z, h))
            if pred is None:  # zone fit failed or beyond cache -> seasonal-naive
                pred = float(fallback.loc[(z, h), "y_pred"])
            out_zone.append(z)
            out_hour.append(h)
            out_pred.append(pred)
        return pd.DataFrame({"zone_id": out_zone, "hour": out_hour, "y_pred": out_pred})


def default_baselines(cfg: Config) -> list[Forecaster]:
    """The Phase-3 baseline forecasters."""
    return [SeasonalNaiveForecaster(), ArimaForecaster(cfg)]
