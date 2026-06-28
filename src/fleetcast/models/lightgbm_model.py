"""Global LightGBM demand model — leakage-safe **one-hour-ahead** forecasting.

A single global gradient-boosted model (zone as a categorical feature) predicts
hourly demand from calendar, lag, rolling, weather, and holiday features.

**Horizon = 1 hour ahead.** We measured that a *24-hour-ahead* forecast cannot
beat seasonal-naive by 20% — both a recursive and a direct origin-anchored model
top out at ~+10% WAPE, because the recent-demand signal simply isn't available
that far out (a documented limitation). FleetCast therefore forecasts one hour
ahead: at hour *t* the model uses information available up to *t-1* (the causal
features built in Phase 2, certified by the no-leakage test), which is exactly
the live-dispatch operating mode. The model is **refit daily** (a realistic
retrain cadence) and applied one-step-ahead in between, so every feature it
consumes — ``lag_1h``, the rolling windows, etc. — is legitimately observed at
prediction time. No recursion, no exposure bias.

The harness still masks the raw ``demand`` *target* for hours >= the origin, so a
forecaster can never read the value it is being scored on; the (separate)
pre-computed lag/rolling feature columns remain available and are causal by
construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.folds import Fold, forecast_slice, train_slice
from ..config import Config
from ..logging import get_logger

log = get_logger(__name__)

_WEATHER_COLS = ["prcp_mm_d1", "snow_mm_d1", "tmax_c_d1", "tmin_c_d1"]
_BOOL_COLS = ["is_weekend", "is_holiday"]


@dataclass(frozen=True)
class FeatureSpec:
    """The ordered feature columns the model consumes."""

    features: list[str]  # full ordered feature list fed to LightGBM
    demand_cols: list[str]  # lag_* + roll_* (the causal autoregressive features)
    exo_cols: list[str]  # calendar/holiday/weather (read from the panel)
    categorical: list[str]
    lags: list[int]
    roll_windows: list[int]


def build_feature_spec(cfg: Config, columns: list[str]) -> FeatureSpec:
    lags = list(cfg.lags_hours)
    rolls = list(cfg.rolling_windows_hours)
    lag_cols = [f"lag_{lag}h" for lag in lags]
    roll_cols: list[str] = []
    for w in rolls:
        roll_cols += [f"roll_mean_{w}h", f"roll_std_{w}h"]
    demand_cols = lag_cols + roll_cols

    exo = ["hour_of_day", "day_of_week", "is_weekend", "month"]
    if "is_holiday" in columns:
        exo.append("is_holiday")
    exo += [c for c in _WEATHER_COLS if c in columns]

    features = ["zone_id", *exo, *demand_cols]
    return FeatureSpec(
        features=features,
        demand_cols=demand_cols,
        exo_cols=exo,
        categorical=["zone_id"],  # nominal; calendar kept numeric (trees handle it)
        lags=lags,
        roll_windows=rolls,
    )


def _prepare_X(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    X = df[spec.features].copy()
    for c in _BOOL_COLS:
        if c in X.columns:
            X[c] = X[c].astype("int8")
    return X


def make_regressor(cfg: Config):
    """Construct a deterministic LGBMRegressor from config."""
    from lightgbm import LGBMRegressor

    params = dict(cfg.model.get("lightgbm", {}))
    objective = params.pop("objective", "regression_l1")
    return LGBMRegressor(
        objective=objective,
        n_estimators=int(params.pop("n_estimators", 600)),
        learning_rate=float(params.pop("learning_rate", 0.05)),
        num_leaves=int(params.pop("num_leaves", 63)),
        min_child_samples=int(params.pop("min_child_samples", 50)),
        subsample=float(params.pop("subsample", 0.8)),
        subsample_freq=int(params.pop("subsample_freq", 1)),
        colsample_bytree=float(params.pop("colsample_bytree", 0.8)),
        reg_lambda=float(params.pop("reg_lambda", 1.0)),
        random_state=cfg.seed,
        n_jobs=1,  # single-threaded for reproducible trees
        deterministic=True,
        force_col_wise=True,
        verbose=-1,
        **params,
    )


def _fit(train_rows: pd.DataFrame, cfg: Config, spec: FeatureSpec):
    """Fit on rows with full lag history (warm-up rows dropped)."""
    longest_lag = max(spec.lags)
    train = train_rows[train_rows[f"lag_{longest_lag}h"].notna()]
    model = make_regressor(cfg)
    model.fit(
        _prepare_X(train, spec),
        train["demand"].to_numpy(dtype=float),
        categorical_feature=spec.categorical,
    )
    return model


class LightGBMForecaster:
    """Global one-hour-ahead LightGBM forecaster for the rolling-origin harness.

    Each fold scores a single hour *t* directly from its causal (pre-computed)
    features, which use only data strictly before *t* (the Phase-2 no-leakage
    guarantee) — a genuine one-step-ahead forecast. The model is **refit only
    every ``refit_every_hours``** (a realistic daily retrain cadence) and reused
    for the hours in between, so 336 hourly folds cost ~14 fits, not 336.
    """

    name = "lightgbm"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._spec: FeatureSpec | None = None
        self._model = None
        self._fit_origin: pd.Timestamp | None = None
        self._refit_every = pd.Timedelta(hours=int(cfg.backtest.get("refit_every_hours", 24)))

    def predict_fold(self, panel: pd.DataFrame, fold: Fold) -> pd.DataFrame:
        spec = build_feature_spec(self.cfg, list(panel.columns))
        self._spec = spec
        # Refit at the first fold and whenever the retrain cadence elapses.
        if self._fit_origin is None or (fold.train_end - self._fit_origin) >= self._refit_every:
            self._model = _fit(train_slice(panel, fold), self.cfg, spec)
            self._fit_origin = fold.train_end

        test = forecast_slice(panel, fold)
        yhat = np.clip(self._model.predict(_prepare_X(test, spec)), 0.0, None)
        return pd.DataFrame(
            {"zone_id": test["zone_id"].to_numpy(), "hour": test["hour"].to_numpy(), "y_pred": yhat}
        )


# --------------------------------------------------------------------------
# Final deployment model (fit on all data) — used by conformal (Phase 5) & UI
# --------------------------------------------------------------------------
def train_final_model(panel: pd.DataFrame, cfg: Config):
    """Fit the global model on the full feature table; return (model, spec)."""
    spec = build_feature_spec(cfg, list(panel.columns))
    model = _fit(panel, cfg, spec)
    return model, spec


def save_model(model, spec: FeatureSpec, processed_dir: Path) -> Path:
    """Persist the booster (text) + feature spec (json)."""
    model_path = processed_dir / "lightgbm_model.txt"
    model.booster_.save_model(str(model_path))
    (processed_dir / "lightgbm_spec.json").write_text(
        json.dumps(
            {
                "features": spec.features,
                "demand_cols": spec.demand_cols,
                "exo_cols": spec.exo_cols,
                "categorical": spec.categorical,
                "lags": spec.lags,
                "roll_windows": spec.roll_windows,
            },
            indent=2,
        )
    )
    return model_path


def load_model(processed_dir: Path):
    """Reload the persisted booster + feature spec (for conformal / the dashboard)."""
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(processed_dir / "lightgbm_model.txt"))
    spec_d = json.loads((processed_dir / "lightgbm_spec.json").read_text())
    spec = FeatureSpec(
        features=spec_d["features"],
        demand_cols=spec_d["demand_cols"],
        exo_cols=spec_d["exo_cols"],
        categorical=spec_d["categorical"],
        lags=spec_d["lags"],
        roll_windows=spec_d["roll_windows"],
    )
    return booster, spec


def predict_demand(booster, spec: FeatureSpec, df: pd.DataFrame) -> np.ndarray:
    """Non-negative demand predictions for rows of a feature frame."""
    return np.clip(booster.predict(_prepare_X(df, spec)), 0.0, None)
