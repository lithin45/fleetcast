"""Global LightGBM model: the WAPE gate, determinism, leakage-safety, persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fleetcast.backtest.folds import forecast_slice, make_folds
from fleetcast.backtest.forecasters import SeasonalNaiveForecaster
from fleetcast.backtest.run import run_backtest
from fleetcast.features.build import build_features, load_features
from fleetcast.models.lightgbm_model import (
    LightGBMForecaster,
    build_feature_spec,
    load_model,
    predict_demand,
    save_model,
    train_final_model,
)

MONTHS3 = ["2024-01", "2024-02", "2024-03"]


def _panel(env):
    build_features(env.cfg)
    return load_features(env.cfg)


def test_lightgbm_beats_seasonal_naive_gate(make_synthetic_env):
    """THE GATE (runs in CI on synthetic): LightGBM beats seasonal-naive by >=20% WAPE."""
    env = make_synthetic_env(months=MONTHS3, n_zones=8, weather_enabled=True)
    from . import synthetic

    synthetic.write_synthetic_weather(env.raw_dir, MONTHS3)
    # One-step-ahead over the last 7 days (168 hourly folds, model refit daily).
    env.cfg.raw["backtest"]["n_folds"] = 168
    panel = _panel(env)

    res = run_backtest(panel, [SeasonalNaiveForecaster(), LightGBMForecaster(env.cfg)], env.cfg)
    sn = res.metrics["seasonal_naive"]["pooled"]["wape"]
    lgbm = res.metrics["lightgbm"]["pooled"]["wape"]
    improvement = (sn - lgbm) / sn
    target = float(env.cfg.eval["wape_improvement_over_seasonal_naive"])
    assert improvement >= target, (
        f"LightGBM only beat seasonal-naive by {improvement:.1%} (< {target:.0%})"
    )


def test_lightgbm_predictions_complete_and_nonnegative(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS3, n_zones=4)
    env.cfg.raw["backtest"]["n_folds"] = 2
    panel = _panel(env)
    fold = make_folds(panel["hour"], env.cfg)[-1]
    preds = LightGBMForecaster(env.cfg).predict_fold(panel, fold)
    assert len(preds) == len(forecast_slice(panel, fold))
    assert preds["y_pred"].notna().all()
    assert (preds["y_pred"] >= 0).all()


def test_lightgbm_is_deterministic(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS3, n_zones=4)
    env.cfg.raw["backtest"]["n_folds"] = 2
    panel = _panel(env)
    r1 = run_backtest(panel, [LightGBMForecaster(env.cfg)], env.cfg)
    r2 = run_backtest(panel, [LightGBMForecaster(env.cfg)], env.cfg)
    pd.testing.assert_frame_equal(r1.predictions, r2.predictions)


def test_feature_spec_includes_expected_columns(make_synthetic_env):
    env = make_synthetic_env(months=["2024-01"], n_zones=3, weather_enabled=True)
    from . import synthetic

    synthetic.write_synthetic_weather(env.raw_dir, ["2024-01"])
    panel = _panel(env)
    spec = build_feature_spec(env.cfg, list(panel.columns))
    assert "zone_id" in spec.features and spec.categorical == ["zone_id"]
    assert {"lag_1h", "lag_24h", "lag_168h"} <= set(spec.features)
    assert {"roll_mean_24h", "roll_std_24h"} <= set(spec.features)
    assert "tmax_c_d1" in spec.features and "is_holiday" in spec.features
    assert "demand" not in spec.features and "hour" not in spec.features  # never features


def test_final_model_save_load_roundtrip(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS3, n_zones=4)
    panel = _panel(env)
    model, spec = train_final_model(panel, env.cfg)
    save_model(model, spec, env.cfg.processed_dir)

    booster, spec2 = load_model(env.cfg.processed_dir)
    assert spec2.features == spec.features
    sample = panel[panel["lag_168h"].notna()].head(100)
    from fleetcast.models.lightgbm_model import _prepare_X

    sk_pred = np.clip(model.predict(_prepare_X(sample, spec)), 0.0, None)
    ld_pred = predict_demand(booster, spec2, sample)
    assert np.allclose(sk_pred, ld_pred, rtol=1e-6, atol=1e-6)
