"""Rolling-origin backtest: fold time-ordering (gate), metrics, baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fleetcast.backtest.forecasters as fmod
from fleetcast.backtest import metrics as M
from fleetcast.backtest.folds import forecast_slice, make_folds, train_slice
from fleetcast.backtest.forecasters import ArimaForecaster, SeasonalNaiveForecaster
from fleetcast.backtest.run import run_backtest
from fleetcast.config import load_config
from fleetcast.features.build import build_features, load_features

MONTHS3 = ["2024-01", "2024-02", "2024-03"]


def _panel(env):
    build_features(env.cfg)
    return load_features(env.cfg)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_metric_formulas():
    y = np.array([10.0, 0.0, 5.0, 20.0])
    yhat = np.array([12.0, 1.0, 5.0, 15.0])
    assert M.wape(y, yhat) == np.float64(8 / 35)
    assert M.mae(y, yhat) == 2.0
    assert M.rmse(y, yhat) == np.sqrt(7.5)
    # MAPE only over y>0: (0.2 + 0 + 0.25) / 3
    assert abs(M.mape(y, yhat) - 0.15) < 1e-12


def test_wape_is_nan_when_actuals_all_zero():
    assert np.isnan(M.wape(np.zeros(3), np.array([1.0, 2.0, 3.0])))


# --------------------------------------------------------------------------
# Fold structure — the leakage gate
# --------------------------------------------------------------------------
def test_folds_strictly_time_ordered(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS3, n_zones=3)
    panel = _panel(env)
    folds = make_folds(panel["hour"], env.cfg)
    assert len(folds) >= 1

    panel_min = panel["hour"].min()
    step = pd.Timedelta(hours=int(env.cfg.backtest["step_hours"]))
    horizon = int(env.cfg.backtest["horizon_hours"])
    for k, f in enumerate(folds):
        tr, te = train_slice(panel, f), forecast_slice(panel, f)
        # GATE: every train hour is strictly before the cutoff == first test hour.
        assert tr["hour"].max() < f.train_end
        assert f.train_end == f.test_start
        assert te["hour"].min() >= f.test_start
        assert te["hour"].max() < f.test_end
        assert f.horizon_hours == horizon
        # Expanding window: training always starts at the panel's first hour.
        assert tr["hour"].min() == panel_min
        # No train/test overlap.
        assert set(tr["hour"]).isdisjoint(set(te["hour"]))
        if k > 0:
            assert f.test_start - folds[k - 1].test_start == step


def test_folds_reduce_when_series_too_short():
    cfg = load_config()  # default min_train=1344, horizon=1, step=1, n_folds=336
    n_folds = int(cfg.backtest["n_folds"])
    hours = pd.date_range("2024-01-01", periods=1400, freq="h")
    folds = make_folds(hours, cfg)
    # 1400h can't fit all n_folds while honoring min_train -> reduced, never overlapping.
    assert 1 <= len(folds) < n_folds
    for f in folds:
        assert (f.train_end - hours[0]) >= pd.Timedelta(hours=1344)
    starts = [f.test_start for f in folds]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def test_seasonal_naive_predicts_same_hour_last_week(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS3, n_zones=3)
    panel = _panel(env)
    fold = make_folds(panel["hour"], env.cfg)[-1]
    preds = SeasonalNaiveForecaster().predict_fold(panel, fold)

    demand = {(int(r.zone_id), r.hour): r.demand for r in panel.itertuples(index=False)}
    assert len(preds) == len(forecast_slice(panel, fold))
    for r in preds.itertuples(index=False):
        assert r.y_pred == demand[(int(r.zone_id), r.hour - pd.Timedelta(hours=168))]


def test_backtest_complete_and_reproducible(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS3, n_zones=3)
    env.cfg.raw["backtest"]["n_folds"] = 2
    panel = _panel(env)

    r1 = run_backtest(panel, [SeasonalNaiveForecaster()], env.cfg)
    r2 = run_backtest(panel, [SeasonalNaiveForecaster()], env.cfg)
    pd.testing.assert_frame_equal(r1.predictions, r2.predictions)

    # Every test zone-hour got a prediction; pooled WAPE is finite.
    assert r1.predictions["y_pred"].notna().all()
    n_test = sum(len(forecast_slice(panel, f)) for f in r1.folds)
    assert len(r1.predictions) == n_test
    assert np.isfinite(r1.metrics["seasonal_naive"]["pooled"]["wape"])


def test_arima_runs_nonnegative_and_reproducible(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS3, n_zones=2)
    env.cfg.raw["backtest"]["n_folds"] = 1
    panel = _panel(env)
    fold = make_folds(panel["hour"], env.cfg)[-1]

    p1 = ArimaForecaster(env.cfg).predict_fold(panel, fold)
    p2 = ArimaForecaster(env.cfg).predict_fold(panel, fold)
    assert len(p1) == len(forecast_slice(panel, fold))
    assert p1["y_pred"].notna().all()
    assert (p1["y_pred"] >= 0).all()
    pd.testing.assert_frame_equal(p1, p2)  # SARIMAX fit is deterministic


def test_high_volume_zones_selects_top(make_synthetic_env):
    env = make_synthetic_env(months=["2024-01"], n_zones=10)
    panel = _panel(env)
    hv = M.high_volume_zones(panel, 0.5)
    # Rank-based: top (1-0.5)*10 = 5 zones; synthetic level rises with zone id.
    assert len(hv) == 5
    assert 10 in hv and 1 not in hv


# --------------------------------------------------------------------------
# Harness hardening — target isolation + merge validation (adversarial)
# --------------------------------------------------------------------------
class _CheatForecaster:
    """Tries to read the demand target it is being scored on."""

    name = "cheat"

    def predict_fold(self, panel, fold):
        test = forecast_slice(panel, fold)
        return pd.DataFrame(
            {
                "zone_id": test["zone_id"].to_numpy(),
                "hour": test["hour"].to_numpy(),
                "y_pred": test["demand"].to_numpy(),
            }
        )


class _DropForecaster:
    name = "drop"

    def predict_fold(self, panel, fold):
        sn = SeasonalNaiveForecaster().predict_fold(panel, fold)
        return sn.iloc[: len(sn) // 2]  # under-cover the panel


class _DupForecaster:
    name = "dup"

    def predict_fold(self, panel, fold):
        sn = SeasonalNaiveForecaster().predict_fold(panel, fold)
        return pd.concat([sn, sn.iloc[:1]], ignore_index=True)  # duplicate one key


def _one_fold_env(make_synthetic_env, n_zones=3):
    env = make_synthetic_env(months=MONTHS3, n_zones=n_zones)
    env.cfg.raw["backtest"]["n_folds"] = 1
    return env


def test_target_is_masked_from_forecasters(make_synthetic_env):
    """A forecaster reading test-window demand gets NaN (masked) -> harness rejects."""
    env = _one_fold_env(make_synthetic_env)
    panel = _panel(env)
    with pytest.raises(ValueError):
        run_backtest(panel, [_CheatForecaster()], env.cfg)


def test_missing_predictions_fail_loudly(make_synthetic_env):
    env = _one_fold_env(make_synthetic_env)
    panel = _panel(env)
    with pytest.raises(ValueError, match="did not predict every test zone-hour"):
        run_backtest(panel, [_DropForecaster()], env.cfg)


def test_duplicate_predictions_fail_loudly(make_synthetic_env):
    env = _one_fold_env(make_synthetic_env)
    panel = _panel(env)
    with pytest.raises(ValueError, match="duplicate"):
        run_backtest(panel, [_DupForecaster()], env.cfg)


def test_arima_falls_back_to_seasonal_naive(make_synthetic_env, monkeypatch):
    env = _one_fold_env(make_synthetic_env, n_zones=2)
    panel = _panel(env)
    fold = make_folds(panel["hour"], env.cfg)[-1]
    # Force every SARIMAX fit to "fail" -> the forecaster must fall back cleanly.
    monkeypatch.setattr(fmod, "_fit_forecast_arima", lambda task: None)
    arima = ArimaForecaster(env.cfg)
    ap = arima.predict_fold(panel, fold).sort_values(["zone_id", "hour"]).reset_index(drop=True)
    sn = (
        SeasonalNaiveForecaster()
        .predict_fold(panel, fold)
        .sort_values(["zone_id", "hour"])
        .reset_index(drop=True)
    )
    # Fallback must reproduce seasonal-naive values (zone_id dtype is incidental).
    pd.testing.assert_frame_equal(ap, sn, check_dtype=False)


def test_pooled_metrics_match_independent_recompute(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS3, n_zones=3)
    env.cfg.raw["backtest"]["n_folds"] = 2
    panel = _panel(env)
    r = run_backtest(panel, [SeasonalNaiveForecaster()], env.cfg)
    preds = r.predictions
    y, yhat = preds["y_true"].to_numpy(float), preds["y_pred"].to_numpy(float)
    exp_wape = np.sum(np.abs(y - yhat)) / np.sum(np.abs(y))
    assert r.metrics["seasonal_naive"]["pooled"]["wape"] == pytest.approx(exp_wape)
