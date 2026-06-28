"""Split-conformal intervals: quantile formula, interval shape, and the coverage gate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fleetcast.backtest.run import run_backtest
from fleetcast.conformal.run import run_conformal
from fleetcast.conformal.split_conformal import (
    apply_intervals,
    conformal_quantile,
    conformalize,
    coverage_report,
)
from fleetcast.features.build import build_features, load_features
from fleetcast.models.lightgbm_model import LightGBMForecaster

from . import synthetic

MONTHS3 = ["2024-01", "2024-02", "2024-03"]


def test_conformal_quantile_is_kth_smallest():
    scores = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # n=10, alpha=0.1 -> k = ceil(11*0.9) = ceil(9.9) = 10 -> 10th smallest = 10
    assert conformal_quantile(scores, 0.10) == 10.0
    # alpha=0.5 -> k = ceil(11*0.5) = 6 -> 6th smallest = 6
    assert conformal_quantile(scores, 0.50) == 6.0


def test_conformal_quantile_infinite_when_too_few_points():
    # n=5, alpha=0.1 -> k = ceil(6*0.9) = 6 > 5 -> inf
    assert conformal_quantile(np.array([1.0, 2, 3, 4, 5]), 0.10) == float("inf")


def test_apply_intervals_clips_lower_at_zero():
    df = pd.DataFrame({"zone_id": [1, 1], "y_true": [3.0, 50.0], "y_pred": [2.0, 40.0]})
    out = apply_intervals(df, {1: 5.0})
    assert (out["lower"] >= 0).all()
    assert out.loc[0, "lower"] == 0.0  # 2 - 5 = -3 -> clipped to 0
    assert out.loc[1, "upper"] == 45.0


def test_coverage_report_counts_correctly():
    df = pd.DataFrame(
        {
            "zone_id": [1, 1, 1, 1],
            "y_true": [10.0, 10, 10, 100],  # last is outside
            "lower": [5.0, 5, 5, 5],
            "upper": [15.0, 15, 15, 15],
        }
    )
    rep = coverage_report(df)
    assert rep["empirical_coverage"] == 0.75
    assert rep["per_zone"][1]["n"] == 4


def test_conformalize_raises_on_too_few_calibration_points(make_synthetic_env):
    """A too-small calibration set yields q=inf — must fail loudly, not inflate coverage."""
    cfg = make_synthetic_env().cfg
    hours = pd.date_range("2024-01-08", periods=8, freq="h")  # 4 calib hours/zone < 9 needed
    rows = [
        {"zone_id": z, "hour": h, "y_true": 5.0, "y_pred": 4.0, "model": "lightgbm", "fold": 0}
        for z in (1, 2)
        for h in hours
    ]
    with pytest.raises(ValueError, match="unbounded"):
        conformalize(pd.DataFrame(rows), cfg)


def test_conformal_coverage_gate_on_synthetic(make_synthetic_env):
    """THE COVERAGE GATE (CI): split conformal achieves ~90% on held-out residuals."""
    env = make_synthetic_env(months=MONTHS3, n_zones=8, weather_enabled=True)
    synthetic.write_synthetic_weather(env.raw_dir, MONTHS3)
    env.cfg.raw["backtest"]["n_folds"] = 168
    build_features(env.cfg)
    panel = load_features(env.cfg)

    res = run_backtest(panel, [LightGBMForecaster(env.cfg)], env.cfg)
    report = conformalize(res.predictions, env.cfg)
    nominal = float(env.cfg.conformal["nominal_coverage"])
    # Split conformal guarantees ~>= nominal under exchangeability; allow a small band.
    assert abs(report["empirical_coverage"] - nominal) <= 0.04
    assert report["mean_interval_width"] > 0


def test_run_conformal_writes_metrics_and_quantiles(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS3, n_zones=6)
    env.cfg.raw["backtest"]["n_folds"] = 120
    build_features(env.cfg)
    panel = load_features(env.cfg)
    res = run_backtest(panel, [LightGBMForecaster(env.cfg)], env.cfg)

    report = run_conformal(env.cfg, predictions=res.predictions)
    meta = json.loads((env.cfg.processed_dir / "conformal_metrics.json").read_text())
    assert "empirical_coverage" in meta
    assert meta["empirical_coverage"] == report["empirical_coverage"]
    qdf = pd.read_parquet(env.cfg.processed_dir / "conformal_quantiles.parquet")
    assert set(qdf.columns) == {"zone_id", "q90"} and len(qdf) == 6
