"""Feature table: density, shape, calendar/lag/rolling correctness, weather/holiday."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fleetcast.features.build import build_features, load_features, panel_window
from fleetcast.features.sql import feature_select

from . import synthetic

# Two months so the 168h (1-week) lag/rolling windows have history.
MONTHS = ["2024-01", "2024-02"]


def _build(env):
    build_features(env.cfg)
    return load_features(env.cfg).sort_values(["zone_id", "hour"]).reset_index(drop=True)


def test_panel_is_dense(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS, n_zones=5)
    df = _build(env)
    n_zones = df["zone_id"].nunique()
    # Pin the expected hours to the CONFIG window (not derived from the output),
    # so a spine-range regression that uniformly drops hours is caught.
    start_s, end_s = panel_window(env.cfg)
    start, end_excl = pd.Timestamp(start_s), pd.Timestamp(end_s)
    expected_hours = int((end_excl - start).total_seconds() // 3600)
    assert df["hour"].nunique() == expected_hours
    assert df["hour"].min() == start
    assert df["hour"].max() == end_excl - pd.Timedelta(hours=1)
    # Exactly zones x hours rows, every zone present for every hour.
    assert len(df) == n_zones * expected_hours
    assert (df.groupby("zone_id").size() == expected_hours).all()
    # Hours are contiguous (1h step) within each zone — gap-free panel.
    for _, g in df.groupby("zone_id"):
        diffs = g["hour"].diff().dropna().dt.total_seconds().unique()
        assert set(diffs) == {3600.0}
    # Zero-filled: demand is non-negative and never null.
    assert df["demand"].notna().all()
    assert (df["demand"] >= 0).all()


def test_feature_table_columns(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS, n_zones=4)
    df = _build(env)
    expected = {
        "zone_id",
        "hour",
        "demand",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "month",
        "lag_1h",
        "lag_24h",
        "lag_168h",
        "roll_mean_24h",
        "roll_std_24h",
        "roll_mean_168h",
        "roll_std_168h",
        "is_holiday",  # default config enables the holiday flag
    }
    assert expected <= set(df.columns)


def test_calendar_features_correct(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS, n_zones=2)
    df = _build(env)
    h = df["hour"].dt
    assert (df["hour_of_day"] == h.hour).all()
    # day_of_week is isodow: Mon=1 … Sun=7  (pandas weekday is Mon=0 … Sun=6)
    assert (df["day_of_week"] == h.weekday + 1).all()
    assert (df["is_weekend"] == (h.weekday >= 5)).all()
    assert (df["month"] == h.month).all()


def test_lag_features_match_shifted_demand(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS, n_zones=3)
    df = _build(env)
    demand = {(int(r.zone_id), r.hour): r.demand for r in df.itertuples(index=False)}
    # Require lag_168h history so both lookups land inside the panel.
    sample = df[df["lag_168h"].notna()].sample(40, random_state=0)
    for r in sample.itertuples(index=False):
        assert r.lag_24h == demand[(int(r.zone_id), r.hour - pd.Timedelta(hours=24))]
        assert r.lag_168h == demand[(int(r.zone_id), r.hour - pd.Timedelta(hours=168))]


def test_rolling_excludes_current_row(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS, n_zones=2)
    df = _build(env)
    # Independent pandas recompute: prior-24h mean, current row EXCLUDED via shift(1).
    g = df.groupby("zone_id")["demand"]
    expected_mean = g.transform(lambda s: s.shift(1).rolling(24, min_periods=1).mean())
    got = df["roll_mean_24h"]
    both = got.notna() & expected_mean.notna()
    assert np.allclose(got[both], expected_mean[both], rtol=1e-9, atol=1e-9)
    # The first row of each zone has no prior data -> null.
    firsts = df.groupby("zone_id").head(1)
    assert firsts["roll_mean_24h"].isna().all()


def test_rolling_std_matches_sample_std(make_synthetic_env):
    """Validate roll_std VALUES (guards STDDEV_SAMP vs _POP / wrong-window swaps)."""
    env = make_synthetic_env(months=MONTHS, n_zones=2)
    df = _build(env)
    g = df.groupby("zone_id")["demand"]
    for w in (24, 168):
        # ddof=1 == STDDEV_SAMP; min_periods=2 because sample std needs >=2 points.
        expected = g.transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=2).std(ddof=1))
        got = df[f"roll_std_{w}h"]
        both = got.notna() & expected.notna()
        assert both.sum() > 0
        assert np.allclose(got[both], expected[both], rtol=1e-6, atol=1e-6)


def test_weather_joins_prior_day_causally(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS, n_zones=2, weather_enabled=True)
    synthetic.write_synthetic_weather(env.raw_dir, MONTHS)
    df = _build(env)
    for col in ("prcp_mm_d1", "snow_mm_d1", "tmax_c_d1", "tmin_c_d1"):
        assert col in df.columns
    # We fetch one extra prior day, so every row has prior-day weather (no nulls).
    assert df["tmax_c_d1"].notna().all()

    # Each row's tmax_c_d1 must equal the PRIOR calendar day's TMAX (causal lag).
    wx = pd.read_csv(env.raw_dir / "weather_daily.csv", parse_dates=["DATE"])
    tmax_by_date = {d.date(): t for d, t in zip(wx["DATE"], wx["TMAX"], strict=True)}
    sample = df.sample(50, random_state=0)
    for r in sample.itertuples(index=False):
        prior_day = r.hour.date() - pd.Timedelta(days=1)  # date - timedelta -> date
        assert np.isclose(r.tmax_c_d1, tmax_by_date[prior_day])
    # All 24 hours of a day share one (prior-day) value.
    one_day = df[
        (df["zone_id"] == df["zone_id"].iloc[0])
        & (df["hour"].dt.date == df["hour"].dt.date.iloc[0])
    ]
    assert one_day["tmax_c_d1"].nunique() == 1


def test_build_without_weather_when_file_missing(make_synthetic_env):
    """weather_enabled=True but no CSV -> build still succeeds, sans weather cols."""
    env = make_synthetic_env(months=MONTHS, n_zones=2, weather_enabled=True)
    # Deliberately do NOT write a weather CSV.
    df = _build(env)
    assert not any(c.endswith("_d1") for c in df.columns)
    meta = json.loads((env.cfg.processed_dir / "features_meta.json").read_text())
    assert meta["weather_used"] is False


def test_features_meta_sidecar(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS, n_zones=4)
    build_features(env.cfg)
    meta = json.loads((env.cfg.processed_dir / "features_meta.json").read_text())
    assert meta["is_dense"] is True
    assert meta["n_rows"] == meta["expected_dense_rows"]
    assert meta["warmup_null_rows"] == 4 * 168  # each zone's first 168h lack lag_168h
    assert meta["weather_used"] is False
    assert meta["config_hash"] == env.cfg.config_hash


def test_holiday_flag_marks_known_holidays(make_synthetic_env):
    env = make_synthetic_env(months=["2024-01"], n_zones=2)
    df = _build(env)
    dates = df["hour"].dt.date.astype(str)
    # 2024-01-01 New Year's Day and 2024-01-15 MLK Day are US federal holidays.
    assert df.loc[dates == "2024-01-01", "is_holiday"].all()
    assert df.loc[dates == "2024-01-15", "is_holiday"].all()
    # A plain Tuesday is not.
    assert not df.loc[dates == "2024-01-09", "is_holiday"].any()


def test_build_is_deterministic(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS, n_zones=3)
    a = _build(env)
    b = _build(env)
    pd.testing.assert_frame_equal(a, b)


def test_feature_select_used_in_no_leakage_is_demand_only(make_synthetic_env):
    """feature_select must not project weather/holiday (those are outer joins)."""
    env = make_synthetic_env(months=MONTHS)
    sql = feature_select(env.cfg, source="panel")
    assert "is_holiday" not in sql and "weather" not in sql
    assert "ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING" in sql
