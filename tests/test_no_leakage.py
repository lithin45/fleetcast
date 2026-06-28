"""No-temporal-leakage gate (HARD GATE).

The contract: a feature computed at hour *t* may use only information available
strictly before *t*. We prove this two ways against the *exact same* windowed SQL
(`feature_select`) used in production:

1. **Perturbation invariance** — spike ``demand`` at a target hour *t* and confirm
   every lag/rolling feature value *at row t* is byte-identical to before (they
   look only backward), while the change DOES propagate to *future* rows
   (t+1's lag_1h, etc.). A feature that peeked at the present would change.
2. **Independent recompute** — lag/rolling values match a hand-rolled
   past-only computation (current row excluded).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from fleetcast.data import duck
from fleetcast.features.build import build_features, load_features
from fleetcast.features.sql import feature_select

from . import synthetic

LAG_COLS = ["lag_1h", "lag_24h", "lag_168h"]
ROLL_COLS = ["roll_mean_24h", "roll_std_24h", "roll_mean_168h", "roll_std_168h"]
FEATURE_COLS = LAG_COLS + ROLL_COLS


def _mini_panel(zones=(1, 2), n_hours: int = 24 * 14, seed: int = 7) -> pd.DataFrame:
    """A controlled dense, gap-free zone-hour panel (>= 168h of history)."""
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1)
    rows = []
    for z in zones:
        for i in range(n_hours):
            rows.append((z, start + timedelta(hours=i), int(rng.integers(0, 50))))
    return pd.DataFrame(rows, columns=["zone_id", "hour", "demand"])


def _run_features(panel: pd.DataFrame, cfg) -> pd.DataFrame:
    con = duck.connect()
    con.register("mini_panel", panel)
    out = con.execute(feature_select(cfg, source="mini_panel")).df()
    con.close()
    return out.sort_values(["zone_id", "hour"]).reset_index(drop=True)


def test_features_do_not_see_the_present(make_synthetic_env):
    cfg = make_synthetic_env().cfg
    panel = _mini_panel()
    base = _run_features(panel, cfg)

    # Target a row with full history (>168h) and a successor in the same zone.
    target_zone = 1
    zrows = base[base["zone_id"] == target_zone].reset_index(drop=True)
    i = 200  # >= 168, leaves room for successors
    target_hour = zrows.loc[i, "hour"]

    # Spike demand at exactly (target_zone, target_hour).
    perturbed = panel.copy()
    mask = (perturbed["zone_id"] == target_zone) & (perturbed["hour"] == target_hour)
    assert mask.sum() == 1
    perturbed.loc[mask, "demand"] += 10_000
    after = _run_features(perturbed, cfg)

    b = base[base["zone_id"] == target_zone].reset_index(drop=True)
    a = after[after["zone_id"] == target_zone].reset_index(drop=True)

    # (1) Features AT the perturbed row are unchanged — they look only backward.
    for col in FEATURE_COLS:
        assert _eq(a.loc[i, col], b.loc[i, col]), f"{col} at t leaked the present"

    # (2) Features at ALL earlier rows are unchanged.
    for col in FEATURE_COLS:
        assert _series_eq(a.loc[: i - 1, col], b.loc[: i - 1, col]), f"{col} changed in the past"

    # (3) The change DOES propagate forward (machinery is live, not constant):
    #     row t+1's lag_1h must now differ by exactly the spike.
    assert a.loc[i + 1, "lag_1h"] - b.loc[i + 1, "lag_1h"] == 10_000
    # row t+24's lag_24h likewise.
    assert a.loc[i + 24, "lag_24h"] - b.loc[i + 24, "lag_24h"] == 10_000

    # (4) The other zone is completely unaffected.
    bz2 = base[base["zone_id"] == 2].reset_index(drop=True)
    az2 = after[after["zone_id"] == 2].reset_index(drop=True)
    pd.testing.assert_frame_equal(az2, bz2)


def test_rolling_and_lags_match_past_only_recompute(make_synthetic_env):
    cfg = make_synthetic_env().cfg
    panel = _mini_panel()
    feats = _run_features(panel, cfg)

    g_demand = panel.sort_values(["zone_id", "hour"]).groupby("zone_id")["demand"]
    # lag_24h == demand shifted 24 within zone.
    exp_lag24 = g_demand.shift(24).reset_index(drop=True)
    got_lag24 = feats.sort_values(["zone_id", "hour"])["lag_24h"].reset_index(drop=True)
    assert _series_eq(got_lag24, exp_lag24)

    # roll_mean_168h == mean of the prior 168 hours, current row EXCLUDED (shift(1)).
    exp_roll = g_demand.transform(
        lambda s: s.shift(1).rolling(168, min_periods=1).mean()
    ).reset_index(drop=True)
    got_roll = feats.sort_values(["zone_id", "hour"])["roll_mean_168h"].reset_index(drop=True)
    both = got_roll.notna() & exp_roll.notna()
    assert np.allclose(got_roll[both], exp_roll[both], rtol=1e-9, atol=1e-9)


def test_weather_perturbation_only_affects_next_day(make_synthetic_env):
    """Weather is joined as the PRIOR day's summary, so changing weather on date D
    may move features only on D+1 — never on D or earlier. This closes the gap that
    feature_select() (demand-only) leaves: it gates the weather family directly."""
    months = ["2024-01", "2024-02"]
    env = make_synthetic_env(months=months, n_zones=2, weather_enabled=True)
    csv = synthetic.write_synthetic_weather(env.raw_dir, months)

    build_features(env.cfg)
    base = load_features(env.cfg).sort_values(["zone_id", "hour"]).reset_index(drop=True)

    # Spike TMAX on a single interior date D, rebuild.
    target = "2024-01-20"
    wx = pd.read_csv(csv)
    wx.loc[wx["DATE"] == target, "TMAX"] = 999.0
    wx.to_csv(csv, index=False)
    build_features(env.cfg)
    after = load_features(env.cfg).sort_values(["zone_id", "hour"]).reset_index(drop=True)

    changed = after["tmax_c_d1"] != base["tmax_c_d1"]
    changed_dates = set(after.loc[changed, "hour"].dt.date.astype(str))
    # ONLY the day AFTER D sees the change — strictly forward in time.
    assert changed_dates == {"2024-01-21"}


# --- small NaN-aware comparison helpers ---
def _eq(x, y) -> bool:
    if pd.isna(x) and pd.isna(y):
        return True
    return bool(x == y)


def _series_eq(a: pd.Series, b: pd.Series) -> bool:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    return bool(((a == b) | (a.isna() & b.isna())).all())
