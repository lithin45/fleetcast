"""The committed sql/features.sql must stay in sync with the generator, and the
generated SQL must carry the structural leakage-safety guarantees.
"""

from __future__ import annotations

from fleetcast.config import load_config
from fleetcast.features.build import holiday_dates_for, panel_window
from fleetcast.features.sql import build_feature_sql, feature_select, render_canonical_sql
from fleetcast.paths import repo_root


def test_committed_canonical_sql_matches_generator():
    """`sql/features.sql` == render_canonical_sql(default config). Regenerate with
    `uv run python -m fleetcast.features.sql` if this fails after a config change."""
    cfg = load_config()
    start, end_excl = panel_window(cfg)
    rendered = render_canonical_sql(cfg, holiday_dates_for(cfg), start, end_excl)
    committed = (repo_root() / "sql" / "features.sql").read_text()
    assert rendered == committed, "sql/features.sql is stale — regenerate it"


def test_leakage_safety_markers_present():
    cfg = load_config()
    sel = feature_select(cfg, source="panel")
    # Rolling windows must end at '1 PRECEDING' (current row excluded) for EVERY window.
    for w in cfg.rolling_windows_hours:
        assert f"ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING" in sel
    # Lags are plain backward LAG()s.
    for lag in cfg.lags_hours:
        assert f"LAG(demand, {lag})" in sel
    # No forward-looking constructs.
    assert "FOLLOWING" not in sel
    assert "LEAD(" not in sel


def test_zero_fill_and_gapfree_spine_present():
    cfg = load_config()
    start, end_excl = panel_window(cfg)
    sql = build_feature_sql(cfg, trips_expr="trips", start=start, end_excl=end_excl)
    assert "COALESCE(h.demand, 0)" in sql  # zero-filled dense panel
    assert "range(TIMESTAMP" in sql  # gap-free hour spine
    assert "CROSS JOIN" in sql  # dense zone x hour lattice
