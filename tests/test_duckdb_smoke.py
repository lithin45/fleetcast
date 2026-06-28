"""DuckDB reads the parquet and aggregates to a zone-hour panel (offline).

Uses deterministic synthetic data so it runs in CI without network. The same
``run_smoke`` path is exercised against real TLC parquet by the network test.
"""

from __future__ import annotations

from fleetcast.data import duck
from fleetcast.data.smoke import run_smoke


def test_duckdb_can_read_synthetic_parquet(make_synthetic_env):
    env = make_synthetic_env(months=["2024-01"], n_zones=6)
    con = duck.connect()
    rel = duck.read_trips_relation(env.cfg)
    n = con.execute(f"SELECT count(*) FROM {rel}").fetchone()[0]
    con.close()
    assert n > 0


def test_read_trips_relation_fails_loudly_on_missing_file(make_synthetic_env):
    env = make_synthetic_env(months=["2024-01"])
    # Ask for a month we never generated -> DuckDB should raise.
    env.cfg.raw["taxi"]["months"] = ["2024-01", "2099-12"]
    con = duck.connect()
    try:
        con.execute(f"SELECT count(*) FROM {duck.read_trips_relation(env.cfg)}")
        raised = False
    except Exception:
        raised = True
    finally:
        con.close()
    assert raised


def test_smoke_report_shapes(make_synthetic_env):
    env = make_synthetic_env(months=["2024-01"], n_zones=8, top_n=5)
    report = run_smoke(env.cfg)

    assert report.total_trips > 0
    assert report.distinct_pickup_zones == 8
    assert len(report.top_zone_ids) == 5
    # January 2024 has 31 days -> 744 hours.
    assert report.n_hours == 31 * 24
    assert report.dense_zone_hours == 5 * 31 * 24
    # Observed zone-hours can never exceed the dense grid.
    assert 0 < report.observed_zone_hours <= report.dense_zone_hours


def test_top_zones_are_highest_volume(make_synthetic_env):
    # Synthetic base level increases with zone index, so the highest IDs win.
    env = make_synthetic_env(months=["2024-01"], n_zones=10, top_n=3)
    report = run_smoke(env.cfg)
    assert set(report.top_zone_ids) == {8, 9, 10}
