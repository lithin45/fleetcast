"""Build the dense zone-hour feature table from raw trips (Phase 2).

Runs the DuckDB feature query (see :mod:`fleetcast.features.sql`), writes the
result to ``data/processed/features.parquet`` plus a JSON metadata sidecar, and
dumps the exact generated SQL to ``data/processed/features.sql`` for inspection.
Weather and the holiday flag are included only when available.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from ..config import Config, load_config
from ..data import duck
from ..data.sources import months_date_range
from ..logging import get_logger
from . import sql as fsql

log = get_logger(__name__)

FEATURES_PARQUET = "features.parquet"
FEATURES_META = "features_meta.json"
FEATURES_SQL_DUMP = "features.sql"


def panel_window(cfg: Config) -> tuple[str, str]:
    """Return (start, end_excl) timestamps as 'YYYY-MM-DD HH:MM:SS' strings."""
    start, end = months_date_range(cfg.months)
    end_excl = end + timedelta(days=1)
    fmt = "%Y-%m-%d 00:00:00"
    return start.strftime(fmt), end_excl.strftime(fmt)


def holiday_dates_for(cfg: Config) -> list[date]:
    """US federal holidays falling within the configured window (empty if disabled)."""
    if not cfg.add_holiday_flag:
        return []
    import holidays

    start, end = months_date_range(cfg.months)
    end_excl = end + timedelta(days=1)
    years = list(range(start.year, end_excl.year + 1))
    us = holidays.US(years=years)
    return sorted(d for d in us if start <= d < end_excl)


def _weather_expr(cfg: Config) -> str | None:
    """DuckDB relation for the daily weather CSV, or None if unavailable."""
    if not cfg.weather_enabled:
        return None
    path = cfg.raw_dir / "weather_daily.csv"
    if not path.is_file():
        log.warning("weather enabled but %s missing; building WITHOUT weather", path.name)
        return None
    return f"read_csv_auto('{path.as_posix()}', header=true)"


def build_features(cfg: Config | None = None) -> Path:
    """Run the feature query and persist the feature table. Returns its path."""
    cfg = cfg or load_config()
    start, end_excl = panel_window(cfg)
    holidays_in_range = holiday_dates_for(cfg)
    weather_expr = _weather_expr(cfg)

    sql = fsql.build_feature_sql(
        cfg,
        trips_expr=duck.read_trips_relation(cfg),
        start=start,
        end_excl=end_excl,
        weather_expr=weather_expr,
        holiday_dates=holidays_in_range,
    )

    out = cfg.processed_dir / FEATURES_PARQUET
    con = duck.connect()
    con.execute(f"CREATE TABLE features AS {sql}")
    con.execute(
        f"COPY (SELECT * FROM features ORDER BY zone_id, hour) "
        f"TO '{out.as_posix()}' (FORMAT PARQUET)"
    )

    report = _summarize(cfg, con, start, end_excl, weather_expr is not None, holidays_in_range)
    con.close()

    # Persist the generated SQL + metadata for inspection / reproducibility.
    (cfg.processed_dir / FEATURES_SQL_DUMP).write_text(sql)
    (cfg.processed_dir / FEATURES_META).write_text(json.dumps(report, indent=2, default=str))
    _print_report(report)
    return out


def _summarize(
    cfg: Config,
    con: duckdb.DuckDBPyConnection,
    start: str,
    end_excl: str,
    weather_used: bool,
    holidays_in_range: list[date],
) -> dict:
    cols = [c[0] for c in con.execute("DESCRIBE features").fetchall()]
    n_rows, n_zones, min_h, max_h = con.execute(
        "SELECT count(*), count(DISTINCT zone_id), min(hour), max(hour) FROM features"
    ).fetchone()
    n_hours = con.execute("SELECT count(DISTINCT hour) FROM features").fetchone()[0]
    # Warm-up nulls: rows where the longest lag isn't available yet.
    longest_lag = max(cfg.lags_hours)
    null_lag = con.execute(
        f"SELECT count(*) FROM features WHERE lag_{longest_lag}h IS NULL"
    ).fetchone()[0]
    total_pickups = con.execute("SELECT sum(demand) FROM features").fetchone()[0]

    return {
        "config_hash": cfg.config_hash,
        "service": cfg.service,
        "months": cfg.months,
        "window": {"start": start, "end_excl": end_excl},
        "n_rows": int(n_rows),
        "n_zones": int(n_zones),
        "n_hours": int(n_hours),
        "expected_dense_rows": int(n_zones) * int(n_hours),
        "is_dense": int(n_rows) == int(n_zones) * int(n_hours),
        "min_hour": str(min_h),
        "max_hour": str(max_h),
        "total_pickups": int(total_pickups),
        "warmup_null_rows": int(null_lag),
        "weather_used": weather_used,
        "holiday_flag": cfg.add_holiday_flag,
        "n_holidays_in_range": len(holidays_in_range),
        "lags_hours": cfg.lags_hours,
        "rolling_windows_hours": cfg.rolling_windows_hours,
        "columns": cols,
    }


def _print_report(r: dict) -> None:
    bar = "─" * 64
    print(bar)
    print(f"FleetCast feature table  ·  {r['service']}  ·  config_hash={r['config_hash']}")
    print(bar)
    print(f"  rows                : {r['n_rows']:>14,}")
    print(f"  zones x hours       : {r['n_zones']} x {r['n_hours']}  (dense={r['is_dense']})")
    print(f"  window              : {r['min_hour']}  →  {r['max_hour']}")
    print(f"  total pickups       : {r['total_pickups']:>14,}")
    print(f"  warm-up null rows   : {r['warmup_null_rows']:>14,}  (insufficient lag history)")
    print(
        f"  weather / holidays  : {r['weather_used']} / {r['holiday_flag']} "
        f"({r['n_holidays_in_range']} in range)"
    )
    print(f"  feature columns     : {', '.join(r['columns'])}")
    print(bar)


def load_features(cfg: Config | None = None) -> pd.DataFrame:
    """Load the persisted feature table (raises if not yet built)."""
    cfg = cfg or load_config()
    path = cfg.processed_dir / FEATURES_PARQUET
    if not path.is_file():
        raise FileNotFoundError(f"feature table not found: {path} (run `make features`)")
    return pd.read_parquet(path)


def main() -> None:
    """Entry point for ``python -m fleetcast.features.build`` / ``make features``."""
    build_features()


if __name__ == "__main__":
    main()
