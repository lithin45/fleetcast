"""DuckDB feature-engineering SQL — the project's SQL centerpiece.

Everything that turns raw TLC trips into a model-ready feature table is expressed
as a single DuckDB query built here:

1.  ``top_zones``  — the N highest-volume pickup zones.
2.  ``hour_spine`` — every hour in range via ``range()`` (half-open, gap-free).
3.  ``grid``       — the dense zone x hour lattice (CROSS JOIN).
4.  ``hourly``     — observed pickup counts per zone-hour.
5.  ``panel``      — ``grid`` LEFT JOIN ``hourly`` with ``COALESCE(..., 0)`` so
                     **zero-demand hours are filled** (the key to gap-free lags).
6.  ``featured``   — calendar + lag + rolling features via **window functions**.

**Leakage safety is structural, not incidental:**

* Lags use ``LAG(demand, n)``. Because ``panel`` is dense and gap-free, lagging by
  *n rows* is exactly lagging by *n hours*, and ``LAG`` only ever looks backward.
* Rolling stats use ``... ROWS BETWEEN w PRECEDING AND 1 PRECEDING`` — the
  ``1 PRECEDING`` upper bound **excludes the current row**, so a feature at time
  *t* can never see ``demand`` at *t*. The no-leakage test asserts exactly this.
* Weather is joined as the **prior day's** realized daily summary
  (``f.hour::DATE = w.d + INTERVAL 1 DAY``); same-day NOAA aggregates (TMAX/PRCP)
  are end-of-day quantities and would leak, so they are deliberately lagged a day.

The generator is config-driven (any set of lag/rolling windows), and
:func:`render_canonical_sql` renders a portable, annotated copy committed at
``sql/features.sql`` for documentation.
"""

from __future__ import annotations

from datetime import date

from ..config import Config


# --------------------------------------------------------------------------
# Feature expression builders (config-driven)
# --------------------------------------------------------------------------
def _lag_exprs(cfg: Config) -> list[str]:
    return [f"LAG(demand, {lag}) OVER w AS lag_{lag}h" for lag in cfg.lags_hours]


def _rolling_exprs(cfg: Config) -> list[str]:
    out: list[str] = []
    for w in cfg.rolling_windows_hours:
        out.append(f"AVG(demand)         OVER w{w} AS roll_mean_{w}h")
        out.append(f"STDDEV_SAMP(demand) OVER w{w} AS roll_std_{w}h")
    return out


def _window_defs(cfg: Config) -> list[str]:
    defs = ["w AS (PARTITION BY zone_id ORDER BY hour)"]
    for w in cfg.rolling_windows_hours:
        defs.append(
            f"w{w} AS (PARTITION BY zone_id ORDER BY hour "
            f"ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING)"
        )
    return defs


def feature_select(cfg: Config, source: str = "panel") -> str:
    """The windowed feature core: calendar + lags + rolling over a panel relation.

    ``source`` must expose ``(zone_id, hour, demand)`` for one dense, gap-free
    zone-hour panel. Returns a standalone ``SELECT`` (no trailing ``;``), so it is
    reusable both inside the production CTE chain and directly over a registered
    table (the no-leakage test does the latter).
    """
    lags = _lag_exprs(cfg)
    rolls = _rolling_exprs(cfg)

    projection: list[str] = [
        "zone_id",
        "hour",
        "demand",
        # calendar features — row-local, computed from the hour itself (no leakage)
        "-- calendar (row-local; uses only the row's own timestamp)\n        "
        "hour(hour)          AS hour_of_day",
        "isodow(hour)        AS day_of_week",  # 1=Mon ... 7=Sun
        "(isodow(hour) >= 6) AS is_weekend",
        "month(hour)         AS month",
        # lags
        "-- lags: dense gap-free panel => LAG(n rows) == lag(n hours), backward-only\n        "
        + lags[0],
        *lags[1:],
        # rolling
        "-- rolling stats over the PRIOR window only; '... AND 1 PRECEDING' EXCLUDES\n        "
        "-- the current row, so demand[t] can never enter its own features (no leakage)\n        "
        + rolls[0],
        *rolls[1:],
    ]
    body = ",\n        ".join(projection)
    windows = ",\n        ".join(_window_defs(cfg))
    return f"SELECT\n        {body}\n    FROM {source}\n    WINDOW\n        {windows}"


# --------------------------------------------------------------------------
# Full query assembly
# --------------------------------------------------------------------------
def _holiday_in_list(holiday_dates: list[date]) -> str:
    quoted = ", ".join(f"DATE '{d.isoformat()}'" for d in holiday_dates)
    return f"(f.hour::DATE IN ({quoted}))" if quoted else "FALSE"


def build_feature_sql(
    cfg: Config,
    *,
    trips_expr: str,
    start: str,
    end_excl: str,
    weather_expr: str | None = None,
    holiday_dates: list[date] | None = None,
) -> str:
    """Assemble the full feature query.

    ``trips_expr`` is a DuckDB relation over the raw trips (e.g. ``read_parquet([...])``);
    ``weather_expr`` is an optional relation over the daily weather CSV. ``start`` /
    ``end_excl`` bound the dense panel (half-open). Holiday/weather columns are
    appended only when requested, so the pipeline runs with or without weather.
    """
    lo_id, hi_id = cfg.location_id_bounds
    pcol = {"yellow": "tpep_pickup_datetime", "green": "lpep_pickup_datetime"}[cfg.service]

    ctes = f"""\
WITH top_zones AS (   -- N highest-volume pickup zones over the window
    SELECT PULocationID AS zone_id
    FROM {trips_expr}
    WHERE PULocationID BETWEEN {lo_id} AND {hi_id}
      AND {pcol} >= TIMESTAMP '{start}'
      AND {pcol} <  TIMESTAMP '{end_excl}'
    GROUP BY zone_id
    ORDER BY count(*) DESC, zone_id
    LIMIT {cfg.top_n_zones}
),
hour_spine AS (   -- every hour in [start, end_excl): half-open, gap-free
    SELECT hour
    FROM range(TIMESTAMP '{start}', TIMESTAMP '{end_excl}', INTERVAL 1 HOUR) t(hour)
),
grid AS (   -- dense zone x hour lattice
    SELECT z.zone_id, s.hour
    FROM top_zones z CROSS JOIN hour_spine s
),
hourly AS (   -- observed pickup counts per zone-hour
    SELECT
        PULocationID               AS zone_id,
        date_trunc('hour', {pcol}) AS hour,
        count(*)                   AS demand
    FROM {trips_expr}
    WHERE PULocationID IN (SELECT zone_id FROM top_zones)
      AND {pcol} >= TIMESTAMP '{start}'
      AND {pcol} <  TIMESTAMP '{end_excl}'
    GROUP BY zone_id, hour
),
panel AS (   -- DENSE, zero-filled target: the prerequisite for gap-free lags
    SELECT g.zone_id, g.hour, COALESCE(h.demand, 0) AS demand
    FROM grid g
    LEFT JOIN hourly h USING (zone_id, hour)
),
featured AS (
    {feature_select(cfg, source="panel")}
)"""

    # Optional weather CTE.
    weather_cte = ""
    weather_join = ""
    weather_cols = ""
    if weather_expr is not None:
        weather_cte = f""",
weather AS (   -- daily Central Park weather, broadcast across each day's 24 hours
    SELECT
        "DATE"::DATE              AS d,
        TRY_CAST("PRCP" AS DOUBLE) AS prcp_mm,
        TRY_CAST("SNOW" AS DOUBLE) AS snow_mm,
        TRY_CAST("TMAX" AS DOUBLE) AS tmax_c,
        TRY_CAST("TMIN" AS DOUBLE) AS tmin_c
    FROM {weather_expr}
)"""
        # CAUSAL join: each row gets the PRIOR day's realized daily weather, which
        # is always fully known before any hour of the current day. Joining the
        # SAME day's summary would leak (TMAX/PRCP are end-of-day aggregates). The
        # `_d1` suffix makes the one-day lag self-documenting.
        weather_join = "\nLEFT JOIN weather w ON f.hour::DATE = w.d + INTERVAL 1 DAY"
        weather_cols = (
            ",\n    w.prcp_mm AS prcp_mm_d1, w.snow_mm AS snow_mm_d1,"
            "\n    w.tmax_c AS tmax_c_d1, w.tmin_c AS tmin_c_d1"
        )

    # Optional holiday flag.
    holiday_col = ""
    if cfg.add_holiday_flag:
        holiday_col = f",\n    {_holiday_in_list(holiday_dates or [])} AS is_holiday"

    final = f"""
SELECT
    f.*{holiday_col}{weather_cols}
FROM featured f{weather_join}
ORDER BY f.zone_id, f.hour"""

    return f"{ctes}{weather_cte}{final}\n"


# --------------------------------------------------------------------------
# Canonical, portable rendering committed to sql/features.sql (documentation)
# --------------------------------------------------------------------------
def render_canonical_sql(cfg: Config, holiday_dates: list[date], start: str, end_excl: str) -> str:
    """Render a portable copy (relative paths, glob) for sql/features.sql."""
    trips = f"read_parquet('data/raw/{cfg.service}_tripdata_*.parquet')"
    weather = "read_csv_auto('data/raw/weather_daily.csv', header=true)"
    header = (
        "-- FleetCast feature engineering — DuckDB window functions (leakage-safe).\n"
        "-- AUTO-RENDERED from src/fleetcast/features/sql.py for the DEFAULT config.\n"
        "-- Lags are backward-only; rolling windows end at '1 PRECEDING' (current row\n"
        "-- EXCLUDED). Regenerate with: uv run python -m fleetcast.features.sql\n\n"
    )
    return header + build_feature_sql(
        cfg,
        trips_expr=trips,
        start=start,
        end_excl=end_excl,
        weather_expr=weather,
        holiday_dates=holiday_dates,
    )


def main() -> None:
    """Regenerate sql/features.sql from the default config."""
    from ..config import load_config
    from ..paths import repo_root
    from .build import holiday_dates_for, panel_window

    cfg = load_config()
    start, end_excl = panel_window(cfg)
    out = repo_root() / "sql" / "features.sql"
    out.write_text(render_canonical_sql(cfg, holiday_dates_for(cfg), start, end_excl))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
