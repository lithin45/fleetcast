"""DuckDB smoke test: read the raw TLC parquet and report zone-hour panel sizes.

This is the Phase-1 acceptance check. It proves the SQL engine can read the
downloaded parquet directly, identify the top-N pickup zones, and aggregate to
a zone-hour grid — printing both the *observed* (non-empty) zone-hours and the
*dense* (zero-filled) panel size that Phase 2 will materialise.

Run: ``python -m fleetcast.data.smoke`` (wired to ``make data`` after download).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from ..config import Config, load_config
from ..logging import get_logger
from . import duck
from .sources import months_date_range

log = get_logger(__name__)


@dataclass
class SmokeReport:
    """Structured result of the smoke test (asserted on in tests)."""

    total_trips: int
    distinct_pickup_zones: int
    top_zone_ids: list[int]
    n_hours: int
    observed_zone_hours: int
    dense_zone_hours: int


def run_smoke(cfg: Config | None = None) -> SmokeReport:
    """Aggregate raw trips to zone-hours via DuckDB and return a report."""
    cfg = cfg or load_config()
    lo_id, hi_id = cfg.location_id_bounds
    start, end = months_date_range(cfg.months)
    # Half-open upper bound = day after the last day of the range.
    end_excl = end + timedelta(days=1)

    pcol = duck.pickup_col(cfg.service)
    trips = duck.read_trips_relation(cfg)
    con = duck.connect()

    # A reusable "clean trips" CTE: valid zones, timestamps inside the range.
    clean_cte = f"""
        WITH clean AS (
            SELECT
                PULocationID                              AS zone_id,
                date_trunc('hour', {pcol})                AS pickup_hour
            FROM {trips}
            WHERE PULocationID BETWEEN {lo_id} AND {hi_id}
              AND {pcol} >= TIMESTAMP '{start} 00:00:00'
              AND {pcol} <  TIMESTAMP '{end_excl} 00:00:00'
        )
    """

    total_trips = con.execute(f"{clean_cte} SELECT count(*) FROM clean").fetchone()[0]
    distinct_zones = con.execute(
        f"{clean_cte} SELECT count(DISTINCT zone_id) FROM clean"
    ).fetchone()[0]

    top_zone_ids = [
        int(r[0])
        for r in con.execute(
            f"""{clean_cte}
            SELECT zone_id
            FROM clean
            GROUP BY zone_id
            ORDER BY count(*) DESC, zone_id
            LIMIT {cfg.top_n_zones}"""
        ).fetchall()
    ]

    if top_zone_ids:
        observed_zone_hours = con.execute(
            f"""{clean_cte}
            SELECT count(*) FROM (
                SELECT zone_id, pickup_hour
                FROM clean
                WHERE zone_id IN ({", ".join(map(str, top_zone_ids))})
                GROUP BY zone_id, pickup_hour
            )"""
        ).fetchone()[0]
    else:
        # No trips fell in range -> no top zones; avoid an invalid `IN ()`.
        observed_zone_hours = 0

    n_hours = int((end_excl - start).total_seconds() // 3600)
    dense_zone_hours = len(top_zone_ids) * n_hours
    con.close()

    report = SmokeReport(
        total_trips=int(total_trips),
        distinct_pickup_zones=int(distinct_zones),
        top_zone_ids=top_zone_ids,
        n_hours=n_hours,
        observed_zone_hours=int(observed_zone_hours),
        dense_zone_hours=int(dense_zone_hours),
    )
    _print_report(cfg, report)
    return report


def _print_report(cfg: Config, r: SmokeReport) -> None:
    bar = "─" * 64
    print(bar)
    print(f"FleetCast DuckDB smoke test  ·  {cfg.service} taxi  ·  months={','.join(cfg.months)}")
    print(bar)
    print(f"  raw trips (in range)        : {r.total_trips:>14,}")
    print(f"  distinct pickup zones       : {r.distinct_pickup_zones:>14,}")
    print(f"  top-{cfg.top_n_zones} zones (by volume)    : {r.top_zone_ids}")
    print(f"  hours in range              : {r.n_hours:>14,}")
    print(f"  observed zone-hours (top-N) : {r.observed_zone_hours:>14,}")
    print(f"  dense zone-hours (zero-fill): {r.dense_zone_hours:>14,}")
    density = r.observed_zone_hours / r.dense_zone_hours if r.dense_zone_hours else 0.0
    print(f"  panel density               : {density:>13.1%}")
    print(bar)


def main() -> None:
    """Entry point for ``python -m fleetcast.data.smoke``."""
    run_smoke()


if __name__ == "__main__":
    main()
