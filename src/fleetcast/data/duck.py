"""Shared DuckDB helpers.

DuckDB is the SQL engine at the heart of FleetCast: it reads the TLC parquet
directly and does all zone-hour aggregation + causal feature engineering with
window functions. Centralising the connection and the service-specific column
names keeps that SQL clean and consistent across modules.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from ..config import Config
from .sources import trip_data_filename

# TLC pickup-datetime column differs by service.
_PICKUP_COL = {"yellow": "tpep_pickup_datetime", "green": "lpep_pickup_datetime"}


def pickup_col(service: str) -> str:
    """Return the pickup-datetime column name for a TLC service."""
    try:
        return _PICKUP_COL[service.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported taxi service: {service!r}") from exc


def connect(threads: int = 4) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with deterministic settings."""
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={int(threads)}")
    # Preserve insertion order for reproducible outputs.
    con.execute("PRAGMA preserve_insertion_order=true")
    return con


def trip_parquet_paths(cfg: Config) -> list[Path]:
    """Local parquet paths for the configured service/months."""
    return [cfg.raw_dir / trip_data_filename(cfg.service, m) for m in cfg.months]


def read_trips_relation(cfg: Config) -> str:
    """A DuckDB ``read_parquet([...])`` expression over the configured months.

    Using an explicit file list (not a glob) means non-contiguous month
    selections work and missing files fail loudly.
    """
    paths = trip_parquet_paths(cfg)
    quoted = ", ".join(f"'{p.as_posix()}'" for p in paths)
    # union_by_name tolerates minor schema drift across monthly files.
    return f"read_parquet([{quoted}], union_by_name=true)"
