"""Canonical URLs and small helpers for the public data sources FleetCast uses.

Keeping every external URL in one module makes the data provenance auditable and
the download layer trivial to mock in tests. All sources are public and keyless:

* NYC TLC Trip Record Data (Parquet, monthly)            — CloudFront mirror
* TLC taxi-zone lookup (CSV) + zone geometry (Shapefile) — CloudFront mirror
* NOAA NCEI GHCN-Daily "daily-summaries" access service  — keyless, metric CSV
* Open-Meteo historical archive (ERA5)                   — keyless fallback
"""

from __future__ import annotations

from calendar import monthrange
from datetime import date

# TLC CloudFront mirror (the canonical download host linked from nyc.gov/tlc).
TLC_BASE = "https://d37ci6vzurychx.cloudfront.net"
ZONE_LOOKUP_URL = f"{TLC_BASE}/misc/taxi_zone_lookup.csv"
ZONE_SHAPEFILE_URL = f"{TLC_BASE}/misc/taxi_zones.zip"

# NOAA NCEI access service: returns a tiny, date-filtered, metric CSV.
NCEI_ACCESS_URL = "https://www.ncei.noaa.gov/access/services/data/v1"

# Open-Meteo historical reanalysis archive (keyless fallback).
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def trip_data_url(service: str, month: str) -> str:
    """URL for one monthly TLC parquet, e.g. yellow_tripdata_2024-01.parquet."""
    return f"{TLC_BASE}/trip-data/{service}_tripdata_{month}.parquet"


def trip_data_filename(service: str, month: str) -> str:
    """Local filename for a monthly TLC parquet."""
    return f"{service}_tripdata_{month}.parquet"


def month_bounds(month: str) -> tuple[date, date]:
    """Return (first_day, last_day) for a 'YYYY-MM' month string, inclusive."""
    year, mon = (int(x) for x in month.split("-"))
    last = monthrange(year, mon)[1]
    return date(year, mon, 1), date(year, mon, last)


def months_date_range(months: list[str]) -> tuple[date, date]:
    """Return the inclusive (start, end) date spanning a list of months."""
    starts, ends = zip(*(month_bounds(m) for m in sorted(months)), strict=True)
    return min(starts), max(ends)
