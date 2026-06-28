"""Deterministic synthetic TLC-shaped data for offline tests and CI.

Real TLC parquet is ~50 MB/month and needs network, so the test-suite and the CI
eval gate run against a small, *deterministic* synthetic dataset that mimics the
TLC schema and — crucially — embeds real structure the models must learn:

* a smooth **diurnal** profile (rush-hour peaks),
* a **weekly** profile (weekday vs weekend),
* per-zone base levels, and
* **AR(1)** autocorrelation so lag features beat same-hour-last-week.

This is what lets a small CI run still demonstrate "LightGBM beats seasonal-naive
by >=20% WAPE" without downloading anything.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fleetcast.data.sources import month_bounds, trip_data_filename

# Columns we emit — a faithful subset of the TLC yellow schema.
_DOW_PROFILE = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 0.85, 0.7])  # Mon..Sun


def _diurnal(hours: np.ndarray) -> np.ndarray:
    """Two-bump diurnal demand multiplier (morning + evening rush)."""
    morning = np.exp(-0.5 * ((hours - 8) / 2.2) ** 2)
    evening = np.exp(-0.5 * ((hours - 18) / 2.6) ** 2)
    base = 0.25 + 1.4 * morning + 1.7 * evening
    return base


def synthesize_counts(
    zones: list[int],
    start: datetime,
    end_excl: datetime,
    seed: int = 1234,
) -> dict[tuple[int, datetime], int]:
    """Return a dict {(zone, hour): count} with diurnal/weekly/AR(1) structure."""
    rng = np.random.default_rng(seed)
    n_hours = int((end_excl - start).total_seconds() // 3600)
    timeline = [start + timedelta(hours=h) for h in range(n_hours)]
    hours = np.array([t.hour for t in timeline])
    dows = np.array([t.weekday() for t in timeline])
    diurnal = _diurnal(hours)
    weekly = _DOW_PROFILE[dows]

    counts: dict[tuple[int, datetime], int] = {}
    for zi, zone in enumerate(zones):
        base_level = 6.0 * (1.0 + 0.7 * zi / max(len(zones) - 1, 1))  # zone heterogeneity
        mu = base_level * diurnal * weekly
        # AR(1) multiplicative noise so lag-1 carries real information.
        ar = np.zeros(n_hours)
        eps = rng.normal(0.0, 0.25, size=n_hours)
        for t in range(1, n_hours):
            ar[t] = 0.6 * ar[t - 1] + eps[t]
        rate = np.clip(mu * np.exp(ar), 0.01, None)
        draws = rng.poisson(rate)
        for t, c in enumerate(draws):
            if c:
                counts[(zone, timeline[t])] = int(c)
    return counts


def write_synthetic_month(
    raw_dir: Path,
    service: str,
    month: str,
    zones: list[int],
    seed: int = 1234,
) -> Path:
    """Write one synthetic monthly parquet in TLC-yellow shape; return its path."""
    first, last = month_bounds(month)
    start = datetime(first.year, first.month, first.day)
    end_excl = datetime(last.year, last.month, last.day) + timedelta(days=1)
    # Per-month seed offset keeps months independent but reproducible.
    counts = synthesize_counts(zones, start, end_excl, seed=seed + int(month.replace("-", "")))

    rng = np.random.default_rng(seed + 7)
    pickups: list[datetime] = []
    pulocs: list[int] = []
    for (zone, hour), c in counts.items():
        minutes = rng.integers(0, 60, size=c)
        seconds = rng.integers(0, 60, size=c)
        for m, s in zip(minutes, seconds, strict=True):
            pickups.append(hour + timedelta(minutes=int(m), seconds=int(s)))
            pulocs.append(zone)

    n = len(pickups)
    pickup_col = "tpep_pickup_datetime" if service == "yellow" else "lpep_pickup_datetime"
    dropoff_col = "tpep_dropoff_datetime" if service == "yellow" else "lpep_dropoff_datetime"
    table = pa.table(
        {
            pickup_col: pa.array(pickups, type=pa.timestamp("us")),
            dropoff_col: pa.array(
                [p + timedelta(minutes=12) for p in pickups], type=pa.timestamp("us")
            ),
            "PULocationID": pa.array(pulocs, type=pa.int32()),
            "DOLocationID": pa.array(rng.integers(1, 264, size=n).tolist(), type=pa.int32()),
            "passenger_count": pa.array(rng.integers(1, 4, size=n).tolist(), type=pa.int64()),
            "trip_distance": pa.array(rng.uniform(0.5, 8.0, size=n).round(2).tolist()),
        }
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / trip_data_filename(service, month)
    pq.write_table(table, dest)
    return dest


def write_synthetic_dataset(
    raw_dir: Path,
    service: str,
    months: list[str],
    zones: list[int],
    seed: int = 1234,
) -> list[Path]:
    """Write a synthetic parquet for each month; return the paths."""
    return [write_synthetic_month(raw_dir, service, m, zones, seed=seed) for m in months]


def write_synthetic_weather(raw_dir: Path, months: list[str], seed: int = 1234) -> Path:
    """Write a NOAA-shaped daily weather CSV spanning the given months."""
    import csv

    from fleetcast.data.sources import months_date_range

    rng = np.random.default_rng(seed + 99)
    start, end = months_date_range(months)
    start = start - timedelta(days=1)  # prior day, joined causally to panel day-1
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "weather_daily.csv"
    n_days = (end - start).days + 1
    with dest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["STATION", "DATE", "PRCP", "SNOW", "TMAX", "TMIN"])
        for i in range(n_days):
            d = start + timedelta(days=i)
            tmax = round(float(rng.uniform(-2, 30)), 1)
            w.writerow(
                [
                    "USW00094728",
                    d.isoformat(),
                    round(float(rng.exponential(2.0)), 1),  # PRCP mm
                    0.0,
                    tmax,
                    round(tmax - float(rng.uniform(3, 9)), 1),  # TMIN < TMAX
                ]
            )
    return dest


def write_synthetic_zone_lookup(raw_dir: Path, n_zones: int = 263) -> Path:
    """Write a minimal taxi_zone_lookup.csv covering LocationID 1..n_zones."""
    import csv

    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "taxi_zone_lookup.csv"
    with dest.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["LocationID", "Borough", "Zone", "service_zone"])
        for i in range(1, n_zones + 1):
            w.writerow([i, "Manhattan", f"Zone {i}", "Yellow Zone"])
    return dest
