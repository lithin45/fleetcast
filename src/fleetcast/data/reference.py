"""Loaders for the TLC taxi-zone reference data (lookup table + geometry).

The lookup maps ``LocationID -> Borough / Zone / service_zone`` and is joined to
forecasts so the dashboard can label zones; the GeoJSON provides the choropleth
geometry. Both are downloaded by :mod:`fleetcast.data.download`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config

ZONE_LOOKUP_COLUMNS = ["LocationID", "Borough", "Zone", "service_zone"]


def zone_lookup_path(cfg: Config) -> Path:
    return cfg.raw_dir / "taxi_zone_lookup.csv"


def zone_geojson_path(cfg: Config) -> Path:
    return cfg.raw_dir / "taxi_zones.geojson"


def load_zone_lookup(cfg: Config) -> pd.DataFrame:
    """Load the taxi-zone lookup CSV as a DataFrame with a typed LocationID.

    Raises ``FileNotFoundError`` if it has not been downloaded yet (run
    ``make data`` first).
    """
    path = zone_lookup_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"zone lookup not found: {path} (run `make data`)")
    df = pd.read_csv(path)
    missing = set(ZONE_LOOKUP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"zone lookup missing columns: {sorted(missing)}")
    df["LocationID"] = df["LocationID"].astype(int)
    return df


def zone_names(cfg: Config) -> dict[int, str]:
    """Return a ``{LocationID: "Borough — Zone"}`` mapping for labelling."""
    df = load_zone_lookup(cfg)
    return {int(r.LocationID): f"{r.Borough} — {r.Zone}" for r in df.itertuples(index=False)}
