"""Download the raw public data FleetCast needs.

Every download is *idempotent* (skips if the target already exists unless
``force=True``) and *atomic* (writes to a ``.part`` temp file then renames) so an
interrupted ``make data`` can simply be re-run. Weather is **optional**: any
failure is logged and the function returns ``None`` so the pipeline proceeds
without weather features.

Run as a module: ``python -m fleetcast.data.download`` (wired to ``make data``).
"""

from __future__ import annotations

import io
import zipfile
from datetime import timedelta
from pathlib import Path

import requests

from ..config import Config, load_config
from ..logging import get_logger
from ..paths import ensure_dir
from . import sources

log = get_logger(__name__)

_TIMEOUT = 60  # seconds per request
_CHUNK = 1 << 20  # 1 MiB streaming chunks


# --------------------------------------------------------------------------
# Generic atomic downloader
# --------------------------------------------------------------------------
def download_file(url: str, dest: Path, *, force: bool = False) -> Path:
    """Stream ``url`` to ``dest`` atomically; skip if it already exists."""
    dest = Path(dest)
    if dest.exists() and not force:
        log.info("skip (cached): %s", dest.name)
        return dest
    ensure_dir(dest.parent)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s -> %s", url, dest.name)
    with requests.get(url, stream=True, timeout=_TIMEOUT) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if chunk:
                    fh.write(chunk)
    tmp.replace(dest)
    log.info("done: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


# --------------------------------------------------------------------------
# TLC trip data + zone reference
# --------------------------------------------------------------------------
def download_trip_data(cfg: Config, *, force: bool = False) -> list[Path]:
    """Download one parquet per configured month. Returns the local paths."""
    out: list[Path] = []
    for month in cfg.months:
        url = sources.trip_data_url(cfg.service, month)
        dest = cfg.raw_dir / sources.trip_data_filename(cfg.service, month)
        out.append(download_file(url, dest, force=force))
    return out


def download_zone_lookup(cfg: Config, *, force: bool = False) -> Path:
    """Download the TLC taxi-zone lookup CSV (LocationID -> Borough/Zone)."""
    return download_file(sources.ZONE_LOOKUP_URL, cfg.raw_dir / "taxi_zone_lookup.csv", force=force)


def download_zone_geometry(cfg: Config, *, force: bool = False) -> Path:
    """Download the TLC taxi-zone shapefile and convert it to WGS84 GeoJSON.

    The shapefile ships in EPSG:2263 (NY State Plane, feet); we reproject to
    EPSG:4326 so the geometry drops straight into a web choropleth. The GeoJSON
    is cached; the raw zip is kept too for provenance.
    """
    zip_path = download_file(
        sources.ZONE_SHAPEFILE_URL, cfg.raw_dir / "taxi_zones.zip", force=force
    )
    geojson_path = cfg.raw_dir / "taxi_zones.geojson"
    if geojson_path.exists() and not force:
        log.info("skip (cached): %s", geojson_path.name)
        return geojson_path

    # Imported lazily so the rest of `make data` works even if GDAL/geo stack
    # is unavailable; geometry is only required for the dashboard.
    import geopandas as gpd

    extract_dir = ensure_dir(cfg.raw_dir / "taxi_zones_shp")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    # The TLC zip nests the shapefile under a `taxi_zones/` folder, so search
    # recursively rather than assuming a flat layout.
    try:
        shp = next(extract_dir.rglob("*.shp"))
    except StopIteration as exc:
        raise FileNotFoundError(f"no .shp found under {extract_dir}") from exc
    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    gdf.to_file(geojson_path, driver="GeoJSON")
    log.info("wrote %s (%d zones, WGS84)", geojson_path.name, len(gdf))
    return geojson_path


# --------------------------------------------------------------------------
# Weather (optional — never fail the pipeline)
# --------------------------------------------------------------------------
def _fetch_noaa(cfg: Config) -> str:
    w = cfg.weather
    start, end = sources.months_date_range(cfg.months)
    # Fetch one extra prior day: weather is joined as the PRIOR day's realized
    # summary (causal), so day-1 of the panel needs the day before it.
    start = start - timedelta(days=1)
    params = {
        "dataset": "daily-summaries",
        "stations": w["station_id"],
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "dataTypes": ",".join(w["elements"]),
        "format": "csv",
        "units": "metric",
    }
    resp = requests.get(sources.NCEI_ACCESS_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    if not resp.text.strip() or "DATE" not in resp.text[:200]:
        raise ValueError("NOAA returned an empty/invalid CSV")
    return resp.text


def _fetch_open_meteo(cfg: Config) -> str:
    """Fallback: Open-Meteo hourly archive, aggregated to a NOAA-shaped daily CSV."""
    import csv

    fb = cfg.weather["fallback"]
    start, end = sources.months_date_range(cfg.months)
    start = start - timedelta(days=1)  # prior-day weather is joined causally
    params = {
        "latitude": fb["latitude"],
        "longitude": fb["longitude"],
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum",
        "timezone": "America/New_York",
    }
    resp = requests.get(sources.OPEN_METEO_ARCHIVE_URL, params=params, timeout=_TIMEOUT)
    resp.raise_for_status()
    daily = resp.json()["daily"]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["STATION", "DATE", "PRCP", "SNOW", "TMAX", "TMIN"])
    for i, day in enumerate(daily["time"]):
        writer.writerow(
            [
                "OPEN_METEO",
                day,
                daily["precipitation_sum"][i],
                daily["snowfall_sum"][i],
                daily["temperature_2m_max"][i],
                daily["temperature_2m_min"][i],
            ]
        )
    return buf.getvalue()


def download_weather(cfg: Config, *, force: bool = False) -> Path | None:
    """Download daily Central Park weather; return path or ``None`` on failure.

    Tries NOAA NCEI first, then the Open-Meteo fallback. If both fail (or weather
    is disabled), logs a warning and returns ``None`` — the pipeline must still run.
    """
    if not cfg.weather_enabled:
        log.info("weather disabled in config; skipping")
        return None

    dest = cfg.raw_dir / "weather_daily.csv"
    if dest.exists() and not force:
        log.info("skip (cached): %s", dest.name)
        return dest

    for name, fetch in (("NOAA NCEI", _fetch_noaa), ("Open-Meteo", _fetch_open_meteo)):
        try:
            csv_text = fetch(cfg)
            tmp = dest.with_suffix(".csv.part")
            tmp.write_text(csv_text)
            tmp.replace(dest)
            n_rows = csv_text.count("\n") - 1
            log.info("weather via %s: %d daily rows -> %s", name, n_rows, dest.name)
            return dest
        except Exception as exc:
            log.warning("weather source %s failed: %s", name, exc)

    log.warning("ALL weather sources failed; proceeding WITHOUT weather features")
    return None


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def download_all(cfg: Config | None = None, *, force: bool = False) -> dict[str, object]:
    """Download everything `make data` needs and return a manifest of paths."""
    cfg = cfg or load_config()
    log.info(
        "fetching %s taxi data | months=%s | top_n_zones=%d | config_hash=%s",
        cfg.service,
        ",".join(cfg.months),
        cfg.top_n_zones,
        cfg.config_hash,
    )
    manifest: dict[str, object] = {
        "trip_data": download_trip_data(cfg, force=force),
        "zone_lookup": download_zone_lookup(cfg, force=force),
        "zone_geometry": download_zone_geometry(cfg, force=force),
        "weather": download_weather(cfg, force=force),
    }
    log.info("data download complete")
    return manifest


def main() -> None:
    """Entry point for ``python -m fleetcast.data.download`` / ``make data``."""
    import argparse

    parser = argparse.ArgumentParser(description="Download FleetCast raw data.")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()
    download_all(force=args.force)


if __name__ == "__main__":
    main()
