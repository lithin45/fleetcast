"""Download layer: idempotency, the weather-is-optional guarantee, and real fetches."""

from __future__ import annotations

import pytest

from fleetcast.data import download


def test_download_file_skips_when_cached(tmp_path, monkeypatch):
    dest = tmp_path / "already.bin"
    dest.write_bytes(b"cached")

    def boom(*a, **k):  # would be called only if the cache check failed
        raise AssertionError("network was hit despite cache")

    monkeypatch.setattr(download.requests, "get", boom)
    out = download.download_file("http://example.invalid/x", dest)
    assert out == dest
    assert dest.read_bytes() == b"cached"


def test_weather_disabled_returns_none(make_synthetic_env):
    env = make_synthetic_env(weather_enabled=False)
    assert download.download_weather(env.cfg) is None


def test_weather_returns_none_when_all_sources_fail(make_synthetic_env, monkeypatch):
    """The core guarantee: weather failures must never raise."""
    env = make_synthetic_env(weather_enabled=True)

    def fail(_cfg):
        raise RuntimeError("simulated outage")

    monkeypatch.setattr(download, "_fetch_noaa", fail)
    monkeypatch.setattr(download, "_fetch_open_meteo", fail)
    assert download.download_weather(env.cfg) is None  # logged, not raised


def test_weather_falls_back_to_open_meteo(make_synthetic_env, monkeypatch):
    env = make_synthetic_env(weather_enabled=True)
    monkeypatch.setattr(download, "_fetch_noaa", lambda _c: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(
        download,
        "_fetch_open_meteo",
        lambda _c: "STATION,DATE,PRCP,SNOW,TMAX,TMIN\nOM,2024-01-01,0,0,5,1\n",
    )
    out = download.download_weather(env.cfg)
    assert out is not None and out.exists()
    assert "OM,2024-01-01" in out.read_text()


# --------------------------------------------------------------------------
# Real network fetches (skipped when FLEETCAST_OFFLINE=1)
# --------------------------------------------------------------------------
@pytest.mark.network
def test_download_real_zone_lookup(make_synthetic_env):
    import csv

    env = make_synthetic_env()
    path = download.download_zone_lookup(env.cfg, force=True)
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))  # header is quoted in the real CSV
    assert "LocationID" in rows[0]
    loc_ids = {int(r["LocationID"]) for r in rows}
    assert len(loc_ids) >= 260  # 263 official taxi zones (+ EWR/unknown codes)
    assert 1 in loc_ids


@pytest.mark.network
@pytest.mark.slow
def test_download_real_zone_geometry_is_wgs84(make_synthetic_env):
    import geopandas as gpd

    env = make_synthetic_env()
    geojson = download.download_zone_geometry(env.cfg, force=True)
    gdf = gpd.read_file(geojson)
    assert len(gdf) >= 260
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326


@pytest.mark.network
def test_download_real_weather(make_synthetic_env):
    env = make_synthetic_env(months=["2024-01"], weather_enabled=True)
    path = download.download_weather(env.cfg, force=True)
    assert path is not None
    header = path.read_text().splitlines()[0]
    assert "PRCP" in header and "TMAX" in header
