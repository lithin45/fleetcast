"""Generate static SVG visuals for the README (no extra deps, no matplotlib).

Renders, from the precomputed forecasts + zone geometry:
* ``docs/choropleth.svg`` — predicted pickups per zone at a representative hour,
* ``docs/interval_plot.svg`` — a zone's 24h forecast with its 90% interval + actuals.

Run: ``python -m fleetcast.ui.static_viz``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from ..config import Config, load_config
from ..logging import get_logger
from ..paths import repo_root
from .precompute import build_forecasts, forecasts_path
from .render import color_for

log = get_logger(__name__)


def _rgb(c: list[int]) -> str:
    return f"rgb({c[0]},{c[1]},{c[2]})"


def _poly_paths(geom, project) -> list[str]:
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    out = []
    for p in polys:
        pts = [project(x, y) for x, y in p.exterior.coords]
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"
        out.append(d)
    return out


def render_choropleth(cfg: Config, when: pd.Timestamp | None = None) -> str:
    import geopandas as gpd

    fc = pd.read_parquet(forecasts_path(cfg))
    zones = sorted(int(z) for z in fc["zone_id"].unique())
    gdf = gpd.read_file(repo_root() / "data" / "raw" / "taxi_zones.geojson")
    gdf["LocationID"] = gdf["LocationID"].astype(int)
    gdf = gdf[gdf["LocationID"].isin(zones)].to_crs(4326)

    when = when or pd.Timestamp(
        sorted(fc["hour"].dt.date.unique())[len(fc["hour"].dt.date.unique()) // 2]
    ) + pd.Timedelta(hours=18)
    snap = fc[fc["hour"] == when].set_index("zone_id")
    vmax = float(fc["y_pred"].quantile(0.97)) or 1.0

    minx, miny, maxx, maxy = gdf.total_bounds
    W, H, pad = 720, 470, 24
    midlat = math.radians((miny + maxy) / 2)
    sx = (W - 2 * pad) / ((maxx - minx) * math.cos(midlat))
    sy = (H - 2 * pad) / (maxy - miny)
    s = min(sx, sy)
    xoff = pad + ((W - 2 * pad) - (maxx - minx) * math.cos(midlat) * s) / 2
    yoff = pad

    def project(lon, lat):
        x = xoff + (lon - minx) * math.cos(midlat) * s
        y = yoff + (maxy - lat) * s
        return x, y

    paths = []
    for r in gdf.itertuples():
        pred = snap["y_pred"].get(int(r.LocationID), float("nan"))
        fill = _rgb(color_for(pred, vmax))
        title = f"{r.zone}: {pred:.0f} pickups" if pred == pred else r.zone
        for d in _poly_paths(r.geometry, project):
            paths.append(
                f'<path d="{d}" fill="{fill}" stroke="#33373d" stroke-width="0.6">'
                f"<title>{title}</title></path>"
            )

    # color legend
    legend = []
    lx, ly, lw = W - 180, H - 30, 150
    for i in range(lw):
        legend.append(
            f'<rect x="{lx + i}" y="{ly}" width="1" height="10" fill="{_rgb(color_for(i / lw * vmax, vmax))}"/>'
        )
    legend_txt = (
        f'<text x="{lx}" y="{ly - 4}" font-size="11" fill="#aab">0</text>'
        f'<text x="{lx + lw}" y="{ly - 4}" font-size="11" fill="#aab" text-anchor="end">{vmax:.0f}+ pickups</text>'
    )

    return (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">'
        f'<rect width="{W}" height="{H}" fill="#15181c" rx="8"/>'
        f'<text x="{pad}" y="26" font-size="16" fill="#fff" font-weight="bold">'
        f"FleetCast: predicted pickups per zone, {when:%Y-%m-%d %H:00}</text>"
        + "".join(paths)
        + "".join(legend)
        + legend_txt
        + "</svg>"
    )


def render_interval_plot(cfg: Config, zone: int | None = None, date=None) -> str:
    fc = pd.read_parquet(forecasts_path(cfg))
    if zone is None:
        zone = int(fc.groupby("zone_id")["y_pred"].mean().idxmax())  # busiest zone
    if date is None:
        date = sorted(fc["hour"].dt.date.unique())[len(fc["hour"].dt.date.unique()) // 2]
    day = fc[(fc["zone_id"] == zone) & (fc["hour"].dt.date == date)].sort_values("hour")
    if day.empty:
        day = fc[fc["zone_id"] == zone].sort_values("hour").head(24)

    W, H, pad = 720, 320, 44
    ymax = float(max(day["upper"].max(), day["y_true"].max())) * 1.1 or 1.0
    n = len(day)

    def px(i):
        return pad + i / max(n - 1, 1) * (W - 2 * pad)

    def py(v):
        return H - pad - (v / ymax) * (H - 2 * pad)

    upper = [(px(i), py(v)) for i, v in enumerate(day["upper"])]
    lower = [(px(i), py(v)) for i, v in enumerate(day["lower"])]
    band = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower[::-1]) + " Z"
    pred = "M" + " L".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(day["y_pred"]))
    dots = "".join(
        f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="2.6" fill="#111"/>'
        for i, v in enumerate(day["y_true"])
    )
    grid = "".join(
        f'<line x1="{pad}" y1="{py(g)}" x2="{W - pad}" y2="{py(g)}" stroke="#e6e6e6" stroke-width="1"/>'
        f'<text x="{pad - 6}" y="{py(g) + 4}" font-size="10" fill="#888" text-anchor="end">{g:.0f}</text>'
        for g in [0, ymax / 2, ymax]
    )
    hours = "".join(
        f'<text x="{px(i)}" y="{H - pad + 16}" font-size="10" fill="#888" text-anchor="middle">{day["hour"].dt.hour.iloc[i]:02d}</text>'
        for i in range(0, n, max(1, n // 6))
    )
    cov = day["covered"].mean()
    return (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">'
        f'<rect width="{W}" height="{H}" fill="#ffffff" rx="8"/>'
        f'<text x="{pad}" y="24" font-size="15" fill="#222" font-weight="bold">'
        f"Zone {zone}, {date}: forecast, 90% interval, and actuals "
        f'<tspan fill="#2a7">({cov:.0%} covered)</tspan></text>'
        + grid
        + f'<path d="{band}" fill="rgba(253,141,60,0.30)" stroke="none"/>'
        + f'<path d="{pred}" fill="none" stroke="#bd0026" stroke-width="2.2"/>'
        + dots
        + hours
        + f'<text x="{pad}" y="{H - 8}" font-size="11" fill="#888">hour of day, '
        f"red = predicted, band = 90% interval, dots = actual</text></svg>"
    )


def generate_readme_visuals(cfg: Config | None = None) -> list[Path]:
    cfg = cfg or load_config()
    build_forecasts(cfg)
    docs = repo_root() / "docs"
    docs.mkdir(exist_ok=True)
    outputs = []
    for name, svg in (
        ("choropleth.svg", render_choropleth(cfg)),
        ("interval_plot.svg", render_interval_plot(cfg)),
    ):
        path = docs / name
        path.write_text(svg)
        outputs.append(path)
        log.info("wrote %s (%d bytes)", path.name, len(svg))
    return outputs


def main() -> None:
    generate_readme_visuals()


if __name__ == "__main__":
    main()
