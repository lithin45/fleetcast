"""Pure (Streamlit-free) rendering helpers for the dashboard — importable & testable.

Keeps the data→visual transforms (colormap, choropleth GeoJSON assembly) out of
``app.py`` so they can be unit-tested without a Streamlit runtime.
"""

from __future__ import annotations

from itertools import pairwise

import pandas as pd

# YlOrRd-ish sequential color anchors for the choropleth.
_ANCHORS = [(0.0, (255, 255, 204)), (0.5, (253, 141, 60)), (1.0, (189, 0, 38))]


def color_for(value: float, vmax: float) -> list[int]:
    """Map a demand value to an ``[r, g, b]`` color on the sequential scale."""
    if value != value or vmax <= 0:  # NaN or degenerate
        return list(_ANCHORS[0][1])
    t = min(1.0, max(0.0, float(value) / vmax))
    for (t0, c0), (t1, c1) in pairwise(_ANCHORS):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return [round(c0[i] + f * (c1[i] - c0[i])) for i in range(3)]
    return list(_ANCHORS[-1][1])


def _round1(v: float):
    return round(float(v), 1) if v == v else None  # None for NaN (JSON-safe)


def choropleth_geojson(geojson: dict, snap: pd.DataFrame, vmax: float) -> dict:
    """Inject the per-zone snapshot (predicted/interval/actual + an ``[r,g,b]`` fill
    color) into a GeoJSON ``FeatureCollection`` whose features carry a ``LocationID``
    property. Pure dict-in/dict-out, so the dashboard needs no GeoPandas/GDAL."""
    features = []
    for f in geojson["features"]:
        props = dict(f["properties"])
        zid = int(props["LocationID"])
        pred = float(snap["y_pred"].get(zid, float("nan")))
        props["predicted"] = _round1(pred)
        props["lower"] = _round1(snap["lower"].get(zid, float("nan")))
        props["upper"] = _round1(snap["upper"].get(zid, float("nan")))
        props["actual"] = _round1(snap["y_true"].get(zid, float("nan")))
        props["fill_color"] = color_for(pred, vmax)
        features.append({**f, "properties": props})
    return {**geojson, "features": features}
