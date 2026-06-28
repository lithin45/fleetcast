"""FleetCast Streamlit dashboard, an interactive choropleth of forecast demand.

Pick a date and hour and see predicted NYC taxi pickups per zone (color scale)
with the conformal 90% prediction interval and the realized actual. The forecasts
are a precomputed snapshot (data/forecasts/forecasts.parquet), so the app loads
instantly and runs nothing heavy. Run `make train` then `make demo` locally to
regenerate them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from render import choropleth_geojson  # local module next to this script (Streamlit style)

st.set_page_config(page_title="FleetCast", page_icon="🚕", layout="wide")

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"
FORECASTS = ROOT / "data" / "forecasts" / "forecasts.parquet"
GEOJSON = ROOT / "data" / "raw" / "taxi_zones.geojson"


@st.cache_data(show_spinner=False)
def load_forecasts():
    return pd.read_parquet(FORECASTS) if FORECASTS.is_file() else None


@st.cache_data(show_spinner=False)
def load_geo(zone_ids: tuple[int, ...]):
    """Filtered GeoJSON FeatureCollection + a {LocationID: zone name} map (no GeoPandas)."""
    gj = json.loads(GEOJSON.read_text())
    keep = set(zone_ids)
    feats = [f for f in gj["features"] if int(f["properties"]["LocationID"]) in keep]
    names = {int(f["properties"]["LocationID"]): f["properties"].get("zone", "") for f in feats}
    return {"type": "FeatureCollection", "features": feats}, names


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    out: dict = {}
    for name in ("backtest_metrics.json", "conformal_metrics.json"):
        p = PROCESSED / name
        if p.is_file():
            out[name] = json.loads(p.read_text())
    return out


st.title("🚕 FleetCast, NYC taxi demand forecasting")
st.caption(
    "One hour ahead pickups per taxi zone with distribution free 90% prediction intervals. "
    "Rolling origin backtested; intervals calibrated by split conformal."
)

fc = load_forecasts()
if fc is None:
    st.warning(
        "No precomputed forecasts found. Generate them locally with:\n\n"
        "```\nmake train && make demo\n```"
    )
    st.stop()

st.caption(
    "Read only demo. The forecasts below are a precomputed snapshot from the project's pipeline; "
    "the live training pipeline runs locally from the repo, nothing heavy runs on this page."
)

zones = sorted(int(z) for z in fc["zone_id"].unique())
geojson_all, zone_name = load_geo(tuple(zones))
metrics = load_metrics()

# --- headline metrics ---
bt = metrics.get("backtest_metrics.json", {}).get("metrics", {})
cf = metrics.get("conformal_metrics.json", {})
c1, c2, c3, c4 = st.columns(4)
if "lightgbm" in bt and "seasonal_naive" in bt:
    sn, lg = bt["seasonal_naive"]["pooled"]["wape"], bt["lightgbm"]["pooled"]["wape"]
    c1.metric("LightGBM WAPE", f"{lg:.3f}", f"{(sn - lg) / sn:+.0%} vs seasonal naive")
    c2.metric("Seasonal naive WAPE", f"{sn:.3f}")
if cf:
    c3.metric("Conformal coverage", f"{cf['empirical_coverage']:.1%}", "target 90% +/- 3%")
    c4.metric("Mean interval width", f"{cf['mean_interval_width']:.0f} pickups")

st.divider()

# --- selectors ---
dates = sorted(fc["hour"].dt.date.unique())
left, right = st.columns([2, 1])
with right:
    date = st.selectbox("Date", dates, index=len(dates) // 2, format_func=str)
    hour = st.slider("Hour of day", 0, 23, 18)
ts = pd.Timestamp(date) + pd.Timedelta(hours=hour)
snap = fc[fc["hour"] == ts].set_index("zone_id")
vmax = float(np.quantile(fc["y_pred"], 0.97)) or 1.0

# --- choropleth ---
geojson = choropleth_geojson(geojson_all, snap, vmax)
layer = pdk.Layer(
    "GeoJsonLayer",
    geojson,
    pickable=True,
    stroked=True,
    filled=True,
    get_fill_color="properties.fill_color",
    get_line_color=[60, 60, 60],
    line_width_min_pixels=0.5,
    opacity=0.85,
)
tooltip = {
    "html": "<b>{zone}</b> ({borough})<br/>"
    "Predicted: <b>{predicted}</b> pickups<br/>"
    "90% interval: [{lower}, {upper}]<br/>"
    "Actual: {actual}",
    "style": {"backgroundColor": "#1e1e1e", "color": "white", "fontSize": "12px"},
}
view = pdk.ViewState(latitude=40.758, longitude=-73.978, zoom=10.3, pitch=0)
with left:
    st.markdown(f"#### Predicted pickups, {date} {hour:02d}:00")
    st.pydeck_chart(
        pdk.Deck(layers=[layer], initial_view_state=view, map_style=None, tooltip=tooltip),
        use_container_width=True,
    )
    st.caption(
        "Color = predicted demand (light to dark = low to high). Hover a zone for its interval."
    )

# --- per-zone interval plot ---
st.divider()
default_zone = int(snap["y_pred"].idxmax()) if len(snap) else zones[0]
zsel = st.selectbox(
    "Zone detail, 24h forecast vs actual with the 90% interval",
    zones,
    index=zones.index(default_zone),
    format_func=lambda z: f"{z}, {zone_name.get(z, '')}",
)
day = fc[(fc["zone_id"] == zsel) & (fc["hour"].dt.date == date)].sort_values("hour")

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=day["hour"], y=day["upper"], line={"width": 0}, showlegend=False, hoverinfo="skip")
)
fig.add_trace(
    go.Scatter(
        x=day["hour"],
        y=day["lower"],
        fill="tonexty",
        fillcolor="rgba(253,141,60,0.25)",
        line={"width": 0},
        name="90% interval",
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=day["hour"], y=day["y_pred"], line={"color": "#bd0026", "width": 2}, name="predicted"
    )
)
fig.add_trace(
    go.Scatter(
        x=day["hour"],
        y=day["y_true"],
        mode="markers",
        marker={"color": "#222", "size": 6},
        name="actual",
    )
)
fig.update_layout(
    height=340,
    margin={"l": 10, "r": 10, "t": 30, "b": 10},
    xaxis_title=None,
    yaxis_title="pickups",
    legend={"orientation": "h"},
)
st.plotly_chart(fig, use_container_width=True)
