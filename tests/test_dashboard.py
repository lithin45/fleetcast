"""Dashboard data flow: colormap, choropleth GeoJSON assembly, forecast precompute."""

from __future__ import annotations

import pandas as pd

from fleetcast.backtest.run import run_backtest, save_backtest
from fleetcast.conformal.run import run_conformal
from fleetcast.features.build import build_features, load_features
from fleetcast.models.lightgbm_model import LightGBMForecaster
from fleetcast.ui.precompute import build_forecasts
from fleetcast.ui.render import choropleth_geojson, color_for

MONTHS3 = ["2024-01", "2024-02", "2024-03"]


def test_color_for_scales_and_handles_nan():
    assert color_for(0.0, 100.0) == [255, 255, 204]  # low end
    assert color_for(100.0, 100.0) == [189, 0, 38]  # high end
    mid = color_for(50.0, 100.0)
    assert mid == [253, 141, 60]
    assert color_for(float("nan"), 100.0) == [255, 255, 204]  # NaN -> low color
    assert color_for(10.0, 0.0) == [255, 255, 204]  # degenerate vmax


def test_choropleth_geojson_attaches_properties():
    def _feat(loc, zone):
        return {
            "type": "Feature",
            "properties": {"LocationID": loc, "zone": zone, "borough": "X"},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        }

    geojson = {"type": "FeatureCollection", "features": [_feat(1, "A"), _feat(2, "B")]}
    snap = pd.DataFrame(
        {
            "zone_id": [1, 2],
            "y_pred": [10.0, 50],
            "lower": [5.0, 40],
            "upper": [15.0, 60],
            "y_true": [12.0, 55],
        }
    ).set_index("zone_id")

    gj = choropleth_geojson(geojson, snap, vmax=50.0)
    assert gj["type"] == "FeatureCollection" and len(gj["features"]) == 2
    props = {f["properties"]["LocationID"]: f["properties"] for f in gj["features"]}
    assert props[1]["predicted"] == 10.0 and props[2]["upper"] == 60.0
    assert len(props[1]["fill_color"]) == 3  # [r,g,b]
    # original geometry preserved, source dict not mutated
    assert geojson["features"][0]["properties"].get("fill_color") is None


def test_build_forecasts_from_synthetic_pipeline(make_synthetic_env):
    env = make_synthetic_env(months=MONTHS3, n_zones=4)
    env.cfg.raw["backtest"]["n_folds"] = 120
    build_features(env.cfg)
    panel = load_features(env.cfg)
    res = run_backtest(panel, [LightGBMForecaster(env.cfg)], env.cfg)
    save_backtest(res, env.cfg)
    run_conformal(env.cfg, predictions=res.predictions)

    path = build_forecasts(env.cfg, force=True)
    df = pd.read_parquet(path)
    assert {"zone_id", "hour", "y_true", "y_pred", "lower", "upper", "covered"} <= set(df.columns)
    assert (df["lower"] <= df["upper"]).all()
    assert (df["lower"] >= 0).all()
    assert df["zone_id"].nunique() == 4
