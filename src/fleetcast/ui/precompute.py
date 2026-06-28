"""Precompute the per-zone-hour forecast table the dashboard serves.

Combines the LightGBM one-step-ahead backtest predictions (the held-out display
window, with actuals to compare against) with the per-zone conformal half-widths
into a single ``forecasts.parquet`` so the Streamlit app loads instantly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.run import PREDICTIONS_PARQUET
from ..config import Config, load_config
from ..conformal.run import CONFORMAL_QUANTILES_PARQUET
from ..logging import get_logger

log = get_logger(__name__)

FORECASTS_PARQUET = "forecasts.parquet"


def forecasts_path(cfg: Config) -> Path:
    return cfg.forecasts_dir / FORECASTS_PARQUET


def build_forecasts(cfg: Config | None = None, force: bool = False) -> Path:
    """Write ``data/forecasts/forecasts.parquet`` from the train-stage artifacts."""
    cfg = cfg or load_config()
    out = forecasts_path(cfg)
    if out.is_file() and not force:
        log.info("forecasts already built: %s", out.name)
        return out

    pred_path = cfg.processed_dir / PREDICTIONS_PARQUET
    q_path = cfg.processed_dir / CONFORMAL_QUANTILES_PARQUET
    for p in (pred_path, q_path):
        if not p.is_file():
            raise FileNotFoundError(f"{p} missing — run `make train` before `make demo`")

    preds = pd.read_parquet(pred_path)
    lgbm = preds[preds["model"] == "lightgbm"][["zone_id", "hour", "y_true", "y_pred"]].copy()
    q90 = pd.read_parquet(q_path).set_index("zone_id")["q90"]

    lgbm["q90"] = lgbm["zone_id"].map(q90)
    lgbm["lower"] = np.clip(lgbm["y_pred"] - lgbm["q90"], 0.0, None)
    lgbm["upper"] = lgbm["y_pred"] + lgbm["q90"]
    lgbm["covered"] = (lgbm["y_true"] >= lgbm["lower"]) & (lgbm["y_true"] <= lgbm["upper"])
    lgbm = lgbm.sort_values(["hour", "zone_id"]).reset_index(drop=True)

    lgbm.to_parquet(out, index=False)
    log.info(
        "wrote %s: %d zone-hours, %d zones, %s -> %s",
        out.name,
        len(lgbm),
        lgbm["zone_id"].nunique(),
        lgbm["hour"].min(),
        lgbm["hour"].max(),
    )
    return out


def main() -> None:
    build_forecasts(force=True)


if __name__ == "__main__":
    main()
