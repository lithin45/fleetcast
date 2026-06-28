"""Run the rolling-origin backtest and report per-model metrics.

Loops folds x forecasters, collecting predictions on **identical folds**, then
summarizes WAPE/MAE/RMSE (and high-volume MAPE) per model. Predictions and the
metrics summary are persisted for the dashboard and the eval gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import Config, load_config
from ..features.build import load_features
from ..logging import get_logger
from . import metrics as M
from .folds import Fold, forecast_slice, make_folds
from .forecasters import Forecaster, default_baselines

log = get_logger(__name__)

PREDICTIONS_PARQUET = "backtest_predictions.parquet"
METRICS_JSON = "backtest_metrics.json"


@dataclass
class BacktestResult:
    predictions: pd.DataFrame  # model, fold, zone_id, hour, y_true, y_pred
    metrics: dict
    folds: list[Fold]
    high_volume_zones: list[int] = field(default_factory=list)


def _validated_merge(
    truth: pd.DataFrame, preds: pd.DataFrame, model: str, fold_index: int
) -> pd.DataFrame:
    """Left-merge predictions onto truth, failing loudly if a forecaster returns
    duplicate or missing (zone_id, hour) keys (which would silently bias or NaN-out
    the metrics). Enforces 'every test zone-hour matched exactly once'."""
    try:
        merged = truth.merge(preds, on=["zone_id", "hour"], how="left", validate="one_to_one")
    except Exception as exc:
        raise ValueError(
            f"forecaster {model!r} returned duplicate (zone_id, hour) keys in fold {fold_index}"
        ) from exc
    if len(merged) != len(truth) or merged["y_pred"].isna().any():
        raise ValueError(
            f"forecaster {model!r} did not predict every test zone-hour in fold {fold_index} "
            f"(missing {int(merged['y_pred'].isna().sum())} of {len(truth)})"
        )
    return merged


def run_backtest(
    panel: pd.DataFrame,
    forecasters: list[Forecaster],
    cfg: Config,
) -> BacktestResult:
    """Evaluate every forecaster on the same rolling-origin folds."""
    panel = panel.sort_values(["zone_id", "hour"]).reset_index(drop=True)
    folds = make_folds(panel["hour"], cfg)
    log.info(
        "backtest: %d folds, horizon=%dh, step=%dh, %d zones, %d forecasters",
        len(folds),
        folds[0].horizon_hours,
        int(cfg.backtest["step_hours"]),
        panel["zone_id"].nunique(),
        len(forecasters),
    )

    high_vol = M.high_volume_zones(panel, float(cfg.eval["high_volume_zone_quantile"]))

    frames: list[pd.DataFrame] = []
    try:
        for fold in folds:
            truth = forecast_slice(panel, fold)[["zone_id", "hour", "demand"]].rename(
                columns={"demand": "y_true"}
            )
            # Target masking: hide actual demand at/after the origin so a forecaster
            # CANNOT see the values it is being scored on. Lag/rolling feature columns
            # (causal, pre-computed) are untouched; only the raw target is masked.
            masked = panel.copy()
            masked.loc[masked["hour"] >= fold.test_start, "demand"] = float("nan")

            for fc in forecasters:
                preds = fc.predict_fold(masked, fold)
                merged = _validated_merge(truth, preds, fc.name, fold.index)
                merged["model"] = fc.name
                merged["fold"] = fold.index
                frames.append(merged)
            log.info("fold %d/%d done (cutoff=%s)", fold.index + 1, len(folds), fold.train_end)
    finally:
        for fc in forecasters:
            if hasattr(fc, "close"):
                fc.close()

    predictions = pd.concat(frames, ignore_index=True)
    metrics = {name: M.summarize_model(g, high_vol) for name, g in predictions.groupby("model")}
    return BacktestResult(predictions, metrics, folds, high_vol)


def save_backtest(result: BacktestResult, cfg: Config) -> tuple[Path, Path]:
    """Persist predictions (parquet) and the metrics summary (json)."""
    pred_path = cfg.processed_dir / PREDICTIONS_PARQUET
    metrics_path = cfg.processed_dir / METRICS_JSON
    result.predictions.to_parquet(pred_path, index=False)
    payload = {
        "config_hash": cfg.config_hash,
        "n_folds": len(result.folds),
        "horizon_hours": result.folds[0].horizon_hours,
        "high_volume_zones": result.high_volume_zones,
        "folds": [
            {
                "index": f.index,
                "train_end": str(f.train_end),
                "test_start": str(f.test_start),
                "test_end": str(f.test_end),
            }
            for f in result.folds
        ],
        "metrics": result.metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, default=str))
    return pred_path, metrics_path


def print_backtest(result: BacktestResult) -> None:
    bar = "─" * 72
    print(bar)
    print(
        f"FleetCast rolling-origin backtest  ·  {len(result.folds)} folds  ·  "
        f"horizon {result.folds[0].horizon_hours}h"
    )
    print(bar)
    print(f"  {'model':<16}{'WAPE':>9}{'MAE':>9}{'RMSE':>9}{'MAPE(hi-vol)':>15}")
    print("  " + "-" * 56)
    for name, m in result.metrics.items():
        p = m["pooled"]
        mape_hv = p.get("mape_high_volume", float("nan"))
        print(f"  {name:<16}{p['wape']:>9.4f}{p['mae']:>9.3f}{p['rmse']:>9.3f}{mape_hv:>15.4f}")
    print(bar)


def run_baselines(cfg: Config | None = None) -> BacktestResult:
    """Load features, run the baseline backtest, persist + print results."""
    cfg = cfg or load_config()
    panel = load_features(cfg)
    result = run_backtest(panel, default_baselines(cfg), cfg)
    save_backtest(result, cfg)
    print_backtest(result)
    return result


def main() -> None:
    """Entry point for ``python -m fleetcast.backtest.run`` / ``make backtest``."""
    run_baselines()


if __name__ == "__main__":
    main()
