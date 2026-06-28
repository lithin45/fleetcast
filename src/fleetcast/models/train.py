"""Train stage: backtest the global LightGBM against the baselines on identical
folds, persist predictions + metrics, and fit the final deployment model.

This is the comprehensive modelling step (``make train``): it produces everything
``make eval`` gates on and the model the dashboard serves.
"""

from __future__ import annotations

import pandas as pd

from ..backtest.forecasters import ArimaForecaster, SeasonalNaiveForecaster
from ..backtest.run import BacktestResult, print_backtest, run_backtest, save_backtest
from ..config import Config, load_config
from ..features.build import load_features
from ..logging import get_logger
from ..seed import seed_everything
from .lightgbm_model import LightGBMForecaster, save_model, train_final_model

log = get_logger(__name__)


def run_train(cfg: Config | None = None) -> BacktestResult:
    """Run the LightGBM-vs-baselines backtest and fit the final model."""
    cfg = cfg or load_config()
    seed_everything(cfg.seed)
    panel: pd.DataFrame = load_features(cfg)

    forecasters = [
        SeasonalNaiveForecaster(),
        ArimaForecaster(cfg),
        LightGBMForecaster(cfg),
    ]
    result = run_backtest(panel, forecasters, cfg)
    save_backtest(result, cfg)
    print_backtest(result)

    # Conformal calibration on the one-step-ahead backtest residuals (Phase 5):
    # writes conformal_metrics.json (the eval coverage gate) + per-zone quantiles.
    from ..conformal.run import run_conformal

    run_conformal(cfg, predictions=result.predictions)

    # Final model fit on ALL data (used by the dashboard).
    model, spec = train_final_model(panel, cfg)
    path = save_model(model, spec, cfg.processed_dir)
    log.info("saved final LightGBM model -> %s (%d features)", path.name, len(spec.features))
    return result


def main() -> None:
    run_train()


if __name__ == "__main__":
    main()
