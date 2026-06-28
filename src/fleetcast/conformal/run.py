"""Run conformal calibration on the backtest predictions and persist the results.

Writes ``conformal_metrics.json`` (the ``make eval`` coverage gate reads
``empirical_coverage``) and ``conformal_quantiles.parquet`` (per-zone deployment
half-widths the dashboard applies to its forecasts).
"""

from __future__ import annotations

import json

import pandas as pd

from ..backtest.run import PREDICTIONS_PARQUET
from ..config import Config, load_config
from ..logging import get_logger
from .split_conformal import conformalize

log = get_logger(__name__)

CONFORMAL_METRICS_JSON = "conformal_metrics.json"
CONFORMAL_QUANTILES_PARQUET = "conformal_quantiles.parquet"


def run_conformal(cfg: Config | None = None, predictions: pd.DataFrame | None = None) -> dict:
    """Calibrate intervals, verify coverage, persist metrics + deployment quantiles."""
    cfg = cfg or load_config()
    if predictions is None:
        predictions = pd.read_parquet(cfg.processed_dir / PREDICTIONS_PARQUET)

    report = conformalize(predictions, cfg)

    # allow_nan=False so a degenerate (inf/nan) report fails loudly rather than
    # round-tripping non-standard `Infinity`/`NaN` literals silently through the gate.
    (cfg.processed_dir / CONFORMAL_METRICS_JSON).write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    qdf = pd.DataFrame(
        [{"zone_id": z, "q90": q} for z, q in report["deployment_quantiles"].items()]
    )
    qdf.to_parquet(cfg.processed_dir / CONFORMAL_QUANTILES_PARQUET, index=False)

    _print_report(report)
    return report


def _print_report(r: dict) -> None:
    bar = "─" * 64
    print(bar)
    print(f"FleetCast conformal intervals  ·  nominal {r['nominal_coverage']:.0%}")
    print(bar)
    print(f"  method               : {r['method']}")
    print(f"  calibration / test   : {r['n_calibration']} / {r['n_test']} points")
    cov = r["empirical_coverage"]
    nominal, tol = r["nominal_coverage"], r["coverage_tolerance"]
    ok = "✅" if abs(cov - nominal) <= tol else "❌"
    print(f"  empirical coverage   : {cov:.4f}  (target {nominal:.2f} +/- {tol:.2f})  {ok}")
    print(f"  mean interval width  : {r['mean_interval_width']:.2f} pickups")
    print(
        f"  per-zone coverage    : min {r['per_zone_coverage_min']:.3f} · "
        f"median {r['per_zone_coverage_median']:.3f} · max {r['per_zone_coverage_max']:.3f} "
        f"({r['n_zones_outside_band']}/{r['n_zones']} zones outside band)"
    )
    print(bar)


def main() -> None:
    run_conformal()


if __name__ == "__main__":
    main()
