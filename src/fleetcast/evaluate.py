"""The eval gate (``make eval``).

Reads the backtest metrics (and, from Phase 5, the conformal coverage), prints a
clean per-metric, per-model table, and **exits non-zero** if a hard gate is
missed — the WAPE improvement over seasonal-naive (>= 20%) and, once conformal
lands, the 90% +/- 3% coverage. MAPE on high-volume zones is reported (the spec
treats it as a reported metric, not a hard gate, since low-volume zones are noisy).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .backtest.run import METRICS_JSON
from .config import Config, load_config
from .logging import get_logger

log = get_logger(__name__)

CONFORMAL_JSON = "conformal_metrics.json"


@dataclass
class GateResult:
    name: str
    value: float
    threshold: float
    passed: bool
    detail: str


def _load(cfg: Config) -> dict:
    path = cfg.processed_dir / METRICS_JSON
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found — run `make train` first")
    return json.loads(path.read_text())


def evaluate(cfg: Config | None = None) -> tuple[int, list[GateResult]]:
    """Print the eval table + enforce gates. Returns (exit_code, gates)."""
    cfg = cfg or load_config()
    payload = _load(cfg)
    metrics = payload["metrics"]
    gates: list[GateResult] = []

    _print_table(payload, metrics)

    # --- Hard gate 1: WAPE improvement over seasonal-naive (MANDATORY) ---
    target = float(cfg.eval["wape_improvement_over_seasonal_naive"])
    if {"seasonal_naive", "lightgbm"} <= metrics.keys():
        sn = metrics["seasonal_naive"]["pooled"]["wape"]
        lgbm = metrics["lightgbm"]["pooled"]["wape"]
        improvement = (sn - lgbm) / sn if sn else float("nan")
        gates.append(
            GateResult(
                "WAPE improvement vs seasonal-naive",
                improvement,
                target,
                improvement >= target,
                f"seasonal_naive={sn:.4f} -> lightgbm={lgbm:.4f} ({improvement:+.1%})",
            )
        )
    else:
        # The gate cannot be skipped silently — a missing model fails the build.
        missing = {"seasonal_naive", "lightgbm"} - metrics.keys()
        gates.append(
            GateResult(
                "WAPE improvement vs seasonal-naive",
                float("nan"),
                target,
                False,
                f"missing metrics for {sorted(missing)} — run `make train`",
            )
        )

    # --- Hard gate 2: conformal coverage (Phase 5, if available) ---
    conf_path = cfg.processed_dir / CONFORMAL_JSON
    if conf_path.is_file():
        conf = json.loads(conf_path.read_text())
        nominal = float(cfg.conformal["nominal_coverage"])
        tol = float(cfg.conformal["coverage_tolerance"])
        cov = float(conf["empirical_coverage"])
        gates.append(
            GateResult(
                "Conformal coverage @ 90%",
                cov,
                nominal,
                abs(cov - nominal) <= tol,
                f"empirical={cov:.4f} (target {nominal:.2f} +/- {tol:.2f})",
            )
        )

    # --- Reported (not gated): MAPE on high-volume zones ---
    if "lightgbm" in metrics:
        mape_hv = metrics["lightgbm"]["pooled"].get("mape_high_volume", float("nan"))
        log.info(
            "reported: LightGBM MAPE (high-volume) = %.4f (target < %.2f, not a hard gate)",
            mape_hv,
            float(cfg.eval["mape_high_volume_max"]),
        )

    _print_gates(gates)
    # No gates, or any failed gate, is a non-zero exit (a missed/absent gate fails CI).
    exit_code = 0 if (gates and all(g.passed for g in gates)) else 1
    return exit_code, gates


def _print_table(payload: dict, metrics: dict) -> None:
    bar = "=" * 72
    print(bar)
    print(
        f"FleetCast evaluation  ·  {payload.get('n_folds')} folds  ·  "
        f"horizon {payload.get('horizon_hours')}h"
    )
    print(bar)
    print(f"  {'model':<16}{'WAPE':>9}{'MAE':>9}{'RMSE':>9}{'MAPE(hi-vol)':>15}")
    print("  " + "-" * 56)
    order = ["seasonal_naive", "arima", "lightgbm"]
    for name in [*order, *(m for m in metrics if m not in order)]:
        if name not in metrics:
            continue
        p = metrics[name]["pooled"]
        print(
            f"  {name:<16}{p['wape']:>9.4f}{p['mae']:>9.3f}{p['rmse']:>9.3f}"
            f"{p.get('mape_high_volume', float('nan')):>15.4f}"
        )
    print(bar)


def _print_gates(gates: list[GateResult]) -> None:
    if not gates:
        print("  (no gates evaluated — is the model trained?)")
        return
    for g in gates:
        status = "PASS ✅" if g.passed else "FAIL ❌"
        print(f"  [{status}] {g.name}: {g.detail}")
    print("=" * 72)


def run_eval(cfg: Config | None = None) -> int:
    code, _ = evaluate(cfg)
    return code


def main() -> None:
    raise SystemExit(run_eval())


if __name__ == "__main__":
    main()
