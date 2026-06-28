"""Distribution-free prediction intervals via **split conformal** on the
rolling-origin one-step-ahead residuals.

Why this method (vs. MAPIE's single-series EnbPI): FleetCast's forecaster is a
*global, panel, one-step-ahead, refit-daily* model, and the rolling-origin
backtest already yields the exact out-of-sample one-step-ahead residuals that a
calibrated interval needs. Split (inductive) conformal on those residuals is the
deployment-matched, distribution-free choice and gives a finite-sample marginal
coverage guarantee. We use MAPIE's ``AbsoluteConformityScore`` for the
conformity scores and apply the standard inductive-conformal quantile, **per
zone**, so each zone's interval reflects its own error scale.

For a calibration set of ``n`` absolute residuals and miscoverage ``alpha``, the
interval half-width is the ``k``-th smallest residual with
``k = ceil((n + 1)(1 - alpha))`` — the classic split-conformal quantile, which
guarantees coverage ``>= 1 - alpha`` under exchangeability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..logging import get_logger

log = get_logger(__name__)


def _conformity_scores(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Absolute residuals (via MAPIE's conformity-score utility)."""
    from mapie.conformity_scores import AbsoluteConformityScore

    return AbsoluteConformityScore().get_conformity_scores(y_true, y_pred)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The split-conformal interval half-width: the k-th smallest |residual| with
    k = ceil((n+1)(1-alpha)). Returns +inf if the level demands more data."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")  # not enough calibration points for this level
    return float(scores[k - 1])


def calibrate_per_zone(df: pd.DataFrame, alpha: float) -> dict[int, float]:
    """Per-zone conformal half-width from a calibration frame (zone_id,y_true,y_pred)."""
    q: dict[int, float] = {}
    for zone, g in df.groupby("zone_id"):
        scores = _conformity_scores(g["y_true"].to_numpy(float), g["y_pred"].to_numpy(float))
        q[int(zone)] = conformal_quantile(scores, alpha)
    return q


def _require_finite(
    q_by_zone: dict[int, float], counts: dict[int, int], alpha: float, which: str
) -> None:
    """Fail loudly if any zone's quantile is unbounded (too few calibration points).

    An unbounded quantile would otherwise ship an infinite-width interval AND
    inflate pooled coverage toward 1.0 — i.e. the gate would pass in the unsafe
    direction. Better to surface it.
    """
    need = int(np.ceil(1.0 / alpha)) - 1
    bad = {z: counts.get(z, 0) for z, v in q_by_zone.items() if not np.isfinite(v)}
    if bad:
        raise ValueError(
            f"{which} conformal quantile is unbounded for zones {sorted(bad)} "
            f"(only {sorted(set(bad.values()))} calibration residuals; need >= {need} "
            f"per zone for {1 - alpha:.0%} intervals — increase n_folds or lower nominal coverage)"
        )


def apply_intervals(df: pd.DataFrame, q_by_zone: dict[int, float]) -> pd.DataFrame:
    """Add lower/upper interval columns; the lower bound is clipped at 0 (demand >= 0)."""
    out = df.copy()
    out["q"] = out["zone_id"].map(q_by_zone)
    out["lower"] = np.clip(out["y_pred"] - out["q"], 0.0, None)
    out["upper"] = out["y_pred"] + out["q"]
    return out


def coverage_report(df_with_intervals: pd.DataFrame) -> dict:
    """Empirical coverage + sharpness (overall and per zone)."""
    d = df_with_intervals
    covered = (d["y_true"] >= d["lower"]) & (d["y_true"] <= d["upper"])
    per_zone = {
        int(z): {
            "coverage": float(((g["y_true"] >= g["lower"]) & (g["y_true"] <= g["upper"])).mean()),
            "mean_width": float((g["upper"] - g["lower"]).mean()),
            "n": len(g),
        }
        for z, g in d.groupby("zone_id")
    }
    return {
        "empirical_coverage": float(covered.mean()),
        "mean_interval_width": float((d["upper"] - d["lower"]).mean()),
        "n": len(d),
        "per_zone": per_zone,
    }


def conformalize(predictions: pd.DataFrame, cfg: Config) -> dict:
    """Calibrate split-conformal intervals on the one-step-ahead backtest residuals
    and verify coverage on a held-out (later) split.

    ``predictions`` is the backtest predictions frame; only the LightGBM rows are
    used. The held-out hours are split time-ordered: the earlier half calibrates,
    the later half tests coverage (the gate). A separate *deployment* quantile is
    calibrated on ALL residuals for the dashboard.
    """
    alpha = 1.0 - float(cfg.conformal["nominal_coverage"])
    lgbm = predictions[predictions["model"] == "lightgbm"][["zone_id", "hour", "y_true", "y_pred"]]
    if lgbm.empty:
        raise ValueError("no LightGBM predictions to conformalize — run `make train` first")

    hours = sorted(lgbm["hour"].unique())
    n_cal = max(1, len(hours) // 2)
    cal_hours, test_hours = set(hours[:n_cal]), set(hours[n_cal:])
    cal = lgbm[lgbm["hour"].isin(cal_hours)]
    test = lgbm[lgbm["hour"].isin(test_hours)]

    # Calibrate on the earlier split, evaluate coverage on the later split.
    q_eval = calibrate_per_zone(cal, alpha)
    _require_finite(q_eval, cal.groupby("zone_id").size().to_dict(), alpha, "evaluation")
    report = coverage_report(apply_intervals(test, q_eval))

    # Deployment quantile: calibrate on ALL held-out residuals (most data).
    q_deploy = calibrate_per_zone(lgbm, alpha)
    _require_finite(q_deploy, lgbm.groupby("zone_id").size().to_dict(), alpha, "deployment")

    nominal = float(cfg.conformal["nominal_coverage"])
    tol = float(cfg.conformal["coverage_tolerance"])
    # Surface conditional (per-zone) coverage spread — pooled coverage can be on
    # target while individual zones miss (marginal vs conditional coverage).
    pz_cov = sorted(v["coverage"] for v in report["per_zone"].values())
    report.update(
        {
            "nominal_coverage": nominal,
            "coverage_tolerance": tol,
            "alpha": alpha,
            "n_calibration": len(cal),
            "n_test": len(test),
            "method": "split_conformal_per_zone_on_rolling_origin_residuals",
            "per_zone_coverage_min": float(pz_cov[0]),
            "per_zone_coverage_median": float(np.median(pz_cov)),
            "per_zone_coverage_max": float(pz_cov[-1]),
            "n_zones_outside_band": int(sum(abs(c - nominal) > tol for c in pz_cov)),
            "n_zones": len(pz_cov),
            "deployment_quantiles": {int(z): float(v) for z, v in q_deploy.items()},
        }
    )
    log.info(
        "conformal: empirical coverage = %.4f on %d held-out points (target %.2f +/- %.2f)",
        report["empirical_coverage"],
        report["n"],
        report["nominal_coverage"],
        report["coverage_tolerance"],
    )
    return report
