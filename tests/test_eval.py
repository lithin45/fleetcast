"""The eval gate: pass/fail logic and non-zero exit on a missed threshold."""

from __future__ import annotations

import json

import pytest

from fleetcast.backtest.run import METRICS_JSON
from fleetcast.evaluate import evaluate


def _write_metrics(cfg, sn_wape: float, lgbm_wape: float) -> None:
    def block(wape):
        return {
            "pooled": {
                "wape": wape,
                "mae": wape * 100,
                "rmse": wape * 150,
                "mape_high_volume": 0.12,
            }
        }

    payload = {
        "n_folds": 14,
        "horizon_hours": 24,
        "metrics": {
            "seasonal_naive": block(sn_wape),
            "arima": block(sn_wape * 2),
            "lightgbm": block(lgbm_wape),
        },
    }
    (cfg.processed_dir / METRICS_JSON).write_text(json.dumps(payload))


def test_eval_passes_when_improvement_met(make_synthetic_env):
    cfg = make_synthetic_env().cfg
    _write_metrics(cfg, sn_wape=0.1697, lgbm_wape=0.1306)  # +23%
    code, gates = evaluate(cfg)
    assert code == 0
    wape_gate = next(g for g in gates if "WAPE" in g.name)
    assert wape_gate.passed and wape_gate.value >= 0.20


def test_eval_fails_when_improvement_short(make_synthetic_env):
    cfg = make_synthetic_env().cfg
    _write_metrics(cfg, sn_wape=0.1697, lgbm_wape=0.1600)  # only ~6%
    code, gates = evaluate(cfg)
    assert code == 1
    assert not next(g for g in gates if "WAPE" in g.name).passed


def test_eval_raises_without_metrics(make_synthetic_env):
    cfg = make_synthetic_env().cfg
    with pytest.raises(FileNotFoundError):
        evaluate(cfg)


def _block(wape):
    return {"pooled": {"wape": wape, "mae": 1, "rmse": 1, "mape_high_volume": 0.12}}


def test_eval_fails_when_lightgbm_missing(make_synthetic_env):
    """A missing model must FAIL the gate, never silently pass."""
    cfg = make_synthetic_env().cfg
    payload = {"n_folds": 14, "metrics": {"seasonal_naive": _block(0.17), "arima": _block(0.40)}}
    (cfg.processed_dir / METRICS_JSON).write_text(json.dumps(payload))
    code, gates = evaluate(cfg)
    assert code == 1
    assert not next(g for g in gates if "WAPE" in g.name).passed


def test_eval_no_keyerror_when_seasonal_naive_missing(make_synthetic_env):
    cfg = make_synthetic_env().cfg
    payload = {"n_folds": 14, "metrics": {"lightgbm": _block(0.13)}}
    (cfg.processed_dir / METRICS_JSON).write_text(json.dumps(payload))
    code, gates = evaluate(cfg)  # must not raise KeyError
    assert code == 1
    assert not next(g for g in gates if "WAPE" in g.name).passed


def test_eval_fails_on_bad_conformal_coverage(make_synthetic_env):
    cfg = make_synthetic_env().cfg
    _write_metrics(cfg, sn_wape=0.1697, lgbm_wape=0.1306)  # WAPE gate passes
    (cfg.processed_dir / "conformal_metrics.json").write_text(
        json.dumps({"empirical_coverage": 0.70})  # far from 0.90 +/- 0.03
    )
    code, gates = evaluate(cfg)
    assert code == 1
    assert not next(g for g in gates if "coverage" in g.name.lower()).passed
