"""Config loading, validation, hashing, and path resolution."""

from __future__ import annotations

import pytest
import yaml

from fleetcast.config import load_config


def test_loads_default_config():
    cfg = load_config()
    assert cfg.service in ("yellow", "green")
    assert len(cfg.months) >= 1
    assert cfg.top_n_zones > 0
    assert 0 < cfg.conformal["nominal_coverage"] < 1
    assert cfg.seed == 1234


def test_config_hash_is_stable_and_short():
    a = load_config().config_hash
    b = load_config().config_hash
    assert a == b
    assert len(a) == 16


def test_paths_are_absolute_and_created(tmp_path):
    cfg = load_config()
    assert cfg.raw_dir.is_absolute()
    assert cfg.processed_dir.exists()  # ensure_dir created it


def test_lag_and_rolling_windows_present():
    cfg = load_config()
    assert 168 in cfg.lags_hours  # same hour last week
    assert 24 in cfg.rolling_windows_hours


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda c: c["taxi"].update(service="purple"), "service"),
        (lambda c: c["taxi"].update(months=[]), "months"),
        (lambda c: c["taxi"].update(months=["2024/01"]), "YYYY-MM"),
        (lambda c: c["taxi"].update(months=["2024-13"]), "month"),
        (lambda c: c["taxi"].update(months=["2024-00"]), "month"),
        (lambda c: c["taxi"].update(months=["abcd-ef"]), "month"),
        (lambda c: c["taxi"].update(top_n_zones=0), "positive"),
        (lambda c: c.pop("seed"), "seed"),
    ],
)
def test_invalid_config_rejected(tmp_path, mutate, msg):
    raw = yaml.safe_load((load_config().source_path).read_text())
    mutate(raw)
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises((ValueError, KeyError)):
        load_config(bad)
