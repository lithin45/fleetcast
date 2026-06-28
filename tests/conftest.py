"""Shared pytest fixtures.

Two goals:
* make the suite runnable fully **offline** (CI default) against deterministic
  synthetic data, and
* still allow **real** network downloads locally via ``@pytest.mark.network``
  (skipped when ``FLEETCAST_OFFLINE=1``).
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from fleetcast.config import Config, load_config

from . import synthetic

OFFLINE = os.environ.get("FLEETCAST_OFFLINE", "0") == "1"


def pytest_collection_modifyitems(config, items):
    """Skip network-marked tests when running offline."""
    if not OFFLINE:
        return
    skip = pytest.mark.skip(reason="FLEETCAST_OFFLINE=1 set; skipping network test")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@dataclass
class SyntheticEnv:
    """A self-contained synthetic data + config environment in a temp dir."""

    cfg: Config
    raw_dir: Path
    zones: list[int]
    months: list[str]


@pytest.fixture
def make_synthetic_env(tmp_path: Path) -> Callable[..., SyntheticEnv]:
    """Factory: build a synthetic TLC dataset + a Config pointing at it.

    Returns a callable so individual tests can pick their own months/zone count.
    """

    def _build(
        months: list[str] | None = None,
        n_zones: int = 6,
        top_n: int | None = None,
        weather_enabled: bool = False,
        seed: int = 1234,
    ) -> SyntheticEnv:
        months = months or ["2024-01"]
        zones = list(range(1, n_zones + 1))
        raw_dir = tmp_path / "raw"
        synthetic.write_synthetic_dataset(raw_dir, "yellow", months, zones, seed=seed)
        synthetic.write_synthetic_zone_lookup(raw_dir, n_zones=263)

        base = copy.deepcopy(load_config().raw)
        base["taxi"]["months"] = months
        base["taxi"]["top_n_zones"] = top_n or n_zones
        base["weather"]["enabled"] = weather_enabled
        base["paths"]["raw_dir"] = str(raw_dir)
        base["paths"]["processed_dir"] = str(tmp_path / "processed")
        base["paths"]["forecasts_dir"] = str(tmp_path / "forecasts")
        base["seed"] = seed

        cfg_path = tmp_path / "data.yaml"
        cfg_path.write_text(yaml.safe_dump(base))
        cfg = load_config(cfg_path)
        return SyntheticEnv(cfg=cfg, raw_dir=raw_dir, zones=zones, months=months)

    return _build
