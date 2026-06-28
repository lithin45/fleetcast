"""Typed configuration loader for FleetCast.

The single source of truth is ``config/data.yaml``. This module loads it into a
lightweight typed wrapper, resolves all paths against the repo root, and exposes
a ``config_hash`` (SHA-256 of the raw file bytes) that downstream artifacts embed
for reproducibility — if the config changes, the hash changes, and stale feature
tables can be detected.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import ensure_dir, repo_root, resolve

DEFAULT_CONFIG_PATH = repo_root() / "config" / "data.yaml"

# 'YYYY-MM' with a real 01-12 month.
_MONTH_RE = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


@dataclass(frozen=True)
class Config:
    """Parsed, validated FleetCast configuration.

    ``raw`` holds the full nested mapping; typed properties expose the fields the
    pipeline actually uses. Path properties return absolute, resolved paths.
    """

    raw: dict[str, Any]
    source_path: Path
    config_hash: str = field(default="", compare=False)

    # --- scope ---
    @property
    def service(self) -> str:
        return str(self.raw["taxi"]["service"]).lower()

    @property
    def months(self) -> list[str]:
        return [str(m) for m in self.raw["taxi"]["months"]]

    @property
    def top_n_zones(self) -> int:
        return int(self.raw["taxi"]["top_n_zones"])

    @property
    def location_id_bounds(self) -> tuple[int, int]:
        f = self.raw["taxi"]["filters"]
        return int(f["min_location_id"]), int(f["max_location_id"])

    # --- weather ---
    @property
    def weather_enabled(self) -> bool:
        return bool(self.raw["weather"]["enabled"])

    @property
    def weather(self) -> dict[str, Any]:
        return dict(self.raw["weather"])

    # --- features ---
    @property
    def lags_hours(self) -> list[int]:
        return [int(x) for x in self.raw["features"]["lags_hours"]]

    @property
    def rolling_windows_hours(self) -> list[int]:
        return [int(x) for x in self.raw["features"]["rolling_windows_hours"]]

    @property
    def add_holiday_flag(self) -> bool:
        return bool(self.raw["features"]["add_holiday_flag"])

    # --- backtest / conformal / eval ---
    @property
    def backtest(self) -> dict[str, Any]:
        return dict(self.raw["backtest"])

    @property
    def conformal(self) -> dict[str, Any]:
        return dict(self.raw["conformal"])

    @property
    def eval(self) -> dict[str, Any]:
        return dict(self.raw["eval"])

    @property
    def baselines(self) -> dict[str, Any]:
        return dict(self.raw.get("baselines", {}))

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.raw.get("model", {}))

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    # --- paths (absolute) ---
    @property
    def raw_dir(self) -> Path:
        return ensure_dir(resolve(self.raw["paths"]["raw_dir"]))

    @property
    def processed_dir(self) -> Path:
        return ensure_dir(resolve(self.raw["paths"]["processed_dir"]))

    @property
    def forecasts_dir(self) -> Path:
        return ensure_dir(resolve(self.raw["paths"]["forecasts_dir"]))


def _validate(raw: dict[str, Any]) -> None:
    """Fail fast on obviously broken config rather than deep in the pipeline."""
    for key in ("taxi", "weather", "features", "backtest", "conformal", "eval", "seed", "paths"):
        if key not in raw:
            raise ValueError(f"config missing required top-level key: {key!r}")
    if raw["taxi"]["service"] not in ("yellow", "green"):
        raise ValueError("taxi.service must be 'yellow' or 'green'")
    if not raw["taxi"]["months"]:
        raise ValueError("taxi.months must list at least one 'YYYY-MM' month")
    for m in raw["taxi"]["months"]:
        # Validate the components, not just the shape, so bad months (e.g.
        # '2024-13', 'abcd-ef') fail fast here rather than deep in the pipeline.
        if not (isinstance(m, str) and _MONTH_RE.fullmatch(m)):
            raise ValueError(f"taxi.months entries must be 'YYYY-MM' with month 01-12, got {m!r}")
    if int(raw["taxi"]["top_n_zones"]) <= 0:
        raise ValueError("taxi.top_n_zones must be positive")


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the configuration.

    Resolution order: explicit ``path`` arg → ``FLEETCAST_CONFIG`` env var →
    ``config/data.yaml`` at the repo root.
    """
    cfg_path = Path(path) if path else Path(os.environ.get("FLEETCAST_CONFIG", DEFAULT_CONFIG_PATH))
    cfg_path = cfg_path if cfg_path.is_absolute() else resolve(cfg_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    data = cfg_path.read_bytes()
    raw = yaml.safe_load(data)
    _validate(raw)
    digest = hashlib.sha256(data).hexdigest()[:16]
    return Config(raw=raw, source_path=cfg_path, config_hash=digest)
