"""FleetCast conformal package — split-conformal prediction intervals."""

from __future__ import annotations

from .run import run_conformal
from .split_conformal import (
    apply_intervals,
    calibrate_per_zone,
    conformal_quantile,
    conformalize,
    coverage_report,
)

__all__ = [
    "apply_intervals",
    "calibrate_per_zone",
    "conformal_quantile",
    "conformalize",
    "coverage_report",
    "run_conformal",
]
