"""Rolling-origin (expanding-window) fold generation — the backtest methodology core.

Each fold trains on **all** history strictly before a cutoff and forecasts the next
``horizon`` hours; the origin then rolls forward by ``step`` and we repeat. The
window is *expanding* (train always starts at the panel's first hour). Folds tile
the **end** of the series so the most recent data is always evaluated.

Leakage safety is structural: for every fold, ``max(train.hour) < train_end ==
test_start <= min(test.hour)``. The backtest test-suite asserts exactly this.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ..config import Config
from ..logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold. ``train_end`` is the exclusive cutoff and equals
    ``test_start`` (the first forecast hour); ``test_end`` is exclusive."""

    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def horizon_hours(self) -> int:
        return int((self.test_end - self.test_start).total_seconds() // 3600)


def make_folds(hours: Sequence[pd.Timestamp], cfg: Config) -> list[Fold]:
    """Build expanding-window rolling-origin folds over a contiguous hourly index.

    Folds tile the end of the series with ``n_folds`` windows of length
    ``horizon_hours`` stepping by ``step_hours``; the first cutoff is pushed late
    enough to honor ``min_train_hours``. If the series is too short for the
    requested ``n_folds``, the count is reduced (with a warning) rather than
    fabricating overlapping windows.
    """
    timeline = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(hours))))
    n = len(timeline)
    bt = cfg.backtest
    horizon = int(bt["horizon_hours"])
    step = int(bt["step_hours"])
    n_folds = int(bt["n_folds"])
    min_train = int(bt["min_train_hours"])

    # Index of the first cutoff so that the n_folds windows tile the end exactly.
    first_cut = n - horizon - (n_folds - 1) * step
    if first_cut < min_train:
        feasible = (n - horizon - min_train) // step + 1
        if feasible < 1:
            raise ValueError(
                f"not enough history for one fold: have {n}h, need "
                f"min_train({min_train}) + horizon({horizon}) = {min_train + horizon}h"
            )
        log.warning(
            "reducing n_folds %d -> %d to honor min_train_hours=%d", n_folds, feasible, min_train
        )
        n_folds = feasible
        first_cut = n - horizon - (n_folds - 1) * step

    folds: list[Fold] = []
    for i in range(n_folds):
        cut = first_cut + i * step
        ts = timeline[cut]
        folds.append(Fold(i, ts, ts, ts + pd.Timedelta(hours=horizon)))
    return folds


def train_slice(panel: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Rows strictly before the cutoff — the expanding training window."""
    return panel[panel["hour"] < fold.train_end]


def forecast_slice(panel: pd.DataFrame, fold: Fold) -> pd.DataFrame:
    """Rows in ``[test_start, test_end)`` — the forecast window."""
    return panel[(panel["hour"] >= fold.test_start) & (panel["hour"] < fold.test_end)]
