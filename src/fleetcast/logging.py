"""Project-wide logging configuration.

A single ``get_logger`` helper installs a clean, timestamped handler once. Log
level is controlled by the ``FLEETCAST_LOG_LEVEL`` environment variable
(default ``INFO``).
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.environ.get("FLEETCAST_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("fleetcast")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger namespaced under ``fleetcast``."""
    _configure_root()
    short = name.split(".")[-1] if name.startswith("fleetcast") else name
    return logging.getLogger(f"fleetcast.{short}")
