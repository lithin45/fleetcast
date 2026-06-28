"""CLI wiring: every pipeline subcommand is registered."""

from __future__ import annotations

import pytest

from fleetcast.cli import build_parser, main


def test_all_subcommands_registered():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    choices = set(actions[0].choices)
    expected = {"data", "smoke", "features", "backtest", "train", "eval", "forecasts", "demo"}
    assert expected <= choices


def test_missing_command_errors():
    with pytest.raises(SystemExit):
        main([])
