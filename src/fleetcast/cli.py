"""FleetCast command-line interface.

A single entry point (``fleetcast <command>``) that the Makefile and Docker
services call. Each subcommand maps to one pipeline stage:
``data → features → backtest → train → eval → forecasts → demo``.
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .seed import seed_everything


# --------------------------------------------------------------------------
# Command implementations
# --------------------------------------------------------------------------
def cmd_data(args: argparse.Namespace) -> int:
    """Download raw data, then run the DuckDB zone-hour smoke test."""
    from .data.download import download_all
    from .data.smoke import run_smoke

    cfg = load_config()
    seed_everything(cfg.seed)
    download_all(cfg, force=args.force)
    if not args.no_smoke:
        run_smoke(cfg)
    return 0


def cmd_smoke(_args: argparse.Namespace) -> int:
    """Run only the DuckDB smoke test against already-downloaded data."""
    from .data.smoke import run_smoke

    cfg = load_config()
    run_smoke(cfg)
    return 0


def cmd_features(_args: argparse.Namespace) -> int:
    """Build the dense zone-hour panel + causal features (DuckDB SQL)."""
    from .features.build import build_features

    cfg = load_config()
    seed_everything(cfg.seed)
    build_features(cfg)
    return 0


def cmd_backtest(_args: argparse.Namespace) -> int:
    """Run the rolling-origin backtest of the baselines."""
    from .backtest.run import run_baselines

    cfg = load_config()
    seed_everything(cfg.seed)
    run_baselines(cfg)
    return 0


def cmd_train(_args: argparse.Namespace) -> int:
    """Backtest the global LightGBM vs baselines + fit the final model."""
    from .models.train import run_train

    run_train(load_config())
    return 0


def cmd_eval(_args: argparse.Namespace) -> int:
    """Print the eval table and enforce the WAPE/coverage gates (non-zero on miss)."""
    from .evaluate import run_eval

    return run_eval(load_config())


def cmd_forecasts(args: argparse.Namespace) -> int:
    """Precompute the per-zone-hour forecast + interval table for the dashboard."""
    from .ui.precompute import build_forecasts

    build_forecasts(load_config(), force=args.force)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Precompute forecasts (if needed) and launch the Streamlit choropleth."""
    import subprocess
    from pathlib import Path

    from .ui.precompute import build_forecasts

    build_forecasts(load_config())
    app = Path(__file__).resolve().parent / "ui" / "app.py"
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app),
            "--server.address",
            args.address,
            "--server.port",
            str(args.port),
            "--server.headless",
            "true",
        ]
    )


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fleetcast", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="download raw TLC/weather data + smoke test")
    p_data.add_argument("--force", action="store_true", help="re-download even if cached")
    p_data.add_argument("--no-smoke", action="store_true", help="skip the DuckDB smoke test")
    p_data.set_defaults(func=cmd_data)

    p_smoke = sub.add_parser("smoke", help="DuckDB zone-hour smoke test only")
    p_smoke.set_defaults(func=cmd_smoke)

    p_features = sub.add_parser("features", help="build dense zone-hour panel + causal features")
    p_features.set_defaults(func=cmd_features)

    p_backtest = sub.add_parser("backtest", help="rolling-origin backtest of the baselines")
    p_backtest.set_defaults(func=cmd_backtest)

    p_train = sub.add_parser("train", help="backtest LightGBM vs baselines + fit final model")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("eval", help="print eval table + enforce WAPE/coverage gates")
    p_eval.set_defaults(func=cmd_eval)

    p_forecasts = sub.add_parser("forecasts", help="precompute per-zone-hour forecasts + intervals")
    p_forecasts.add_argument("--force", action="store_true", help="rebuild even if cached")
    p_forecasts.set_defaults(func=cmd_forecasts)

    p_demo = sub.add_parser("demo", help="launch the Streamlit choropleth dashboard")
    p_demo.add_argument("--address", default="0.0.0.0", help="server address")
    p_demo.add_argument("--port", type=int, default=8501, help="server port")
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
