"""Repo-root and filesystem path helpers.

All paths in FleetCast are resolved relative to the repository root so the
pipeline behaves identically whether invoked from the host, a Docker container,
or CI. The root is discovered by walking up from this file until a ``pyproject.toml``
(or ``.git``) marker is found.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return the repository root (directory containing ``pyproject.toml``).

    Falls back to the current working directory if no marker is found, which
    keeps unit tests that run in temp dirs from blowing up.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else (repo_root() / p)
