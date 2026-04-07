"""Shared import/bootstrap helpers for direct workflow execution and notebooks."""

from __future__ import annotations

from pathlib import Path
import sys


def _prepend_path(path: Path) -> Path:
    """Prepend one directory to ``sys.path`` when needed."""
    path = path.resolve()
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return path


def ensure_scripts_on_path_from_entry(entry_file: str | Path) -> Path:
    """Ensure the repository ``scripts/`` directory is importable for one CLI entrypoint."""
    scripts_dir = Path(entry_file).resolve().parents[1]
    return _prepend_path(scripts_dir)


def bootstrap_notebook_scripts_dir(start: str | Path | None = None) -> Path:
    """Locate and prepend the repository ``scripts/`` directory from a notebook session."""
    current = Path.cwd().resolve() if start is None else Path(start).resolve()
    for candidate in (current, *current.parents):
        scripts_dir = candidate / "scripts"
        if (scripts_dir / "config.py").exists():
            return _prepend_path(scripts_dir)
    raise FileNotFoundError(
        "Could not locate scripts/config.py from the current notebook session."
    )


__all__ = [
    "bootstrap_notebook_scripts_dir",
    "ensure_scripts_on_path_from_entry",
]
