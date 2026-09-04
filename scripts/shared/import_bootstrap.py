"""Shared import/bootstrap helpers for direct workflow execution."""

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


def _prepend_repo_import_paths(repo_root: Path) -> Path:
    """Expose both ``scripts/`` helpers and the installable ``src/unibm`` package."""
    scripts_dir = _prepend_path(repo_root / "scripts")
    _prepend_path(repo_root / "src")
    return scripts_dir


def ensure_scripts_on_path_from_entry(entry_file: str | Path) -> Path:
    """Ensure repo-local ``scripts/`` and ``src/`` imports work for one CLI entrypoint."""
    repo_root = Path(entry_file).resolve().parents[2]
    return _prepend_repo_import_paths(repo_root)


__all__ = ["ensure_scripts_on_path_from_entry"]
