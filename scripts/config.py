"""Repository path resolution shared by domain scripts and the vignette.

The code repo and manuscript repo may live either as siblings or with the
manuscript nested inside the code repo. `DIR_WORK` anchors the code repo, while
`DIR_MANUSCRIPT` can optionally point to the manuscript repo explicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_FALLBACK_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_FALLBACK_REPO_ROOT / ".env")
_DEFAULT_REPO_ROOT = Path(os.environ.get("DIR_WORK", _FALLBACK_REPO_ROOT)).expanduser().resolve()


def _looks_like_code_repo(path: Path) -> bool:
    """Return whether a path resembles the UniBM code repo root."""
    return (path / "scripts").is_dir() and (path / "pyproject.toml").is_file()


def _resolve_code_root(path: Path) -> Path:
    """Resolve the canonical code repo root from a requested workspace path."""
    if _looks_like_code_repo(path):
        return path
    nested_repo = path / _FALLBACK_REPO_ROOT.name
    if nested_repo != path and _looks_like_code_repo(nested_repo):
        return nested_repo
    return path


def _resolve_manuscript_root(*, requested_root: Path | None, code_root: Path) -> Path:
    """Resolve the manuscript repo root from explicit, sibling, or nested layouts."""
    explicit = os.environ.get("DIR_MANUSCRIPT")
    if explicit:
        return Path(explicit).expanduser().resolve()

    candidates: list[Path] = []
    if requested_root is not None:
        candidates.append(requested_root / "UniBM_manuscript")
    candidates.extend(
        [
            code_root.parent / "UniBM_manuscript",
            code_root / "UniBM_manuscript",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _common_root(*paths: Path) -> Path:
    """Return the common parent covering all provided paths."""
    return Path(os.path.commonpath([str(path.resolve()) for path in paths]))


def resolve_repo_dirs(dir_work: Path | str | None = None) -> dict[str, Path]:
    """Return the canonical repo directories and export them to the environment."""
    requested_root = Path(dir_work).expanduser().resolve() if dir_work else _DEFAULT_REPO_ROOT
    work = _resolve_code_root(requested_root)
    manuscript = _resolve_manuscript_root(requested_root=requested_root, code_root=work)
    workspace = _common_root(work, manuscript)
    dirs = {
        "DIR_WORK": work,
        "DIR_WORKSPACE": workspace,
        "DIR_SCRIPTS": work / "scripts",
        "DIR_DATA": work / "data",
        "DIR_DATA_RAW": work / "data" / "raw",
        "DIR_DATA_RAW_GHCN": work / "data" / "raw" / "ghcn",
        "DIR_DATA_RAW_USGS": work / "data" / "raw" / "usgs",
        "DIR_DATA_RAW_FEMA": work / "data" / "raw" / "fema",
        "DIR_DATA_DERIVED": work / "data" / "derived",
        "DIR_DATA_METADATA": work / "data" / "metadata",
        "DIR_DATA_METADATA_APPLICATION": work / "data" / "metadata" / "application",
        "DIR_OUT": work / "out",
        "DIR_OUT_BENCHMARK": work / "out" / "benchmark",
        "DIR_OUT_BENCHMARK_CACHE": work / "out" / "benchmark" / "cache",
        "DIR_OUT_APPLICATIONS": work / "out" / "applications",
        "DIR_MANUSCRIPT": manuscript,
        "DIR_MANUSCRIPT_FIGURE": manuscript / "Figure",
        "DIR_MANUSCRIPT_TABLE": manuscript / "Table",
    }
    for key, value in dirs.items():
        os.environ[key] = str(value)
    return dirs
