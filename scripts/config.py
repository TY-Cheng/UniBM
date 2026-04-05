"""Repository path resolution shared by scripts, workflows, and the vignette.

The repo keeps one `.env` file at the root with `DIR_WORK` as the primary
anchor. Every other path is derived from that anchor so local scripts, notebook
sessions, and manuscript builders all resolve the same data/output locations.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_FALLBACK_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_FALLBACK_REPO_ROOT / ".env")
_DEFAULT_REPO_ROOT = Path(os.environ.get("DIR_WORK", _FALLBACK_REPO_ROOT)).expanduser().resolve()


def resolve_repo_dirs(dir_work: Path | str | None = None) -> dict[str, Path]:
    """Return the canonical repo directories and export them to the environment."""
    work = Path(dir_work).expanduser().resolve() if dir_work else _DEFAULT_REPO_ROOT
    dirs = {
        "DIR_WORK": work,
        "DIR_SCRIPTS": work / "scripts",
        "DIR_DATA": work / "data",
        "DIR_DATA_RAW": work / "data" / "raw",
        "DIR_DATA_RAW_GHCN": work / "data" / "raw" / "ghcn",
        "DIR_DATA_DERIVED": work / "data" / "derived",
        "DIR_DATA_METADATA": work / "data" / "metadata",
        "DIR_OUT": work / "out",
        "DIR_OUT_BENCHMARK": work / "out" / "benchmark",
        "DIR_OUT_BENCHMARK_CACHE": work / "out" / "benchmark" / "cache",
        "DIR_OUT_APPLICATIONS": work / "out" / "applications",
        "DIR_MANUSCRIPT": work / "UniBM_manuscript",
        "DIR_MANUSCRIPT_FIGURE": work / "UniBM_manuscript" / "Figure",
        "DIR_MANUSCRIPT_TABLE": work / "UniBM_manuscript" / "Table",
    }
    for key, value in dirs.items():
        os.environ.setdefault(key, str(value))
    return dirs
