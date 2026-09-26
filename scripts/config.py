"""Repository path resolution shared by domain scripts.

Code and data roots are derived from this file. UNIBM_REPORT_DIR selects the
final report destination; unset or blank values use the local out/reports tree.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_FALLBACK_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_FALLBACK_REPO_ROOT / ".env")
_DEFAULT_REPO_ROOT = _FALLBACK_REPO_ROOT


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


def _resolve_report_root(code_root: Path) -> Path:
    """Resolve and validate the one report destination without creating it."""
    explicit = os.environ.get("UNIBM_REPORT_DIR", "").strip()
    requested = Path(explicit).expanduser() if explicit else Path("out/reports")
    candidate = code_root / requested
    if candidate.is_symlink():
        raise ValueError(f"UNIBM_REPORT_DIR output directory must not be a symlink: {candidate}")
    report = candidate.resolve()
    if not explicit and report != candidate:
        raise ValueError(f"Default report directory must not follow a symlink: {candidate}")
    if report == Path(report.anchor):
        raise ValueError("UNIBM_REPORT_DIR must not be the filesystem root.")
    for directory in (report, report / "Figure", report / "Table"):
        if directory.is_symlink():
            raise ValueError(
                f"UNIBM_REPORT_DIR output directory must not be a symlink: {directory}"
            )
        ancestor = directory
        while not ancestor.exists() and not ancestor.is_symlink():
            ancestor = ancestor.parent
        if not ancestor.is_dir() or not os.access(ancestor, os.W_OK | os.X_OK):
            raise ValueError(f"UNIBM_REPORT_DIR is not a writable directory: {ancestor}")
        if directory != report and directory.exists():
            for child in directory.iterdir():
                if child.is_symlink():
                    raise ValueError(f"UNIBM_REPORT_DIR output must not be a symlink: {child}")
    manifest = report / "report_subset_manifest.json"
    if manifest.is_symlink() or (manifest.exists() and not manifest.is_file()):
        raise ValueError(f"UNIBM_REPORT_DIR manifest must be a regular file: {manifest}")
    return report


def _resolve_data_root(*, code_root: Path) -> Path:
    """Return the repository-local canonical data root."""
    return code_root / "data"


def _common_root(*paths: Path) -> Path:
    """Use the common parent, or the code root when Windows drives differ."""
    resolved = [path.resolve() for path in paths]
    try:
        return Path(os.path.commonpath([str(path) for path in resolved]))
    except ValueError:
        # Absolute paths on different Windows drives have no common ancestor.
        return resolved[0]


def resolve_repo_dirs(dir_work: Path | str | None = None) -> dict[str, Path]:
    """Return the canonical repository directories."""
    requested_root = Path(dir_work).expanduser().resolve() if dir_work else _DEFAULT_REPO_ROOT
    work = _resolve_code_root(requested_root)
    report = _resolve_report_root(work)
    data_root = _resolve_data_root(code_root=work)
    workspace = _common_root(work, report)
    dirs = {
        "DIR_WORK": work,
        "DIR_WORKSPACE": workspace,
        "DIR_SCRIPTS": work / "scripts",
        "DIR_DATA": data_root,
        "DIR_DATA_RAW": data_root / "raw",
        "DIR_DATA_RAW_GHCN": data_root / "raw" / "ghcn",
        "DIR_DATA_RAW_USGS": data_root / "raw" / "usgs",
        "DIR_DATA_RAW_FEMA": data_root / "raw" / "fema",
        "DIR_DATA_RAW_CPI": data_root / "raw" / "cpi",
        "DIR_DATA_PROCESSED": data_root / "processed",
        "DIR_DATA_METADATA": data_root / "metadata",
        "DIR_DATA_METADATA_APPLICATION": data_root / "metadata" / "application",
        "DIR_OUT": work / "out",
        "DIR_OUT_BENCHMARK": work / "out" / "benchmark",
        "DIR_OUT_BENCHMARK_CACHE": work / "out" / "benchmark" / "cache",
        "DIR_OUT_APPLICATIONS": work / "out" / "applications",
        "DIR_REPORT": report,
        "DIR_REPORT_FIGURE": report / "Figure",
        "DIR_REPORT_TABLE": report / "Table",
    }
    return dirs


if __name__ == "__main__":
    print(f"Report destination: {resolve_repo_dirs()['DIR_REPORT']}")
