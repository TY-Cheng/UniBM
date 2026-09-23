"""Index the curated report artifacts; this is not a record of numerical provenance."""
# ruff: noqa: E402

from __future__ import annotations

if __package__ in {None, ""}:
    import importlib.util
    from pathlib import Path

    _helper_path = Path(__file__).resolve().parents[1] / "shared" / "import_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("_shared_import_bootstrap", _helper_path)
    if _spec is None or _spec.loader is None:  # pragma: no cover - import bootstrap failure
        raise ImportError(f"Could not load import bootstrap helper from {_helper_path}.")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _module.ensure_scripts_on_path_from_entry(__file__)

import json
from pathlib import Path
import subprocess

from config import resolve_repo_dirs
from data_prep.constants import ANALYSIS_END_DATE
from shared.runtime import status


def _relative(path: Path, *, root: Path) -> str:
    """Serialize a workspace-relative path, or an absolute path outside it."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _git_output(repo_root: Path, *args: str) -> str:
    """Return the code checkout state recorded when the index is written."""
    return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()


def _figure_entry(
    *,
    label: str,
    placement: str,
    generated_by: str,
    paths: list[Path],
    root: Path,
) -> dict[str, object]:
    return {
        "label": label,
        "kind": "figure",
        "placement": placement,
        "generated_by": generated_by,
        "artifact_paths": [_relative(path, root=root) for path in paths],
    }


def _table_entry(
    *,
    label: str,
    placement: str,
    generated_by: str,
    path: Path,
    root: Path,
    note: str | None = None,
) -> dict[str, object]:
    entry: dict[str, object] = {
        "label": label,
        "kind": "table",
        "placement": placement,
        "generated_by": generated_by,
        "artifact_paths": [_relative(path, root=root)],
    }
    if note:
        entry["note"] = note
    return entry


def build_report_subset_manifest(root: Path | str = ".") -> Path:
    """Write expected paths, labels, producers, and placements for the report subset.

    Version fields describe this index's creation state, not the versions used
    to calculate its listed artifacts. Writing the index neither regenerates
    nor verifies those files, which may not yet exist at the destination.
    """
    dirs = resolve_repo_dirs(root)
    repo_root = dirs["DIR_WORK"]
    workspace_root = dirs["DIR_WORKSPACE"]
    report_dir = dirs["DIR_REPORT"]
    figure_dir = dirs["DIR_REPORT_FIGURE"]
    table_dir = dirs["DIR_REPORT_TABLE"]
    manifest_path = report_dir / "report_subset_manifest.json"
    report_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        _figure_entry(
            label="fig:benchmark-evi-main",
            placement="primary",
            generated_by="scripts/benchmark/evi_report.py",
            paths=[figure_dir / "benchmark_evi_summary.pdf"],
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:benchmark-evi-targets",
            placement="primary",
            generated_by="scripts/benchmark/evi_report.py",
            paths=[figure_dir / "benchmark_evi_targets.pdf"],
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:benchmark-ei-main",
            placement="primary",
            generated_by="scripts/benchmark/ei_report.py",
            paths=[figure_dir / "benchmark_ei_summary.pdf"],
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:benchmark-ei-targets",
            placement="primary",
            generated_by="scripts/benchmark/ei_report.py",
            paths=[figure_dir / "benchmark_ei_targets.pdf"],
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:application-streamflow",
            placement="primary",
            generated_by="scripts/application/build.py",
            paths=[
                figure_dir / "application_composite_tx_streamflow.pdf",
                figure_dir / "application_composite_fl_streamflow.pdf",
            ],
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:application-nfip",
            placement="primary",
            generated_by="scripts/application/build.py",
            paths=[
                figure_dir / "application_composite_tx_nfip_claims.pdf",
                figure_dir / "application_composite_fl_nfip_claims.pdf",
            ],
            root=workspace_root,
        ),
        _table_entry(
            label="tab:application-summary",
            placement="primary-supporting",
            generated_by="scripts/application/build.py",
            path=table_dir / "application_summary.tex",
            note="Report-facing snapshot aligned to the curated four-case subset.",
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:benchmark-evi-shrinkage",
            placement="supplementary",
            generated_by="scripts/benchmark/evi_report.py",
            paths=[figure_dir / "benchmark_evi_shrinkage_sensitivity.pdf"],
            root=workspace_root,
        ),
        _table_entry(
            label="tab:benchmark-evi-summary",
            placement="supplementary",
            generated_by="scripts/benchmark/evi_report.py",
            path=table_dir / "benchmark_evi_summary.tex",
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:benchmark-ei-shrinkage",
            placement="supplementary",
            generated_by="scripts/benchmark/ei_report.py",
            paths=[figure_dir / "benchmark_ei_shrinkage_sensitivity.pdf"],
            root=workspace_root,
        ),
        _table_entry(
            label="tab:benchmark-ei-summary",
            placement="supplementary",
            generated_by="scripts/benchmark/ei_report.py",
            path=table_dir / "benchmark_ei_summary.tex",
            root=workspace_root,
        ),
        _table_entry(
            label="tab:application-selection-sensitivity",
            placement="supplementary",
            generated_by="scripts/application/build.py",
            path=table_dir / "application_selection_sensitivity.tex",
            root=workspace_root,
        ),
        _table_entry(
            label="tab:application-extrapolation",
            placement="supplementary",
            generated_by="scripts/application/build.py",
            path=table_dir / "application_extrapolation.tex",
            root=workspace_root,
        ),
        _table_entry(
            label="tab:application-streamflow-gev-check",
            placement="supplementary",
            generated_by="scripts/application/build.py",
            path=table_dir / "application_streamflow_gev_check.tex",
            root=workspace_root,
        ),
        _table_entry(
            label="tab:application-usgs-screening",
            placement="supplementary",
            generated_by="scripts/application/build.py",
            path=table_dir / "application_usgs_screening.tex",
            root=workspace_root,
        ),
    ]
    payload = {
        "report_scope": "curated four-case report subset",
        "analysis_end_date": ANALYSIS_END_DATE,
        "manifest_code_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "manifest_code_worktree_dirty": bool(
            _git_output(repo_root, "status", "--porcelain", "--untracked-files=normal")
        ),
        "workspace_root": _relative(workspace_root, root=workspace_root),
        "code_repo_root": _relative(repo_root, root=workspace_root),
        "report_root": _relative(report_dir, root=workspace_root),
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest_path


def main() -> None:
    manifest_path = build_report_subset_manifest()
    status("report", f"report_subset_manifest: {manifest_path}")


__all__ = ["build_report_subset_manifest"]


if __name__ == "__main__":
    main()
