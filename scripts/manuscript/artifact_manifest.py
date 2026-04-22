"""Paper-subset artifact manifest for the manuscript-facing build."""
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

from config import resolve_repo_dirs
from shared.runtime import status


def _relative(path: Path, *, root: Path) -> str:
    """Return a stable repo-relative path for manifest serialization."""
    return str(path.resolve().relative_to(root.resolve()))


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


def build_paper_subset_manifest(root: Path | str = ".") -> Path:
    """Write the paper-subset artifact manifest for the current manuscript."""
    dirs = resolve_repo_dirs(root)
    repo_root = dirs["DIR_WORK"]
    workspace_root = dirs["DIR_WORKSPACE"]
    manuscript_dir = dirs["DIR_MANUSCRIPT"]
    figure_dir = dirs["DIR_MANUSCRIPT_FIGURE"]
    table_dir = dirs["DIR_MANUSCRIPT_TABLE"]
    manifest_path = manuscript_dir / "paper_subset_manifest.json"
    manuscript_dir.mkdir(parents=True, exist_ok=True)

    entries = [
        _figure_entry(
            label="fig:benchmark-evi-main",
            placement="main",
            generated_by="scripts/benchmark/evi_report.py",
            paths=[figure_dir / "benchmark_summary.pdf"],
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:benchmark-evi-targets",
            placement="main",
            generated_by="scripts/benchmark/evi_report.py",
            paths=[figure_dir / "benchmark_targets.pdf"],
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:benchmark-ei-main",
            placement="main",
            generated_by="scripts/benchmark/ei_report.py",
            paths=[figure_dir / "benchmark_ei_summary.pdf"],
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:benchmark-ei-targets",
            placement="main",
            generated_by="scripts/benchmark/ei_report.py",
            paths=[figure_dir / "benchmark_ei_targets.pdf"],
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:application-streamflow",
            placement="main",
            generated_by="scripts/application/build.py",
            paths=[
                figure_dir / "application_composite_tx_streamflow.pdf",
                figure_dir / "application_composite_fl_streamflow.pdf",
            ],
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:application-nfip",
            placement="main",
            generated_by="scripts/application/build.py",
            paths=[
                figure_dir / "application_composite_tx_nfip_claims.pdf",
                figure_dir / "application_composite_fl_nfip_claims.pdf",
            ],
            root=workspace_root,
        ),
        _table_entry(
            label="tab:application-summary-main",
            placement="main-supporting",
            generated_by="scripts/application/build.py",
            path=table_dir / "application_summary_main.tex",
            note="Paper-facing snapshot aligned to the curated four-case subset.",
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:benchmark-evi-shrinkage",
            placement="appendix",
            generated_by="scripts/benchmark/evi_report.py",
            paths=[figure_dir / "benchmark_shrinkage_sensitivity.pdf"],
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:benchmark-evi-stress",
            placement="appendix",
            generated_by="scripts/benchmark/evi_report.py",
            paths=[figure_dir / "benchmark_stress_summary.pdf"],
            root=workspace_root,
        ),
        _table_entry(
            label="tab:benchmark-evi-summary-main",
            placement="appendix",
            generated_by="scripts/benchmark/evi_report.py",
            path=table_dir / "benchmark_evi_summary_main.tex",
            root=workspace_root,
        ),
        _table_entry(
            label="tab:benchmark-stress-main",
            placement="appendix",
            generated_by="scripts/benchmark/evi_report.py",
            path=table_dir / "benchmark_stress_main.tex",
            root=workspace_root,
        ),
        _table_entry(
            label="tab:benchmark-record-length-main",
            placement="appendix",
            generated_by="scripts/benchmark/evi_report.py",
            path=table_dir / "benchmark_record_length_main.tex",
            root=workspace_root,
        ),
        _figure_entry(
            label="fig:benchmark-ei-shrinkage",
            placement="appendix",
            generated_by="scripts/benchmark/ei_report.py",
            paths=[figure_dir / "benchmark_ei_shrinkage_sensitivity.pdf"],
            root=workspace_root,
        ),
        _table_entry(
            label="tab:benchmark-ei-summary-main",
            placement="appendix",
            generated_by="scripts/benchmark/ei_report.py",
            path=table_dir / "benchmark_ei_summary_main.tex",
            root=workspace_root,
        ),
        _table_entry(
            label="tab:application-selection-sensitivity-main",
            placement="appendix",
            generated_by="scripts/application/build.py",
            path=table_dir / "application_selection_sensitivity_main.tex",
            root=workspace_root,
        ),
        _table_entry(
            label="tab:application-usgs-screening-main",
            placement="appendix",
            generated_by="scripts/application/build.py",
            path=table_dir / "application_usgs_screening_main.tex",
            root=workspace_root,
        ),
    ]
    payload = {
        "paper_scope": "curated four-case manuscript subset",
        "workspace_root": str(workspace_root),
        "code_repo_root": _relative(repo_root, root=workspace_root),
        "manuscript_repo_root": _relative(manuscript_dir, root=workspace_root),
        "manuscript_source": _relative(manuscript_dir / "0_manuscript.tex", root=workspace_root),
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest_path


def main() -> None:
    manifest_path = build_paper_subset_manifest()
    status("manuscript", f"paper_subset_manifest: {manifest_path}")


__all__ = ["build_paper_subset_manifest"]


if __name__ == "__main__":
    main()
