"""Remove only named outputs owned by the current repository workflows."""

from pathlib import Path

from benchmark.ei_benchmark import (
    _output_paths as ei_output_paths,
    _resolve_benchmark_n_obs as ei_n_obs,
)
from benchmark.evi_benchmark import (
    _output_paths as evi_output_paths,
    _resolve_benchmark_n_obs as evi_n_obs,
)
from config import resolve_repo_dirs
from application.specs import APPLICATIONS
from application.docs_cases import EXTRA_KEYS


def clean_generated(root: Path | str = ".") -> list[Path]:
    """Keep caches, other sample-size runs, and all unrecognized files intact."""
    dirs = resolve_repo_dirs(root)
    benchmark = dirs["DIR_OUT_BENCHMARK"]
    applications = dirs["DIR_OUT_APPLICATIONS"]
    figures = dirs["DIR_REPORT_FIGURE"]
    tables = dirs["DIR_REPORT_TABLE"]
    paths = [
        *evi_output_paths(benchmark, n_obs=evi_n_obs()).values(),
        *ei_output_paths(benchmark, n_obs=ei_n_obs()).values(),
        benchmark / "benchmark_evi_shrinkage_sensitivity.csv",
        benchmark / "benchmark_ei_shrinkage_sensitivity.csv",
        applications / "application_summary.json",
        applications / "report.html",
        dirs["DIR_REPORT"] / "report_subset_manifest.json",
    ]
    paths.extend(
        applications / "cases" / f"{stem}{suffix}"
        for stem in [spec.figure_stem for spec in APPLICATIONS]
        + [f"{key}_normalized" for key in EXTRA_KEYS]
        for suffix in (".png", ".json")
    )
    paths.extend(
        applications / f"application_{name}.csv"
        for name in (
            "series_registry",
            "screening",
            "summary",
            "design_life_levels",
            "design_life_intervals",
            "evi_methods",
            "ei_methods",
            "usgs_site_screening",
            "streamflow_gev_check",
        )
    )
    paths.extend(
        figures / f"{name}.pdf"
        for name in (
            "benchmark_evi_summary",
            "benchmark_evi_overview",
            "benchmark_evi_targets",
            "benchmark_evi_interval_sharpness",
            "benchmark_evi_shrinkage_sensitivity",
            "benchmark_ei_summary",
            "benchmark_ei_overview",
            "benchmark_ei_targets",
            "benchmark_ei_interval_sharpness",
            "benchmark_ei_shrinkage_sensitivity",
            "application_overview",
        )
    )
    paths.extend(
        figures / f"application_{kind}_{case}.pdf"
        for case in ("tx_streamflow", "fl_streamflow", "tx_nfip_claims", "fl_nfip_claims")
        for kind in ("ts", "evi", "target", "ei", "design_life", "composite")
    )
    paths.extend(
        tables / f"{name}.tex"
        for name in (
            "benchmark_evi_summary",
            "benchmark_evi_interval",
            "benchmark_evi_overview",
            "benchmark_ei_summary",
            "benchmark_ei_interval",
            "benchmark_ei_overview",
            "application_summary",
            "application_streamflow_gev_check",
            "application_design_life_levels",
            "application_ei",
            "application_selection_sensitivity",
            "application_extrapolation",
            "application_usgs_screening",
        )
    )
    # Validate the entire deletion set before touching any file. Never follow links.
    for path in paths:
        if any(parent.is_symlink() for parent in path.parents):
            raise ValueError(f"Generated output parent must not be a symlink: {path}")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Generated output must be a regular file: {path}")
    removed = []
    for path in paths:
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


if __name__ == "__main__":
    for removed in clean_generated():
        print(f"Removed {removed}")
