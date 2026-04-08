"""Application-side workflow facade for manuscript-ready real-data analyses."""
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

from application.fit import (
    build_application_bundle,
    build_application_bundles,
)
from application.inputs import (
    build_application_inputs,
    ensure_ghcn_raw_data,
    ensure_nfip_raw_data,
    ensure_usgs_raw_data,
    load_usgs_frozen_sites,
)
from application.outputs import (
    application_ei_method_rows,
    application_design_life_level_table,
    application_method_rows,
    application_summary_record,
    application_summary_table,
    build_application_outputs,
    plot_application_composite,
    plot_application_ei,
    plot_application_design_life_levels,
    plot_application_overview,
    plot_application_scaling,
    plot_application_target_stability,
    plot_application_time_series,
    seasonal_monthly_pit_unit_frechet,
    write_application_figures,
)
from application.specs import (
    APPLICATIONS,
    ApplicationBundle,
    ApplicationPreparedInputs,
    ApplicationSpec,
)
from shared.runtime import status


__all__ = [
    "APPLICATIONS",
    "ApplicationBundle",
    "ApplicationPreparedInputs",
    "ApplicationSpec",
    "application_ei_method_rows",
    "application_design_life_level_table",
    "application_method_rows",
    "application_summary_record",
    "application_summary_table",
    "build_application_bundle",
    "build_application_bundles",
    "build_application_inputs",
    "build_application_outputs",
    "ensure_ghcn_raw_data",
    "ensure_nfip_raw_data",
    "ensure_usgs_raw_data",
    "load_usgs_frozen_sites",
    "plot_application_composite",
    "plot_application_ei",
    "plot_application_design_life_levels",
    "plot_application_overview",
    "plot_application_scaling",
    "plot_application_target_stability",
    "plot_application_time_series",
    "seasonal_monthly_pit_unit_frechet",
    "write_application_figures",
]


def main() -> None:
    outputs = build_application_outputs()
    for name, path in outputs.items():
        status("application", f"{name}: {path}")


if __name__ == "__main__":
    main()
