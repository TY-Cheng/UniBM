"""Application-side workflow facade for manuscript-ready real-data analyses."""
# ruff: noqa: E402

from __future__ import annotations

if __package__ in {None, ""}:
    from import_bootstrap import ensure_scripts_on_path_from_entry

    ensure_scripts_on_path_from_entry(__file__)

from workflows.application_fit import (
    build_application_bundle,
    build_application_bundles,
)
from workflows.application_inputs import (
    build_application_inputs,
    ensure_ghcn_raw_data,
    ensure_nfip_raw_data,
    ensure_usgs_raw_data,
    load_usgs_frozen_sites,
)
from workflows.application_outputs import (
    application_ei_method_rows,
    application_method_rows,
    plot_application_ei,
    plot_application_overview,
    plot_application_return_levels,
    plot_application_scaling,
    plot_application_target_stability,
    plot_application_time_series,
    application_summary_record,
    application_summary_table,
    build_application_outputs,
    write_application_figures,
)
from workflows.application_specs import (
    APPLICATIONS,
    ApplicationBundle,
    ApplicationPreparedInputs,
    ApplicationSpec,
)
from workflows.workflow_runtime import status


__all__ = [
    "APPLICATIONS",
    "ApplicationBundle",
    "ApplicationPreparedInputs",
    "ApplicationSpec",
    "application_ei_method_rows",
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
    "plot_application_ei",
    "plot_application_overview",
    "plot_application_return_levels",
    "plot_application_scaling",
    "plot_application_target_stability",
    "plot_application_time_series",
    "write_application_figures",
]


def main() -> None:
    outputs = build_application_outputs()
    for name, path in outputs.items():
        status("application", f"{name}: {path}")


if __name__ == "__main__":
    main()
