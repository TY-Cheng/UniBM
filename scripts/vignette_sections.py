"""Reusable section builders for the generated research vignette."""

from __future__ import annotations

try:  # pragma: no cover - exercised through script execution
    from vignette_cells import code_cell, md_cell
except ImportError:  # pragma: no cover - exercised through module import
    from .vignette_cells import code_cell, md_cell


def _python_list(values: list[str]) -> str:
    return "[" + ", ".join(repr(value) for value in values) + "]"


def single_application_section(
    *,
    heading: str,
    application_key: str,
    variable_prefix: str,
    include_target: bool,
    interpretation_code: str | None = None,
) -> list[dict]:
    """Build one notebook section for a single application case."""
    body_lines = [
        f"{variable_prefix}_bundle = application_bundle_map[{application_key!r}]",
        f"{variable_prefix}_summary = application_summary[",
        f'    application_summary["application"] == "{application_key}"',
        "].copy()",
        f"{variable_prefix}_return_levels = application_return_levels[",
        f'    application_return_levels["application"] == "{application_key}"',
        "].copy()",
        f"{variable_prefix}_ei = application_ei_methods[",
        f'    application_ei_methods["application"] == "{application_key}"',
        "].copy()",
        f"{variable_prefix}_methods = application_methods[",
        f'    application_methods["application"] == "{application_key}"',
        "].copy()",
        f"{variable_prefix}_screening = application_screening[",
        f'    application_screening["name"] == "{application_key}"',
        "].copy()",
        "",
        f"display({variable_prefix}_screening)",
        f"display({variable_prefix}_summary)",
        f"display({variable_prefix}_return_levels)",
        f"display({variable_prefix}_ei)",
        f"display({variable_prefix}_methods)",
        "",
        f"plot_application_time_series({variable_prefix}_bundle)",
        "plt.show()",
        f"plot_application_scaling({variable_prefix}_bundle)",
        "plt.show()",
    ]
    if include_target:
        body_lines.extend(
            [
                f"plot_application_target_stability({variable_prefix}_bundle)",
                "plt.show()",
            ]
        )
    body_lines.extend(
        [
            f"plot_application_ei({variable_prefix}_bundle)",
            "plt.show()",
            f"plot_application_return_levels({variable_prefix}_bundle)",
            "plt.show()",
        ]
    )
    cells = [md_cell(heading), code_cell("\n".join(body_lines))]
    if interpretation_code is not None:
        cells.append(code_cell(interpretation_code))
    return cells


def group_application_section(
    *,
    heading: str,
    application_keys: list[str],
    variable_prefix: str,
    include_registry: bool = False,
    include_target: bool = False,
    interpretation_code: str | None = None,
) -> list[dict]:
    """Build one notebook section for a grouped application comparison."""
    app_list = _python_list(application_keys)
    body_lines = [
        f"{variable_prefix}_bundles = [application_bundle_map[key] for key in {app_list}]",
        f"{variable_prefix}_screening = application_screening[",
        f'    application_screening["name"].isin({app_list})',
        "].copy()",
        f"{variable_prefix}_summary = application_summary[",
        f'    application_summary["application"].isin({app_list})',
        "].copy()",
        f"{variable_prefix}_return_levels = application_return_levels[",
        f'    application_return_levels["application"].isin({app_list})',
        "].copy()",
        f"{variable_prefix}_ei = application_ei_methods[",
        f'    application_ei_methods["application"].isin({app_list})',
        "].copy()",
        f"{variable_prefix}_methods = application_methods[",
        f'    application_methods["application"].isin({app_list})',
        "].copy()",
    ]
    if include_registry:
        body_lines.extend(
            [
                f"{variable_prefix}_registry = series_registry[",
                f'    series_registry["application"].isin({app_list})',
                "].copy()",
                f'display({variable_prefix}_registry[["application", "role", "series_name", "series_basis", "n_obs"]])',
            ]
        )
    body_lines.extend(
        [
            f"display({variable_prefix}_screening)",
            f"display({variable_prefix}_summary)",
            f"display({variable_prefix}_return_levels)",
            f"display({variable_prefix}_ei)",
            f"display({variable_prefix}_methods)",
            "",
        ]
    )
    for idx, _ in enumerate(application_keys):
        bundle_expr = f"{variable_prefix}_bundles[{idx}]"
        body_lines.extend(
            [
                f"plot_application_time_series({bundle_expr})",
                "plt.show()",
                f"plot_application_scaling({bundle_expr})",
                "plt.show()",
            ]
        )
        if include_target:
            body_lines.extend(
                [
                    f"plot_application_target_stability({bundle_expr})",
                    "plt.show()",
                ]
            )
        body_lines.extend(
            [
                f"plot_application_ei({bundle_expr})",
                "plt.show()",
                f"plot_application_return_levels({bundle_expr})",
                "plt.show()",
            ]
        )
    cells = [md_cell(heading), code_cell("\n".join(body_lines))]
    if interpretation_code is not None:
        cells.append(code_cell(interpretation_code))
    return cells


__all__ = ["group_application_section", "single_application_section"]
