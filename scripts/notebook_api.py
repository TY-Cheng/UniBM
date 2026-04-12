"""Compact notebook-facing facade for vignette and exploratory sessions."""

from __future__ import annotations

from application.build import (
    build_application_bundles,
    build_application_outputs,
    plot_application_composite,
    plot_application_design_life_levels,
    plot_application_ei,
    plot_application_overview,
    plot_application_scaling,
    plot_application_target_stability,
    plot_application_time_series,
    seasonal_monthly_pit_unit_frechet,
)
from benchmark.design import CORE_METHODS, UNIVERSAL_BENCHMARK_SET
from benchmark.ei_report import (
    build_ei_benchmark_manuscript_outputs,
    ei_core_story_table,
    ei_interval_story_table,
    ei_story_latex,
    ei_targets_story_table,
    plot_ei_core_panels,
    plot_ei_interval_sharpness_scatter,
    plot_ei_targets_panels,
)
from benchmark.evi_external import (
    interval_sharpness_story_latex,
    interval_sharpness_story_table,
    plot_interval_sharpness_scatter,
    plot_target_plus_external_panels,
    target_plus_external_story_latex,
    target_plus_external_story_table,
)
from benchmark.evi_report import (
    benchmark_story_latex,
    benchmark_story_table,
    benchmark_table,
    build_evi_benchmark_manuscript_outputs,
    plot_benchmark_panels,
)

__all__ = [
    "CORE_METHODS",
    "UNIVERSAL_BENCHMARK_SET",
    "benchmark_story_latex",
    "benchmark_story_table",
    "benchmark_table",
    "build_application_bundles",
    "build_application_outputs",
    "build_ei_benchmark_manuscript_outputs",
    "build_evi_benchmark_manuscript_outputs",
    "ei_core_story_table",
    "ei_interval_story_table",
    "ei_story_latex",
    "ei_targets_story_table",
    "interval_sharpness_story_latex",
    "interval_sharpness_story_table",
    "plot_application_composite",
    "plot_application_design_life_levels",
    "plot_application_ei",
    "plot_application_overview",
    "plot_application_scaling",
    "plot_application_target_stability",
    "plot_application_time_series",
    "plot_benchmark_panels",
    "plot_ei_core_panels",
    "plot_ei_interval_sharpness_scatter",
    "plot_ei_targets_panels",
    "plot_interval_sharpness_scatter",
    "plot_target_plus_external_panels",
    "seasonal_monthly_pit_unit_frechet",
    "target_plus_external_story_latex",
    "target_plus_external_story_table",
]
