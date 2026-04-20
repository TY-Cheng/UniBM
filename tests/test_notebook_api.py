from __future__ import annotations
# ruff: noqa: E402

import unittest

try:
    from . import _path_setup as test_paths
except ImportError:  # pragma: no cover
    import _path_setup as test_paths

test_paths.ensure_repo_import_paths()

import notebook_api


class NotebookApiTests(unittest.TestCase):
    def test_notebook_api_exports_expected_symbols(self) -> None:
        expected = {
            "CORE_METHODS",
            "UNIVERSAL_BENCHMARK_SET",
            "benchmark_story_latex",
            "benchmark_story_table",
            "benchmark_table",
            "build_application_bundles",
            "build_application_outputs",
            "plot_application_composite",
            "plot_application_overview",
            "plot_application_time_series",
            "plot_application_scaling",
            "plot_application_ei",
            "plot_application_design_life_levels",
            "seasonal_monthly_pit_unit_frechet",
            "plot_benchmark_panels",
            "plot_ei_core_panels",
            "plot_interval_sharpness_scatter",
        }
        self.assertTrue(expected.issubset(set(notebook_api.__all__)))
        for name in expected:
            self.assertTrue(hasattr(notebook_api, name))


if __name__ == "__main__":
    unittest.main()
