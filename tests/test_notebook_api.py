from __future__ import annotations

import unittest

from scripts.workflows import notebook_api


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
            "build_ei_benchmark_manuscript_outputs",
            "build_evi_benchmark_manuscript_outputs",
            "plot_application_overview",
            "plot_application_time_series",
            "plot_application_scaling",
            "plot_application_ei",
            "plot_application_return_levels",
            "plot_benchmark_panels",
            "plot_ei_core_panels",
            "plot_interval_sharpness_scatter",
        }
        self.assertTrue(expected.issubset(set(notebook_api.__all__)))
        for name in expected:
            self.assertTrue(hasattr(notebook_api, name))


if __name__ == "__main__":
    unittest.main()
