from __future__ import annotations
# ruff: noqa: E402

import unittest

import numpy as np
import pandas as pd

try:
    from . import _path_setup as test_paths
except ImportError:  # pragma: no cover
    import _path_setup as test_paths

test_paths.ensure_repo_import_paths()

import application.build as application
from data_prep.ghcn import PreparedSeries


def _make_prepared(series: pd.Series, *, name: str, provider: str, role: str) -> PreparedSeries:
    annual_maxima = series.groupby(series.index.year).max()
    return PreparedSeries(
        name=name,
        value_name="value",
        series=series,
        annual_maxima=annual_maxima,
        metadata={"provider": provider, "series_role": role},
    )


class ApplicationFacadeTests(unittest.TestCase):
    def test_facade_exports_expected_symbols(self) -> None:
        expected = {
            "ApplicationSpec",
            "ApplicationPreparedInputs",
            "ApplicationBundle",
            "build_application_inputs",
            "build_application_bundle",
            "build_application_bundles",
            "build_application_outputs",
            "application_summary_table",
            "load_usgs_frozen_sites",
            "plot_application_composite",
        }
        self.assertTrue(expected.issubset(set(application.__all__)))
        for name in expected:
            self.assertTrue(hasattr(application, name))

    def test_facade_builds_bundle_and_summary_table(self) -> None:
        rs = np.random.default_rng(17)
        index = pd.date_range("2000-01-01", periods=365 * 25, freq="D")
        values = pd.Series(rs.pareto(2.4, index.size) + 1.0, index=index, dtype=float)
        prepared = _make_prepared(values, name="facade", provider="synthetic", role="shared")
        inputs = application.ApplicationPreparedInputs(display=prepared, evi=prepared, ei=prepared)
        spec = application.ApplicationSpec(
            key="facade_synthetic",
            provider="ghcn",
            label="Facade synthetic",
            figure_stem="facade_synthetic",
            raw_key="none",
            ylabel="value",
            time_series_title="Facade synthetic series",
            scaling_title="Facade synthetic scaling",
            scaling_ylabel="log median block maximum",
        )

        bundle = application.build_application_bundle(spec, inputs)
        table = application.application_summary_table([bundle])

        self.assertEqual(
            table.columns.tolist(),
            ["Application", "xi", "theta_bb", "Mean cluster size", "10y_dll", "50y_dll"],
        )
        self.assertEqual(table.shape, (1, 6))
        self.assertEqual(table.loc[0, "Application"], "Facade synthetic")
        self.assertIn("[", table.loc[0, "xi"])
        self.assertIn("[", table.loc[0, "theta_bb"])
        self.assertNotEqual(table.loc[0, "Mean cluster size"], "NA")
        self.assertIn("[", table.loc[0, "10y_dll"])
        self.assertIn("[", table.loc[0, "50y_dll"])


if __name__ == "__main__":
    unittest.main()
