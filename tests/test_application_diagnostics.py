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

from data_prep.ghcn import PreparedSeries
from application.build import ApplicationPreparedInputs, ApplicationSpec, build_application_bundle
from application.diagnostics import (
    application_design_life_interval_record,
)


def _make_prepared(series: pd.Series, *, name: str, provider: str, role: str) -> PreparedSeries:
    annual_maxima = series.groupby(series.index.year).max()
    return PreparedSeries(
        name=name,
        value_name="value",
        series=series,
        annual_maxima=annual_maxima,
        metadata={"provider": provider, "series_role": role},
    )


def _make_stream_bundle() -> object:
    rng = np.random.default_rng(17)
    index = pd.date_range("2000-01-01", periods=365 * 18, freq="D")
    values = pd.Series(rng.pareto(2.1, index.size) + 1.0, index=index, dtype=float)
    prepared = _make_prepared(values, name="synthetic_stream", provider="usgs", role="shared")
    inputs = ApplicationPreparedInputs(display=prepared, evi=prepared, ei=prepared)
    spec = ApplicationSpec(
        key="tx_streamflow",
        provider="usgs",
        label="Texas streamflow",
        figure_stem="synthetic_stream",
        raw_key="none",
        ylabel="value",
        time_series_title="Synthetic streamflow",
        scaling_title="Synthetic streamflow scaling",
        scaling_ylabel="log median block maximum",
        observations_per_year=365.25,
        target_stability_title="Synthetic target stability",
    )
    return build_application_bundle(spec, inputs)


class ApplicationDiagnosticsTests(unittest.TestCase):
    def test_design_life_interval_record_contains_point_estimate(self) -> None:
        bundle = _make_stream_bundle()

        record = application_design_life_interval_record(bundle)

        self.assertEqual(record["tau"], 0.5)
        self.assertGreater(float(record["dll10"]), 0.0)
        self.assertGreater(float(record["dll50"]), float(record["dll10"]))
        self.assertLessEqual(float(record["dll10_lo"]), float(record["dll10"]))
        self.assertGreaterEqual(float(record["dll10_hi"]), float(record["dll10"]))
        self.assertLessEqual(float(record["dll50_lo"]), float(record["dll50"]))
        self.assertGreaterEqual(float(record["dll50_hi"]), float(record["dll50"]))


if __name__ == "__main__":
    unittest.main()
