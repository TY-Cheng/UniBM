from __future__ import annotations
# ruff: noqa: E402

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_prep.ghcn import PreparedSeries
from application.build import ApplicationPreparedInputs, ApplicationSpec, build_application_bundle
from application.diagnostics import (
    application_design_life_interval_record,
    application_stationarity_records,
    scaling_residual_record,
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
    def test_stationarity_records_are_deterministic(self) -> None:
        bundle = _make_stream_bundle()

        record_a = application_stationarity_records(bundle)
        record_b = application_stationarity_records(bundle)

        self.assertEqual(record_a["application"], "tx_streamflow")
        self.assertEqual(record_a["severity_clock"], "calendar-day discharge")
        self.assertEqual(record_a["severity_pettitt_break"], record_b["severity_pettitt_break"])
        self.assertTrue(np.isfinite(float(record_a["severity_mk_p"])))
        self.assertTrue(np.isfinite(float(record_a["annual_maxima_mk_p"])))

    def test_scaling_residual_record_reports_finite_ranges(self) -> None:
        bundle = _make_stream_bundle()

        record = scaling_residual_record(bundle)

        self.assertGreater(int(record["plateau_points"]), 0)
        self.assertTrue(np.isfinite(float(record["residual_sd"])))
        self.assertTrue(np.isfinite(float(record["xi_range_lo"])))
        self.assertLessEqual(float(record["xi_range_lo"]), float(record["xi_range_hi"]))
        self.assertLessEqual(float(record["dll10_range_lo"]), float(record["dll10_range_hi"]))
        self.assertLessEqual(float(record["dll50_range_lo"]), float(record["dll50_range_hi"]))

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
