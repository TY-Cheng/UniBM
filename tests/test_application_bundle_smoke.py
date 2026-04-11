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
from application.build import (
    ApplicationPreparedInputs,
    ApplicationSpec,
    build_application_bundle,
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


class ApplicationBundleSmokeTests(unittest.TestCase):
    def test_standard_application_bundle_runs_end_to_end(self) -> None:
        rs = np.random.default_rng(7)
        index = pd.date_range("2000-01-01", periods=365 * 30, freq="D")
        values = pd.Series(rs.pareto(2.5, index.size) + 1.0, index=index, dtype=float)
        prepared = _make_prepared(values, name="synthetic", provider="synthetic", role="shared")
        inputs = ApplicationPreparedInputs(display=prepared, evi=prepared, ei=prepared)
        spec = ApplicationSpec(
            key="synthetic",
            provider="ghcn",
            label="Synthetic",
            figure_stem="synthetic",
            raw_key="none",
            ylabel="value",
            time_series_title="Synthetic series",
            scaling_title="Synthetic scaling",
            scaling_ylabel="log median block maximum",
        )

        bundle = build_application_bundle(spec, inputs)

        self.assertTrue(np.isfinite(bundle.evi_fit.slope))
        self.assertTrue(np.isfinite(bundle.ei_primary.theta_hat))
        self.assertTrue(np.isfinite(bundle.ei_comparator.theta_hat))

    def test_fema_style_application_bundle_uses_positive_evi_and_zero_filled_ei(self) -> None:
        rs = np.random.default_rng(11)
        index = pd.date_range("2005-01-01", periods=365 * 25, freq="D")
        zero_filled = pd.Series(0.0, index=index, dtype=float)
        active_mask = rs.random(index.size) < 0.08
        zero_filled.loc[active_mask] = (rs.pareto(1.8, active_mask.sum()) + 1.0) * 1000.0
        positive_only = zero_filled[zero_filled > 0.0]

        inputs = ApplicationPreparedInputs(
            display=_make_prepared(
                zero_filled,
                name="NFIP display",
                provider="fema",
                role="display",
            ),
            evi=_make_prepared(
                positive_only,
                name="NFIP active-day payouts",
                provider="fema",
                role="evi",
            ),
            ei=_make_prepared(
                zero_filled,
                name="NFIP payout waves",
                provider="fema",
                role="ei",
            ),
        )
        spec = ApplicationSpec(
            key="synthetic_nfip",
            provider="fema",
            label="Synthetic NFIP",
            figure_stem="synthetic_nfip",
            raw_key="none",
            ylabel="usd",
            time_series_title="Synthetic NFIP",
            scaling_title="Synthetic NFIP scaling",
            scaling_ylabel="log median positive payouts",
            design_life_level_basis="claim_active_day",
            design_life_level_label="claim-active-day design-life (years)",
            ei_allow_zeros=True,
        )

        bundle = build_application_bundle(spec, inputs)

        self.assertLess(bundle.prepared.evi.series.size, bundle.prepared.ei.series.size)
        self.assertTrue((bundle.prepared.evi.series > 0.0).all())
        self.assertTrue((bundle.prepared.ei.series >= 0.0).all())
        self.assertTrue(np.isfinite(bundle.evi_fit.slope))
        self.assertTrue(np.isfinite(bundle.ei_primary.theta_hat))
        self.assertTrue(np.isfinite(bundle.ei_comparator.theta_hat))

    def test_non_fema_bundle_can_preserve_zero_inflated_ei_calendar(self) -> None:
        rs = np.random.default_rng(29)
        index = pd.date_range("1990-01-01", periods=365 * 25, freq="D")
        zero_filled = pd.Series(0.0, index=index, dtype=float)
        active_mask = rs.random(index.size) < 0.25
        zero_filled.loc[active_mask] = rs.gamma(shape=2.5, scale=4.0, size=active_mask.sum())

        prepared = _make_prepared(
            zero_filled,
            name="zero-inflated precipitation",
            provider="ghcn",
            role="shared",
        )
        inputs = ApplicationPreparedInputs(display=prepared, evi=prepared, ei=prepared)
        spec = ApplicationSpec(
            key="zero_inflated_precip",
            provider="ghcn",
            label="Zero-inflated precipitation",
            figure_stem="zero_inflated_precip",
            raw_key="none",
            ylabel="value",
            time_series_title="Zero-inflated precipitation",
            scaling_title="Zero-inflated scaling",
            scaling_ylabel="log median block maximum",
            ei_allow_zeros=True,
        )

        bundle = build_application_bundle(spec, inputs)

        self.assertGreater(int((bundle.prepared.ei.series == 0.0).sum()), 0)
        self.assertTrue(np.isfinite(bundle.ei_primary.theta_hat))
        self.assertTrue(np.isfinite(bundle.ei_comparator.theta_hat))

    def test_evi_only_application_bundle_skips_formal_ei_fits(self) -> None:
        rs = np.random.default_rng(31)
        index = pd.date_range("1995-01-01", periods=365 * 15, freq="D")
        values = pd.Series(
            rs.gamma(shape=2.0, scale=3.5, size=index.size), index=index, dtype=float
        )

        prepared = _make_prepared(values, name="evi_only", provider="ghcn", role="shared")
        inputs = ApplicationPreparedInputs(display=prepared, evi=prepared, ei=prepared)
        spec = ApplicationSpec(
            key="evi_only",
            provider="ghcn",
            label="EVI-only",
            figure_stem="evi_only",
            raw_key="none",
            ylabel="value",
            time_series_title="EVI-only",
            scaling_title="EVI-only scaling",
            scaling_ylabel="log median block maximum",
            formal_ei=False,
        )

        bundle = build_application_bundle(spec, inputs)

        self.assertTrue(np.isfinite(bundle.evi_fit.slope))
        self.assertIsNone(bundle.ei_bundle)
        self.assertIsNone(bundle.ei_bb_sliding_fgls)
        with self.assertRaises(ValueError):
            _ = bundle.ei_primary


if __name__ == "__main__":
    unittest.main()
