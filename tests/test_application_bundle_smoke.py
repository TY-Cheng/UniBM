from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from data_prep.ghcn import PreparedSeries
from workflows.application import (
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
            return_level_basis="claim_active_day",
            return_level_label="claim-active-day return period (years)",
        )

        bundle = build_application_bundle(spec, inputs)

        self.assertLess(bundle.prepared.evi.series.size, bundle.prepared.ei.series.size)
        self.assertTrue((bundle.prepared.evi.series > 0.0).all())
        self.assertTrue((bundle.prepared.ei.series >= 0.0).all())
        self.assertTrue(np.isfinite(bundle.evi_fit.slope))
        self.assertTrue(np.isfinite(bundle.ei_primary.theta_hat))
        self.assertTrue(np.isfinite(bundle.ei_comparator.theta_hat))


if __name__ == "__main__":
    unittest.main()
