from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.unibm.diagnostics import (
    _as_finite_1d,
    _kernel_bandwidth,
    _quantile_summary_label,
    _rolling_min_summary,
    _select_stable_sd_index,
    empirical_cdf,
    estimate_extremal_index_reciprocal,
    kernel_cdf,
    marginal_cdf,
    target_stability_summary,
)


class UniBmDiagnosticsTests(unittest.TestCase):
    def test_as_finite_1d_filters_nonfinite_values(self) -> None:
        values = _as_finite_1d([1.0, np.nan, 2.0, np.inf, -3.0])
        np.testing.assert_allclose(values, np.array([1.0, 2.0, -3.0], dtype=float))

    def test_kernel_cdf_handles_empty_singleton_and_array_queries(self) -> None:
        empty = kernel_cdf([])
        self.assertTrue(np.isnan(empty(1.0)))
        singleton = kernel_cdf([2.0])
        self.assertEqual(singleton(1.5), 0.0)
        self.assertEqual(singleton(2.0), 1.0)
        estimator = kernel_cdf([1.0, 2.0, 3.0], bandwidth=0.25)
        result = estimator(np.array([1.0, 2.0]))
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all((result >= 0.0) & (result <= 1.0)))

    def test_kernel_bandwidth_and_marginal_cdf_validate_arguments(self) -> None:
        sample = np.array([1.0, 2.0, 4.0, 8.0], dtype=float)
        self.assertGreater(_kernel_bandwidth(sample, "scott"), 0.0)
        self.assertGreater(_kernel_bandwidth(sample, "silverman"), 0.0)
        self.assertGreater(_kernel_bandwidth(sample, 0.5), 0.0)
        with self.assertRaisesRegex(ValueError, "Unsupported bandwidth rule"):
            _kernel_bandwidth(sample, "bad-rule")
        with self.assertRaisesRegex(ValueError, "Only the Gaussian kernel"):
            kernel_cdf(sample, kernel="epanechnikov")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "Unsupported marginal CDF method"):
            marginal_cdf(sample, method="unsupported")  # type: ignore[arg-type]

    def test_empirical_cdf_and_quantile_label_behave_as_expected(self) -> None:
        empty = empirical_cdf([])
        self.assertTrue(np.isnan(empty(1.0)))
        singleton = empirical_cdf([3.0])
        self.assertEqual(singleton(2.0), 0.0)
        self.assertEqual(singleton(3.0), 1.0)
        estimator = empirical_cdf([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(estimator(2.0), 2.0 / 5.0)
        np.testing.assert_allclose(estimator(np.array([0.0, 5.0])), np.array([0.0, 4.0 / 5.0]))
        self.assertEqual(_quantile_summary_label(0.5), "median")
        self.assertEqual(_quantile_summary_label(0.8), "quantile_tau_0.80")
        empirical = marginal_cdf([1.0, 2.0, 3.0], method="empirical")
        self.assertAlmostEqual(empirical(2.0), 2.0 / 4.0)

    def test_rolling_min_summary_and_stable_sd_selector(self) -> None:
        scores = pd.Series([4.0, 3.0, 2.0, 5.0, 1.0, 6.0], dtype=float)
        block_sizes = np.array([2, 3], dtype=int)
        means, standard_deviations = _rolling_min_summary(scores, block_sizes)
        self.assertEqual(means.shape, (2,))
        self.assertEqual(standard_deviations.shape, (2,))
        self.assertTrue(np.all(np.isfinite(means)))
        self.assertEqual(_select_stable_sd_index(np.array([np.nan, 1.5, 0.5])), 2)
        with self.assertRaisesRegex(ValueError, "No finite standard deviations"):
            _select_stable_sd_index(np.array([np.nan, np.nan]))

    def test_estimate_extremal_index_reciprocal_returns_finite_diagnostics(self) -> None:
        index = pd.date_range("2001-01-01", periods=365 * 8, freq="D")
        values = pd.Series(np.linspace(0.1, 10.0, index.size), index=index, dtype=float)
        fit = estimate_extremal_index_reciprocal(
            values,
            num_step=8,
            min_block_size=5,
            max_block_size=20,
            geom=False,
            cdf_method="empirical",
        )
        self.assertGreater(fit.block_sizes.size, 0)
        self.assertTrue(np.all(np.isfinite(fit.northrop_values)))
        self.assertTrue(np.all(np.isfinite(fit.bb_values)))
        self.assertGreaterEqual(fit.northrop_estimate, 1.0)
        self.assertGreaterEqual(fit.bb_estimate, 1.0)
        with self.assertRaisesRegex(ValueError, "at least two finite non-negative observations"):
            estimate_extremal_index_reciprocal(
                pd.Series([np.nan, -1.0], index=pd.date_range("2000-01-01", periods=2, freq="D"))
            )

        long_index = pd.date_range("1990-01-01", periods=6001, freq="D")
        long_values = pd.Series(
            np.linspace(0.1, 20.0, long_index.size), index=long_index, dtype=float
        )
        long_fit = estimate_extremal_index_reciprocal(long_values, cdf_method="empirical")
        self.assertGreater(long_fit.block_sizes.size, 0)

    def test_target_stability_summary_returns_expected_columns(self) -> None:
        rs = np.random.default_rng(31)
        values = rs.pareto(2.5, 512) + 1.0
        block_sizes = np.array([4, 8, 16], dtype=int)
        summary = target_stability_summary(values, block_sizes, sliding=True, quantile=0.8)
        self.assertEqual(
            list(summary.columns),
            ["block_size", "quantile_tau_0.80", "mean", "mode"],
        )
        self.assertEqual(summary.shape[0], block_sizes.size)


if __name__ == "__main__":
    unittest.main()
