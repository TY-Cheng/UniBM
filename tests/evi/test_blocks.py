from __future__ import annotations

import unittest
import warnings

import numpy as np

from unibm.evi.blocks import block_maxima, block_summary_curve
from unibm.evi.targets import _quantile_summary_label, target_stability_summary


class EviBlocksTests(unittest.TestCase):
    def test_block_maxima_supports_sliding_and_disjoint_with_nonfinite_filtering(self) -> None:
        values = np.array([1.0, 5.0, np.nan, 2.0, 4.0, 3.0], dtype=float)
        np.testing.assert_allclose(
            block_maxima(values, 2, sliding=True), np.array([5.0, 4.0, 4.0])
        )
        np.testing.assert_allclose(block_maxima(values, 2, sliding=False), np.array([5.0, 4.0]))
        self.assertEqual(block_maxima(values, 10, sliding=True).size, 0)

    def test_block_summary_curve_warns_on_negative_and_nonpositive_values(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            curve = block_summary_curve(
                np.array([-1.0, 0.0, 0.0, 0.0, 1.0, 2.0], dtype=float),
                np.array([2, 3], dtype=int),
                sliding=False,
                target="quantile",
            )
        self.assertEqual(curve.block_sizes.size, 2)
        self.assertTrue(any("negative observations" in str(w.message) for w in caught))

        with warnings.catch_warnings(record=True) as caught_zeros:
            warnings.simplefilter("always")
            zero_curve = block_summary_curve(
                np.zeros(40, dtype=float),
                np.array([2, 4, 8], dtype=int),
                sliding=True,
                target="mean",
            )
        self.assertFalse(np.any(zero_curve.positive_mask))
        self.assertTrue(any("non-positive" in str(w.message) for w in caught_zeros))

    def test_quantile_summary_label_behaves_as_expected(self) -> None:
        self.assertEqual(_quantile_summary_label(0.5), "median")
        self.assertEqual(_quantile_summary_label(0.8), "quantile_tau_0.80")

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
