from __future__ import annotations

import unittest
import warnings

import numpy as np

from unibm.core import (
    _aligned_bootstrap_covariance,
    _fit_linear_model,
    _fit_scaling_model,
    block_maxima,
    block_summary_curve,
    estimate_evi_quantile,
    estimate_design_life_level,
    estimate_target_scaling,
    generate_block_sizes,
    predict_block_quantile,
    select_penultimate_window,
)
from unibm.models import BlockSummaryCurve, PlateauWindow


class UniBmCoreTests(unittest.TestCase):
    @staticmethod
    def _positive_sample(size: int = 512, seed: int = 101) -> np.ndarray:
        rs = np.random.default_rng(seed)
        return rs.pareto(2.2, size) + 1.0

    def test_generate_block_sizes_validates_and_supports_linear_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least 32 observations"):
            generate_block_sizes(16)
        adjusted = generate_block_sizes(64, min_block_size=8, max_block_size=8, num_step=4)
        self.assertTrue(np.all(adjusted >= 8))
        block_sizes = generate_block_sizes(
            128,
            num_step=5,
            min_block_size=4,
            max_block_size=20,
            geom=False,
        )
        np.testing.assert_allclose(block_sizes, np.array([4, 8, 12, 16, 20]))

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

    def test_fit_linear_model_and_bootstrap_covariance_alignment(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        y = 0.5 + 2.0 * x
        ols = _fit_linear_model(x, y)
        self.assertAlmostEqual(ols["slope"], 2.0, places=6)
        covariance = np.eye(4) * 0.1
        gls = _fit_linear_model(x, y, covariance=covariance)
        self.assertAlmostEqual(gls["slope"], 2.0, places=6)
        gls_no_shrink = _fit_linear_model(x, y, covariance=covariance, covariance_shrinkage=0.0)
        self.assertAlmostEqual(gls_no_shrink["slope"], 2.0, places=6)

        curve = BlockSummaryCurve(
            block_sizes=np.array([4, 8, 16], dtype=int),
            counts=np.array([10, 8, 4], dtype=int),
            values=np.array([2.0, 3.0, 5.0], dtype=float),
            positive_mask=np.array([True, True, True], dtype=bool),
        )
        plateau = PlateauWindow(
            start=1,
            stop=3,
            score=0.0,
            mask=np.array([False, True, True], dtype=bool),
            x=np.log(np.array([8, 16], dtype=float)),
            y=np.log(np.array([3.0, 5.0], dtype=float)),
        )
        cov = np.arange(9, dtype=float).reshape(3, 3)
        aligned = _aligned_bootstrap_covariance(
            {"covariance": cov, "block_sizes": np.array([4, 8, 16], dtype=int)},
            curve,
            plateau,
        )
        np.testing.assert_allclose(aligned, cov[1:, 1:])
        self.assertIsNone(_aligned_bootstrap_covariance(None, curve, plateau))
        self.assertIsNone(_aligned_bootstrap_covariance({"covariance": None}, curve, plateau))
        self.assertIsNone(
            _aligned_bootstrap_covariance({"covariance": np.asarray([])}, curve, plateau)
        )
        self.assertIsNone(
            _aligned_bootstrap_covariance(
                {"covariance": np.eye(2), "block_sizes": np.array([4, 8, 16], dtype=int)},
                curve,
                plateau,
            )
        )
        self.assertIsNone(
            _aligned_bootstrap_covariance(
                {"covariance": np.eye(3), "block_sizes": np.array([4, 12, 16], dtype=int)},
                curve,
                plateau,
            )
        )
        self.assertIsNone(_aligned_bootstrap_covariance({"covariance": np.eye(2)}, curve, plateau))

    def test_select_penultimate_window_and_scaling_fits(self) -> None:
        values = self._positive_sample()
        block_sizes = np.array([4, 8, 16, 32, 64], dtype=int)
        curve = block_summary_curve(values, block_sizes, sliding=True, target="quantile")
        plateau = select_penultimate_window(curve.log_block_sizes, curve.log_values, min_points=3)
        self.assertGreaterEqual(plateau.stop - plateau.start, 3)
        reset_plateau = select_penultimate_window(
            np.log(np.array([4.0, 8.0, 16.0, 32.0, 64.0])),
            np.log(np.array([2.0, 2.5, 3.2, 4.0, 5.0])),
            min_points=5,
            trim_fraction=0.49,
        )
        self.assertEqual(reset_plateau.start, 0)
        self.assertEqual(reset_plateau.stop, 5)

        fit = _fit_scaling_model(
            values,
            target="quantile",
            quantile=0.5,
            sliding=True,
            curve=curve,
            plateau=plateau,
            bootstrap_reps=0,
            plateau_points=3,
        )
        self.assertTrue(np.isfinite(fit.slope))

        mean_fit = estimate_target_scaling(
            values,
            target="mean",
            sliding=False,
            block_sizes=block_sizes,
            bootstrap_reps=0,
            plateau_points=3,
        )
        mode_fit = estimate_target_scaling(
            values,
            target="mode",
            sliding=True,
            block_sizes=block_sizes,
            bootstrap_reps=0,
            plateau_points=3,
        )
        self.assertTrue(np.isfinite(mean_fit.slope))
        self.assertTrue(np.isfinite(mode_fit.slope))

        with self.assertRaisesRegex(ValueError, "At least 32 finite observations"):
            _fit_scaling_model(np.array([1.0, 2.0, 3.0]), target="quantile")
        with self.assertRaisesRegex(ValueError, "Not enough positive block summaries"):
            _fit_scaling_model(np.zeros(64, dtype=float), target="quantile")

    def test_evi_prediction_and_design_life_level_helpers(self) -> None:
        values = self._positive_sample(seed=202)
        fit = estimate_evi_quantile(
            values,
            block_sizes=np.array([4, 8, 16, 32, 64], dtype=int),
            bootstrap_reps=0,
            plateau_points=3,
        )
        prediction = predict_block_quantile(fit, 10.0)
        self.assertTrue(np.isfinite(prediction))
        levels = estimate_design_life_level(
            fit,
            np.array([1.0, 10.0]),
            observations_per_year=365.25,
        )
        self.assertEqual(levels.shape, (2,))
        with self.assertRaisesRegex(ValueError, "quantile-based ScalingFit"):
            predict_block_quantile(
                estimate_target_scaling(
                    values,
                    target="mean",
                    block_sizes=np.array([4, 8, 16, 32, 64], dtype=int),
                    bootstrap_reps=0,
                    plateau_points=3,
                ),
                10.0,
            )
        with self.assertRaisesRegex(ValueError, "Block size must be positive"):
            predict_block_quantile(fit, 0.0)


if __name__ == "__main__":
    unittest.main()
