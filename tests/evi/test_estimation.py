from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from unibm._block_grid import generate_block_sizes, validate_block_sizes
from unibm.evi._regression import (
    _aligned_bootstrap_covariance,
    _fit_linear_model,
    _fit_scaling_model,
)
from unibm.evi.blocks import block_summary_curve
from unibm.evi.design import (
    _design_block_sizes,
    estimate_design_life_level,
    estimate_design_life_level_interval,
    predict_block_quantile,
)
from unibm.evi.estimation import estimate_evi_quantile, estimate_target_scaling
from unibm.evi.models import BlockSummaryCurve, PlateauWindow
from unibm.evi.selection import select_penultimate_window


def _evi_bootstrap_result(
    block_sizes: np.ndarray,
    covariance: np.ndarray | None,
) -> dict[str, object]:
    return {
        "block_sizes": block_sizes,
        "covariance": covariance,
        "target": "quantile",
        "quantile": 0.5,
        "sliding": True,
    }


class EviEstimationTests(unittest.TestCase):
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

    def test_validate_block_sizes_rejects_invalid_custom_grids(self) -> None:
        np.testing.assert_array_equal(
            validate_block_sizes(np.array([4.0, 8.0, 16.0]), n_obs=32),
            np.array([4, 8, 16]),
        )
        invalid_grids = (
            np.array([4.0, 8.5, 16.0]),
            np.array([4, 8, 8, 16]),
            np.array([4, 16, 8, 32]),
            np.array([1, 4, 8]),
            np.array([4, 8, 33]),
        )
        for block_sizes in invalid_grids:
            with self.subTest(block_sizes=block_sizes):
                with self.assertRaisesRegex(ValueError, "block_sizes"):
                    validate_block_sizes(block_sizes, n_obs=32)

    def test_fit_linear_model_and_bootstrap_covariance_alignment(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        y = 0.5 + 2.0 * x
        ols = _fit_linear_model(x, y)
        self.assertAlmostEqual(ols["slope"], 2.0, places=6)
        covariance = np.eye(4) * 0.1
        gls = _fit_linear_model(x, y, covariance=covariance)
        self.assertAlmostEqual(gls["slope"], 2.0, places=6)

        curve = BlockSummaryCurve(
            target="quantile",
            quantile=0.5,
            sliding=True,
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
        covariance_factor = np.array(
            [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0]],
            dtype=float,
        )
        cov = covariance_factor @ covariance_factor.T
        aligned = _aligned_bootstrap_covariance(
            _evi_bootstrap_result(np.array([4, 8, 16], dtype=int), cov),
            curve,
            plateau,
        )
        np.testing.assert_allclose(aligned, cov[1:, 1:])
        self.assertIsNone(_aligned_bootstrap_covariance(None, curve, plateau))
        self.assertIsNone(
            _aligned_bootstrap_covariance(
                _evi_bootstrap_result(np.array([4, 8, 16], dtype=int), None),
                curve,
                plateau,
            )
        )
        with self.assertRaisesRegex(ValueError, "block_sizes"):
            _aligned_bootstrap_covariance(
                {
                    "covariance": np.array([]),
                    "target": "quantile",
                    "quantile": 0.5,
                    "sliding": True,
                },
                curve,
                plateau,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            _aligned_bootstrap_covariance(
                _evi_bootstrap_result(np.array([4, 8, 16]), np.eye(2)),
                curve,
                plateau,
            )
        curve_with_gap = BlockSummaryCurve(
            target="quantile",
            quantile=0.5,
            sliding=True,
            block_sizes=np.array([4, 8, 16, 32], dtype=int),
            counts=np.array([10, 8, 4, 2], dtype=int),
            values=np.array([2.0, -1.0, 5.0, 7.0], dtype=float),
            positive_mask=np.array([True, False, True, True], dtype=bool),
        )
        gap_plateau = PlateauWindow(
            start=0,
            stop=2,
            score=0.0,
            mask=np.array([True, True, False], dtype=bool),
            x=curve_with_gap.log_block_sizes[:2],
            y=curve_with_gap.log_values[:2],
        )
        reordered = covariance_factor @ covariance_factor.T
        aligned_reordered = _aligned_bootstrap_covariance(
            _evi_bootstrap_result(
                np.array([32, 4, 16], dtype=int),
                reordered,
            ),
            curve_with_gap,
            gap_plateau,
        )
        expected = reordered[np.ix_([1, 2, 0], [1, 2, 0])][:2, :2]
        np.testing.assert_allclose(aligned_reordered, expected)

    def test_fit_linear_model_condition_number_and_covariance_validation(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        y = 1.5 + 0.25 * x
        with mock.patch("unibm.evi._regression.np.linalg.cond", side_effect=np.linalg.LinAlgError):
            model = _fit_linear_model(x, y)
        self.assertEqual(model["condition_number"], float("inf"))
        with self.assertRaisesRegex(ValueError, "shape must match"):
            _fit_linear_model(x, y, covariance=np.eye(3))

    def test_public_evi_records_explicit_ols_contract(self) -> None:
        fit = estimate_evi_quantile(
            self._positive_sample(),
            regression="OLS",
            block_sizes=np.array([4, 8, 16, 32, 64], dtype=int),
            bootstrap_reps=0,
            plateau_points=3,
        )

        self.assertEqual(fit.regression_policy, "OLS")
        self.assertEqual(fit.regression, "OLS")
        self.assertEqual(fit.ci_variant, "hc0")
        self.assertIsNone(fit.bootstrap)

    def test_public_evi_rejects_invalid_regression_bootstrap_combinations(self) -> None:
        values = self._positive_sample()
        block_sizes = np.array([4, 8, 16, 32, 64], dtype=int)

        with self.assertRaisesRegex(ValueError, "regression must be 'OLS', 'FGLS', or 'AUTO'"):
            estimate_evi_quantile(values, regression="typo", block_sizes=block_sizes)
        with self.assertRaisesRegex(ValueError, "OLS does not use bootstrap"):
            estimate_evi_quantile(
                values,
                regression="OLS",
                block_sizes=block_sizes,
                bootstrap_reps=2,
            )
        with self.assertRaisesRegex(ValueError, "OLS does not accept bootstrap_result"):
            estimate_evi_quantile(
                values,
                regression="OLS",
                block_sizes=block_sizes,
                bootstrap_result={"block_sizes": block_sizes, "covariance": np.eye(5)},
            )
        with self.assertRaisesRegex(ValueError, "bootstrap_reps must be an integer at least 2"):
            estimate_evi_quantile(
                values,
                regression="FGLS",
                block_sizes=block_sizes,
                bootstrap_reps=1,
            )
        with self.assertRaisesRegex(ValueError, "bootstrap_reps must be None"):
            estimate_evi_quantile(
                values,
                regression="FGLS",
                block_sizes=block_sizes,
                bootstrap_reps=0,
                bootstrap_result={"block_sizes": block_sizes, "covariance": np.eye(5)},
            )

    def test_public_evi_fails_closed_without_supplied_covariance(self) -> None:
        values = self._positive_sample()
        block_sizes = np.array([4, 8, 16, 32, 64], dtype=int)
        missing_covariance = _evi_bootstrap_result(block_sizes, None)

        for regression in ("FGLS", "AUTO"):
            with self.subTest(regression=regression):
                with self.assertRaisesRegex(ValueError, "supplied bootstrap_result"):
                    estimate_evi_quantile(
                        values,
                        regression=regression,
                        block_sizes=block_sizes,
                        bootstrap_result=missing_covariance,
                        plateau_points=3,
                    )

    def test_public_evi_fgls_bootstraps_when_small_sample_geometry_is_feasible(self) -> None:
        fit = estimate_evi_quantile(
            self._positive_sample(size=48, seed=818),
            regression="FGLS",
            block_sizes=np.array([2, 3, 4, 6, 8], dtype=int),
            bootstrap_reps=20,
            plateau_points=3,
            random_state=17,
        )

        self.assertEqual(fit.regression_policy, "FGLS")
        self.assertEqual(fit.regression, "FGLS")
        self.assertEqual(fit.ci_variant, "bootstrap_cov")
        self.assertIsNotNone(fit.bootstrap)
        self.assertIsNotNone(fit.bootstrap["covariance"])
        self.assertEqual(fit.covariance_shrinkage_policy, "fixed")
        self.assertEqual(fit.covariance_shrinkage, 0.37)
        self.assertEqual(fit.bootstrap_block_length_policy, "default")
        self.assertEqual(fit.bootstrap_block_length, 12)
        self.assertEqual(fit.bootstrap_reps_requested, 20)
        self.assertEqual(fit.bootstrap_reps_used, 20)

    def test_public_evi_rejects_mismatched_bootstrap_estimator_identity(self) -> None:
        values = self._positive_sample(seed=811)
        block_sizes = np.array([4, 8, 16, 32, 64], dtype=int)
        bootstrap_result = {
            "block_sizes": block_sizes,
            "covariance": np.eye(block_sizes.size),
            "target": "quantile",
            "quantile": 0.5,
            "sliding": True,
        }

        fit = estimate_evi_quantile(
            values,
            regression="FGLS",
            block_sizes=block_sizes,
            bootstrap_result=bootstrap_result,
            plateau_points=3,
        )
        self.assertEqual(fit.regression, "FGLS")
        for field, replacement in (
            ("target", "mean"),
            ("quantile", 0.9),
            ("sliding", False),
        ):
            with self.subTest(field=field, case="mismatch"):
                mismatched = dict(bootstrap_result)
                mismatched[field] = replacement
                with self.assertRaisesRegex(ValueError, field):
                    estimate_evi_quantile(
                        values,
                        regression="FGLS",
                        block_sizes=block_sizes,
                        bootstrap_result=mismatched,
                        plateau_points=3,
                    )
            with self.subTest(field=field, case="missing"):
                missing = dict(bootstrap_result)
                del missing[field]
                with self.assertRaisesRegex(ValueError, field):
                    estimate_evi_quantile(
                        values,
                        regression="FGLS",
                        block_sizes=block_sizes,
                        bootstrap_result=missing,
                        plateau_points=3,
                    )

    def test_public_evi_rejects_mismatched_curve_or_plateau_reuse(self) -> None:
        values = self._positive_sample(seed=812)
        block_sizes = np.array([4, 8, 16, 32, 64], dtype=int)
        curve = block_summary_curve(
            values,
            block_sizes,
            target="quantile",
            quantile=0.5,
            sliding=True,
        )
        plateau = select_penultimate_window(
            curve.log_block_sizes,
            curve.log_values,
            min_points=3,
        )

        for field, replacement in (
            ("target", "mean"),
            ("quantile", 0.9),
            ("sliding", False),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    estimate_evi_quantile(
                        values,
                        regression="OLS",
                        curve=replace(curve, **{field: replacement}),
                        plateau=plateau,
                        bootstrap_reps=0,
                        plateau_points=3,
                    )

        bad_mask = plateau.mask.copy()
        bad_mask[:] = False
        for field, replacement in (
            ("mask", bad_mask),
            ("x", plateau.x + 1.0),
            ("y", plateau.y + 1.0),
            ("start", plateau.start + 1),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "plateau"):
                    estimate_evi_quantile(
                        values,
                        regression="OLS",
                        curve=curve,
                        plateau=replace(plateau, **{field: replacement}),
                        bootstrap_reps=0,
                        plateau_points=3,
                    )

    def test_public_evi_rejects_invalid_quantiles(self) -> None:
        values = self._positive_sample(seed=813)
        block_sizes = np.array([4, 8, 16, 32, 64], dtype=int)
        for quantile in (0.0, 1.0, np.nan, np.inf, True):
            with self.subTest(quantile=quantile):
                with self.assertRaisesRegex(ValueError, "quantile"):
                    estimate_evi_quantile(
                        values,
                        regression="OLS",
                        quantile=quantile,
                        block_sizes=block_sizes,
                        bootstrap_reps=0,
                        plateau_points=3,
                    )

    def test_public_evi_auto_falls_back_only_for_internal_covariance_unavailability(self) -> None:
        values = self._positive_sample(size=32, seed=828)
        block_sizes = np.array([4, 8, 16], dtype=int)

        auto_fit = estimate_evi_quantile(
            values,
            regression="AUTO",
            block_sizes=block_sizes,
            bootstrap_reps=2,
            plateau_points=2,
        )
        self.assertEqual(auto_fit.regression_policy, "AUTO")
        self.assertEqual(auto_fit.regression, "OLS")
        self.assertEqual(auto_fit.ci_variant, "hc0")
        self.assertIsNone(auto_fit.bootstrap["covariance"])

        with self.assertRaisesRegex(ValueError, "FGLS requires usable bootstrap covariance"):
            estimate_evi_quantile(
                values,
                regression="FGLS",
                block_sizes=block_sizes,
                bootstrap_reps=2,
                plateau_points=2,
            )

    def test_public_evi_validates_supplied_covariance(self) -> None:
        values = self._positive_sample(seed=919)
        block_sizes = np.array([4, 8, 16, 32, 64], dtype=int)

        singular = estimate_evi_quantile(
            values,
            regression="FGLS",
            block_sizes=block_sizes,
            bootstrap_result=_evi_bootstrap_result(
                block_sizes[::-1],
                np.ones((5, 5), dtype=float),
            ),
            covariance_shrinkage=0.0,
            plateau_points=3,
        )
        self.assertEqual(singular.regression, "FGLS")

        near_psd = np.ones((5, 5), dtype=float)
        near_psd[np.triu_indices(5, 1)] += 1e-14
        corrected = estimate_evi_quantile(
            values,
            regression="FGLS",
            block_sizes=block_sizes,
            bootstrap_result=_evi_bootstrap_result(block_sizes, near_psd),
            covariance_shrinkage=0.0,
            plateau_points=3,
        )
        self.assertEqual(corrected.regression, "FGLS")

        invalid_covariances = {
            "finite": np.full((5, 5), np.nan),
            "positive scale": np.zeros((5, 5)),
            "symmetric": np.triu(np.ones((5, 5))),
            "positive semidefinite": np.full((5, 5), 2.0) - np.eye(5),
        }
        for message, covariance in invalid_covariances.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    estimate_evi_quantile(
                        values,
                        regression="FGLS",
                        block_sizes=block_sizes,
                        bootstrap_result=_evi_bootstrap_result(block_sizes, covariance),
                        plateau_points=3,
                    )

        baseline = estimate_evi_quantile(
            values,
            regression="OLS",
            block_sizes=block_sizes,
            bootstrap_reps=0,
            plateau_points=3,
        )
        selected_labels = set(int(label) for label in baseline.plateau_block_sizes)
        unused_index = next(
            idx for idx, label in enumerate(block_sizes) if int(label) not in selected_labels
        )
        invalid_full_grid = np.eye(block_sizes.size, dtype=float)
        invalid_full_grid[unused_index, unused_index] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            estimate_evi_quantile(
                values,
                regression="FGLS",
                block_sizes=block_sizes,
                bootstrap_result=_evi_bootstrap_result(block_sizes, invalid_full_grid),
                plateau_points=3,
            )
        with self.assertRaisesRegex(ValueError, "block_sizes"):
            estimate_evi_quantile(
                values,
                regression="FGLS",
                block_sizes=block_sizes,
                bootstrap_result={
                    "covariance": np.eye(5),
                    "target": "quantile",
                    "quantile": 0.5,
                    "sliding": True,
                },
                plateau_points=3,
            )

    def test_public_evi_rejects_invalid_covariance_shrinkage(self) -> None:
        values = self._positive_sample()
        block_sizes = np.array([4, 8, 16, 32, 64], dtype=int)

        for shrinkage in (-0.1, 1.1, np.nan, np.inf, True, "auto", "AUTO", "bad"):
            with self.subTest(shrinkage=shrinkage):
                with self.assertRaisesRegex(ValueError, "covariance_shrinkage"):
                    estimate_evi_quantile(
                        values,
                        regression="OLS",
                        block_sizes=block_sizes,
                        covariance_shrinkage=shrinkage,
                        plateau_points=3,
                    )

    def test_scaling_fit_entrypoints_return_finite_results(self) -> None:
        values = self._positive_sample()
        block_sizes = np.array([4, 8, 16, 32, 64], dtype=int)
        curve = block_summary_curve(values, block_sizes, sliding=True, target="quantile")
        plateau = select_penultimate_window(curve.log_block_sizes, curve.log_values, min_points=3)
        fit = _fit_scaling_model(
            values,
            target="quantile",
            regression="OLS",
            quantile=0.5,
            sliding=True,
            curve=curve,
            plateau=plateau,
            bootstrap_reps=0,
            plateau_points=3,
        )
        self.assertTrue(np.isfinite(fit.slope))

        quantile_fit = estimate_evi_quantile(
            values,
            regression="OLS",
            block_sizes=block_sizes,
            bootstrap_reps=0,
            plateau_points=3,
        )
        mean_fit = estimate_target_scaling(
            values,
            regression="OLS",
            target="mean",
            sliding=False,
            block_sizes=block_sizes,
            bootstrap_reps=0,
            plateau_points=3,
        )
        mode_fit = estimate_target_scaling(
            values,
            regression="OLS",
            target="mode",
            sliding=True,
            block_sizes=block_sizes,
            bootstrap_reps=0,
            plateau_points=3,
        )
        self.assertTrue(np.isfinite(quantile_fit.slope))
        self.assertTrue(np.isfinite(mean_fit.slope))
        self.assertTrue(np.isfinite(mode_fit.slope))
        auto_fit = _fit_scaling_model(
            values,
            target="mean",
            regression="AUTO",
            sliding=True,
            block_sizes=None,
            num_step=4,
            min_block_size=4,
            max_block_size=16,
            plateau_points=3,
            bootstrap_reps=2,
            super_block_size=32,
            random_state=3,
        )
        self.assertIsNotNone(auto_fit.bootstrap)
        self.assertTrue(np.isfinite(auto_fit.slope))

        with self.assertRaisesRegex(ValueError, "At least 32 finite observations"):
            _fit_scaling_model(np.array([1.0, 2.0, 3.0]), target="quantile", regression="OLS")
        with (
            self.assertWarnsRegex(RuntimeWarning, "non-positive block summaries"),
            self.assertRaisesRegex(ValueError, "Not enough positive block summaries"),
        ):
            _fit_scaling_model(np.zeros(64, dtype=float), target="quantile", regression="OLS")

    def test_prediction_and_design_life_level_helpers(self) -> None:
        values = self._positive_sample(seed=202)
        fit = estimate_evi_quantile(
            values,
            regression="OLS",
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
        np.testing.assert_array_equal(
            _design_block_sizes(np.array([1.0, 10.0]), 365.25),
            np.array([366.0, 3653.0]),
        )
        self.assertAlmostEqual(levels[0], predict_block_quantile(fit, 366.0))
        self.assertAlmostEqual(levels[1], predict_block_quantile(fit, 3653.0))
        interval_lo, interval_hi = estimate_design_life_level_interval(
            fit,
            np.array([1.0, 10.0]),
            observations_per_year=365.25,
        )
        self.assertEqual(levels.shape, (2,))
        self.assertEqual(interval_lo.shape, (2,))
        self.assertEqual(interval_hi.shape, (2,))
        self.assertTrue(np.all(interval_lo <= levels))
        self.assertTrue(np.all(levels <= interval_hi))
        scalar_lo, scalar_hi = estimate_design_life_level_interval(
            fit,
            10.0,
            observations_per_year=365.25,
        )
        self.assertLessEqual(scalar_lo, float(levels[1]))
        self.assertGreaterEqual(scalar_hi, float(levels[1]))
        with self.assertRaisesRegex(ValueError, "quantile-based ScalingFit"):
            predict_block_quantile(
                estimate_target_scaling(
                    values,
                    regression="OLS",
                    target="mean",
                    block_sizes=np.array([4, 8, 16, 32, 64], dtype=int),
                    bootstrap_reps=0,
                    plateau_points=3,
                ),
                10.0,
            )
        with self.assertRaisesRegex(ValueError, "quantile-based ScalingFit"):
            estimate_design_life_level(
                estimate_target_scaling(
                    values,
                    regression="OLS",
                    target="mean",
                    block_sizes=np.array([4, 8, 16, 32, 64], dtype=int),
                    bootstrap_reps=0,
                    plateau_points=3,
                ),
                10.0,
            )
        with self.assertRaisesRegex(ValueError, "same tau"):
            estimate_design_life_level(fit, 10.0, tau=0.9)
        with self.assertRaisesRegex(ValueError, "same tau"):
            estimate_design_life_level_interval(fit, 10.0, tau=0.9)
        with self.assertRaisesRegex(ValueError, "Block size must be positive"):
            predict_block_quantile(fit, 0.0)
        with self.assertRaisesRegex(ValueError, "Design-life years must be positive"):
            estimate_design_life_level_interval(fit, 0.0)

    def test_design_life_helpers_reject_nonfinite_or_inverted_inputs(self) -> None:
        fit = estimate_evi_quantile(
            self._positive_sample(seed=909),
            regression="OLS",
            block_sizes=np.array([4, 8, 16, 32, 64], dtype=int),
            bootstrap_reps=0,
            plateau_points=3,
        )

        with self.assertRaisesRegex(ValueError, "Block size must be positive and finite"):
            predict_block_quantile(fit, np.nan)
        for function in (estimate_design_life_level, estimate_design_life_level_interval):
            with self.subTest(function=function.__name__, parameter="years"):
                with self.assertRaisesRegex(ValueError, "years must be positive and finite"):
                    function(fit, np.nan)
            with self.subTest(function=function.__name__, parameter="observations_per_year"):
                with self.assertRaisesRegex(
                    ValueError, "observations_per_year must be finite and positive"
                ):
                    function(fit, 10.0, observations_per_year=np.inf)
        for z_crit in (-1.0, np.nan):
            with self.subTest(z_crit=z_crit):
                with self.assertRaisesRegex(ValueError, "z_crit must be finite and non-negative"):
                    estimate_design_life_level_interval(fit, 10.0, z_crit=z_crit)
        with self.assertRaisesRegex(ValueError, "finite coefficient covariance"):
            estimate_design_life_level_interval(
                replace(fit, cov_beta=np.array([[1.0, np.nan], [np.nan, 1.0]])),
                10.0,
            )


if __name__ == "__main__":
    unittest.main()
