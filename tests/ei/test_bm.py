from __future__ import annotations

import unittest

import numpy as np

from unibm.ei.bm import (
    EI_DEFAULT_COVARIANCE_SHRINKAGE,
    _bb_wald_fit,
    _fit_pooled_z_model,
    _northrop_profile_fit,
    _pooled_z_fit,
    _regularize_ei_covariance,
    estimate_native_bm_ei,
    estimate_pooled_bm_ei,
)
from unibm.ei.models import EiPathBundle, EiPreparedBundle, EiStableWindow
from unibm.ei.preparation import prepare_ei_bundle
from unibm.ei.selection import extract_stable_path_window


def _ei_bootstrap_result(
    block_sizes: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, object]:
    return {
        "block_sizes": block_sizes,
        "covariance": covariance,
        "base_path": "bb",
        "sliding": True,
    }


class EiBmTests(unittest.TestCase):
    @staticmethod
    def _positive_sample(size: int = 512, seed: int = 404) -> np.ndarray:
        rs = np.random.default_rng(seed)
        return rs.pareto(2.0, size) + 1.0

    def test_pooled_model_helpers(self) -> None:
        z = np.array([0.2, 0.25, 0.22, 0.24], dtype=float)
        covariance = np.eye(4) * 0.05
        regularized = _regularize_ei_covariance(
            covariance, covariance_shrinkage=EI_DEFAULT_COVARIANCE_SHRINKAGE
        )
        self.assertEqual(regularized.shape, (4, 4))
        ols = _fit_pooled_z_model(z)
        gls = _fit_pooled_z_model(z, covariance=covariance)
        self.assertTrue(np.isfinite(ols["intercept"]))
        self.assertTrue(np.isfinite(gls["intercept"]))
        z_hat, se, variant = _pooled_z_fit(z, covariance=covariance)
        self.assertTrue(np.isfinite(z_hat))
        self.assertTrue(np.isfinite(se))
        self.assertEqual(variant, "bootstrap_cov")

    def test_ei_preparation_rejects_multidimensional_series(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            prepare_ei_bundle(
                np.ones((8, 8), dtype=float),
                allow_zeros=False,
                block_sizes=np.array([4, 8, 16, 32], dtype=int),
            )

    def test_native_and_pooled_bm_estimators(self) -> None:
        sample = self._positive_sample()
        bundle = prepare_ei_bundle(
            sample,
            block_sizes=np.array([4, 8, 16, 32], dtype=int),
            allow_zeros=False,
        )
        path = bundle.paths[("bb", True)]
        selected_levels, _ = extract_stable_path_window(path)
        bootstrap_result = _ei_bootstrap_result(
            selected_levels,
            np.eye(selected_levels.size) * 0.05,
        )
        pooled = estimate_pooled_bm_ei(
            bundle,
            base_path="bb",
            sliding=True,
            regression="FGLS",
            bootstrap_result=bootstrap_result,
            covariance_shrinkage=1.0,
        )
        native_bb = estimate_native_bm_ei(bundle, base_path="bb", sliding=True)
        native_northrop = estimate_native_bm_ei(
            bundle,
            base_path="northrop",
            sliding=True,
            use_adjusted_chandwich=True,
        )
        self.assertTrue(np.isfinite(pooled.theta_hat))
        self.assertTrue(np.isfinite(native_bb.theta_hat))
        self.assertTrue(np.isfinite(native_northrop.theta_hat))
        self.assertIsNotNone(pooled.z_standard_error)
        self.assertAlmostEqual(
            pooled.standard_error,
            pooled.theta_hat * pooled.z_standard_error,
        )
        self.assertIsNone(native_bb.z_standard_error)
        self.assertIsNone(native_northrop.z_standard_error)

        with self.assertRaisesRegex(ValueError, "FGLS requires bootstrap covariance"):
            estimate_pooled_bm_ei(
                bundle,
                base_path="bb",
                sliding=True,
                regression="FGLS",
            )

    def test_pooled_fgls_constrains_theta_to_parameter_space(self) -> None:
        block_sizes = np.array([4, 8], dtype=int)
        z_path = np.array([0.8, 0.01], dtype=float)
        path = EiPathBundle(
            base_path="bb",
            sliding=True,
            block_sizes=block_sizes,
            theta_path=np.exp(-z_path),
            eir_path=np.exp(z_path),
            z_path=z_path,
            sample_counts=np.array([32, 32], dtype=int),
            sample_statistics={},
            stable_window=EiStableWindow(4, 8),
            selected_level=4,
        )
        bundle = EiPreparedBundle(
            values=np.ones(32, dtype=float),
            block_sizes=block_sizes,
            paths={("bb", True): path},
            threshold_candidates={},
        )
        covariance = np.array([[4.0, 1.5], [1.5, 1.0]], dtype=float)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))

        fit = estimate_pooled_bm_ei(
            bundle,
            base_path="bb",
            sliding=True,
            regression="FGLS",
            bootstrap_result=_ei_bootstrap_result(block_sizes, covariance),
            covariance_shrinkage=0.0,
        )

        self.assertEqual(fit.theta_hat, 1.0)
        self.assertLessEqual(fit.confidence_interval[0], fit.theta_hat)
        self.assertGreaterEqual(fit.confidence_interval[1], fit.theta_hat)
        self.assertEqual(fit.ci_variant, "bootstrap_cov_boundary")
        self.assertIsNotNone(fit.z_standard_error)
        self.assertAlmostEqual(fit.standard_error, fit.z_standard_error)

    def test_fgls_subsets_full_grid_covariance_by_block_size(self) -> None:
        block_sizes = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64], dtype=int)
        bundle = prepare_ei_bundle(
            self._positive_sample(size=1024),
            block_sizes=block_sizes,
            allow_zeros=False,
        )
        selected_levels, _ = extract_stable_path_window(bundle.paths[("bb", True)])
        self.assertLess(selected_levels.size, block_sizes.size)
        bootstrap_levels = block_sizes[::-1]
        full_covariance = np.diag(np.linspace(0.01, 0.09, block_sizes.size))
        lookup = {int(level): idx for idx, level in enumerate(bootstrap_levels)}
        selected_idx = np.asarray([lookup[int(level)] for level in selected_levels], dtype=int)

        full_grid = estimate_pooled_bm_ei(
            bundle,
            base_path="bb",
            sliding=True,
            regression="FGLS",
            bootstrap_result=_ei_bootstrap_result(bootstrap_levels, full_covariance),
        )
        selected_grid = estimate_pooled_bm_ei(
            bundle,
            base_path="bb",
            sliding=True,
            regression="FGLS",
            bootstrap_result=_ei_bootstrap_result(
                selected_levels,
                full_covariance[np.ix_(selected_idx, selected_idx)],
            ),
        )

        self.assertEqual(full_grid.ci_variant, selected_grid.ci_variant)
        self.assertTrue(full_grid.ci_variant.startswith("bootstrap_cov"))
        self.assertAlmostEqual(full_grid.theta_hat, selected_grid.theta_hat)
        np.testing.assert_allclose(
            full_grid.confidence_interval, selected_grid.confidence_interval
        )

    def test_pooled_bm_rejects_unknown_regression(self) -> None:
        bundle = prepare_ei_bundle(
            self._positive_sample(),
            block_sizes=np.array([4, 8, 16, 32], dtype=int),
            allow_zeros=False,
        )

        with self.assertRaisesRegex(ValueError, "regression must be 'OLS' or 'FGLS'"):
            estimate_pooled_bm_ei(
                bundle,
                base_path="bb",
                sliding=True,
                regression="typo",
            )

    def test_pooled_bm_rejects_mismatched_bootstrap_estimator_identity(self) -> None:
        bundle = prepare_ei_bundle(
            self._positive_sample(seed=818),
            block_sizes=np.array([4, 8, 16, 32], dtype=int),
            allow_zeros=False,
        )
        selected_levels, _ = extract_stable_path_window(bundle.paths[("bb", True)])
        bootstrap_result = {
            "block_sizes": selected_levels,
            "covariance": np.eye(selected_levels.size),
            "base_path": "bb",
            "sliding": True,
        }

        for field, replacement in (("base_path", "northrop"), ("sliding", False)):
            with self.subTest(field=field, case="mismatch"):
                mismatched = dict(bootstrap_result)
                mismatched[field] = replacement
                with self.assertRaisesRegex(ValueError, field):
                    estimate_pooled_bm_ei(
                        bundle,
                        base_path="bb",
                        sliding=True,
                        regression="FGLS",
                        bootstrap_result=mismatched,
                    )
            with self.subTest(field=field, case="missing"):
                missing = dict(bootstrap_result)
                del missing[field]
                with self.assertRaisesRegex(ValueError, field):
                    estimate_pooled_bm_ei(
                        bundle,
                        base_path="bb",
                        sliding=True,
                        regression="FGLS",
                        bootstrap_result=missing,
                    )

        with self.assertRaisesRegex(ValueError, "OLS does not accept bootstrap_result"):
            estimate_pooled_bm_ei(
                bundle,
                base_path="bb",
                sliding=True,
                regression="OLS",
                bootstrap_result=bootstrap_result,
            )

    def test_pooled_bm_uses_shared_covariance_validation(self) -> None:
        bundle = prepare_ei_bundle(
            self._positive_sample(),
            block_sizes=np.array([4, 8, 16, 32], dtype=int),
            allow_zeros=False,
        )
        selected_levels, _ = extract_stable_path_window(bundle.paths[("bb", True)])

        singular = estimate_pooled_bm_ei(
            bundle,
            base_path="bb",
            sliding=True,
            regression="FGLS",
            bootstrap_result=_ei_bootstrap_result(
                selected_levels,
                np.ones((selected_levels.size, selected_levels.size)),
            ),
            covariance_shrinkage=0.0,
        )
        self.assertEqual(singular.regression, "FGLS")

        with self.assertRaisesRegex(ValueError, "positive scale"):
            estimate_pooled_bm_ei(
                bundle,
                base_path="bb",
                sliding=True,
                regression="FGLS",
                bootstrap_result=_ei_bootstrap_result(
                    selected_levels,
                    np.zeros((selected_levels.size, selected_levels.size)),
                ),
            )
        for invalid in (np.nan, "auto"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "covariance_shrinkage"):
                    estimate_pooled_bm_ei(
                        bundle,
                        base_path="bb",
                        sliding=True,
                        regression="OLS",
                        covariance_shrinkage=invalid,
                    )

    def test_fixed_b_helpers_return_finite_values(self) -> None:
        stats = np.array([1.4, 1.5, 1.3, 1.45], dtype=float)
        theta_hat, interval, se = _bb_wald_fit(stats, block_size=8)
        self.assertTrue(np.isfinite(theta_hat))
        self.assertTrue(np.isfinite(se))
        self.assertLessEqual(interval[0], theta_hat)
        northrop_plain = _northrop_profile_fit(stats, adjusted=False)
        northrop_adjusted = _northrop_profile_fit(stats, adjusted=True)
        self.assertTrue(np.isfinite(northrop_plain[0]))
        self.assertTrue(np.isfinite(northrop_adjusted[0]))
        expected_plain_se = northrop_plain[0] / np.sqrt(stats.size)
        self.assertAlmostEqual(northrop_plain[2], expected_plain_se)
        hessian = -stats.size / northrop_adjusted[0] ** 2
        scores = (1.0 / northrop_adjusted[0]) - stats
        expected_adjusted_se = np.sqrt(np.sum(scores**2)) / abs(hessian)
        self.assertAlmostEqual(northrop_adjusted[2], expected_adjusted_se)
        with self.assertRaisesRegex(ValueError, "at least two finite positive statistics"):
            _bb_wald_fit(np.array([1.0]), block_size=8)
        with self.assertRaisesRegex(ValueError, "at least two finite positive statistics"):
            _northrop_profile_fit(np.array([1.0]), adjusted=False)

    def test_northrop_boundary_estimate_is_inside_profile_interval(self) -> None:
        theta_hat, interval, _, _ = _northrop_profile_fit(
            np.array([0.5, 0.75, 0.8, 0.9]),
            adjusted=False,
        )

        self.assertEqual(theta_hat, 1.0)
        self.assertLessEqual(interval[0], theta_hat)
        self.assertGreaterEqual(interval[1], theta_hat)


if __name__ == "__main__":
    unittest.main()
