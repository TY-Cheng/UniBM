from __future__ import annotations

import unittest

import numpy as np

from unibm.ei.bm import (
    EI_DEFAULT_COVARIANCE_SHRINKAGE,
    _bb_wald_fit,
    _build_bm_estimate,
    _fit_pooled_z_model,
    _northrop_profile_fit,
    _pooled_z_fit,
    _regularize_ei_covariance,
    estimate_native_bm_ei,
    estimate_pooled_bm_ei,
)
from unibm.ei.paths import extract_stable_path_window
from unibm.ei.preparation import prepare_ei_bundle


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

    def test_native_and_pooled_bm_estimators(self) -> None:
        sample = self._positive_sample()
        bundle = prepare_ei_bundle(sample, block_sizes=np.array([4, 8, 16, 32], dtype=int))
        path = bundle.paths[("bb", True)]
        selected_levels, _ = extract_stable_path_window(path)
        bootstrap_result = {
            "block_sizes": selected_levels,
            "covariance": np.eye(selected_levels.size) * 0.05,
        }
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

        built = _build_bm_estimate(
            "bb_sliding_fgls",
            path,
            regression="FGLS",
            bootstrap_result={"block_sizes": np.array([999]), "covariance": np.eye(1)},
        )
        self.assertEqual(built.ci_variant, "ols")

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


if __name__ == "__main__":
    unittest.main()
