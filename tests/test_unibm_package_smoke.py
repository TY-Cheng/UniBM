from __future__ import annotations

import unittest
from importlib import resources

import numpy as np

import unibm
from unibm.extremal_index import estimate_pooled_bm_ei, prepare_ei_bundle


class UniBmPackageSmokeTests(unittest.TestCase):
    @staticmethod
    def _sample(seed: int = 321, size: int = 512) -> np.ndarray:
        return np.random.default_rng(seed).pareto(2.0, size) + 1.0

    def test_top_level_package_import_supports_minimal_evi_workflow(self) -> None:
        sample = self._sample()
        fit = unibm.estimate_evi_quantile(sample, quantile=0.5, sliding=True, bootstrap_reps=0)
        design_life = unibm.estimate_design_life_level(fit, years=np.array([10.0, 50.0]))

        self.assertTrue(np.isfinite(fit.slope))
        self.assertEqual(len(fit.confidence_interval), 2)
        self.assertEqual(len(fit.plateau_bounds), 2)
        np.testing.assert_equal(design_life.shape, (2,))
        self.assertTrue(np.all(np.isfinite(design_life)))
        self.assertEqual(unibm.__version__, "0.1.0")

    def test_top_level_package_import_supports_minimal_formal_ei_workflow(self) -> None:
        sample = self._sample(seed=654)
        bundle = prepare_ei_bundle(sample)
        fit = estimate_pooled_bm_ei(bundle, base_path="bb", sliding=True, regression="OLS")

        self.assertTrue(np.isfinite(fit.theta_hat))
        self.assertEqual(len(fit.confidence_interval), 2)
        self.assertIsNotNone(fit.stable_window)
        self.assertEqual(fit.regression, "OLS")
        self.assertEqual(fit.base_path, "bb")
        self.assertGreater(len(fit.path_level), 0)
        self.assertEqual(len(fit.path_level), len(fit.path_theta))
        self.assertTrue(resources.files("unibm").joinpath("py.typed").is_file())


if __name__ == "__main__":
    unittest.main()
