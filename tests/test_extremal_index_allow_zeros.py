from __future__ import annotations

import unittest

import numpy as np

from unibm.ei import estimate_k_gaps, estimate_pooled_bm_ei, prepare_ei_bundle


class ExtremalIndexAllowZerosTests(unittest.TestCase):
    def test_prepare_ei_bundle_can_preserve_zero_filled_time_axis(self) -> None:
        base = np.array(
            [0, 0, 3, 0, 0, 5, 0, 7, 0, 0, 4, 0, 6, 0, 0, 8],
            dtype=float,
        )
        vec = np.tile(base, 12)
        bundle = prepare_ei_bundle(vec, allow_zeros=True)
        self.assertEqual(bundle.values.size, vec.size)
        primary = estimate_pooled_bm_ei(bundle, base_path="bb", sliding=True, regression="OLS")
        comparator = estimate_k_gaps(bundle)
        self.assertTrue(np.isfinite(primary.theta_hat))
        self.assertTrue(np.isfinite(comparator.theta_hat))
        self.assertGreater(primary.theta_hat, 0.0)
        self.assertLessEqual(primary.theta_hat, 1.0)


if __name__ == "__main__":
    unittest.main()
