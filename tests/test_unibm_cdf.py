from __future__ import annotations

import unittest

import numpy as np

from unibm.cdf import empirical_cdf


class UniBmCdfTests(unittest.TestCase):
    def test_empirical_cdf_behaves_as_expected(self) -> None:
        empty = empirical_cdf([])
        self.assertTrue(np.isnan(empty(1.0)))
        singleton = empirical_cdf([3.0])
        self.assertEqual(singleton(2.0), 0.0)
        self.assertEqual(singleton(3.0), 1.0)
        estimator = empirical_cdf([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(estimator(2.0), 2.0 / 5.0)
        result = estimator(np.array([2.0, 3.0]))
        self.assertEqual(result.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_empirical_cdf_propagates_nan_queries(self) -> None:
        for values in ([3.0], [1.0, 2.0, 3.0]):
            with self.subTest(values=values):
                estimator = empirical_cdf(values)
                self.assertTrue(np.isnan(estimator(np.nan)))
                result = estimator(np.array([0.0, np.nan, 4.0]))
                self.assertTrue(np.isnan(result[1]))

    def test_empirical_cdf_flattens_sample_and_omits_nonfinite_values(self) -> None:
        estimator = empirical_cdf(np.array([[1.0, 2.0], [3.0, np.nan]]))

        self.assertAlmostEqual(estimator(2.0), 2.0 / 4.0)


if __name__ == "__main__":
    unittest.main()
