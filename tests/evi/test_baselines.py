from __future__ import annotations

import unittest

import numpy as np

from unibm.evi.baselines import (
    _dedh_moment_path,
    _dedh_standard_error,
    _finite_positive,
    _hill_path,
    _hill_standard_error,
    _max_spectrum_curve,
    _max_spectrum_path,
    _normalize_standard_error,
    _pickands_path,
    _pickands_standard_error,
    _positive_finite_in_order,
    _select_from_path,
    _weighted_slope_with_se,
    candidate_max_spectrum_scales,
    candidate_tail_counts,
    estimate_dedh_moment_evi,
    estimate_hill_evi,
    estimate_max_spectrum_evi,
    estimate_pickands_evi,
    select_stable_integer_window,
    select_stable_tail_window,
    wald_confidence_interval,
)


class EviBaselinesTests(unittest.TestCase):
    @staticmethod
    def _pareto_sample(size: int = 4096, seed: int = 17) -> np.ndarray:
        rs = np.random.default_rng(seed)
        return rs.pareto(2.0, size) + 1.0

    def test_finite_positive_orders_descending(self) -> None:
        sample = np.array([np.nan, 2.0, 5.0, 1.0, np.inf, 4.0, 3.0, 6.0, 7.0, 8.0], dtype=float)
        ordered = _finite_positive(sample)
        np.testing.assert_allclose(
            ordered,
            np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=float),
        )
        in_order = _positive_finite_in_order(sample)
        np.testing.assert_allclose(in_order, np.array([2.0, 5.0, 1.0, 4.0, 3.0, 6.0, 7.0, 8.0]))

    def test_confidence_intervals_and_stable_window_selection(self) -> None:
        lo, hi = wald_confidence_interval(1.0, 0.5, ci_level=0.95)
        self.assertLess(lo, 1.0)
        self.assertGreater(hi, 1.0)
        self.assertTrue(np.isnan(_normalize_standard_error(-1.0)))
        self.assertEqual(_normalize_standard_error(0.25), 0.25)

        tail_counts = candidate_tail_counts(64, min_count=8, max_fraction=0.3, num=6)
        self.assertTrue(np.all(tail_counts >= 8))
        chosen, window, stable = select_stable_integer_window(
            np.array([8, 12, 16, 20], dtype=int),
            np.array([1.0, 1.1, 1.12, 1.11], dtype=float),
            min_window=4,
        )
        self.assertEqual(chosen, 14)
        self.assertEqual((window.lo, window.hi), (8, 20))
        self.assertEqual(stable.size, 4)
        alias_chosen, _, _ = select_stable_tail_window(
            np.array([8, 12, 16], dtype=int),
            np.array([1.0, 1.1, 1.2], dtype=float),
            min_window=4,
        )
        self.assertEqual(alias_chosen, 12)

    def test_tail_and_max_spectrum_paths(self) -> None:
        ordered = _finite_positive(self._pareto_sample(size=512))
        k_values = np.array([8, 12, 16, 24], dtype=int)
        self.assertTrue(np.all(np.isfinite(_hill_path(ordered, k_values))))
        self.assertTrue(np.any(np.isfinite(_pickands_path(ordered, k_values))))
        self.assertTrue(np.all(np.isfinite(_dedh_moment_path(ordered, k_values))))
        self.assertTrue(np.isnan(_hill_standard_error(np.nan, 8)))
        self.assertTrue(np.isnan(_pickands_standard_error(np.nan, 8)))
        self.assertTrue(np.isfinite(_pickands_standard_error(0.0, 8)))
        self.assertTrue(np.isnan(_dedh_standard_error(np.nan, 8)))

        sample = self._pareto_sample(size=1024, seed=23)
        scales = candidate_max_spectrum_scales(sample.size, min_scale=1, min_blocks=4)
        y_values, n_blocks = _max_spectrum_curve(sample, scales)
        start_scales, xi_path, j_max = _max_spectrum_path(
            scales, y_values, n_blocks, min_scale_count=3
        )
        self.assertGreaterEqual(j_max, int(scales[-1]))
        self.assertEqual(start_scales.shape, xi_path.shape)

    def test_public_baseline_estimators_return_finite_estimates(self) -> None:
        sample = self._pareto_sample()
        hill = estimate_hill_evi(sample, k_values=np.array([16, 24, 32, 48, 64], dtype=int))
        pickands = estimate_pickands_evi(sample, k_values=np.array([8, 12, 16, 24, 32], dtype=int))
        dedh = estimate_dedh_moment_evi(sample, k_values=np.array([16, 24, 32, 48, 64], dtype=int))
        max_spec = estimate_max_spectrum_evi(sample, min_scale_count=3)
        for fit in (hill, pickands, dedh, max_spec):
            self.assertTrue(np.isfinite(fit.xi_hat))
            self.assertIsNotNone(fit.selected_level)
            self.assertGreater(len(fit.path_level), 0)
        self.assertEqual(hill.selected_k, hill.selected_level)

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        y = np.array([2.0, 4.1, 6.0, 8.2], dtype=float)
        slope, se = _weighted_slope_with_se(x, y, np.ones_like(x))
        self.assertTrue(np.isfinite(slope))
        self.assertTrue(np.isfinite(se))
        with self.assertRaisesRegex(ValueError, "produced no finite path estimates"):
            _select_from_path("bad_path", np.array([1, 2]), np.array([np.nan, np.nan]))


if __name__ == "__main__":
    unittest.main()
