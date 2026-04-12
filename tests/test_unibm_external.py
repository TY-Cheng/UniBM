from __future__ import annotations

import unittest

import numpy as np

from unibm.evi.external import (
    _dedh_moment_path,
    _finite_positive,
    _hill_path,
    _hill_standard_error,
    _max_spectrum_curve,
    _max_spectrum_path,
    _normalize_standard_error,
    _pickands_standard_error,
    _pickands_path,
    _positive_finite_in_order,
    _select_from_path,
    _weighted_slope_with_se,
    _dedh_standard_error,
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


class UniBmExternalTests(unittest.TestCase):
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

    def test_wald_confidence_interval_and_se_normalization(self) -> None:
        lo, hi = wald_confidence_interval(1.0, 0.5, ci_level=0.95)
        self.assertLess(lo, 1.0)
        self.assertGreater(hi, 1.0)
        nan_lo, nan_hi = wald_confidence_interval(np.nan, 0.5)
        self.assertTrue(np.isnan(nan_lo))
        self.assertTrue(np.isnan(nan_hi))
        self.assertTrue(np.isnan(_normalize_standard_error(-1.0)))
        self.assertEqual(_normalize_standard_error(0.25), 0.25)
        with self.assertRaisesRegex(ValueError, "ci_level must lie strictly between 0 and 1"):
            wald_confidence_interval(1.0, 0.5, ci_level=1.2)

    def test_candidate_grids_and_stable_window_selection(self) -> None:
        small_tail_counts = candidate_tail_counts(10, min_count=8, max_fraction=0.2, num=3)
        self.assertEqual(small_tail_counts.tolist(), [7])
        tail_counts = candidate_tail_counts(64, min_count=8, max_fraction=0.3, num=6)
        self.assertTrue(np.all(tail_counts >= 8))
        scales = candidate_max_spectrum_scales(256, min_scale=1, min_blocks=2)
        self.assertTrue(np.all(scales >= 1))
        self.assertEqual(candidate_max_spectrum_scales(1).size, 0)
        chosen, window, stable = select_stable_integer_window(
            np.array([8, 12, 16, 20], dtype=int),
            np.array([1.0, 1.1, 1.12, 1.11], dtype=float),
            min_window=4,
        )
        self.assertEqual(chosen, 14)
        self.assertEqual((window.lo, window.hi), (8, 20))
        self.assertEqual(stable.size, 4)
        short_chosen, short_window, short_stable = select_stable_integer_window(
            np.array([8, 12, 16], dtype=int),
            np.array([1.0, 1.1, 1.2], dtype=float),
            min_window=4,
        )
        self.assertEqual(short_chosen, 12)
        self.assertEqual((short_window.lo, short_window.hi), (8, 16))
        self.assertEqual(short_stable.size, 3)
        alias_chosen, _, _ = select_stable_tail_window(
            np.array([8, 12, 16], dtype=int),
            np.array([1.0, 1.1, 1.2], dtype=float),
            min_window=4,
        )
        self.assertEqual(alias_chosen, 12)
        with self.assertRaisesRegex(ValueError, "non-empty and aligned"):
            select_stable_integer_window(np.array([], dtype=int), np.array([], dtype=float))

    def test_threshold_paths_return_finite_values(self) -> None:
        ordered = _finite_positive(self._pareto_sample(size=512))
        k_values = np.array([8, 12, 16, 24], dtype=int)
        self.assertTrue(np.all(np.isfinite(_hill_path(ordered, k_values))))
        self.assertTrue(np.any(np.isfinite(_pickands_path(ordered, k_values))))
        self.assertTrue(np.all(np.isfinite(_dedh_moment_path(ordered, k_values))))
        flat = np.ones(64, dtype=float)
        pickands_bad = _pickands_path(flat, np.array([8, 20], dtype=int))
        self.assertTrue(np.isnan(pickands_bad).all())
        dedh_bad = _dedh_moment_path(flat, np.array([8, 12], dtype=int))
        self.assertTrue(np.isnan(dedh_bad).all())
        self.assertTrue(np.isnan(_hill_standard_error(np.nan, 8)))
        self.assertTrue(np.isnan(_pickands_standard_error(np.nan, 8)))
        self.assertTrue(np.isfinite(_pickands_standard_error(0.0, 8)))
        self.assertTrue(np.isnan(_dedh_standard_error(np.nan, 8)))

    def test_weighted_slope_and_max_spectrum_helpers(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        y = np.array([2.0, 4.1, 6.0, 8.2], dtype=float)
        slope, se = _weighted_slope_with_se(x, y, np.ones_like(x))
        self.assertTrue(np.isfinite(slope))
        self.assertTrue(np.isfinite(se))
        nan_slope, nan_se = _weighted_slope_with_se(x[:2], y[:2], np.ones(2))
        self.assertTrue(np.isnan(nan_slope))
        self.assertTrue(np.isnan(nan_se))
        bad_slope, bad_se = _weighted_slope_with_se(x, y, np.array([1.0, 0.0, np.inf, np.nan]))
        self.assertTrue(np.isnan(bad_slope))
        self.assertTrue(np.isnan(bad_se))

        sample = self._pareto_sample(size=1024, seed=23)
        scales = candidate_max_spectrum_scales(sample.size, min_scale=1, min_blocks=4)
        y_values, n_blocks = _max_spectrum_curve(sample, scales)
        self.assertEqual(y_values.shape, scales.shape)
        self.assertEqual(n_blocks.shape, scales.shape)
        sparse_y, sparse_blocks = _max_spectrum_curve(sample[:8], np.array([1, 5], dtype=int))
        self.assertEqual(int(sparse_blocks[-1]), 0)
        self.assertTrue(np.isnan(sparse_y[-1]))
        start_scales, xi_path, j_max = _max_spectrum_path(
            scales, y_values, n_blocks, min_scale_count=3
        )
        self.assertGreaterEqual(j_max, int(scales[-1]))
        self.assertEqual(start_scales.shape, xi_path.shape)
        with self.assertRaisesRegex(ValueError, "at least three usable dyadic scales"):
            _max_spectrum_path(np.array([1, 2]), np.array([0.1, 0.2]), np.array([8, 4]))
        with self.assertRaisesRegex(ValueError, "produced no finite path estimates"):
            _select_from_path("bad_path", np.array([1, 2]), np.array([np.nan, np.nan]))

    def test_public_external_estimators_return_finite_estimates(self) -> None:
        sample = self._pareto_sample()
        hill_auto = estimate_hill_evi(sample)
        hill = estimate_hill_evi(sample, k_values=np.array([16, 24, 32, 48, 64], dtype=int))
        pickands_auto = estimate_pickands_evi(sample)
        pickands = estimate_pickands_evi(sample, k_values=np.array([8, 12, 16, 24, 32], dtype=int))
        dedh_auto = estimate_dedh_moment_evi(sample)
        dedh = estimate_dedh_moment_evi(sample, k_values=np.array([16, 24, 32, 48, 64], dtype=int))
        max_spec_fixed = estimate_max_spectrum_evi(
            sample, scales=np.array([1, 2, 3, 4], dtype=int)
        )
        max_spec = estimate_max_spectrum_evi(sample, min_scale_count=3)

        for fit in (
            hill_auto,
            hill,
            pickands_auto,
            pickands,
            dedh_auto,
            dedh,
            max_spec_fixed,
            max_spec,
        ):
            self.assertTrue(np.isfinite(fit.xi_hat))
            self.assertIsNotNone(fit.selected_level)
            self.assertGreater(len(fit.path_level), 0)
            self.assertGreater(len(fit.path_xi), 0)

        self.assertEqual(hill.tuning_axis, "k")
        self.assertEqual(hill.selected_k, hill.selected_level)
        self.assertEqual(hill.path_k, hill.path_level)
        self.assertEqual(max_spec.tuning_axis, "scale_start")
        self.assertIsNotNone(max_spec.fixed_upper_level)


if __name__ == "__main__":
    unittest.main()
