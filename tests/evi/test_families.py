from __future__ import annotations

import unittest

import numpy as np

from unibm.evi.spectrum import (
    _max_spectrum_curve,
    _max_spectrum_path,
    _weighted_slope_with_se,
    candidate_max_spectrum_scales,
    estimate_max_spectrum_evi,
)
from unibm.evi.tail import (
    _dedh_moment_path,
    _dedh_standard_error,
    _finite_positive,
    _hill_path,
    _hill_standard_error,
    _normalize_standard_error,
    _pickands_path,
    _pickands_standard_error,
    _select_from_path,
    candidate_tail_counts,
    estimate_dedh_moment_evi,
    estimate_hill_evi,
    estimate_pickands_evi,
    select_stable_integer_window,
    wald_confidence_interval,
)
from unibm.evi.blocks import block_maxima


class EviEstimatorFamilyTests(unittest.TestCase):
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

    def test_max_spectrum_preserves_observation_positions(self) -> None:
        sample = np.arange(1.0, 33.0)
        sample[0] = 0.0
        _, block_counts = _max_spectrum_curve(sample, np.array([1, 2, 3], dtype=int))
        np.testing.assert_array_equal(block_counts, np.array([16, 8, 4], dtype=int))

        fit = estimate_max_spectrum_evi(
            sample,
            scales=np.array([1, 2, 3], dtype=int),
            min_scale_count=3,
        )
        self.assertTrue(np.isfinite(fit.xi_hat))

    def test_max_spectrum_rejects_clock_breaking_inputs(self) -> None:
        sample = np.arange(1.0, 33.0)
        nonfinite = sample.copy()
        nonfinite[4] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            estimate_max_spectrum_evi(
                nonfinite,
                scales=np.array([1, 2, 3], dtype=int),
                min_scale_count=3,
            )

        nonpositive_block = sample.copy()
        nonpositive_block[:8] = 0.0
        with self.assertRaisesRegex(ValueError, "block maxima must be strictly positive"):
            estimate_max_spectrum_evi(
                nonpositive_block,
                scales=np.array([1, 2, 3], dtype=int),
                min_scale_count=3,
            )

    def test_max_spectrum_rejects_invalid_custom_scales(self) -> None:
        sample = np.arange(1.0, 33.0)
        invalid_scales = (
            np.array([1.0, 2.5, 3.0]),
            np.array([1, 2, 2]),
            np.array([2, 1, 3]),
            np.array([-1, 1, 2]),
            np.array([1, 2, 5]),
            np.array([[1, 2, 3]]),
        )
        for scales in invalid_scales:
            with self.subTest(scales=scales):
                with self.assertRaisesRegex(ValueError, "scales"):
                    estimate_max_spectrum_evi(sample, scales=scales, min_scale_count=3)

    def test_confidence_intervals_and_stable_window_selection(self) -> None:
        lo, hi = wald_confidence_interval(1.0, 0.5, ci_level=0.95)
        self.assertLess(lo, 1.0)
        self.assertGreater(hi, 1.0)
        with self.assertRaisesRegex(ValueError, "ci_level must lie strictly"):
            wald_confidence_interval(1.0, 0.5, ci_level=1.0)
        self.assertTrue(np.isnan(wald_confidence_interval(np.nan, 0.2)[0]))
        self.assertTrue(np.isnan(_normalize_standard_error(-1.0)))
        self.assertEqual(_normalize_standard_error(0.25), 0.25)

        tail_counts = candidate_tail_counts(64, min_count=8, max_fraction=0.3, num=6)
        self.assertTrue(np.all(tail_counts >= 8))
        with self.assertRaisesRegex(ValueError, "feasible tail-count grid"):
            candidate_tail_counts(10, min_count=8)
        chosen, window, stable = select_stable_integer_window(
            np.array([8, 12, 16, 20], dtype=int),
            np.array([1.0, 1.1, 1.12, 1.11], dtype=float),
            min_window=4,
        )
        self.assertEqual(chosen, 12)
        self.assertEqual((window.lo, window.hi), (8, 20))
        self.assertEqual(stable.size, 4)
        with self.assertRaisesRegex(ValueError, "must be non-empty and aligned"):
            select_stable_integer_window(np.array([], dtype=int), np.array([], dtype=float))

        invalid_windows = (
            (np.array([8.0, 12.5]), np.array([1.0, 1.1]), 2),
            (np.array([12, 8]), np.array([1.0, 1.1]), 2),
            (np.array([8, 12]), np.array([1.0, np.nan]), 2),
            (np.array([8, 12]), np.array([1.0, 1.1]), 1),
        )
        for levels, path, min_window in invalid_windows:
            with self.subTest(levels=levels, path=path, min_window=min_window):
                with self.assertRaises(ValueError):
                    select_stable_integer_window(levels, path, min_window=min_window)

    def test_block_maxima_rejects_invalid_block_sizes(self) -> None:
        sample = np.arange(1.0, 9.0)
        for block_size in (1, 2.5, np.nan, np.inf, True):
            with self.subTest(block_size=block_size):
                with self.assertRaisesRegex(ValueError, "block_size"):
                    block_maxima(sample, block_size)

    def test_tail_and_spectrum_paths(self) -> None:
        ordered = _finite_positive(self._pareto_sample(size=512))
        k_values = np.array([8, 12, 16, 24], dtype=int)
        self.assertTrue(np.all(np.isfinite(_hill_path(ordered, k_values))))
        self.assertTrue(np.any(np.isfinite(_pickands_path(ordered, k_values))))
        self.assertTrue(np.all(np.isfinite(_dedh_moment_path(ordered, k_values))))
        np.testing.assert_allclose(
            _pickands_path(np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]), np.array([3])),
            np.array([np.nan]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            _pickands_path(np.array([10.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]), np.array([1])),
            np.array([np.nan]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            _dedh_moment_path(np.array([5.0] * 10, dtype=float), np.array([2], dtype=int)),
            np.array([np.nan]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            _dedh_moment_path(
                np.array([np.exp(2.0), np.exp(2.0), 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                np.array([2], dtype=int),
            ),
            np.array([np.nan]),
            equal_nan=True,
        )
        self.assertTrue(np.isnan(_hill_standard_error(np.nan, 8)))
        self.assertTrue(np.isnan(_pickands_standard_error(np.nan, 8)))
        self.assertTrue(np.isfinite(_pickands_standard_error(0.0, 8)))
        self.assertTrue(np.isnan(_dedh_standard_error(np.nan, 8)))

        sample = self._pareto_sample(size=1024, seed=23)
        np.testing.assert_array_equal(
            candidate_max_spectrum_scales(1, min_scale=1, min_blocks=2),
            np.empty(0, dtype=int),
        )
        scales = candidate_max_spectrum_scales(sample.size, min_scale=1, min_blocks=4)
        y_values, n_blocks = _max_spectrum_curve(sample, scales)
        sparse_slope, sparse_se = _weighted_slope_with_se(
            np.array([1.0, np.nan]),
            np.array([2.0, 3.0]),
            np.array([1.0, 1.0]),
        )
        self.assertTrue(np.isnan(sparse_slope))
        self.assertTrue(np.isnan(sparse_se))
        with self.assertRaisesRegex(ValueError, "at least two complete blocks"):
            _max_spectrum_curve(
                np.array([1.0] * 8, dtype=float),
                np.array([10], dtype=int),
            )
        with self.assertRaisesRegex(ValueError, "at least three usable dyadic scales"):
            _max_spectrum_path(
                np.array([1, 2], dtype=int),
                np.array([1.0, 2.0], dtype=float),
                np.array([8, 4], dtype=int),
                min_scale_count=3,
            )
        start_scales, xi_path, j_max = _max_spectrum_path(
            scales, y_values, n_blocks, min_scale_count=3
        )
        self.assertGreaterEqual(j_max, int(scales[-1]))
        self.assertEqual(start_scales.shape, xi_path.shape)

    def test_tail_estimators_validate_custom_tail_counts(self) -> None:
        sample = self._pareto_sample(size=64)
        invalid_cases = (
            (estimate_hill_evi, np.array([2.5, 4.0])),
            (estimate_hill_evi, np.array([2, 2, 4])),
            (estimate_hill_evi, np.array([4, 2])),
            (estimate_hill_evi, np.array([0, 2])),
            (estimate_hill_evi, np.array([[2, 4]])),
            (estimate_hill_evi, np.array([64])),
            (estimate_pickands_evi, np.array([17])),
            (estimate_dedh_moment_evi, np.array([1])),
        )
        for estimator, k_values in invalid_cases:
            with self.subTest(estimator=estimator.__name__, k_values=k_values):
                with self.assertRaisesRegex(ValueError, "k_values"):
                    estimator(sample, k_values=k_values)

        for estimator in (estimate_hill_evi, estimate_dedh_moment_evi):
            with self.subTest(estimator=estimator.__name__):
                fit = estimator(sample, k_values=np.array([2, 3, 4, 5]))
                self.assertIn(fit.selected_level, fit.path_level)

        hill = estimate_hill_evi(sample, k_values=np.array([8, 12, 16, 20]))
        self.assertEqual(hill.selected_level, 12)
        self.assertAlmostEqual(hill.standard_error, abs(hill.xi_hat) / np.sqrt(12.0))

    def test_candidate_tail_counts_rejects_invalid_grid_parameters(self) -> None:
        invalid_calls = (
            ("n_obs", lambda: candidate_tail_counts(64.5)),
            ("min_count", lambda: candidate_tail_counts(64, min_count=8.5)),
            ("max_fraction", lambda: candidate_tail_counts(64, max_fraction=0.0)),
            ("max_fraction", lambda: candidate_tail_counts(64, max_fraction=1.5)),
            ("num", lambda: candidate_tail_counts(64, num=0)),
        )
        for parameter, call in invalid_calls:
            with self.subTest(parameter=parameter):
                with self.assertRaisesRegex(ValueError, parameter):
                    call()

    def test_public_estimator_families_return_finite_estimates(self) -> None:
        sample = self._pareto_sample()
        hill = estimate_hill_evi(sample, k_values=np.array([16, 24, 32, 48, 64], dtype=int))
        pickands = estimate_pickands_evi(sample, k_values=np.array([8, 12, 16, 24, 32], dtype=int))
        dedh = estimate_dedh_moment_evi(sample, k_values=np.array([16, 24, 32, 48, 64], dtype=int))
        default_hill = estimate_hill_evi(sample)
        default_pickands = estimate_pickands_evi(sample)
        default_dedh = estimate_dedh_moment_evi(sample)
        max_spec = estimate_max_spectrum_evi(sample, min_scale_count=3)
        for fit in (hill, pickands, dedh, default_hill, default_pickands, default_dedh, max_spec):
            self.assertTrue(np.isfinite(fit.xi_hat))
            self.assertIsNotNone(fit.selected_level)
            self.assertGreater(len(fit.path_level), 0)
        self.assertEqual(hill.selected_k, hill.selected_level)
        self.assertEqual(hill.path_k, hill.path_level)

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        y = np.array([2.0, 4.1, 6.0, 8.2], dtype=float)
        slope, se = _weighted_slope_with_se(x, y, np.ones_like(x))
        self.assertTrue(np.isfinite(slope))
        self.assertTrue(np.isfinite(se))
        with self.assertRaisesRegex(ValueError, "produced no finite path estimates"):
            _select_from_path("bad_path", np.array([1, 2]), np.array([np.nan, np.nan]))
        custom = _select_from_path(
            "custom_path",
            np.array([8, 12, 16, 20], dtype=int),
            np.array([0.4, 0.42, 0.41, 0.43], dtype=float),
            tuning_axis="level",
        )
        self.assertTrue(np.isnan(custom.standard_error))
        self.assertEqual(custom.tuning_axis, "level")


if __name__ == "__main__":
    unittest.main()
