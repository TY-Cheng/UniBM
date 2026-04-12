from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from unibm.ei import (
    estimate_ferro_segers,
    estimate_k_gaps,
    estimate_native_bm_ei,
    estimate_pooled_bm_ei,
    extract_stable_path_window,
    prepare_ei_bundle,
)
from unibm.ei._internal import (
    _central_wald_interval,
    _finite_nonnegative_series,
    _finite_positive_series,
    _intervals_overlap,
    _log_scale_theta_interval,
    _select_between_candidates,
    find_1d_profile_likelihood_intervals,
    scale_1d_pseudo_likelihood,
)
from unibm.ei.bootstrap import bootstrap_bm_ei_path, bootstrap_bm_ei_path_draws
from unibm.ei.models import EiPathBundle, EiPreparedBundle, EiStableWindow, ThresholdCandidate
from unibm.ei.native import (
    EI_DEFAULT_COVARIANCE_SHRINKAGE,
    _bb_wald_fit,
    _build_bm_estimate,
    _fit_pooled_z_model,
    _northrop_profile_fit,
    _pooled_z_fit,
    _regularize_ei_covariance,
)
from unibm.ei.paths import (
    _build_bm_paths_from_values,
    _build_path_from_scores,
    _rolling_window_minima,
    _select_stable_ei_window,
)
from unibm.ei.threshold import (
    _ferro_segers_from_times,
    _inter_exceedance_times,
    _kgaps_profile_fit,
)


def _baseline_select_stable_ei_window(
    block_sizes: np.ndarray,
    z_path: np.ndarray,
    *,
    min_points: int = 4,
    trim_fraction: float = 0.15,
    roughness_penalty: float = 0.75,
    curvature_penalty: float = 0.5,
) -> tuple[EiStableWindow, np.ndarray]:
    levels = np.asarray(block_sizes, dtype=int)
    z = np.asarray(z_path, dtype=float)
    mask = np.isfinite(z)
    levels = levels[mask]
    z = z[mask]
    if levels.size < min_points:
        raise ValueError("Not enough finite EI path values to select a stable window.")
    lo = int(np.floor(levels.size * trim_fraction))
    hi = levels.size - lo
    if hi - lo < min_points:
        lo = 0
        hi = levels.size
    best: tuple[float, int, int] | None = None
    for start in range(lo, hi - min_points + 1):
        for stop in range(start + min_points, hi + 1):
            window = z[start:stop]
            variance = float(np.mean((window - window.mean()) ** 2))
            first_diff = np.diff(window)
            roughness = float(np.mean(np.abs(first_diff))) if first_diff.size else 0.0
            curvature = float(np.mean(np.abs(np.diff(first_diff)))) if first_diff.size > 1 else 0.0
            score = (
                variance
                + float(roughness_penalty) * roughness
                + float(curvature_penalty) * curvature
            ) / np.sqrt(stop - start)
            if best is None or score < best[0]:
                best = (score, start, stop)
    assert best is not None
    _, start, stop = best
    selected_mask = np.zeros(mask.sum(), dtype=bool)
    selected_mask[start:stop] = True
    window = EiStableWindow(int(levels[start]), int(levels[stop - 1]))
    return window, selected_mask


def _baseline_bootstrap_bm_ei_path_draws(
    bootstrap_samples: np.ndarray,
    *,
    block_sizes: np.ndarray,
    allow_zeros: bool = False,
) -> dict[tuple[str, bool], np.ndarray]:
    samples = np.asarray(bootstrap_samples, dtype=float)
    block_sizes = np.asarray(block_sizes, dtype=int)
    draws = {
        ("northrop", True): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
        ("northrop", False): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
        ("bb", True): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
        ("bb", False): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
    }
    for rep, sample in enumerate(samples):
        try:
            sample_values = (
                _finite_nonnegative_series(sample)
                if allow_zeros
                else _finite_positive_series(sample)
            )
            sample_paths = _build_bm_paths_from_values(sample_values, block_sizes)
        except ValueError:
            continue
        for key, path in sample_paths.items():
            draws[key][rep] = path.z_path
    return draws


class UniBmExtremalIndexTests(unittest.TestCase):
    @staticmethod
    def _positive_sample(size: int = 512, seed: int = 404) -> np.ndarray:
        rs = np.random.default_rng(seed)
        return rs.pareto(2.0, size) + 1.0

    @staticmethod
    def _zero_inflated_sample(size: int = 512, seed: int = 405) -> np.ndarray:
        rs = np.random.default_rng(seed)
        values = np.zeros(size, dtype=float)
        mask = rs.random(size) < 0.25
        values[mask] = rs.gamma(shape=2.5, scale=2.0, size=int(mask.sum()))
        return values

    def test_low_level_intervals_and_candidate_selection(self) -> None:
        nan_interval = _central_wald_interval(np.nan, 1.0)
        self.assertTrue(np.isnan(nan_interval[0]))
        self.assertTrue(np.isnan(nan_interval[1]))
        lo, hi = _central_wald_interval(0.6, 0.1, bounded_unit_interval=True)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        raw_lo, raw_hi = _central_wald_interval(0.6, 0.1)
        self.assertLess(raw_lo, 0.6)
        self.assertGreater(raw_hi, 0.6)
        theta_lo, theta_hi = _log_scale_theta_interval(np.log(2.0), 0.1)
        self.assertGreater(theta_hi, theta_lo)
        invalid_theta_interval = _log_scale_theta_interval(np.nan, 0.1)
        self.assertTrue(np.isnan(invalid_theta_interval[0]))
        self.assertTrue(_intervals_overlap((0.1, 0.4), (0.3, 0.8)))
        self.assertFalse(_intervals_overlap((0.1, 0.2), (0.3, 0.4)))
        self.assertFalse(_intervals_overlap((np.nan, 0.2), (0.3, 0.4)))

        preferred = ThresholdCandidate(0.9, 1.0, 0.4, (0.3, 0.5), 0.1, "wald", "default")
        alternative = ThresholdCandidate(0.95, 2.0, 0.7, (0.35, 0.9), 0.2, "wald", "default")
        self.assertIs(_select_between_candidates(preferred, alternative), preferred)
        invalid_preferred = ThresholdCandidate(
            0.9, 1.0, np.nan, (np.nan, np.nan), np.nan, "wald", "default"
        )
        invalid_alternative = ThresholdCandidate(
            0.95, 2.0, np.nan, (np.nan, np.nan), np.nan, "wald", "default"
        )
        self.assertIs(_select_between_candidates(invalid_preferred, alternative), alternative)
        self.assertIs(_select_between_candidates(preferred, invalid_alternative), preferred)

    def test_profile_likelihood_helpers(self) -> None:
        def loglik(theta: float) -> float:
            return -50.0 * (theta - 0.4) ** 2

        adjusted = scale_1d_pseudo_likelihood(
            loglik, mle=0.4, hessian=-100.0, empirical_variance=25.0
        )
        self.assertTrue(np.isfinite(adjusted(0.45)))
        interval = find_1d_profile_likelihood_intervals(loglik, 0.4, 0.01, 0.99)
        self.assertLess(interval[0], 0.4)
        self.assertGreater(interval[1], 0.4)
        with self.assertRaisesRegex(ValueError, "Hessian must be strictly negative"):
            scale_1d_pseudo_likelihood(loglik, mle=0.4, hessian=0.0, empirical_variance=1.0)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            scale_1d_pseudo_likelihood(loglik, mle=0.4, hessian=-1.0, empirical_variance=0.0)

        flat_interval = find_1d_profile_likelihood_intervals(
            lambda theta: 0.0,
            mle=0.4,
            lower_bound_search=0.1,
            upper_bound_search=0.9,
        )
        self.assertEqual(flat_interval, (0.1, 0.9))

        def edge_raising_loglik(theta: float) -> float:
            if np.isclose(theta, 0.1) or np.isclose(theta, 0.9):
                raise OverflowError("edge")
            return -50.0 * (theta - 0.4) ** 2

        edge_interval = find_1d_profile_likelihood_intervals(
            edge_raising_loglik,
            mle=0.4,
            lower_bound_search=0.1,
            upper_bound_search=0.9,
        )
        self.assertLess(edge_interval[0], 0.4)
        self.assertGreater(edge_interval[1], 0.4)

    def test_series_filters_and_window_minima(self) -> None:
        positive = _finite_positive_series(np.arange(1.0, 40.0, dtype=float))
        nonnegative = _finite_nonnegative_series(
            np.concatenate([[0.0], np.arange(1.0, 40.0, dtype=float)])
        )
        self.assertEqual(positive.size, 39)
        self.assertEqual(nonnegative.size, 40)
        with self.assertRaisesRegex(ValueError, "at least 32 positive finite observations"):
            _finite_positive_series(np.arange(1.0, 10.0, dtype=float))
        with self.assertRaisesRegex(ValueError, "at least 32 finite non-negative observations"):
            _finite_nonnegative_series(np.arange(10.0, dtype=float))

        scores = np.array([4.0, 2.0, np.nan, 1.0, 3.0, 5.0], dtype=float)
        np.testing.assert_allclose(
            _rolling_window_minima(scores, 2, sliding=True), np.array([2.0, 1.0, 3.0])
        )
        np.testing.assert_allclose(
            _rolling_window_minima(scores, 2, sliding=False), np.array([2.0, 3.0])
        )
        self.assertEqual(_rolling_window_minima(scores, 10, sliding=True).size, 0)
        self.assertEqual(_rolling_window_minima(scores, 1, sliding=False).size, 0)

    def test_path_builders_and_stable_window_selection(self) -> None:
        block_sizes = np.array([4, 8, 16, 32], dtype=int)
        z_path = np.array([0.1, 0.11, 0.12, 0.11], dtype=float)
        window, mask = _select_stable_ei_window(block_sizes, z_path, min_points=3)
        baseline_window, baseline_mask = _baseline_select_stable_ei_window(
            block_sizes,
            z_path,
            min_points=3,
        )
        self.assertLessEqual(window.lo, window.hi)
        self.assertGreater(mask.sum(), 0)
        self.assertEqual(window, baseline_window)
        np.testing.assert_array_equal(mask, baseline_mask)
        fallback_window, fallback_mask = _select_stable_ei_window(
            block_sizes,
            z_path,
            min_points=4,
            trim_fraction=0.4,
        )
        baseline_fallback_window, baseline_fallback_mask = _baseline_select_stable_ei_window(
            block_sizes,
            z_path,
            min_points=4,
            trim_fraction=0.4,
        )
        self.assertLessEqual(fallback_window.lo, fallback_window.hi)
        self.assertEqual(fallback_mask.sum(), 4)
        self.assertEqual(fallback_window, baseline_fallback_window)
        np.testing.assert_array_equal(fallback_mask, baseline_fallback_mask)
        with self.assertRaisesRegex(ValueError, "Not enough finite EI path values"):
            _select_stable_ei_window(block_sizes[:2], np.array([np.nan, 0.1]), min_points=3)

        sample = self._positive_sample()
        paths = _build_bm_paths_from_values(sample, block_sizes)
        self.assertEqual(
            set(paths), {("northrop", True), ("northrop", False), ("bb", True), ("bb", False)}
        )
        path_with_gap = _build_path_from_scores(
            "bb",
            np.linspace(0.1, 0.9, sample.size),
            np.array([4, 8, 16, 32, sample.size + 1], dtype=int),
            sliding=True,
        )
        self.assertEqual(path_with_gap.sample_counts[-1], 0)
        self.assertTrue(np.isnan(path_with_gap.theta_path[-1]))
        path = _build_path_from_scores(
            "bb", np.linspace(0.1, 0.9, sample.size), block_sizes, sliding=True
        )
        levels, z_values = extract_stable_path_window(path)
        self.assertGreater(levels.size, 0)
        self.assertEqual(levels.shape, z_values.shape)
        with self.assertRaisesRegex(ValueError, "Unknown BM EI base path"):
            _build_path_from_scores(
                "mystery",
                np.linspace(0.1, 0.9, sample.size),
                block_sizes,
                sliding=True,
            )
        bad_path = EiPathBundle(
            base_path=path.base_path,
            sliding=path.sliding,
            block_sizes=path.block_sizes,
            theta_path=path.theta_path,
            eir_path=path.eir_path,
            z_path=path.z_path,
            sample_counts=path.sample_counts,
            sample_statistics=path.sample_statistics,
            stable_window=EiStableWindow(999, 1001),
            selected_level=path.selected_level,
        )
        with self.assertRaisesRegex(ValueError, "Stable EI window did not retain any finite"):
            extract_stable_path_window(bad_path)

    def test_prepare_bundle_and_bootstrap_draws(self) -> None:
        values = self._zero_inflated_sample()
        block_sizes = np.array([4, 8, 16, 32], dtype=int)
        bundle = prepare_ei_bundle(values, block_sizes=block_sizes, allow_zeros=True)
        self.assertEqual(tuple(bundle.block_sizes), (4, 8, 16, 32))
        self.assertIn(0.9, bundle.threshold_candidates)

        bootstrap_samples = np.vstack(
            [
                values,
                np.full(values.size, np.nan, dtype=float),
                np.roll(values, 3),
            ]
        )
        draws = bootstrap_bm_ei_path_draws(
            bootstrap_samples, block_sizes=block_sizes, allow_zeros=True
        )
        baseline_draws = _baseline_bootstrap_bm_ei_path_draws(
            bootstrap_samples,
            block_sizes=block_sizes,
            allow_zeros=True,
        )
        self.assertEqual(draws[("bb", True)].shape, (3, 4))
        self.assertTrue(np.isnan(draws[("bb", True)][1]).all())
        for key in draws:
            np.testing.assert_allclose(draws[key], baseline_draws[key], equal_nan=True)

        boot = bootstrap_bm_ei_path(
            values,
            base_path="bb",
            sliding=True,
            block_sizes=block_sizes,
            reps=4,
            random_state=9,
            allow_zeros=True,
        )
        self.assertEqual(boot["samples"].shape, (4, 4))

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
        self.assertTrue(np.isfinite(pooled.theta_hat))
        native_bb = estimate_native_bm_ei(bundle, base_path="bb", sliding=True)
        native_northrop = estimate_native_bm_ei(
            bundle,
            base_path="northrop",
            sliding=True,
            use_adjusted_chandwich=True,
        )
        self.assertTrue(np.isfinite(native_bb.theta_hat))
        self.assertTrue(np.isfinite(native_northrop.theta_hat))

        built = _build_bm_estimate(
            "bb_sliding_fgls",
            path,
            regression="FGLS",
            bootstrap_result={"block_sizes": np.array([999]), "covariance": np.eye(1)},
        )
        self.assertEqual(built.ci_variant, "ols")

    def test_fixed_b_and_threshold_estimators(self) -> None:
        stats = np.array([1.4, 1.5, 1.3, 1.45], dtype=float)
        theta_hat, interval, se = _bb_wald_fit(stats, block_size=8)
        self.assertTrue(np.isfinite(theta_hat))
        self.assertTrue(np.isfinite(se))
        self.assertLessEqual(interval[0], theta_hat)

        northrop_plain = _northrop_profile_fit(stats, adjusted=False)
        northrop_adjusted = _northrop_profile_fit(stats, adjusted=True)
        self.assertTrue(np.isfinite(northrop_plain[0]))
        self.assertTrue(np.isfinite(northrop_adjusted[0]))

        times_small = np.array([1.0, 2.0, 2.0, 1.0], dtype=float)
        times_large = np.array([3.0, 4.0, 2.0, 5.0], dtype=float)
        self.assertTrue(np.isfinite(_ferro_segers_from_times(times_small)[0]))
        self.assertTrue(np.isfinite(_ferro_segers_from_times(times_large)[0]))
        self.assertEqual(_inter_exceedance_times(np.array([5], dtype=int)).size, 0)
        np.testing.assert_allclose(
            _inter_exceedance_times(np.array([1, 4, 10], dtype=int)), np.array([3.0, 6.0])
        )
        with self.assertRaisesRegex(ValueError, "at least two inter-exceedance times"):
            _ferro_segers_from_times(np.array([1.0], dtype=float))

        candidate = _kgaps_profile_fit(np.array([2.0, 4.0, 3.0]), run_k=1, exceedance_rate=0.1)
        self.assertTrue(np.isfinite(candidate.theta_hat))
        self.assertEqual(candidate.run_k, 1)
        with self.assertRaisesRegex(ValueError, "at least two finite gap observations"):
            _kgaps_profile_fit(np.array([2.0]), run_k=1, exceedance_rate=0.1)

        invalid_theta_checks: list[float] = []

        def fake_interval(loglik_func, mle, lower_bound_search, upper_bound_search, *, alpha):
            invalid_theta_checks.extend(
                [float(loglik_func(0.0)), float(loglik_func(1.0)), float(loglik_func(0.5))]
            )
            return (0.2, 0.4)

        with patch(
            "unibm.ei.threshold.find_1d_profile_likelihood_intervals",
            side_effect=fake_interval,
        ):
            zero_gap_candidate = _kgaps_profile_fit(
                np.array([1.0, 1.0, 1.0]),
                run_k=5,
                exceedance_rate=0.1,
            )
        self.assertEqual(invalid_theta_checks[:2], [-np.inf, -np.inf])
        self.assertTrue(np.isfinite(zero_gap_candidate.theta_hat))

        bundle = prepare_ei_bundle(
            self._positive_sample(seed=707), block_sizes=np.array([4, 8, 16, 32], dtype=int)
        )
        ferro = estimate_ferro_segers(bundle)
        kgaps = estimate_k_gaps(bundle)
        self.assertTrue(np.isfinite(ferro.theta_hat))
        self.assertTrue(np.isfinite(kgaps.theta_hat))

        sparse_bundle = EiPreparedBundle(
            values=np.linspace(1.0, 50.0, 50),
            block_sizes=np.array([4, 8, 16, 32], dtype=int),
            paths={},
            threshold_candidates={
                0.9: np.array([1, 4], dtype=int),
                0.95: np.array([10, 15], dtype=int),
            },
        )
        with self.assertRaisesRegex(
            ValueError, "could not find a threshold with enough exceedances"
        ):
            estimate_ferro_segers(sparse_bundle)
        with self.assertRaisesRegex(
            ValueError, "could not find a threshold with enough exceedances"
        ):
            estimate_k_gaps(sparse_bundle)

        mixed_bundle = EiPreparedBundle(
            values=np.linspace(1.0, 100.0, 100),
            block_sizes=np.array([4, 8, 16, 32], dtype=int),
            paths={},
            threshold_candidates={
                0.9: np.array([1, 4], dtype=int),
                0.95: np.array([10, 20, 35, 60], dtype=int),
            },
        )
        ferro_mixed = estimate_ferro_segers(mixed_bundle)
        kgaps_mixed = estimate_k_gaps(mixed_bundle)
        self.assertEqual(ferro_mixed.selected_threshold_quantile, 0.95)
        self.assertEqual(kgaps_mixed.selected_threshold_quantile, 0.95)


if __name__ == "__main__":
    unittest.main()
