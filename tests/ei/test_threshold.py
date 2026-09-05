from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from unibm.ei._likelihood import (
    find_1d_profile_likelihood_intervals,
    scale_1d_pseudo_likelihood,
)
from unibm.ei._stats import (
    _central_wald_interval,
    _intervals_overlap,
    _log_scale_theta_interval,
)
from unibm.ei.models import EiPreparedBundle, ThresholdCandidate
from unibm.ei.preparation import prepare_ei_bundle
from unibm.ei.threshold import (
    _select_between_candidates,
    _ferro_segers_from_times,
    _inter_exceedance_times,
    _kgaps_profile_fit,
    estimate_ferro_segers,
    estimate_k_gaps,
)


class EiThresholdTests(unittest.TestCase):
    @staticmethod
    def _positive_sample(size: int = 512, seed: int = 707) -> np.ndarray:
        rs = np.random.default_rng(seed)
        return rs.pareto(2.0, size) + 1.0

    def test_low_level_intervals_and_candidate_selection(self) -> None:
        nan_interval = _central_wald_interval(np.nan, 1.0)
        self.assertTrue(np.isnan(nan_interval[0]))
        lo, hi = _central_wald_interval(0.6, 0.1, bounded_unit_interval=True)
        self.assertGreaterEqual(lo, 0.0)
        raw_lo, raw_hi = _central_wald_interval(0.6, 0.1)
        self.assertLess(raw_lo, 0.6)
        self.assertGreater(raw_hi, 0.6)
        theta_lo, theta_hi = _log_scale_theta_interval(np.log(2.0), 0.1)
        self.assertGreater(theta_hi, theta_lo)
        boundary_lo, boundary_hi = _log_scale_theta_interval(0.01, 0.1)
        self.assertAlmostEqual(boundary_lo, np.exp(-0.206))
        self.assertEqual(boundary_hi, 1.0)
        invalid_theta_interval = _log_scale_theta_interval(np.nan, 0.1)
        self.assertTrue(np.isnan(invalid_theta_interval[0]))
        self.assertTrue(_intervals_overlap((0.1, 0.4), (0.3, 0.8)))
        self.assertFalse(_intervals_overlap((np.nan, 0.4), (0.3, 0.8)))

        preferred = ThresholdCandidate(0.9, 1.0, 0.4, (0.3, 0.5), 0.1, "wald", "default")
        alternative = ThresholdCandidate(0.95, 2.0, 0.7, (0.35, 0.9), 0.2, "wald", "default")
        self.assertIs(_select_between_candidates(preferred, alternative), preferred)
        nan_preferred = ThresholdCandidate(
            0.9,
            1.0,
            float("nan"),
            (float("nan"), float("nan")),
            float("nan"),
            "wald",
            "default",
        )
        self.assertIs(_select_between_candidates(nan_preferred, alternative), alternative)
        self.assertIs(_select_between_candidates(preferred, nan_preferred), preferred)

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
        with self.assertRaisesRegex(ValueError, "strictly negative"):
            scale_1d_pseudo_likelihood(loglik, mle=0.4, hessian=0.0, empirical_variance=25.0)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            scale_1d_pseudo_likelihood(loglik, mle=0.4, hessian=-100.0, empirical_variance=0.0)

        flat_interval = find_1d_profile_likelihood_intervals(lambda theta: 1.0, 0.4, 0.01, 0.99)
        self.assertEqual(flat_interval, (0.01, 0.99))

        def unstable_loglik(theta: float) -> float:
            if theta < 0.1:
                raise OverflowError("outside support")
            return -50.0 * (theta - 0.4) ** 2

        guarded_interval = find_1d_profile_likelihood_intervals(unstable_loglik, 0.4, 0.01, 0.99)
        self.assertLessEqual(guarded_interval[0], 0.4)
        self.assertGreaterEqual(guarded_interval[1], 0.4)

    def test_threshold_estimators_and_helpers(self) -> None:
        times_small = np.array([1.0, 2.0, 2.0, 1.0], dtype=float)
        times_large = np.array([3.0, 4.0, 2.0, 5.0], dtype=float)
        self.assertTrue(np.isfinite(_ferro_segers_from_times(times_small)[0]))
        self.assertTrue(np.isfinite(_ferro_segers_from_times(times_large)[0]))
        self.assertEqual(_inter_exceedance_times(np.array([1], dtype=int)).size, 0)
        np.testing.assert_allclose(
            _inter_exceedance_times(np.array([1, 4, 10], dtype=int)), np.array([3.0, 6.0])
        )
        with self.assertRaisesRegex(ValueError, "at least two inter-exceedance times"):
            _ferro_segers_from_times(np.array([1.0]))

        candidate = _kgaps_profile_fit(np.array([2.0, 4.0, 3.0]), run_k=1, exceedance_rate=0.1)
        self.assertTrue(np.isfinite(candidate.theta_hat))
        self.assertEqual(candidate.run_k, 1)
        expected_information = 2.0 * 3.0 / candidate.theta_hat**2
        self.assertAlmostEqual(candidate.standard_error, 1.0 / np.sqrt(expected_information))
        with self.assertRaisesRegex(ValueError, "at least two finite gap observations"):
            _kgaps_profile_fit(np.array([1.0]), run_k=1, exceedance_rate=0.1)

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
        zero_gap_information = 3.0 / (1.0 - zero_gap_candidate.theta_hat) ** 2
        self.assertAlmostEqual(
            zero_gap_candidate.standard_error,
            1.0 / np.sqrt(zero_gap_information),
        )

        bundle = prepare_ei_bundle(
            self._positive_sample(),
            block_sizes=np.array([4, 8, 16, 32], dtype=int),
            allow_zeros=False,
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

    def test_threshold_estimators_default_to_bundle_candidates(self) -> None:
        bundle = prepare_ei_bundle(
            self._positive_sample(),
            block_sizes=np.array([4, 8, 16, 32], dtype=int),
            allow_zeros=False,
            threshold_quantiles=(0.80, 0.90),
        )

        for estimator in (estimate_ferro_segers, estimate_k_gaps):
            with self.subTest(estimator=estimator.__name__):
                fit = estimator(bundle)
                self.assertIn(fit.selected_threshold_quantile, (0.80, 0.90))

    def test_preparation_validates_threshold_quantiles(self) -> None:
        invalid_quantiles = (
            (),
            (0.90, 0.90),
            (0.95, 0.90),
            (0.0, 0.90),
            (0.90, 1.0),
            (0.90, np.nan),
            np.array([[0.90, 0.95]]),
        )
        for threshold_quantiles in invalid_quantiles:
            with self.subTest(threshold_quantiles=threshold_quantiles):
                with self.assertRaisesRegex(ValueError, "threshold_quantiles"):
                    prepare_ei_bundle(
                        self._positive_sample(),
                        block_sizes=np.array([4, 8, 16, 32], dtype=int),
                        allow_zeros=False,
                        threshold_quantiles=threshold_quantiles,
                    )

    def test_threshold_estimators_validate_explicit_subsets(self) -> None:
        bundle = prepare_ei_bundle(
            self._positive_sample(),
            block_sizes=np.array([4, 8, 16, 32], dtype=int),
            allow_zeros=False,
            threshold_quantiles=(0.80, 0.90, 0.95),
        )

        for estimator in (estimate_ferro_segers, estimate_k_gaps):
            with self.subTest(estimator=estimator.__name__, case="single"):
                fit = estimator(bundle, threshold_quantiles=(0.90,))
                self.assertEqual(fit.selected_threshold_quantile, 0.90)
            with self.subTest(estimator=estimator.__name__, case="missing"):
                with self.assertRaisesRegex(ValueError, "threshold_quantiles"):
                    estimator(bundle, threshold_quantiles=(0.90, 0.99))
            with self.subTest(estimator=estimator.__name__, case="reordered"):
                with self.assertRaisesRegex(ValueError, "threshold_quantiles"):
                    estimator(bundle, threshold_quantiles=(0.95, 0.90))

    def test_k_gaps_validates_run_grid(self) -> None:
        bundle = prepare_ei_bundle(
            self._positive_sample(seed=727),
            block_sizes=np.array([4, 8, 16, 32], dtype=int),
            allow_zeros=False,
        )
        self.assertTrue(np.isfinite(estimate_k_gaps(bundle, k_grid=(0, 1)).theta_hat))
        invalid_grids = (
            (),
            (1.9,),
            (1, 1),
            (2, 1),
            (-1, 1),
            (True, 1),
            (np.nan, 1),
        )
        for k_grid in invalid_grids:
            with self.subTest(k_grid=k_grid):
                with self.assertRaisesRegex(ValueError, "k_grid"):
                    estimate_k_gaps(bundle, k_grid=k_grid)


if __name__ == "__main__":
    unittest.main()
