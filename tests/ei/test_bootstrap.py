from __future__ import annotations

import unittest

import numpy as np

from unibm.ei._validation import _finite_nonnegative_series, _finite_positive_series
from unibm.ei.bootstrap import bootstrap_bm_ei_path, bootstrap_bm_ei_path_draws
from unibm.ei.paths import _build_bm_paths_from_values


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


class EiBootstrapTests(unittest.TestCase):
    @staticmethod
    def _zero_inflated_sample(size: int = 512, seed: int = 405) -> np.ndarray:
        rs = np.random.default_rng(seed)
        values = np.zeros(size, dtype=float)
        mask = rs.random(size) < 0.25
        values[mask] = rs.gamma(shape=2.5, scale=2.0, size=int(mask.sum()))
        return values

    def test_bootstrap_draws_match_baseline(self) -> None:
        values = self._zero_inflated_sample()
        block_sizes = np.array([4, 8, 16, 32], dtype=int)
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

    def test_bootstrap_bm_ei_path_returns_expected_shape(self) -> None:
        values = self._zero_inflated_sample()
        block_sizes = np.array([4, 8, 16, 32], dtype=int)
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


if __name__ == "__main__":
    unittest.main()
