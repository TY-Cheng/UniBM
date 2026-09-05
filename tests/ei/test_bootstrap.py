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
        self.assertEqual(draws[("bb", True)].shape, (2, 4))
        for key in draws:
            np.testing.assert_allclose(draws[key], baseline_draws[key], equal_nan=True)

        invalid_samples = bootstrap_samples.copy()
        invalid_samples[0, 10] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            bootstrap_bm_ei_path_draws(
                invalid_samples,
                block_sizes=block_sizes,
                allow_zeros=True,
            )
        with self.assertRaises(TypeError):
            bootstrap_bm_ei_path_draws(bootstrap_samples, block_sizes=block_sizes)

        with self.assertRaisesRegex(ValueError, "block_sizes"):
            bootstrap_bm_ei_path_draws(
                bootstrap_samples,
                block_sizes=np.array([4.0, 8.5, 16.0, 32.0]),
                allow_zeros=True,
            )

    def test_bootstrap_bm_ei_path_returns_covariance_on_full_grid(self) -> None:
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
        self.assertEqual(boot["covariance"].shape, (4, 4))
        self.assertTrue(np.all(np.isfinite(boot["covariance"])))
        self.assertEqual(boot["base_path"], "bb")
        self.assertIs(boot["sliding"], True)

    def test_bootstrap_bm_ei_path_validates_fixed_block_length(self) -> None:
        values = np.arange(1.0, 65.0)
        block_sizes = np.array([4, 8, 16], dtype=int)

        default = bootstrap_bm_ei_path(
            values,
            allow_zeros=False,
            base_path="bb",
            sliding=True,
            block_sizes=block_sizes,
            reps=3,
            random_state=17,
        )
        fixed = bootstrap_bm_ei_path(
            values,
            allow_zeros=False,
            base_path="bb",
            sliding=True,
            block_sizes=block_sizes,
            reps=3,
            random_state=17,
            bootstrap_block_length=7,
        )
        self.assertEqual(default["bootstrap_block_length_policy"], "default")
        self.assertEqual(default["bootstrap_block_length"], 16)
        self.assertEqual(fixed["bootstrap_block_length_policy"], "fixed")
        self.assertEqual(fixed["bootstrap_block_length"], 7)

        for invalid in (0, 65, 1.5, True, "auto", "bad"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "bootstrap_block_length"):
                    bootstrap_bm_ei_path(
                        values,
                        allow_zeros=False,
                        base_path="bb",
                        sliding=True,
                        block_sizes=block_sizes,
                        reps=3,
                        random_state=17,
                        bootstrap_block_length=invalid,
                    )


if __name__ == "__main__":
    unittest.main()
