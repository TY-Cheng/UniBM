from __future__ import annotations

import unittest

import numpy as np

from unibm.ei._validation import _finite_nonnegative_series, _finite_positive_series
from unibm.ei.models import EiPathBundle, EiStableWindow
from unibm.ei.paths import (
    _build_bm_paths_from_values,
    _build_path_from_scores,
    _rolling_window_minima,
    _select_stable_ei_window,
    extract_stable_path_window,
)
from unibm.ei.preparation import prepare_ei_bundle


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


class EiPathsTests(unittest.TestCase):
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

    def test_series_filters_validate_expected_support(self) -> None:
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

    def test_prepare_bundle_and_allow_zeros_path(self) -> None:
        values = self._zero_inflated_sample()
        block_sizes = np.array([4, 8, 16, 32], dtype=int)
        bundle = prepare_ei_bundle(values, block_sizes=block_sizes, allow_zeros=True)
        self.assertEqual(tuple(bundle.block_sizes), (4, 8, 16, 32))
        self.assertIn(0.9, bundle.threshold_candidates)
        self.assertEqual(
            set(bundle.paths),
            {("northrop", True), ("northrop", False), ("bb", True), ("bb", False)},
        )

    def test_rolling_window_minima(self) -> None:
        scores = np.array([4.0, 2.0, np.nan, 1.0, 3.0, 5.0], dtype=float)
        np.testing.assert_allclose(
            _rolling_window_minima(scores, 2, sliding=True), np.array([2.0, 1.0, 3.0])
        )
        np.testing.assert_allclose(
            _rolling_window_minima(scores, 2, sliding=False), np.array([2.0, 3.0])
        )

    def test_path_builders_and_stable_window_selection(self) -> None:
        block_sizes = np.array([4, 8, 16, 32], dtype=int)
        z_path = np.array([0.1, 0.11, 0.12, 0.11], dtype=float)
        window, mask = _select_stable_ei_window(block_sizes, z_path, min_points=3)
        baseline_window, baseline_mask = _baseline_select_stable_ei_window(
            block_sizes,
            z_path,
            min_points=3,
        )
        self.assertEqual(window, baseline_window)
        np.testing.assert_array_equal(mask, baseline_mask)

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


if __name__ == "__main__":
    unittest.main()
