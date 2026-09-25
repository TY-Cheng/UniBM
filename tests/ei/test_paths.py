from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from unibm.ei._validation import _validate_ei_series
from unibm.ei.models import EiPathBundle, EiStableWindow
from unibm.ei.paths import (
    _build_bm_paths_from_values,
    _build_path_from_scores,
    _rolling_window_minima,
)
from unibm.ei.preparation import prepare_ei_bundle
from unibm.ei.selection import extract_stable_path_window, select_stable_path_window


def _baseline_select_stable_ei_window(
    block_sizes: np.ndarray,
    z_path: np.ndarray,
    *,
    min_points: int = 4,
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
    best: tuple[float, int, int] | None = None
    for start in range(levels.size - min_points + 1):
        for stop in range(start + min_points, levels.size + 1):
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

    def test_series_validation_preserves_positions_and_rejects_invalid_support(self) -> None:
        positive = _validate_ei_series(np.arange(1.0, 40.0, dtype=float), allow_zeros=False)
        nonnegative = _validate_ei_series(
            np.concatenate([[0.0], np.arange(1.0, 40.0, dtype=float)]), allow_zeros=True
        )
        np.testing.assert_array_equal(positive, np.arange(1.0, 40.0, dtype=float))
        np.testing.assert_array_equal(
            nonnegative,
            np.concatenate([[0.0], np.arange(1.0, 40.0, dtype=float)]),
        )

        invalid_positive = np.arange(1.0, 40.0, dtype=float)
        invalid_positive[10] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_ei_series(invalid_positive, allow_zeros=False)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            _validate_ei_series(np.concatenate([[0.0], np.arange(1.0, 40.0)]), allow_zeros=False)

        invalid_nonnegative = np.arange(40.0, dtype=float)
        invalid_nonnegative[10] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_ei_series(invalid_nonnegative, allow_zeros=True)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _validate_ei_series(np.concatenate([[-1.0], np.arange(39.0)]), allow_zeros=True)

        with self.assertRaisesRegex(ValueError, "at least 32 observations"):
            _validate_ei_series(np.arange(1.0, 10.0, dtype=float), allow_zeros=False)
        with self.assertRaisesRegex(ValueError, "at least 32 observations"):
            _validate_ei_series(np.arange(10.0, dtype=float), allow_zeros=True)

    def test_prepare_bundle_requires_clock_choice_and_valid_block_grid(self) -> None:
        values = self._positive_sample()
        with self.assertRaises(TypeError):
            prepare_ei_bundle(values)

        invalid_grids = {
            "fractional": np.array([4.0, 8.5, 16.0, 32.0]),
            "duplicate": np.array([4, 8, 8, 16, 32]),
            "unsorted": np.array([4, 16, 8, 32]),
            "too small": np.array([1, 4, 8, 16, 32]),
            "too large": np.array([4, 8, 16, 32, values.size + 1]),
        }
        for label, block_sizes in invalid_grids.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "block_sizes"):
                    prepare_ei_bundle(
                        values,
                        block_sizes=block_sizes,
                        allow_zeros=False,
                    )

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

    def test_default_ei_grid_uses_square_root_cap(self) -> None:
        for n_obs, upper in ((170, 10), (365, 19), (2500, 50)):
            with self.subTest(n_obs=n_obs):
                bundle = prepare_ei_bundle(self._positive_sample(n_obs), allow_zeros=False)
                self.assertEqual(bundle.block_sizes[-1], upper)
        with self.assertRaisesRegex(ValueError, "Not enough finite EI path values"):
            prepare_ei_bundle(self._positive_sample(128), allow_zeros=False)
        with self.assertRaisesRegex(ValueError, "max_block_size must be greater"):
            prepare_ei_bundle(self._positive_sample(91), allow_zeros=False)

    def test_rolling_window_minima(self) -> None:
        scores = np.array([4.0, 2.0, np.nan, 1.0, 3.0, 5.0], dtype=float)
        np.testing.assert_allclose(
            _rolling_window_minima(scores, 2, sliding=True), np.array([2.0, 1.0, 3.0])
        )
        np.testing.assert_allclose(
            _rolling_window_minima(scores, 2, sliding=False), np.array([2.0, 3.0])
        )

    def test_requested_paths_match_full_preparation_without_unrequested_work(self) -> None:
        values = self._positive_sample()
        full = prepare_ei_bundle(values, allow_zeros=False)
        with mock.patch(
            "unibm.ei.paths._build_path_from_scores", wraps=_build_path_from_scores
        ) as build_path:
            selected = prepare_ei_bundle(values, allow_zeros=False, path_keys=(("bb", True),))
        self.assertEqual(build_path.call_count, 1)
        self.assertEqual(set(selected.paths), {("bb", True)})
        actual, expected = selected.paths[("bb", True)], full.paths[("bb", True)]
        np.testing.assert_array_equal(actual.theta_path, expected.theta_path)
        self.assertEqual(actual.stable_window, expected.stable_window)
        for invalid in (
            None,
            (("unknown", True),),
            (("bb", 1),),
            (("bb", True), ("bb", True)),
            (([], True),),
        ):
            with self.subTest(path_keys=invalid):
                with self.assertRaisesRegex(ValueError, "path_keys"):
                    prepare_ei_bundle(values, allow_zeros=False, path_keys=invalid)

    def test_threshold_only_skips_block_grid_and_path_preparation(self) -> None:
        with (
            mock.patch("unibm.ei.preparation.generate_block_sizes") as grid,
            mock.patch("unibm.ei.preparation._build_bm_paths_from_values") as paths,
        ):
            bundle = prepare_ei_bundle(self._positive_sample(), allow_zeros=False, path_keys=())
        grid.assert_not_called()
        paths.assert_not_called()
        self.assertEqual(bundle.paths, {})
        self.assertEqual(bundle.block_sizes.size, 0)
        with self.assertRaisesRegex(ValueError, "block_sizes is not used"):
            prepare_ei_bundle(
                self._positive_sample(), allow_zeros=False, path_keys=(), block_sizes=[8]
            )

    def test_path_builders_and_stable_window_selection(self) -> None:
        block_sizes = np.array([4, 8, 16, 32], dtype=int)
        z_path = np.array([0.1, 0.11, 0.12, 0.11], dtype=float)
        window, mask = select_stable_path_window(block_sizes, z_path, min_points=3)
        baseline_window, baseline_mask = _baseline_select_stable_ei_window(
            block_sizes,
            z_path,
            min_points=3,
        )
        self.assertEqual(window, baseline_window)
        np.testing.assert_array_equal(mask, baseline_mask)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            select_stable_path_window(
                np.array([4, 16, 8, 32], dtype=int),
                z_path,
                min_points=3,
            )

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

    def test_selection_edge_cases_cover_missing_values_and_minimal_windows(self) -> None:
        with self.assertRaisesRegex(ValueError, "Not enough finite EI path values"):
            select_stable_path_window(
                np.array([4, 8], dtype=int),
                np.array([np.nan, 0.1], dtype=float),
                min_points=2,
            )
        with self.assertRaisesRegex(ValueError, "min_points must be an integer at least 2"):
            select_stable_path_window(
                np.array([4], dtype=int),
                np.array([0.2], dtype=float),
                min_points=1,
            )
        window, mask = select_stable_path_window(
            np.array([4, 8, 16], dtype=int),
            np.array([0.1, 0.11, 0.12], dtype=float),
            min_points=3,
        )
        self.assertEqual(window, EiStableWindow(4, 16))
        np.testing.assert_array_equal(mask, np.array([True, True, True]))

    def test_selection_can_reach_either_finite_grid_endpoint(self) -> None:
        levels = np.arange(2, 14)
        z = np.array([np.nan, 1, 1, 1, 1, 2, 4, 8, 16, 32, 64, np.nan])
        for values, expected in ((z, (3, 6)), (z[::-1], (9, 12))):
            with self.subTest(expected=expected):
                window, mask = select_stable_path_window(levels, values)
                self.assertEqual((window.lo, window.hi), expected)
                self.assertEqual(mask.size, 10)
                self.assertEqual(mask.sum(), 4)

    def test_batched_window_scan_keeps_ties_and_finite_label_alignment(self) -> None:
        levels = np.arange(2, 77)
        for start in (0, 40, 70):
            z = np.r_[100 + np.arange(start, dtype=float) ** 2, np.zeros(75 - start)]
            if start:
                z[2] = np.nan
            window, mask = select_stable_path_window(levels, z)
            self.assertEqual((window.lo, window.hi), (levels[start], levels[start + 3]))
            self.assertEqual(mask.sum(), 4)
            np.testing.assert_array_equal(levels[np.isfinite(z)][mask], levels[start : start + 4])

    def test_stable_path_selection_rejects_invalid_tuning_parameters(self) -> None:
        block_sizes = np.array([4, 8, 16], dtype=int)
        z_path = np.array([0.1, 0.11, 0.12], dtype=float)
        for name, value in (("roughness_penalty", -1.0), ("curvature_penalty", np.nan)):
            with self.subTest(name=name, value=value):
                kwargs = {name: value}
                with self.assertRaisesRegex(ValueError, f"{name} must be finite"):
                    select_stable_path_window(
                        block_sizes,
                        z_path,
                        min_points=2,
                        **kwargs,
                    )


if __name__ == "__main__":
    unittest.main()
