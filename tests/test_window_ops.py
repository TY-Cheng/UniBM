from __future__ import annotations

import unittest

import numpy as np

from scripts.unibm.window_ops import (
    circular_sliding_window_maximum,
    sliding_window_extreme_valid,
)


def _baseline_sliding_window_extreme_valid(
    vec: np.ndarray,
    window: int,
    *,
    reducer: str,
) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if window < 2 or arr.size < window:
        return np.asarray([], dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    valid = np.all(np.isfinite(windows), axis=1)
    if reducer == "max":
        return windows.max(axis=1)[valid]
    return windows.min(axis=1)[valid]


def _baseline_circular_sliding_window_maximum(
    vec: np.ndarray,
    window: int,
) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if window < 2 or arr.size < window:
        return np.asarray([], dtype=float)
    wrapped = np.concatenate([arr, arr[: window - 1]])
    windows = np.lib.stride_tricks.sliding_window_view(wrapped, window)[: arr.size]
    return windows.max(axis=1)


class WindowOpsTests(unittest.TestCase):
    def test_sliding_window_extreme_valid_matches_baseline_for_finite_maxima(self) -> None:
        values = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], dtype=float)
        observed = sliding_window_extreme_valid(values, 3, reducer="max")
        expected = _baseline_sliding_window_extreme_valid(values, 3, reducer="max")
        np.testing.assert_allclose(observed, expected)

    def test_sliding_window_extreme_valid_matches_baseline_with_missing_minima(self) -> None:
        values = np.array([1.0, np.nan, 2.0, 5.0, 4.0, np.nan, 3.0], dtype=float)
        observed = sliding_window_extreme_valid(values, 2, reducer="min")
        expected = _baseline_sliding_window_extreme_valid(values, 2, reducer="min")
        np.testing.assert_allclose(observed, expected, equal_nan=True)

    def test_circular_sliding_window_maximum_matches_baseline_for_finite_series(self) -> None:
        values = np.array([1.0, 5.0, 2.0, 4.0], dtype=float)
        observed = circular_sliding_window_maximum(values, 2)
        expected = _baseline_circular_sliding_window_maximum(values, 2)
        np.testing.assert_allclose(observed, expected)

    def test_circular_sliding_window_maximum_matches_baseline_with_missing_values(self) -> None:
        values = np.array([1.0, np.nan, 2.0, 4.0], dtype=float)
        observed = circular_sliding_window_maximum(values, 3)
        expected = _baseline_circular_sliding_window_maximum(values, 3)
        np.testing.assert_allclose(observed, expected, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
