"""Low-level rolling-window extrema helpers shared across EVI and EI."""

from __future__ import annotations

from typing import Literal

import numpy as np


Reducer = Literal["max", "min"]


def _as_1d_float_array(vec: np.ndarray | list[float]) -> np.ndarray:
    """Coerce one input vector to a 1D float array."""
    return np.asarray(vec, dtype=float).reshape(-1)


def _rolling_extreme_finite(
    arr: np.ndarray,
    window: int,
    *,
    reducer: Reducer,
) -> np.ndarray:
    """Return trailing rolling extrema for one numeric array."""
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if window < 2 or arr.size < window:
        return np.asarray([], dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    if reducer == "max":
        return windows.max(axis=1)
    return windows.min(axis=1)


def _finite_window_mask(arr: np.ndarray, window: int) -> np.ndarray:
    """Return which trailing windows contain only finite observations."""
    nonfinite = (~np.isfinite(arr)).astype(np.int64, copy=False)
    prefix = np.empty(arr.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(nonfinite, out=prefix[1:])
    return (prefix[window:] - prefix[:-window]) == 0


def _nan_window_mask(arr: np.ndarray, window: int) -> np.ndarray:
    """Return which trailing windows contain at least one NaN."""
    nan_count = np.isnan(arr).astype(np.int64, copy=False)
    prefix = np.empty(arr.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(nan_count, out=prefix[1:])
    return (prefix[window:] - prefix[:-window]) > 0


def sliding_window_extreme_valid(
    vec: np.ndarray | list[float],
    window: int,
    *,
    reducer: Reducer,
    use_fast_path: bool = True,
) -> np.ndarray:
    """Return sliding-window extrema, dropping windows with non-finite inputs."""
    arr = _as_1d_float_array(vec)
    if window < 2 or arr.size < window:
        return np.asarray([], dtype=float)
    del use_fast_path
    fill_value = -np.inf if reducer == "max" else np.inf
    safe = np.where(np.isfinite(arr), arr, fill_value)
    extrema = _rolling_extreme_finite(safe, window, reducer=reducer)
    return extrema[_finite_window_mask(arr, window)]


def circular_sliding_window_maximum(
    vec: np.ndarray | list[float],
    window: int,
    *,
    use_fast_path: bool = False,
) -> np.ndarray:
    """Return circular sliding maxima for one segment.

    Windows that include a NaN propagate ``nan`` exactly as the old stride-based
    baseline did. Infinite values remain comparable and can therefore survive
    as the window maximum.
    """
    arr = _as_1d_float_array(vec)
    if window < 2 or arr.size < window:
        return np.asarray([], dtype=float)
    del use_fast_path
    wrapped = np.concatenate([arr, arr[: window - 1]])
    safe = np.where(np.isnan(wrapped), -np.inf, wrapped)
    maxima = _rolling_extreme_finite(safe, window, reducer="max")[: arr.size]
    maxima[_nan_window_mask(wrapped, window)[: arr.size]] = np.nan
    return maxima


__all__ = [
    "circular_sliding_window_maximum",
    "sliding_window_extreme_valid",
]
