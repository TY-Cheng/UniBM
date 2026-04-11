"""Low-level rolling-window extrema helpers with guarded finite fast paths."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


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
    """Return trailing rolling extrema for a fully finite array."""
    rolling = pd.Series(arr, copy=False).rolling(window=window, min_periods=window)
    if reducer == "max":
        return rolling.max().to_numpy(dtype=float)[window - 1 :]
    return rolling.min().to_numpy(dtype=float)[window - 1 :]


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
    if use_fast_path and np.isfinite(arr).all():
        return _rolling_extreme_finite(arr, window, reducer=reducer)
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    valid = np.all(np.isfinite(windows), axis=1)
    if reducer == "max":
        return windows.max(axis=1)[valid]
    return windows.min(axis=1)[valid]


def circular_sliding_window_maximum(
    vec: np.ndarray | list[float],
    window: int,
    *,
    use_fast_path: bool = False,
) -> np.ndarray:
    """Return circular sliding maxima for one segment.

    The wrapped bootstrap case keeps the original stride-based path by default.
    Profiling showed that the finite-only rolling fast path is helpful for the
    long one-pass application/EI windows, but not for circular bootstrap
    segments where the wraparound copy dominates and pandas adds overhead.
    """
    arr = _as_1d_float_array(vec)
    if window < 2 or arr.size < window:
        return np.asarray([], dtype=float)
    wrapped = np.concatenate([arr, arr[: window - 1]])
    if use_fast_path and np.isfinite(wrapped).all():
        return _rolling_extreme_finite(wrapped, window, reducer="max")[: arr.size]
    windows = np.lib.stride_tricks.sliding_window_view(wrapped, window)[: arr.size]
    return windows.max(axis=1)


__all__ = [
    "circular_sliding_window_maximum",
    "sliding_window_extreme_valid",
]
