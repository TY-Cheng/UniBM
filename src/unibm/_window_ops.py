"""Low-level rolling-window extrema helpers shared across EVI and EI."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d


Reducer = Literal["max", "min"]


def _as_1d_float_array(vec: np.ndarray | list[float]) -> np.ndarray:
    """Convert to floats and flatten all dimensions without filtering values."""
    return np.asarray(vec, dtype=float).reshape(-1)


def _rolling_extreme_finite(
    arr: np.ndarray,
    window: int,
    *,
    reducer: Reducer,
) -> np.ndarray:
    """Reduce each complete, unpadded window to its maximum or minimum.

    The result has length ``n - window + 1``. A window smaller than two or
    longer than the flattened input returns an empty array. No values are
    filtered here; callers handle non-finite observations separately.
    """
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if window < 2 or arr.size < window:
        return np.asarray([], dtype=float)
    operation = maximum_filter1d if reducer == "max" else minimum_filter1d
    # SciPy centers its filter; this slice retains only complete original windows.
    start = window // 2
    return operation(arr, size=window)[start : start + arr.size - window + 1]


def _finite_window_mask(arr: np.ndarray, window: int) -> np.ndarray:
    """Mark complete windows containing no NaN or infinity in a 1D array.

    Prefix counts avoid constructing a Boolean matrix of overlapping windows.
    The caller supplies a window between one and the input length.
    """
    nonfinite = (~np.isfinite(arr)).astype(np.int64, copy=False)
    prefix = np.empty(arr.size + 1, dtype=np.int64)
    prefix[0] = 0
    np.cumsum(nonfinite, out=prefix[1:])
    return (prefix[window:] - prefix[:-window]) == 0


def sliding_window_extreme_valid(
    vec: np.ndarray | list[float],
    window: int,
    *,
    reducer: Reducer,
) -> np.ndarray:
    """Return extrema only for complete windows whose observations are finite.

    Invalid windows are omitted after forming windows on the original series,
    so gaps never join observations that were originally separated. Input is
    flattened; windows smaller than two or longer than the input return an
    empty array. The surviving extrema retain their original window order.
    """
    arr = _as_1d_float_array(vec)
    if window < 2 or arr.size < window:
        return np.asarray([], dtype=float)
    fill_value = -np.inf if reducer == "max" else np.inf
    safe = np.where(np.isfinite(arr), arr, fill_value)
    extrema = _rolling_extreme_finite(safe, window, reducer=reducer)
    return extrema[_finite_window_mask(arr, window)]


def circular_sliding_window_maximum(
    vec: np.ndarray | list[float],
    window: int,
) -> np.ndarray:
    """Return circular sliding maxima for one segment.

    For a valid window, return one maximum per observation, with windows
    starting near the end wrapping to the beginning of the flattened input.
    Windows containing NaN return NaN. Infinite values remain comparable and
    can therefore survive as the maximum. A window smaller than two or longer
    than the input returns an empty array.
    """
    arr = _as_1d_float_array(vec)
    if window < 2 or arr.size < window:
        return np.asarray([], dtype=float)
    return _circular_sliding_maxima_rows(arr[None, :], window)[0]


def _circular_sliding_maxima_rows(segments: np.ndarray, window: int) -> np.ndarray:
    """Batch circular maxima within rows for an already validated window.

    Each row is a separate segment. Preserve NaN-containing windows as NaN
    and allow infinities as extrema, just as the one-segment entry point does.
    """
    width = segments.shape[1]
    wrapped = np.concatenate([segments, segments[:, : window - 1]], axis=1)
    missing = np.isnan(wrapped)
    safe = np.where(missing, -np.inf, wrapped)
    start = window // 2
    maxima = maximum_filter1d(safe, size=window, axis=1)[:, start : start + width].copy()
    prefix = np.empty((len(segments), wrapped.shape[1] + 1), dtype=np.int64)
    prefix[:, 0] = 0
    np.cumsum(missing, axis=1, out=prefix[:, 1:])
    maxima[(prefix[:, window:] - prefix[:, :-window]) > 0] = np.nan
    return maxima


__all__ = [
    "circular_sliding_window_maximum",
    "sliding_window_extreme_valid",
]
