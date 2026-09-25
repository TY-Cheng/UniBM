"""Small shared numeric helpers."""

from __future__ import annotations

import numpy as np


def candidate_window_batches(n: int, min_points: int):
    """Yield start/stop arrays in the original nested-loop order.

    Batch at most 32 starting positions at once: NumPy can score all their
    windows together without allocating quadratic working memory for long
    user-supplied grids. Stops are exclusive; exact ties still prefer the
    earliest start, then the earliest stop.
    """
    for first in range(0, n - min_points + 1, 32):
        starts = np.arange(first, min(first + 32, n - min_points + 1))
        stops = np.arange(first + min_points, n + 1)
        start, stop = np.broadcast_arrays(starts[:, None], stops[None, :])
        valid = stop - start >= min_points
        yield start[valid], stop[valid]


def prefix_sum(values: np.ndarray) -> np.ndarray:
    """Return n + 1 cumulative sums for a 1D array, starting with zero.

    The sum of ``values[start:stop]`` is then ``prefix[stop] - prefix[start]``.
    Accumulation uses floating point even when the input contains integers.
    """
    prefix = np.empty(values.size + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(values, dtype=float, out=prefix[1:])
    return prefix
