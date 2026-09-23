"""Small shared numeric helpers."""

from __future__ import annotations

import numpy as np


def prefix_sum(values: np.ndarray) -> np.ndarray:
    """Return n + 1 cumulative sums for a 1D array, starting with zero.

    The sum of ``values[start:stop]`` is then ``prefix[stop] - prefix[start]``.
    Accumulation uses floating point even when the input contains integers.
    """
    prefix = np.empty(values.size + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(values, dtype=float, out=prefix[1:])
    return prefix
