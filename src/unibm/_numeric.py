"""Small shared numeric helpers."""

from __future__ import annotations

import numpy as np


def prefix_sum(values: np.ndarray) -> np.ndarray:
    """Return a float prefix sum with a leading zero."""
    prefix = np.empty(values.size + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(values, dtype=float, out=prefix[1:])
    return prefix
