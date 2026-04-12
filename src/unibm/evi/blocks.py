"""Block extraction and block-summary curve construction for EVI fits."""

from __future__ import annotations

import numpy as np

from .._validation import as_1d_float_array, warn_on_negative_values, warn_on_nonpositive_values
from .._window_ops import sliding_window_extreme_valid
from .models import BlockSummaryCurve
from .summaries import summarize_block_maxima


def block_maxima(
    vec: np.ndarray | list[float],
    block_size: int,
    sliding: bool = True,
) -> np.ndarray:
    """Compute sliding or disjoint block maxima."""
    arr = as_1d_float_array(vec)
    if block_size < 2 or arr.size < block_size:
        return np.asarray([], dtype=float)
    if sliding:
        return sliding_window_extreme_valid(arr, block_size, reducer="max")
    n_block = arr.size // block_size
    if n_block < 1:
        return np.asarray([], dtype=float)
    windows = arr[: n_block * block_size].reshape(n_block, block_size)
    maxima = windows.max(axis=1)
    valid = np.all(np.isfinite(windows), axis=1)
    return maxima[valid]


def block_summary_curve(
    vec: np.ndarray | list[float],
    block_sizes: np.ndarray,
    *,
    sliding: bool = True,
    quantile: float = 0.5,
    target: str = "quantile",
) -> BlockSummaryCurve:
    """Summarize block maxima over multiple block sizes."""
    warn_on_negative_values(vec, context="block_summary_curve", stacklevel=3)
    block_sizes = np.asarray(block_sizes, dtype=int)
    values = np.empty(block_sizes.size, dtype=float)
    counts = np.empty(block_sizes.size, dtype=int)
    for idx, block_size in enumerate(block_sizes):
        maxima = block_maxima(vec=vec, block_size=int(block_size), sliding=sliding)
        counts[idx] = maxima.size
        values[idx] = summarize_block_maxima(maxima, target=target, quantile=quantile)
    excluded = values[np.isfinite(values) & (values <= 0) & (counts > 0)]
    if excluded.size:
        warn_on_nonpositive_values(
            excluded,
            context="block_summary_curve",
            noun="block summaries excluded from the log-log regression",
            stacklevel=3,
        )
    positive_mask = np.isfinite(values) & (values > 0) & (counts > 0)
    return BlockSummaryCurve(
        block_sizes=block_sizes,
        counts=counts,
        values=values,
        positive_mask=positive_mask,
    )
