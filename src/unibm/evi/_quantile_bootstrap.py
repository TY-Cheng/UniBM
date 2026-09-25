"""Exact median-unbiased quantiles from resampled segment multiplicities."""

from __future__ import annotations

import numpy as np

from . import _accelerator


CountTable = tuple[np.ndarray, np.ndarray]


def prepare_quantile_counts(bank: np.ndarray, *, max_bytes: int) -> CountTable | None:
    """Cache per-segment cumulative counts when a finite nonnegative bank fits.

    Each column is one segment; row k counts its observations at or below
    distinct value k. Zeros remain observations. Nonfinite or negative banks
    retain ordinary NumPy quantiles. Check a sorting-workspace allowance first,
    then budget the dense table using the actual number of distinct values;
    repeated sliding maxima need not pay for a worst-case unique-value table.
    """
    segments, width = bank.shape
    if (
        bank.size == 0
        or bank.size * 2 * np.dtype(np.int64).itemsize > max_bytes
        or not np.all(np.isfinite(bank))
        or np.any(bank < 0)
    ):
        return None
    values, inverse = np.unique(bank, return_inverse=True)
    if len(values) * (segments + 1) * np.dtype(np.int64).itemsize > max_bytes:
        return None
    segment = np.repeat(np.arange(segments), width)
    counts = np.bincount(
        inverse.ravel() * segments + segment, minlength=len(values) * segments
    ).reshape(-1, segments)
    np.cumsum(counts, axis=0, out=counts)
    return values, counts


def segment_multiplicities(draws: np.ndarray) -> np.ndarray:
    """Count repeated segment indices independently in each bootstrap draw."""
    segments = draws.shape[1]
    codes = draws + np.arange(len(draws))[:, None] * segments
    return np.bincount(codes.ravel(), minlength=len(draws) * segments).reshape(-1, segments)


def _search_rank(prefix: np.ndarray, weights: np.ndarray, rank: int) -> np.ndarray:
    """Find the first cumulative count strictly above a zero-based order rank."""
    low = np.zeros(len(weights), dtype=np.int64)
    high = np.full(len(weights), len(prefix), dtype=np.int64)
    for _ in range(len(prefix).bit_length()):
        mid = (low + high) // 2
        count = np.sum(prefix[mid] * weights, axis=1)
        above = count > rank
        high = np.where(above, mid, high)
        low = np.where(above, low, mid + 1)
    return low


def quantile_from_counts(
    table: CountTable, weights: np.ndarray, *, size: int, quantile: float, max_bytes: int
) -> np.ndarray:
    """Evaluate NumPy's median-unbiased order ranks and interpolation exactly.

    Native kernels search integer count tables without holding the GIL.
    Without them, small banks use dense integer products (not BLAS); large
    ones use a vectorized search to avoid an R-by-distinct-values temporary.
    Interpolation retains NumPy's two-sided arithmetic and boundary clipping.
    """
    values, prefix = table
    q = np.float64(quantile)
    index = size * q + (1 / 3 + q * (1 - 1 / 3 - 1 / 3)) - 1
    floor = int(np.floor(index))
    gamma = float(index - floor)
    lower, upper = max(0, min(floor, size - 1)), max(0, min(floor + 1, size - 1))
    if _accelerator.kernels is not None:
        ranks = np.empty((len(weights), 2), dtype=np.int64)
        _accelerator.kernels.rank_indices(prefix, weights, lower, upper, ranks)
        first, second = ranks[:, 0], ranks[:, 1]
    elif size <= 4096 and len(weights) * len(values) * 8 <= max_bytes:
        cumulative = weights @ prefix.T
        first = np.sum(cumulative <= lower, axis=1)
        second = np.sum(cumulative <= upper, axis=1)
    else:
        first = _search_rank(prefix, weights, lower)
        second = first.copy()
        need_next = np.sum(prefix[first] * weights, axis=1) <= upper
        if np.any(need_next):
            second[need_next] = _search_rank(prefix, weights[need_next], upper)
    a, b = values[first], values[second]
    difference = b - a
    return b - difference * (1 - gamma) if gamma >= 0.5 else a + difference * gamma
