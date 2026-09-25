"""Evaluate the existing log-KDE mode using repeated-value multiplicities."""

from __future__ import annotations

import numpy as np

from . import _accelerator


def prepare_mode_counts(
    bank: np.ndarray, *, max_bytes: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Count mode's positive finite maxima without changing segment membership.

    Invalid maxima retain their positions in the bank but contribute no KDE
    weight, as in the expanded evaluator. Check a sorting-workspace allowance
    first, then size the dense table from the actual distinct positive values.
    Empty positive support yields a zero-column table.
    """
    segments, width = bank.shape
    if bank.size == 0 or bank.size * 2 * np.dtype(np.int64).itemsize > max_bytes:
        return None
    positions = np.flatnonzero(np.isfinite(bank) & (bank > 0))
    values, inverse = np.unique(bank.ravel()[positions], return_inverse=True)
    if len(values) * (segments + 1) * np.dtype(np.int64).itemsize > max_bytes:
        return None
    if not len(values):
        return values, np.empty((segments, 0), dtype=np.int64)
    codes = (positions // width) * len(values) + inverse
    counts = np.bincount(codes, minlength=segments * len(values)).reshape(segments, -1)
    return values, counts


def weighted_density(
    logs: np.ndarray, counts: np.ndarray, grid: np.ndarray, bandwidth: np.ndarray
) -> np.ndarray:
    """Sum Gaussian kernels at each grid point, without normalization or Jacobian.

    Rows are independent bootstrap samples; integer multiplicities are passed
    as floats to the optional GIL-free loop. The NumPy fallback caps each kernel
    temporary at 32 MiB rather than expanding repeated observations.
    """
    density = np.zeros_like(grid)
    if _accelerator.kernels is not None:
        _accelerator.kernels.kde(
            logs, np.ascontiguousarray(counts, dtype=float), grid, bandwidth, density
        )
        return density
    budget = 32 * 1024**2 // (grid.shape[1] * 8)
    rows = max(1, budget // max(1, len(logs)))
    for start in range(0, len(grid), rows):
        stop = min(start + rows, len(grid))
        columns = max(1, budget // (stop - start))
        for column in range(0, len(logs), columns):
            selected = slice(column, column + columns)
            work = grid[start:stop, :, None] - logs[None, None, selected]
            work /= bandwidth[start:stop, None, None]
            np.square(work, out=work)
            work *= -0.5
            np.exp(work, out=work)
            work *= counts[start:stop, None, selected]
            density[start:stop] += work.sum(axis=2)
    return density


def mode_from_counts(
    values: np.ndarray, counts: np.ndarray, *, selected: np.ndarray | None = None
) -> np.ndarray:
    """Evaluate the 256-grid KDE from each resample's positive multiplicities.

    Without ``selected``, retain the established positive-bank arithmetic;
    all count rows must be nonempty. Supplying the corresponding expanded
    maxima preserves their moment sums and quartile interpolation, avoiding
    rounding changes in newly compressed banks. Empty positive rows return
    NaN and singletons return their observation exactly.

    A near-tied nonconstant grid also returns NaN when ``selected`` is given.
    The bootstrap owner must retry such rows with the expanded evaluator,
    retaining its original row/column reduction groups. This guard protects
    the argmax from small differences in floating-point Gaussian sums.
    """
    if selected is not None:
        valid = np.isfinite(selected) & (selected > 0)
        sizes = valid.sum(axis=1)
        summaries = np.full(len(selected), np.nan)
        single = sizes == 1
        if np.any(single):
            summaries[single] = np.max(np.where(valid[single], selected[single], -np.inf), axis=1)
        multiple = sizes > 1
        if not np.any(multiple):
            return summaries
        selected = selected[multiple]
        valid = valid[multiple]
        counts = counts[multiple]
    logs = np.log1p(values)
    n = counts.sum(axis=1)
    cumulative = counts.cumsum(axis=1)

    def rank_values(ranks):
        """Select zero-based order ranks from the sorted finite support."""
        return logs[np.sum(cumulative <= ranks[:, None], axis=1)]

    def quantile(q):
        """Match NumPy's linear quartile interpolation for the bandwidth."""
        rank = (n - 1) * q
        low = np.floor(rank).astype(int)
        gamma = rank - low
        a, b = rank_values(low), rank_values(np.minimum(low + 1, n - 1))
        difference = b - a
        if selected is not None:
            return a + difference * gamma
        return np.where(gamma >= 0.5, b - difference * (1 - gamma), a + difference * gamma)

    iqr = quantile(0.75) - quantile(0.25)
    if selected is None:
        sums = counts @ logs
        second = counts @ (logs * logs)
    else:
        expanded_logs = np.zeros_like(selected)
        np.log1p(selected, out=expanded_logs, where=valid)
        sums = expanded_logs.sum(axis=1)
        second = (expanded_logs * expanded_logs).sum(axis=1)
    means = sums / n
    std = np.sqrt(np.maximum((second - n * means * means) / np.maximum(n - 1, 1), 0.0))
    sigma = np.minimum(std, np.where(iqr > 0, iqr / 1.349, std))
    sigma = np.where(~np.isfinite(sigma) | (sigma <= 0), np.maximum(std, 1e-3), sigma)
    bandwidth = np.maximum(1.059 * sigma * n ** (-0.2), 1e-3)
    low, high = rank_values(np.zeros(len(n), dtype=int)), rank_values(n - 1)
    grid = low[:, None] + (high - low)[:, None] * np.linspace(0.0, 1.0, 256)[None, :]
    density = weighted_density(logs, counts, grid, bandwidth)
    density /= n[:, None]
    density *= np.exp(-grid)  # Jacobian for z = log(1 + x).
    modes = np.expm1(grid[np.arange(len(n)), np.argmax(density, axis=1)])
    if selected is None:
        return modes
    summaries[multiple] = modes
    top = np.partition(density, -2, axis=1)[:, -2:]
    best = top.max(axis=1)
    # Conservative roundoff guard, not a statistical tuning parameter. A
    # constant grid has the same returned value whichever point wins.
    ambiguous = (best - top.min(axis=1) <= 64 * np.finfo(float).eps * selected.shape[1] * best) & (
        high != low
    )
    summaries[np.flatnonzero(multiple)[ambiguous]] = np.nan
    return summaries
