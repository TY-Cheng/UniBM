"""Evaluate the existing log-KDE mode using repeated-value multiplicities."""

from __future__ import annotations

import numpy as np

from . import _accelerator


def prepare_mode_counts(
    bank: np.ndarray, *, max_bytes: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Cache finite positive maxima counts per segment within the table budget.

    Other supports keep the expanded evaluator's filtering and warnings. The
    conservative bound is checked before allocating the dense count matrix.
    """
    segments, width = bank.shape
    if (
        bank.size == 0
        or bank.size * (segments + 1) * 8 > max_bytes
        or not np.all(np.isfinite(bank))
        or np.any(bank <= 0)
    ):
        return None
    values, inverse = np.unique(bank, return_inverse=True)
    codes = np.repeat(np.arange(segments), width) * len(values) + inverse.ravel()
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


def mode_from_counts(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Return the unchanged 256-grid KDE mode for positive resampled maxima.

    Recover bandwidth quantiles and moments from multiplicities. This avoids
    sorting each expanded replicate. Each row has a nonempty positive sample;
    constant and singleton rows collapse to their sole retained value.
    """
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
        return np.where(gamma >= 0.5, b - difference * (1 - gamma), a + difference * gamma)

    iqr = quantile(0.75) - quantile(0.25)
    sums = counts @ logs
    second = counts @ (logs * logs)
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
    return np.expm1(grid[np.arange(len(n)), np.argmax(density, axis=1)])
