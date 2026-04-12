"""Block-maxima summary functionals for the EVI scaling workflow."""

from __future__ import annotations

import warnings

import numpy as np

from .._validation import as_1d_float_array


def estimate_sample_mode(sample: np.ndarray | list[float], *, warn: bool = True) -> float:
    """Estimate a positive-sample mode via a log-scale KDE surrogate."""
    sample_arr = as_1d_float_array(sample)
    if warn:
        excluded = int(np.sum(np.isfinite(sample_arr) & (sample_arr <= 0)))
        if excluded:
            warnings.warn(
                (
                    f"estimate_sample_mode excluded {excluded} non-positive observations. "
                    "The KDE surrogate requires strictly positive support."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
    sample_arr = sample_arr[np.isfinite(sample_arr) & (sample_arr > 0)]
    if sample_arr.size == 0:
        return np.nan
    if sample_arr.size == 1:
        return float(sample_arr[0])
    log_sample = np.log1p(sample_arr)
    iqr = np.subtract(*np.quantile(log_sample, [0.75, 0.25]))
    sigma = min(
        np.std(log_sample, ddof=1),
        iqr / 1.349 if iqr > 0 else np.std(log_sample, ddof=1),
    )
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(np.std(log_sample, ddof=1), 1e-3)
    bandwidth = max(float(1.059 * sigma * log_sample.size ** (-0.2)), 1e-3)
    grid = np.linspace(log_sample.min(), log_sample.max(), 256)
    density = np.zeros(grid.size, dtype=float)
    chunk_size = 8192
    for start in range(0, log_sample.size, chunk_size):
        chunk = log_sample[start : start + chunk_size]
        kernel = np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / bandwidth) ** 2)
        density += kernel.sum(axis=1)
    density /= log_sample.size
    density_on_original_scale = density * np.exp(-grid)
    return float(np.expm1(grid[int(np.nanargmax(density_on_original_scale))]))


def summarize_block_maxima(
    maxima: np.ndarray | list[float],
    *,
    target: str,
    quantile: float = 0.5,
) -> float:
    """Evaluate the requested summary target on one set of block maxima."""
    maxima_arr = np.asarray(maxima, dtype=float)
    maxima_arr = maxima_arr[np.isfinite(maxima_arr)]
    if maxima_arr.size == 0:
        return np.nan
    if target == "quantile":
        return float(np.quantile(maxima_arr, quantile, method="median_unbiased"))
    if target == "mean":
        return float(np.mean(maxima_arr))
    if target == "mode":
        return estimate_sample_mode(maxima_arr, warn=False)
    raise ValueError(f"Unsupported target: {target}")
