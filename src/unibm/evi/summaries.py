"""Block-maxima summary functionals for the EVI scaling workflow."""

from __future__ import annotations

import warnings

import numpy as np

from .._validation import as_1d_float_array
from ._mode import weighted_density


def _validate_quantile(quantile: float) -> float:
    """Return a finite non-boolean quantile strictly inside the unit interval."""
    if isinstance(quantile, (bool, np.bool_)):
        raise ValueError("quantile must be finite and lie strictly between 0 and 1.")
    try:
        value = float(quantile)
    except (TypeError, ValueError) as exc:
        raise ValueError("quantile must be finite and lie strictly between 0 and 1.") from exc
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("quantile must be finite and lie strictly between 0 and 1.")
    return value


def estimate_sample_mode(sample: np.ndarray | list[float], *, warn: bool = True) -> float:
    """Approximate a positive-sample mode using a Gaussian KDE on ``log1p(x)``.

    Nonfinite and nonpositive observations are removed; ``warn`` controls
    warnings about the latter. Evaluate 256 evenly spaced transformed
    points and apply the Jacobian before maximizing density on the original
    scale. Return NaN for no retained values and the value itself for a
    singleton. This grid-based surrogate is not an exact density mode.
    """
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
    values, counts = np.unique(log_sample, return_counts=True)
    density = weighted_density(values, counts[None, :], grid[None, :], np.array([bandwidth]))[0]
    density /= log_sample.size
    # For z = log(1 + x), dz/dx = exp(-z); maximize density in x, not in z.
    density_on_original_scale = density * np.exp(-grid)
    return float(np.expm1(grid[int(np.nanargmax(density_on_original_scale))]))


def summarize_block_maxima(
    maxima: np.ndarray | list[float],
    *,
    target: str,
    quantile: float = 0.5,
) -> float:
    """Return a quantile, arithmetic mean, or KDE mode of finite maxima.

    An empty finite sample returns NaN. Quantiles use NumPy's
    ``median_unbiased`` interpolation with ``0 < quantile < 1``. Quantiles
    and means retain zeros and negatives; the mode surrogate uses only
    positive maxima. An unsupported target raises ValueError for a
    nonempty finite sample.
    """
    maxima_arr = np.asarray(maxima, dtype=float)
    maxima_arr = maxima_arr[np.isfinite(maxima_arr)]
    if maxima_arr.size == 0:
        return np.nan
    if target == "quantile":
        return float(
            np.quantile(
                maxima_arr,
                _validate_quantile(quantile),
                method="median_unbiased",
            )
        )
    if target == "mean":
        return float(np.mean(maxima_arr))
    if target == "mode":
        return estimate_sample_mode(maxima_arr, warn=False)
    raise ValueError(f"Unsupported target: {target}")
