"""Public empirical CDF helper shared across UniBM workflows."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _as_finite_1d(vec: np.ndarray | list[float]) -> np.ndarray:
    """Flatten an array-like sample and omit its non-finite observations."""
    arr = np.asarray(vec, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _empty_cdf_estimator() -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return a CDF estimator that yields NaN on every query."""

    def estimate_empty(q: float | np.ndarray) -> float | np.ndarray:
        """Return NaN with the query's shape, including a float for scalar input."""
        q_arr = np.asarray(q, dtype=float)
        result = np.full(q_arr.shape, np.nan, dtype=float)
        return float(result.item()) if result.ndim == 0 else result

    return estimate_empty


def _singleton_cdf_estimator(point: float) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return the exact CDF of a single-point empirical distribution."""

    def estimate_single(q: float | np.ndarray) -> float | np.ndarray:
        """Return zero below the point and one at or above it, preserving NaNs."""
        q_arr = np.asarray(q, dtype=float)
        result = np.where(np.isnan(q_arr), np.nan, np.where(q_arr >= point, 1.0, 0.0))
        return float(result.item()) if np.ndim(result) == 0 else result

    return estimate_single


def empirical_cdf(
    vec: np.ndarray | list[float],
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return a callable that evaluates right-inclusive empirical ranks.

    Flatten the sample and omit NaN and infinite observations. For at least
    two retained values, a query q returns ``count(x <= q) / (n + 1)``. This
    plotting-position convention stays below one even above the sample
    maximum; it is not the usual empirical CDF with denominator n. Ties use
    the largest rank, which includes every observation equal to q.

    An empty sample gives NaN everywhere. A singleton instead uses its exact
    point-mass CDF (zero below the point, one otherwise). The callable accepts
    scalars or arrays, preserves the query shape, and propagates NaN queries.
    """
    arr = np.sort(_as_finite_1d(vec))
    if arr.size == 0:
        return _empty_cdf_estimator()
    if arr.size == 1:
        return _singleton_cdf_estimator(float(arr[0]))
    normalizer = float(arr.size + 1)

    def estimate(q: float | np.ndarray) -> float | np.ndarray:
        """Evaluate count(x <= q) / (n + 1) while preserving query shape and NaNs."""
        q_arr = np.asarray(q, dtype=float)
        q_flat = q_arr.reshape(-1)
        counts = np.searchsorted(arr, q_flat, side="right").astype(float)
        scaled = counts / normalizer
        scaled[np.isnan(q_flat)] = np.nan
        result = scaled.reshape(q_arr.shape)
        return float(result.item()) if result.ndim == 0 else result

    return estimate


__all__ = [
    "empirical_cdf",
]
