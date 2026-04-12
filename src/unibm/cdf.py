"""Public empirical CDF helper shared across UniBM workflows."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _as_finite_1d(vec: np.ndarray | list[float]) -> np.ndarray:
    """Return a one-dimensional finite float array."""
    arr = np.asarray(vec, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _empty_cdf_estimator() -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return a CDF estimator that yields NaN on every query."""

    def estimate_empty(q: float | np.ndarray) -> float | np.ndarray:
        q_arr = np.asarray(q, dtype=float)
        result = np.full(q_arr.shape, np.nan, dtype=float)
        return float(result.item()) if result.ndim == 0 else result

    return estimate_empty


def _singleton_cdf_estimator(point: float) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return the exact CDF of a single-point empirical distribution."""

    def estimate_single(q: float | np.ndarray) -> float | np.ndarray:
        q_arr = np.asarray(q, dtype=float)
        result = np.where(q_arr >= point, 1.0, 0.0)
        return float(result.item()) if np.ndim(result) == 0 else result

    return estimate_single


def empirical_cdf(
    vec: np.ndarray | list[float],
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return an empirical marginal CDF estimator based on scaled ranks."""
    arr = np.sort(_as_finite_1d(vec))
    if arr.size == 0:
        return _empty_cdf_estimator()
    if arr.size == 1:
        return _singleton_cdf_estimator(float(arr[0]))
    normalizer = float(arr.size + 1)

    def estimate(q: float | np.ndarray) -> float | np.ndarray:
        q_arr = np.asarray(q, dtype=float)
        q_flat = q_arr.reshape(-1)
        counts = np.searchsorted(arr, q_flat, side="right").astype(float)
        result = (counts / normalizer).reshape(q_arr.shape)
        return float(result.item()) if result.ndim == 0 else result

    return estimate


__all__ = [
    "empirical_cdf",
]
