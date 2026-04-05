"""Small validation helpers shared across UniBM modules."""

from __future__ import annotations

import warnings

import numpy as np


def as_1d_float_array(vec: np.ndarray | list[float]) -> np.ndarray:
    """Convert arbitrary array-like input to a flat float array."""
    return np.asarray(vec, dtype=float).reshape(-1)


def warn_on_negative_values(
    vec: np.ndarray | list[float],
    *,
    context: str,
    stacklevel: int = 2,
) -> None:
    """Warn once when negative values enter a positive-support workflow."""
    arr = as_1d_float_array(vec)
    negative_count = int(np.sum(np.isfinite(arr) & (arr < 0)))
    if negative_count:
        warnings.warn(
            (
                f"{context} received {negative_count} negative observations. "
                "UniBM is designed for positive-support data; downstream positive-only "
                "steps may exclude or invalidate those values."
            ),
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def warn_on_nonpositive_values(
    vec: np.ndarray | list[float],
    *,
    context: str,
    noun: str = "values",
    stacklevel: int = 2,
) -> None:
    """Warn when non-positive finite values are excluded by a positive-only step."""
    arr = as_1d_float_array(vec)
    nonpositive_count = int(np.sum(np.isfinite(arr) & (arr <= 0)))
    if nonpositive_count:
        warnings.warn(
            (
                f"{context} excluded {nonpositive_count} non-positive {noun}. "
                "This step requires strictly positive inputs."
            ),
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def positive_finite_values(
    vec: np.ndarray | list[float],
    *,
    context: str,
    minimum_size: int = 0,
    stacklevel: int = 2,
) -> np.ndarray:
    """Return positive finite values after warning about excluded non-positive entries."""
    arr = as_1d_float_array(vec)
    warn_on_nonpositive_values(
        arr, context=context, noun="observations", stacklevel=stacklevel + 1
    )
    positive = arr[np.isfinite(arr) & (arr > 0)]
    if positive.size < minimum_size:
        raise ValueError(
            f"{context} requires at least {minimum_size} positive finite observations."
        )
    return positive


__all__ = [
    "as_1d_float_array",
    "positive_finite_values",
    "warn_on_negative_values",
    "warn_on_nonpositive_values",
]
