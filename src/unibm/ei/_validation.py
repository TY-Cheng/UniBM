"""Private validation helpers specific to formal EI workflows."""

from __future__ import annotations

import numpy as np

from .._validation import as_1d_float_array


def _validate_threshold_quantiles(
    threshold_quantiles: tuple[float, ...] | list[float] | np.ndarray,
) -> tuple[float, ...]:
    """Return a non-empty, strictly increasing quantile grid."""
    try:
        raw = np.asarray(threshold_quantiles)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "threshold_quantiles must be a one-dimensional numeric sequence."
        ) from exc
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("threshold_quantiles must be a non-empty one-dimensional sequence.")
    if np.iscomplexobj(raw) or raw.dtype.kind == "b":
        raise ValueError("threshold_quantiles must contain finite values in (0, 1).")
    try:
        quantiles = raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold_quantiles must contain finite values in (0, 1).") from exc
    if not np.all(np.isfinite(quantiles)) or np.any((quantiles <= 0.0) | (quantiles >= 1.0)):
        raise ValueError("threshold_quantiles must contain finite values in (0, 1).")
    if np.any(np.diff(quantiles) <= 0.0):
        raise ValueError("threshold_quantiles must be strictly increasing with no duplicates.")
    return tuple(float(quantile) for quantile in quantiles)


def _validate_ei_series(
    vec: np.ndarray | list[float],
    *,
    allow_zeros: bool,
) -> np.ndarray:
    """Validate an EI series without changing its temporal positions."""
    values = as_1d_float_array(vec)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "extremal-index benchmark requires every observation to be finite; "
            "handle missing observations before choosing the EI clock."
        )
    if allow_zeros:
        if np.any(values < 0):
            raise ValueError("extremal-index benchmark requires non-negative observations.")
    elif np.any(values <= 0):
        raise ValueError(
            "extremal-index benchmark requires strictly positive observations unless "
            "allow_zeros=True."
        )
    if values.size < 32:
        raise ValueError("extremal-index benchmark requires at least 32 observations.")
    return values


def _finite_positive_series(vec: np.ndarray | list[float]) -> np.ndarray:
    """Validate the strictly positive series used by the EI benchmark."""
    return _validate_ei_series(vec, allow_zeros=False)


def _finite_nonnegative_series(vec: np.ndarray | list[float]) -> np.ndarray:
    """Validate the non-negative series used by application-side EI fits."""
    return _validate_ei_series(vec, allow_zeros=True)
