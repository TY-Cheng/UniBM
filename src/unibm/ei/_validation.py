"""Private validation helpers specific to formal EI workflows."""

from __future__ import annotations

import numpy as np

from .._validation import as_1d_float_array


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
