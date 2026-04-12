"""Private validation helpers specific to formal EI workflows."""

from __future__ import annotations

import numpy as np

from .._validation import positive_finite_values


def _finite_positive_series(vec: np.ndarray | list[float]) -> np.ndarray:
    """Return the positive finite series used by the EI benchmark."""
    return positive_finite_values(
        vec,
        context="extremal-index benchmark",
        minimum_size=32,
        stacklevel=3,
    )


def _finite_nonnegative_series(vec: np.ndarray | list[float]) -> np.ndarray:
    """Return the non-negative finite series used by application-side EI fits."""
    values = np.asarray(vec, dtype=float).reshape(-1)
    finite = values[np.isfinite(values) & (values >= 0)]
    if finite.size < 32:
        raise ValueError(
            "extremal-index benchmark requires at least 32 finite non-negative observations."
        )
    return finite
