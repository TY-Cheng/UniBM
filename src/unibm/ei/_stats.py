"""Private statistical constants and interval transforms for EI workflows."""

from __future__ import annotations

import numpy as np


Z_CRIT_95 = 1.96
EI_ALPHA = 0.05
EI_CI_LEVEL = 1.0 - EI_ALPHA
EI_TINY = 1e-8


def _central_wald_interval(
    center: float,
    standard_error: float,
    *,
    bounded_unit_interval: bool = False,
) -> tuple[float, float]:
    """Return a central 95% Wald interval."""
    if not np.isfinite(center) or not np.isfinite(standard_error) or standard_error < 0:
        return (float("nan"), float("nan"))
    lo = float(center - Z_CRIT_95 * standard_error)
    hi = float(center + Z_CRIT_95 * standard_error)
    if bounded_unit_interval:
        return (max(0.0, lo), min(1.0, hi))
    return (lo, hi)


def _log_scale_theta_interval(
    z_hat: float,
    standard_error: float,
) -> tuple[float, float]:
    """Back-transform a central 95% Wald interval from `z = log(1/theta)`."""
    if not np.isfinite(z_hat) or not np.isfinite(standard_error) or standard_error < 0:
        return (float("nan"), float("nan"))
    z_lo = float(z_hat - Z_CRIT_95 * standard_error)
    z_hi = float(z_hat + Z_CRIT_95 * standard_error)
    return (float(np.exp(-z_hi)), float(np.exp(-z_lo)))


def _intervals_overlap(
    left: tuple[float, float],
    right: tuple[float, float],
) -> bool:
    """Return whether two finite intervals overlap."""
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return False
    return bool(max(left[0], right[0]) <= min(left[1], right[1]))
