"""Shared constants, dataclasses, and low-level helpers for EI estimators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2

from .._validation import positive_finite_values

Z_CRIT_95 = 1.96
EI_ALPHA = 0.05
EI_CI_LEVEL = 1.0 - EI_ALPHA
EI_TINY = 1e-8


def scale_1d_pseudo_likelihood(
    loglik_func: Callable[[float], float],
    mle: float,
    hessian: float,
    empirical_variance: float,
) -> Callable[[float], float]:
    """Apply a 1D Chandler-Bate scale adjustment to a pseudo-log-likelihood."""
    if hessian >= 0:
        raise ValueError("Hessian must be strictly negative at the MLE maximum.")
    if empirical_variance <= 0:
        raise ValueError("Empirical score variance must be strictly positive.")
    scale = float(np.sqrt(-hessian / empirical_variance))

    def adjusted_loglik(theta: float) -> float:
        adjusted_theta = float(mle + scale * (theta - mle))
        return float(loglik_func(adjusted_theta))

    return adjusted_loglik


def find_1d_profile_likelihood_intervals(
    loglik_func: Callable[[float], float],
    mle: float,
    lower_bound_search: float,
    upper_bound_search: float,
    *,
    alpha: float = EI_ALPHA,
) -> tuple[float, float]:
    """Return a central profile-likelihood interval for a scalar parameter."""
    max_loglik = float(loglik_func(mle))
    threshold_value = max_loglik - 0.5 * float(chi2.ppf(1.0 - alpha, df=1))

    def root_func(theta: float) -> float:
        try:
            return float(loglik_func(theta)) - threshold_value
        except (ValueError, ZeroDivisionError, OverflowError):
            return -np.inf

    ci_lower = float(mle)
    ci_upper = float(mle)
    try:
        lower_value = root_func(lower_bound_search)
        mle_value = root_func(mle)
        if lower_value < 0 and mle_value > 0:
            ci_lower = float(brentq(root_func, lower_bound_search, mle, xtol=1e-8))
        elif lower_value >= 0:
            ci_lower = float(lower_bound_search)
    except Exception as exc:  # pragma: no cover - safety fallback
        warnings.warn(f"Failed to find lower profile root: {exc}", stacklevel=2)
        ci_lower = float(lower_bound_search)
    try:
        upper_value = root_func(upper_bound_search)
        mle_value = root_func(mle)
        if upper_value < 0 and mle_value > 0:
            ci_upper = float(brentq(root_func, mle, upper_bound_search, xtol=1e-8))
        elif upper_value >= 0:
            ci_upper = float(upper_bound_search)
    except Exception as exc:  # pragma: no cover - safety fallback
        warnings.warn(f"Failed to find upper profile root: {exc}", stacklevel=2)
        ci_upper = float(upper_bound_search)
    return ci_lower, ci_upper


@dataclass(frozen=True)
class EiStableWindow:
    """Selected stable window on an integer tuning axis."""

    lo: int
    hi: int


@dataclass(frozen=True)
class EiPathBundle:
    """Observed BM-EI path ingredients for one base-path and block scheme.

    ``theta_path`` and ``z_path`` are computed from the observed series over the
    candidate block-size grid. ``stable_window`` and ``selected_level`` record
    where that observed path is judged stable, while ``sample_statistics``
    preserves the per-block-size window statistics reused by native fixed-``b``
    estimators.
    """

    base_path: str
    sliding: bool
    block_sizes: np.ndarray
    theta_path: np.ndarray
    eir_path: np.ndarray
    z_path: np.ndarray
    sample_counts: np.ndarray
    sample_statistics: dict[int, np.ndarray]
    stable_window: EiStableWindow
    selected_level: int


@dataclass(frozen=True)
class ExtremalIndexEstimate:
    """Unified formal-EI result container.

    Most users should read ``theta_hat`` and ``confidence_interval`` first, then
    inspect ``stable_window``, ``regression``, and ``base_path`` to understand
    which formal estimator produced the headline result. ``path_level``,
    ``path_theta``, and ``path_eir`` are retained for path diagnostics and
    plotting rather than for headline reporting.
    """

    method: str
    theta_hat: float
    confidence_interval: tuple[float, float]
    standard_error: float = np.nan
    ci_method: str = "wald"
    ci_variant: str = "default"
    tuning_axis: str = "b"
    selected_level: int | None = None
    stable_window: EiStableWindow | None = None
    path_level: tuple[int, ...] = ()
    path_theta: tuple[float, ...] = ()
    path_eir: tuple[float, ...] = ()
    selected_threshold_quantile: float | None = None
    selected_threshold_value: float | None = None
    selected_run_k: int | None = None
    block_scheme: str | None = None
    base_path: str | None = None
    regression: str | None = None


@dataclass(frozen=True)
class ThresholdCandidate:
    """One threshold-side EI fit before cross-threshold selection."""

    threshold_quantile: float
    threshold_value: float
    theta_hat: float
    confidence_interval: tuple[float, float]
    standard_error: float
    ci_method: str
    ci_variant: str
    run_k: int | None = None


@dataclass(frozen=True)
class EiPreparedBundle:
    """Reusable EI preparation outputs derived from one observed series.

    The bundle stores the cleaned observed values, the candidate block-size
    grid, all BM path variants, and threshold-side exceedance candidates so the
    native BM, pooled BM, and threshold estimators can all reuse the same
    preparation step.
    """

    values: np.ndarray
    block_sizes: np.ndarray
    paths: dict[tuple[str, bool], EiPathBundle]
    threshold_candidates: dict[float, np.ndarray]


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


def _select_between_candidates(
    preferred: ThresholdCandidate,
    alternative: ThresholdCandidate,
) -> ThresholdCandidate:
    """Prefer the first candidate when the intervals overlap, else the second."""
    if not np.isfinite(preferred.theta_hat):
        return alternative
    if not np.isfinite(alternative.theta_hat):
        return preferred
    if _intervals_overlap(preferred.confidence_interval, alternative.confidence_interval):
        return preferred
    return alternative
