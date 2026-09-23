"""Private pseudo-likelihood helpers for EI inference."""

from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2

from ._stats import EI_ALPHA


def scale_1d_pseudo_likelihood(
    loglik_func: Callable[[float], float],
    mle: float,
    hessian: float,
    empirical_variance: float,
) -> Callable[[float], float]:
    """Return a Chandler--Bate adjusted scalar pseudo-log-likelihood.

    Rescale distance from ``mle`` by ``sqrt(-hessian / empirical_variance)``.
    ``hessian`` is the second derivative of the total log-likelihood at its
    maximum; ``empirical_variance`` is the caller's positive score-variance
    estimate. This helper does not estimate dependence between scores.
    """
    if hessian >= 0:
        raise ValueError("Hessian must be strictly negative at the MLE maximum.")
    if empirical_variance <= 0:
        raise ValueError("Empirical score variance must be strictly positive.")
    scale = float(np.sqrt(-hessian / empirical_variance))

    def adjusted_loglik(theta: float) -> float:
        """Evaluate the original likelihood at the rescaled displacement from the MLE."""
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
    """Find scalar likelihood-ratio endpoints using a chi-square(1) cutoff.

    Search each side of ``mle`` within the supplied bounds for a log-likelihood
    drop of ``chi2.ppf(1 - alpha, 1) / 2``. A bound is returned if it remains
    inside the likelihood-ratio region; root-finding exceptions also return
    that side's bound with a warning. Calibration is asymptotic and does not
    provide an exact boundary correction or an equal-tailed interval.
    """
    max_loglik = float(loglik_func(mle))
    threshold_value = max_loglik - 0.5 * float(chi2.ppf(1.0 - alpha, df=1))

    def root_func(theta: float) -> float:
        """Measure log-likelihood above the cutoff; treat evaluation errors as outside."""
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
