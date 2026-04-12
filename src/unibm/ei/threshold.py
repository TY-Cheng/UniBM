"""Threshold-based extremal-index estimators."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ._internal import (
    EI_ALPHA,
    EI_TINY,
    Z_CRIT_95,
    _central_wald_interval,
    _select_between_candidates,
    find_1d_profile_likelihood_intervals,
)
from .models import EiPreparedBundle, ExtremalIndexEstimate, ThresholdCandidate


def _inter_exceedance_times(indices: np.ndarray) -> np.ndarray:
    """Return the raw inter-exceedance times from exceedance indices."""
    indices = np.asarray(indices, dtype=int)
    if indices.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(indices).astype(float)


def _ferro_segers_from_times(times: np.ndarray) -> tuple[float, float]:
    """Return the Ferro-Segers point estimate and asymptotic SE from inter-exceedance times."""
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t) & (t > 0)]
    if t.size < 2:
        raise ValueError("Ferro-Segers requires at least two inter-exceedance times.")
    if np.max(t) <= 2.0:
        obs = np.column_stack([t, t**2])
        moments = obs.mean(axis=0)
        a, b = moments
        theta_hat = float(np.clip(2.0 * a * a / max(b, EI_TINY), EI_TINY, 1.0))
        gradient = np.asarray([4.0 * a / max(b, EI_TINY), -2.0 * a * a / max(b**2, EI_TINY)])
    else:
        x = t - 1.0
        y = (t - 1.0) * (t - 2.0)
        obs = np.column_stack([x, y])
        moments = obs.mean(axis=0)
        a, b = moments
        theta_hat = float(np.clip(2.0 * a * a / max(b, EI_TINY), EI_TINY, 1.0))
        gradient = np.asarray([4.0 * a / max(b, EI_TINY), -2.0 * a * a / max(b**2, EI_TINY)])
    cov_means = np.atleast_2d(np.cov(obs, rowvar=False, ddof=1)) / obs.shape[0]
    standard_error = float(np.sqrt(max(gradient @ cov_means @ gradient, 0.0)))
    return theta_hat, standard_error


def estimate_ferro_segers(
    bundle: EiPreparedBundle,
    *,
    threshold_quantiles: tuple[float, float] = (0.90, 0.95),
) -> ExtremalIndexEstimate:
    """Estimate ``theta`` with the Ferro-Segers intervals estimator."""
    candidates: list[ThresholdCandidate] = []
    for quantile in threshold_quantiles:
        indices = bundle.threshold_candidates[float(quantile)]
        if indices.size < 3:
            continue
        threshold_value = float(np.quantile(bundle.values, quantile))
        theta_hat, standard_error = _ferro_segers_from_times(_inter_exceedance_times(indices))
        candidates.append(
            ThresholdCandidate(
                threshold_quantile=float(quantile),
                threshold_value=threshold_value,
                theta_hat=theta_hat,
                confidence_interval=_central_wald_interval(
                    theta_hat,
                    standard_error,
                    bounded_unit_interval=True,
                ),
                standard_error=standard_error,
                ci_method="wald",
                ci_variant="default",
            )
        )
    if not candidates:
        raise ValueError("Ferro-Segers could not find a threshold with enough exceedances.")
    chosen = candidates[0]
    for candidate in candidates[1:]:
        chosen = _select_between_candidates(chosen, candidate)
    return ExtremalIndexEstimate(
        method="ferro_segers",
        theta_hat=chosen.theta_hat,
        confidence_interval=chosen.confidence_interval,
        standard_error=chosen.standard_error,
        ci_method=chosen.ci_method,
        ci_variant=chosen.ci_variant,
        tuning_axis="u",
        selected_threshold_quantile=chosen.threshold_quantile,
        selected_threshold_value=chosen.threshold_value,
    )


def _kgaps_profile_fit(
    times: np.ndarray, *, run_k: int, exceedance_rate: float
) -> ThresholdCandidate:
    """Fit the K-gaps model for one `(u, K)` combination."""
    raw_gaps = np.maximum(np.asarray(times, dtype=float) - float(run_k), 0.0)
    scaled_gaps = exceedance_rate * raw_gaps
    scaled_gaps = scaled_gaps[np.isfinite(scaled_gaps)]
    if scaled_gaps.size < 2:
        raise ValueError("K-gaps requires at least two finite gap observations.")
    zero_mask = scaled_gaps <= 0
    positive = scaled_gaps[~zero_mask]
    n_zero = int(np.sum(zero_mask))
    n_pos = int(positive.size)

    def loglik(theta: float) -> float:
        theta = float(theta)
        if not (EI_TINY <= theta <= 1.0 - EI_TINY):
            return -np.inf
        value = n_zero * np.log(max(1.0 - theta, EI_TINY))
        if n_pos:
            value += n_pos * (2.0 * np.log(theta)) - theta * float(np.sum(positive))
        return float(value)

    def objective(theta: float) -> float:
        return -loglik(theta)

    optimum = minimize_scalar(objective, bounds=(EI_TINY, 1.0 - EI_TINY), method="bounded")
    theta_hat = float(np.clip(optimum.x, EI_TINY, 1.0 - EI_TINY))
    interval = find_1d_profile_likelihood_intervals(
        loglik,
        theta_hat,
        EI_TINY,
        1.0 - EI_TINY,
        alpha=EI_ALPHA,
    )
    standard_error = (
        float((interval[1] - interval[0]) / (2.0 * Z_CRIT_95))
        if np.all(np.isfinite(interval))
        else float("nan")
    )
    return ThresholdCandidate(
        threshold_quantile=float("nan"),
        threshold_value=float("nan"),
        theta_hat=theta_hat,
        confidence_interval=interval,
        standard_error=standard_error,
        ci_method="profile",
        ci_variant="default",
        run_k=int(run_k),
    )


def estimate_k_gaps(
    bundle: EiPreparedBundle,
    *,
    threshold_quantiles: tuple[float, float] = (0.90, 0.95),
    k_grid: tuple[int, int] = (1, 2),
) -> ExtremalIndexEstimate:
    """Estimate ``theta`` with the K-gaps likelihood."""
    threshold_winners: list[ThresholdCandidate] = []
    for quantile in threshold_quantiles:
        indices = bundle.threshold_candidates[float(quantile)]
        if indices.size < 3:
            continue
        times = _inter_exceedance_times(indices)
        exceedance_rate = float(indices.size / bundle.values.size)
        threshold_value = float(np.quantile(bundle.values, quantile))
        k_candidates: list[ThresholdCandidate] = []
        for run_k in k_grid:
            candidate = _kgaps_profile_fit(
                times, run_k=int(run_k), exceedance_rate=exceedance_rate
            )
            k_candidates.append(
                ThresholdCandidate(
                    threshold_quantile=float(quantile),
                    threshold_value=threshold_value,
                    theta_hat=candidate.theta_hat,
                    confidence_interval=candidate.confidence_interval,
                    standard_error=candidate.standard_error,
                    ci_method=candidate.ci_method,
                    ci_variant=candidate.ci_variant,
                    run_k=int(run_k),
                )
            )
        winner = k_candidates[0]
        for candidate in k_candidates[1:]:
            winner = _select_between_candidates(winner, candidate)
        threshold_winners.append(winner)
    if not threshold_winners:
        raise ValueError("K-gaps could not find a threshold with enough exceedances.")
    chosen = threshold_winners[0]
    for candidate in threshold_winners[1:]:
        chosen = _select_between_candidates(chosen, candidate)
    return ExtremalIndexEstimate(
        method="k_gaps",
        theta_hat=chosen.theta_hat,
        confidence_interval=chosen.confidence_interval,
        standard_error=chosen.standard_error,
        ci_method=chosen.ci_method,
        ci_variant=chosen.ci_variant,
        tuning_axis="u",
        selected_threshold_quantile=chosen.threshold_quantile,
        selected_threshold_value=chosen.threshold_value,
        selected_run_k=chosen.run_k,
    )
