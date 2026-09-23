"""Threshold-based extremal-index estimators."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ._likelihood import find_1d_profile_likelihood_intervals
from ._stats import (
    EI_ALPHA,
    EI_TINY,
    _central_wald_interval,
    _intervals_overlap,
)
from ._validation import _validate_threshold_quantiles
from .models import EiPreparedBundle, ExtremalIndexEstimate, ThresholdCandidate


def _validate_k_grid(k_grid: tuple[int, ...]) -> tuple[int, ...]:
    """Return an increasing tuple of non-negative integer run lengths.

    Reject empty/non-1D grids, duplicates, non-finite or fractional values,
    complex inputs and boolean arrays; preserve the supplied selection order.
    """
    try:
        raw = np.asarray(k_grid)
    except (TypeError, ValueError) as exc:
        raise ValueError("k_grid must contain finite non-negative integers.") from exc
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind == "b" or np.iscomplexobj(raw):
        raise ValueError("k_grid must be a non-empty sequence of finite non-negative integers.")
    try:
        numeric = raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("k_grid must contain finite non-negative integers.") from exc
    if (
        not np.all(np.isfinite(numeric))
        or not np.all(numeric == np.floor(numeric))
        or np.any(numeric < 0)
    ):
        raise ValueError("k_grid must contain finite non-negative integers.")
    if np.any(np.diff(numeric) <= 0):
        raise ValueError("k_grid must be strictly increasing with no duplicates.")
    return tuple(int(value) for value in numeric)


def _resolve_threshold_quantiles(
    bundle: EiPreparedBundle,
    threshold_quantiles: tuple[float, ...] | None,
) -> tuple[float, ...]:
    """Return all prepared threshold quantiles or a validated increasing subset.

    Explicit quantiles must already exist as bundle keys. This does not compute
    new exceedance indices or silently reorder thresholds.
    """
    available = _validate_threshold_quantiles(tuple(bundle.threshold_candidates))
    if threshold_quantiles is None:
        return available
    requested = _validate_threshold_quantiles(threshold_quantiles)
    missing = tuple(
        quantile for quantile in requested if quantile not in bundle.threshold_candidates
    )
    if missing:
        raise ValueError(
            "threshold_quantiles must be present in the prepared bundle; "
            f"missing={missing}, available={available}."
        )
    return requested


def _select_between_candidates(
    preferred: ThresholdCandidate,
    alternative: ThresholdCandidate,
) -> ThresholdCandidate:
    """Keep the preferred candidate when finite estimates have overlapping CIs.

    Otherwise choose the alternative; a non-finite point estimate loses to a
    finite one. If both estimates are non-finite, return the alternative. This
    is a sequential tuning rule, not a calibrated multiple-comparison test.
    """
    if not np.isfinite(preferred.theta_hat):
        return alternative
    if not np.isfinite(alternative.theta_hat):
        return preferred
    if _intervals_overlap(preferred.confidence_interval, alternative.confidence_interval):
        return preferred
    return alternative


def _inter_exceedance_times(indices: np.ndarray) -> np.ndarray:
    """Return successive differences of sorted exceedance indices in observation steps.

    The caller supplies increasing indices on the intended EI clock. Fewer than
    two indices yield an empty float array; no sorting or gap correction occurs.
    """
    indices = np.asarray(indices, dtype=int)
    if indices.size < 2:
        return np.asarray([], dtype=float)
    return np.diff(indices).astype(float)


def _ferro_segers_from_times(times: np.ndarray) -> tuple[float, float]:
    """Return the Ferro--Segers moment estimate and a delta-method theta-scale SE.

    Require two finite positive times after filtering. Use moments of ``T``
    and ``T**2`` when every gap is at most two, otherwise moments of ``T - 1``
    and ``(T - 1) * (T - 2)``. Clip theta to [EI_TINY, 1], but compute the SE
    from the untruncated moment formula. The covariance-of-means approximation
    divides the empirical moment covariance by the number of gaps; it does not
    estimate serial covariances between gaps.
    """
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
    threshold_quantiles: tuple[float, ...] | None = None,
) -> ExtremalIndexEstimate:
    """Estimate theta from Ferro--Segers inter-exceedance-time moments.

    Use ``prepare_ei_bundle`` output and either all its threshold quantiles or
    an increasing subset of existing quantiles. Skip thresholds with fewer than
    three strict exceedances; raise ``ValueError`` if none are usable.

    Traverse thresholds in order, retaining the current candidate when its
    nominal 95% Wald interval overlaps the next candidate's interval, otherwise
    replacing it. Return the selected theta, bounded interval, delta-method SE,
    and threshold metadata in an ``ExtremalIndexEstimate``. The interval
    conditions on the selected threshold and does not include tuning uncertainty
    or serial covariance between successive gaps.
    """
    candidates: list[ThresholdCandidate] = []
    for quantile in _resolve_threshold_quantiles(bundle, threshold_quantiles):
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
    """Fit the zero/exponential K-gap likelihood at a fixed threshold and run length.

    Scale ``max(times - run_k, 0)`` by the exceedance rate. Zero gaps contribute
    ``log(1 - theta)`` and positive gaps contribute ``2 * log(theta) - theta * gap``.
    At least two finite scaled gaps must remain. Return a candidate with nominal
    95% profile endpoints and observed-information SE; threshold fields remain
    NaN for the caller to fill. No serial score covariance is estimated here.
    """
    raw_gaps = np.maximum(np.asarray(times, dtype=float) - float(run_k), 0.0)
    scaled_gaps = exceedance_rate * raw_gaps
    scaled_gaps = scaled_gaps[np.isfinite(scaled_gaps)]
    if scaled_gaps.size < 2:
        raise ValueError("K-gaps requires at least two finite gap observations.")
    zero_mask = scaled_gaps <= 0
    positive = scaled_gaps[~zero_mask]
    n_zero = int(np.sum(zero_mask))
    n_pos = int(positive.size)
    sum_positive = float(np.sum(positive))

    def loglik(theta: float) -> float:
        """Evaluate the K-gap log-likelihood inside the numerically bounded theta domain."""
        theta = float(theta)
        if not (EI_TINY <= theta <= 1.0 - EI_TINY):
            return -np.inf
        value = n_zero * np.log(max(1.0 - theta, EI_TINY))
        if n_pos:
            value += n_pos * (2.0 * np.log(theta)) - theta * sum_positive
        return float(value)

    def objective(theta: float) -> float:
        """Negate the K-gap log-likelihood for bounded scalar minimization."""
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
    observed_information = n_zero / (1.0 - theta_hat) ** 2 + 2.0 * n_pos / theta_hat**2
    standard_error = float(1.0 / np.sqrt(observed_information))
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
    threshold_quantiles: tuple[float, ...] | None = None,
    k_grid: tuple[int, ...] = (1, 2),
) -> ExtremalIndexEstimate:
    """Estimate theta from K-gap likelihoods and select a threshold/run-length pair.

    ``bundle`` supplies strict-exceedance indices on the caller's observation
    clock. ``threshold_quantiles`` selects an increasing subset of prepared
    quantiles or all of them when omitted. ``k_grid`` is an increasing sequence
    of non-negative integer run lengths in observation steps (default 1 and 2).

    At each threshold with at least three exceedances, compare candidates in
    increasing K order: retain the current fit if its nominal 95% profile CI
    overlaps the next, otherwise replace it. Apply the same rule across threshold
    winners. Raise ``ValueError`` if no threshold has enough exceedances.

    Return theta, its profile interval and observed-information SE, and selected
    threshold/K metadata. Inference conditions on this selected pair; it does not
    include tuning uncertainty or a serial-dependence score adjustment.
    """
    k_grid = _validate_k_grid(k_grid)
    threshold_winners: list[ThresholdCandidate] = []
    for quantile in _resolve_threshold_quantiles(bundle, threshold_quantiles):
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
