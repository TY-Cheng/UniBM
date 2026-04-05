"""Formal extremal-index benchmark estimators.

This module is separate from ``diagnostics.py`` because it implements the
benchmark-facing EI estimators rather than exploratory diagnostics. The main
entrypoints here cover:

- threshold estimators: Ferro-Segers and K-gaps;
- native BM estimators: Northrop and BB on one selected block size;
- pooled UniBM BM estimators: OLS/FGLS pooling over a stable block-size window.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2

from .bootstrap import draw_circular_block_bootstrap_samples
from .core import generate_block_sizes
from .diagnostics import empirical_cdf
from ._validation import positive_finite_values

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
    """Apply a 1D Chandler-Bate scale adjustment to a pseudo-log-likelihood.

    This helper is local to the EI module because the current codebase only
    uses the chandwich adjustment for the native Northrop fixed-`b` EI fit.
    """
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
    """Shared block-size path ingredients for one BM-EI base estimator."""

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
    """Unified extremal-index benchmark result container."""

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
    """All per-replicate ingredients reused across EI methods."""

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


def _rolling_window_minima(
    scores: np.ndarray,
    block_size: int,
    *,
    sliding: bool,
) -> np.ndarray:
    """Return sliding or disjoint window minima for one score series."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size < block_size or block_size < 2:
        return np.asarray([], dtype=float)
    if sliding:
        windows = np.lib.stride_tricks.sliding_window_view(scores, block_size)
        valid = np.all(np.isfinite(windows), axis=1)
        return windows.min(axis=1)[valid]
    n_block = scores.size // block_size
    if n_block < 1:
        return np.asarray([], dtype=float)
    windows = scores[: n_block * block_size].reshape(n_block, block_size)
    valid = np.all(np.isfinite(windows), axis=1)
    return windows.min(axis=1)[valid]


def _select_stable_ei_window(
    block_sizes: np.ndarray,
    z_path: np.ndarray,
    *,
    min_points: int = 4,
    trim_fraction: float = 0.15,
    roughness_penalty: float = 0.75,
    curvature_penalty: float = 0.5,
) -> tuple[EiStableWindow, np.ndarray]:
    """Choose the most stable contiguous block-size window on the transformed EI path."""
    levels = np.asarray(block_sizes, dtype=int)
    z = np.asarray(z_path, dtype=float)
    mask = np.isfinite(z)
    levels = levels[mask]
    z = z[mask]
    if levels.size < min_points:
        raise ValueError("Not enough finite EI path values to select a stable window.")
    lo = int(np.floor(levels.size * trim_fraction))
    hi = levels.size - lo
    if hi - lo < min_points:
        lo = 0
        hi = levels.size
    best: tuple[float, int, int] | None = None
    for start in range(lo, hi - min_points + 1):
        for stop in range(start + min_points, hi + 1):
            window = z[start:stop]
            variance = float(np.mean((window - window.mean()) ** 2))
            first_diff = np.diff(window)
            roughness = float(np.mean(np.abs(first_diff))) if first_diff.size else 0.0
            curvature = float(np.mean(np.abs(np.diff(first_diff)))) if first_diff.size > 1 else 0.0
            score = (
                variance
                + float(roughness_penalty) * roughness
                + float(curvature_penalty) * curvature
            ) / np.sqrt(stop - start)
            if best is None or score < best[0]:
                best = (score, start, stop)
    assert best is not None
    _, start, stop = best
    selected_mask = np.zeros(mask.sum(), dtype=bool)
    selected_mask[start:stop] = True
    window = EiStableWindow(int(levels[start]), int(levels[stop - 1]))
    return window, selected_mask


def _build_path_from_scores(
    base_path: str,
    scores: np.ndarray,
    block_sizes: np.ndarray,
    *,
    sliding: bool,
) -> EiPathBundle:
    """Construct the full EI path for one BM base estimator."""
    theta_path = np.full(block_sizes.size, np.nan, dtype=float)
    eir_path = np.full(block_sizes.size, np.nan, dtype=float)
    z_path = np.full(block_sizes.size, np.nan, dtype=float)
    sample_counts = np.zeros(block_sizes.size, dtype=int)
    sample_statistics: dict[int, np.ndarray] = {}
    for idx, block_size in enumerate(np.asarray(block_sizes, dtype=int)):
        minima = _rolling_window_minima(scores, int(block_size), sliding=sliding)
        if minima.size == 0:
            continue
        statistics = float(block_size) * minima
        sample_statistics[int(block_size)] = statistics
        sample_counts[idx] = statistics.size
        mean_stat = float(np.mean(statistics))
        if base_path == "northrop":
            eir = max(mean_stat, 1.0)
            theta = float(1.0 / eir)
        elif base_path == "bb":
            theta = float(max((1.0 / max(mean_stat, EI_TINY)) - 1.0 / float(block_size), EI_TINY))
            theta = min(theta, 1.0)
            eir = float(1.0 / theta)
        else:
            raise ValueError(f"Unknown BM EI base path: {base_path}")
        theta_path[idx] = theta
        eir_path[idx] = eir
        z_path[idx] = float(np.log(eir))
    stable_window, stable_mask = _select_stable_ei_window(block_sizes, z_path)
    selected_level = int(block_sizes[np.isfinite(z_path)][stable_mask][0])
    return EiPathBundle(
        base_path=base_path,
        sliding=bool(sliding),
        block_sizes=np.asarray(block_sizes, dtype=int),
        theta_path=theta_path,
        eir_path=eir_path,
        z_path=z_path,
        sample_counts=sample_counts,
        sample_statistics=sample_statistics,
        stable_window=stable_window,
        selected_level=selected_level,
    )


def _build_bm_paths_from_values(
    values: np.ndarray,
    block_sizes: np.ndarray,
) -> dict[tuple[str, bool], EiPathBundle]:
    """Build the four BM EI paths reused across benchmark methods."""
    cdf_values = np.asarray(empirical_cdf(values)(values), dtype=float)
    cdf_values = np.clip(cdf_values, EI_TINY, 1.0 - EI_TINY)
    northrop_scores = -np.log(cdf_values)
    bb_scores = 1.0 - cdf_values
    return {
        ("northrop", True): _build_path_from_scores(
            "northrop", northrop_scores, block_sizes, sliding=True
        ),
        ("northrop", False): _build_path_from_scores(
            "northrop", northrop_scores, block_sizes, sliding=False
        ),
        ("bb", True): _build_path_from_scores("bb", bb_scores, block_sizes, sliding=True),
        ("bb", False): _build_path_from_scores("bb", bb_scores, block_sizes, sliding=False),
    }


def extract_stable_path_window(path: EiPathBundle) -> tuple[np.ndarray, np.ndarray]:
    """Return the selected stable block levels and transformed values for one path."""
    finite_mask = np.isfinite(path.z_path)
    finite_levels = path.block_sizes[finite_mask]
    finite_z = path.z_path[finite_mask]
    window_mask = (finite_levels >= path.stable_window.lo) & (
        finite_levels <= path.stable_window.hi
    )
    selected_levels = finite_levels[window_mask]
    selected_z = finite_z[window_mask]
    if selected_levels.size == 0:
        raise ValueError("Stable EI window did not retain any finite transformed path values.")
    return selected_levels, selected_z


def prepare_ei_bundle(
    vec: np.ndarray | list[float],
    *,
    block_sizes: np.ndarray | None = None,
    threshold_quantiles: tuple[float, ...] = (0.90, 0.95),
) -> EiPreparedBundle:
    """Build all reusable EI benchmark ingredients for one replicate."""
    values = _finite_positive_series(vec)
    if block_sizes is None:
        block_sizes = generate_block_sizes(values.size)
    block_sizes = np.asarray(block_sizes, dtype=int)
    paths = _build_bm_paths_from_values(values, block_sizes)
    threshold_candidates = {
        float(q): np.flatnonzero(values > np.quantile(values, float(q)))
        for q in threshold_quantiles
    }
    return EiPreparedBundle(
        values=values,
        block_sizes=block_sizes,
        paths=paths,
        threshold_candidates=threshold_candidates,
    )


def bootstrap_bm_ei_path_draws(
    bootstrap_samples: np.ndarray,
    *,
    block_sizes: np.ndarray,
) -> dict[tuple[str, bool], np.ndarray]:
    """Transform one raw bootstrap bank into all four BM-EI z-path draw matrices.

    This keeps the expensive circular bootstrap draw step shared across the
    Northrop/BB and sliding/disjoint variants. Each raw bootstrap sample is
    converted into the four transformed `z = log(1/theta)` paths in one pass.
    """
    samples = np.asarray(bootstrap_samples, dtype=float)
    block_sizes = np.asarray(block_sizes, dtype=int)
    draws = {
        ("northrop", True): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
        ("northrop", False): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
        ("bb", True): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
        ("bb", False): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
    }
    for rep, sample in enumerate(samples):
        try:
            sample_values = _finite_positive_series(sample)
            sample_paths = _build_bm_paths_from_values(sample_values, block_sizes)
        except ValueError:
            continue
        for key, path in sample_paths.items():
            draws[key][rep] = path.z_path
    return draws


def bootstrap_bm_ei_path(
    vec: np.ndarray | list[float],
    *,
    base_path: str,
    sliding: bool,
    block_sizes: np.ndarray,
    reps: int,
    random_state: int,
) -> dict[str, np.ndarray | None]:
    """Bootstrap the transformed BM-EI path on the full block-size grid.

    The workflow layer later subsets these draws to the original replicate's
    selected stable window before estimating a covariance matrix. That keeps
    FGLS from discarding a bootstrap draw just because some unused large block
    size became non-finite.
    """
    values = _finite_positive_series(vec)
    samples = draw_circular_block_bootstrap_samples(
        values,
        reps=reps,
        random_state=random_state,
    ).samples
    z_draws = bootstrap_bm_ei_path_draws(
        samples,
        block_sizes=np.asarray(block_sizes, dtype=int),
    )[(base_path, sliding)]
    return {
        "block_sizes": np.asarray(block_sizes, dtype=int),
        "samples": z_draws,
        "covariance": None,
    }


def _pooled_z_fit(
    z_values: np.ndarray,
    *,
    covariance: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return the intercept-only pooled estimate and its SE on the z-scale."""
    z = np.asarray(z_values, dtype=float)
    if covariance is not None and covariance.shape == (z.size, z.size):
        inv_cov = np.linalg.pinv(covariance)
        ones = np.ones(z.size, dtype=float)
        denom = float(ones @ inv_cov @ ones)
        if denom > 0 and np.isfinite(denom):
            z_hat = float((ones @ inv_cov @ z) / denom)
            se = float(np.sqrt(1.0 / denom))
            return z_hat, se
    z_hat = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(max(z.size, 1))) if z.size >= 2 else 0.0
    return z_hat, se


def _build_bm_estimate(
    method: str,
    path: EiPathBundle,
    *,
    regression: str,
    bootstrap_result: dict[str, np.ndarray | None] | None = None,
) -> ExtremalIndexEstimate:
    """Pool one BM path either by OLS or by FGLS on the transformed scale."""
    selected_levels, selected_z = extract_stable_path_window(path)
    covariance = None
    used_gls = False
    if bootstrap_result is not None:
        raw_cov = bootstrap_result.get("covariance")
        boot_levels = np.asarray(bootstrap_result.get("block_sizes", []), dtype=int)
        if (
            raw_cov is not None
            and raw_cov.shape == (selected_levels.size, selected_levels.size)
            and np.array_equal(boot_levels, selected_levels)
        ):
            covariance = raw_cov
            used_gls = True
    z_hat, se = _pooled_z_fit(
        selected_z,
        covariance=covariance if regression == "FGLS" else None,
    )
    # We pool on z = log(1/theta) rather than directly on theta so the Wald
    # approximation respects the positive reciprocal-EI geometry and the final
    # interval can be mapped back monotonically to (0, 1].
    theta_hat = float(np.exp(-z_hat))
    if regression == "FGLS":
        ci_variant = "bootstrap_cov" if used_gls else "ols_fallback_no_cov"
    else:
        ci_variant = "ols"
    return ExtremalIndexEstimate(
        method=method,
        theta_hat=theta_hat,
        confidence_interval=_log_scale_theta_interval(z_hat, se),
        standard_error=se,
        ci_method="log_wald",
        ci_variant=ci_variant,
        tuning_axis="b",
        selected_level=None,
        stable_window=path.stable_window,
        path_level=tuple(int(level) for level in path.block_sizes[np.isfinite(path.z_path)]),
        path_theta=tuple(float(x) for x in path.theta_path[np.isfinite(path.z_path)]),
        path_eir=tuple(float(x) for x in path.eir_path[np.isfinite(path.z_path)]),
        block_scheme="sliding" if path.sliding else "disjoint",
        base_path=path.base_path,
        regression=regression,
    )


def _northrop_profile_fit(
    statistics: np.ndarray,
    *,
    adjusted: bool,
) -> tuple[float, tuple[float, float], float, str]:
    """Fit the Northrop pseudo-likelihood on one fixed block size."""
    stats = np.asarray(statistics, dtype=float)
    stats = stats[np.isfinite(stats) & (stats > 0)]
    if stats.size < 2:
        raise ValueError("Northrop likelihood requires at least two finite positive statistics.")
    theta_hat = float(np.clip(1.0 / np.mean(stats), EI_TINY, 1.0))

    def loglik(theta: float) -> float:
        theta = float(theta)
        if not (EI_TINY <= theta <= 1.0):
            return -np.inf
        return float(stats.size * np.log(theta) - theta * np.sum(stats))

    interval = (float("nan"), float("nan"))
    ci_variant = "profile"
    try:
        if adjusted:
            hessian = float(-stats.size / (theta_hat**2))
            scores = (1.0 / theta_hat) - stats
            # Chandler-Bate scaling uses the variability of the total score.
            # At the MLE the summed score is zero here, so the OPG estimate is
            # naturally the sum of squared individual score contributions.
            empirical_variance = float(np.sum(scores**2))
            adjusted_loglik = scale_1d_pseudo_likelihood(
                loglik,
                theta_hat,
                hessian=hessian,
                empirical_variance=empirical_variance,
            )
            interval = find_1d_profile_likelihood_intervals(
                adjusted_loglik,
                theta_hat,
                EI_TINY,
                1.0 - EI_TINY,
                alpha=EI_ALPHA,
            )
            ci_variant = "chandwich_adjusted"
        else:
            interval = find_1d_profile_likelihood_intervals(
                loglik,
                theta_hat,
                EI_TINY,
                1.0 - EI_TINY,
                alpha=EI_ALPHA,
            )
    except Exception as exc:  # pragma: no cover - fallback safety
        warnings.warn(
            f"Falling back to unadjusted Northrop profile likelihood: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )
        # Chandwich is the preferred native-Northrop CI variant, but a failed
        # calibration should degrade to the ordinary profile interval rather
        # than dropping the replicate from the benchmark entirely.
        interval = find_1d_profile_likelihood_intervals(
            loglik,
            theta_hat,
            EI_TINY,
            1.0 - EI_TINY,
            alpha=EI_ALPHA,
        )
        ci_variant = "profile_fallback"
    standard_error = (
        float((interval[1] - interval[0]) / (2.0 * Z_CRIT_95))
        if np.all(np.isfinite(interval))
        else float("nan")
    )
    return theta_hat, interval, standard_error, ci_variant


def _bb_wald_fit(
    statistics: np.ndarray, block_size: int
) -> tuple[float, tuple[float, float], float]:
    """Fit the BB fixed-b Wald approximation from the rolling-min statistic sample."""
    stats = np.asarray(statistics, dtype=float)
    stats = stats[np.isfinite(stats) & (stats > 0)]
    if stats.size < 2:
        raise ValueError("BB Wald fit requires at least two finite positive statistics.")
    mean_stat = float(np.mean(stats))
    theta_hat = float(np.clip((1.0 / mean_stat) - (1.0 / float(block_size)), EI_TINY, 1.0))
    standard_error = float(np.std(stats, ddof=1) / np.sqrt(stats.size) / (mean_stat**2))
    return (
        theta_hat,
        _central_wald_interval(theta_hat, standard_error, bounded_unit_interval=True),
        standard_error,
    )


def estimate_native_bm_ei(
    bundle: EiPreparedBundle,
    *,
    base_path: str,
    sliding: bool,
    use_adjusted_chandwich: bool = False,
) -> ExtremalIndexEstimate:
    """Estimate theta with a native single-b BM estimator."""
    path = bundle.paths[(base_path, sliding)]
    selected_level = path.selected_level
    statistics = path.sample_statistics[selected_level]
    if base_path == "northrop":
        theta_hat, interval, standard_error, ci_variant = _northrop_profile_fit(
            statistics,
            adjusted=use_adjusted_chandwich,
        )
        ci_method = "profile"
    else:
        theta_hat, interval, standard_error = _bb_wald_fit(statistics, selected_level)
        ci_variant = "default"
        ci_method = "wald"
    return ExtremalIndexEstimate(
        method=f"{base_path}_{'sliding' if sliding else 'disjoint'}_native",
        theta_hat=theta_hat,
        confidence_interval=interval,
        standard_error=standard_error,
        ci_method=ci_method,
        ci_variant=ci_variant,
        tuning_axis="b",
        selected_level=selected_level,
        stable_window=path.stable_window,
        path_level=tuple(int(level) for level in path.block_sizes[np.isfinite(path.z_path)]),
        path_theta=tuple(float(x) for x in path.theta_path[np.isfinite(path.z_path)]),
        path_eir=tuple(float(x) for x in path.eir_path[np.isfinite(path.z_path)]),
        block_scheme="sliding" if sliding else "disjoint",
        base_path=base_path,
    )


def estimate_pooled_bm_ei(
    bundle: EiPreparedBundle,
    *,
    base_path: str,
    sliding: bool,
    regression: str,
    bootstrap_result: dict[str, np.ndarray | None] | None = None,
) -> ExtremalIndexEstimate:
    """Estimate theta by pooling the BM reciprocal-EI path over a stable window."""
    path = bundle.paths[(base_path, sliding)]
    return _build_bm_estimate(
        f"{base_path}_{'sliding' if sliding else 'disjoint'}_{regression.lower()}",
        path,
        regression=regression,
        bootstrap_result=bootstrap_result,
    )


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
    """Estimate theta with the classical Ferro-Segers intervals estimator."""
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
    """Estimate theta with the K-gaps likelihood and overlap-based tuning selection."""
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


__all__ = [
    "EI_ALPHA",
    "EI_CI_LEVEL",
    "EiPreparedBundle",
    "EiPathBundle",
    "EiStableWindow",
    "ExtremalIndexEstimate",
    "bootstrap_bm_ei_path_draws",
    "extract_stable_path_window",
    "prepare_ei_bundle",
    "bootstrap_bm_ei_path",
    "estimate_ferro_segers",
    "estimate_k_gaps",
    "estimate_native_bm_ei",
    "estimate_pooled_bm_ei",
]
