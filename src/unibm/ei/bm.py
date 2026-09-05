"""Canonical BM-based extremal-index estimators."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from .._bootstrap_precision import matching_precision_metadata

from .._validation import (
    matrix_condition_number,
    regularize_covariance,
    subset_covariance_by_labels,
    validate_covariance_shrinkage,
)
from ._likelihood import find_1d_profile_likelihood_intervals, scale_1d_pseudo_likelihood
from ._stats import (
    EI_ALPHA,
    EI_TINY,
    _central_wald_interval,
    _log_scale_theta_interval,
)
from .models import EiPathBundle, EiPreparedBundle, ExtremalIndexEstimate
from .selection import extract_stable_path_window


EI_DEFAULT_COVARIANCE_SHRINKAGE = 0.37


def _validate_ei_bootstrap_identity(
    bootstrap_result: dict[str, Any],
    path: EiPathBundle,
) -> None:
    """Require bootstrap covariance to identify the fitted EI path."""
    if "base_path" not in bootstrap_result:
        raise ValueError("bootstrap_result must include base_path metadata.")
    if bootstrap_result["base_path"] != path.base_path:
        raise ValueError("bootstrap_result base_path does not match the fitted EI path.")
    if "sliding" not in bootstrap_result:
        raise ValueError("bootstrap_result must include sliding metadata.")
    raw_sliding = bootstrap_result["sliding"]
    if not isinstance(raw_sliding, (bool, np.bool_)) or bool(raw_sliding) != bool(path.sliding):
        raise ValueError("bootstrap_result sliding metadata does not match the fitted EI path.")


def _regularize_ei_covariance(
    covariance: np.ndarray,
    *,
    covariance_shrinkage: float = EI_DEFAULT_COVARIANCE_SHRINKAGE,
) -> np.ndarray:
    """Shrink and ridge-regularize an EI bootstrap covariance matrix."""
    return regularize_covariance(
        covariance,
        covariance_shrinkage=covariance_shrinkage,
        context="EI bootstrap covariance",
    )


def _fit_pooled_z_model(
    z_values: np.ndarray,
    *,
    covariance: np.ndarray | None = None,
    covariance_shrinkage: float = EI_DEFAULT_COVARIANCE_SHRINKAGE,
) -> dict[str, float | bool | np.ndarray]:
    """Fit the pooled intercept-only model on the z-scale."""
    z = np.asarray(z_values, dtype=float)
    X = np.ones((z.size, 1), dtype=float)

    if covariance is not None:
        regularized = _regularize_ei_covariance(
            covariance,
            covariance_shrinkage=covariance_shrinkage,
        )
        inv_cov = np.linalg.pinv(regularized)
        normal_matrix = X.T @ inv_cov @ X
        beta = np.linalg.pinv(normal_matrix) @ (X.T @ inv_cov @ z)
        cov_beta = np.linalg.pinv(normal_matrix)
    else:
        normal_matrix = X.T @ X
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)

    unconstrained_intercept = float(beta[0])
    boundary_active = unconstrained_intercept <= 0.0
    beta = np.asarray([max(0.0, unconstrained_intercept)], dtype=float)
    fitted = X @ beta
    resid = z - fitted
    if covariance is not None and covariance.shape == (z.size, z.size):
        objective = float(resid @ inv_cov @ resid)
    else:
        objective = float(resid @ resid)
        dof = max(z.size - X.shape[1], 1)
        sigma2 = objective / float(dof) if z.size > X.shape[1] else 0.0
        cov_beta = sigma2 * np.linalg.pinv(normal_matrix)
    try:
        condition_number = float(np.linalg.cond(normal_matrix))
    except np.linalg.LinAlgError:
        condition_number = float("inf")

    return {
        "intercept": float(beta[0]),
        "unconstrained_intercept": unconstrained_intercept,
        "boundary_active": boundary_active,
        "standard_error": float(np.sqrt(max(float(cov_beta[0, 0]), 0.0))),
        "objective": objective,
        "condition_number": condition_number,
        "fitted": fitted,
        "cov_beta": cov_beta,
    }


def _pooled_z_fit(
    z_values: np.ndarray,
    *,
    covariance: np.ndarray | None = None,
    covariance_shrinkage: float = EI_DEFAULT_COVARIANCE_SHRINKAGE,
) -> tuple[float, float, str]:
    """Return the pooled estimate, its SE, and the fit variant."""
    z = np.asarray(z_values, dtype=float)
    use_gls = covariance is not None
    model = _fit_pooled_z_model(
        z,
        covariance=covariance if use_gls else None,
        covariance_shrinkage=covariance_shrinkage,
    )
    variant = "bootstrap_cov" if use_gls else "ols"
    if bool(model["boundary_active"]):
        variant = f"{variant}_boundary"
    return float(model["intercept"]), float(model["standard_error"]), variant


def _build_bm_estimate(
    method: str,
    path: EiPathBundle,
    *,
    regression: str,
    bootstrap_result: dict[str, Any] | None = None,
    covariance_shrinkage: float = EI_DEFAULT_COVARIANCE_SHRINKAGE,
) -> ExtremalIndexEstimate:
    """Pool one BM path either by OLS or by FGLS on the transformed scale."""
    selected_levels, selected_z = extract_stable_path_window(path)
    shrinkage_policy = validate_covariance_shrinkage(covariance_shrinkage)
    covariance = None
    bootstrap_reps_used: int | None = None
    if bootstrap_result is not None:
        _validate_ei_bootstrap_identity(bootstrap_result, path)
        raw_cov = bootstrap_result.get("covariance")
        if raw_cov is not None and "block_sizes" not in bootstrap_result:
            raise ValueError("bootstrap_result must include block_sizes labels.")
        if raw_cov is not None:
            covariance = subset_covariance_by_labels(
                raw_cov,
                bootstrap_result["block_sizes"],
                selected_levels,
                context="EI bootstrap covariance",
            )
        realized_shrinkage = float(shrinkage_policy)
        samples = bootstrap_result.get("samples")
        if samples is not None and np.asarray(samples).ndim == 2:
            bootstrap_reps_used = int(np.asarray(samples).shape[0])
    else:
        realized_shrinkage = None
    if regression == "FGLS" and covariance is None:
        raise ValueError("FGLS requires bootstrap covariance covering every selected block size.")
    z_hat, z_standard_error, ci_variant = _pooled_z_fit(
        selected_z,
        covariance=covariance if regression == "FGLS" else None,
        covariance_shrinkage=(0.0 if realized_shrinkage is None else realized_shrinkage),
    )
    theta_hat = float(np.exp(-z_hat))
    standard_error = float(theta_hat * z_standard_error)
    return ExtremalIndexEstimate(
        method=method,
        theta_hat=theta_hat,
        confidence_interval=_log_scale_theta_interval(z_hat, z_standard_error),
        standard_error=standard_error,
        z_standard_error=z_standard_error,
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
        covariance_shrinkage_policy=(None if regression != "FGLS" else "fixed"),
        covariance_shrinkage=(None if regression != "FGLS" else realized_shrinkage),
        covariance_condition_number_raw=(
            None if covariance is None else matrix_condition_number(covariance)
        ),
        covariance_condition_number_regularized=(
            None
            if covariance is None
            else matrix_condition_number(
                _regularize_ei_covariance(
                    covariance,
                    covariance_shrinkage=(
                        0.0 if realized_shrinkage is None else realized_shrinkage
                    ),
                )
            )
        ),
        bootstrap_block_length_policy=(
            None
            if bootstrap_result is None
            else bootstrap_result.get("bootstrap_block_length_policy")
        ),
        bootstrap_block_length=(
            None if bootstrap_result is None else bootstrap_result.get("bootstrap_block_length")
        ),
        bootstrap_reps_requested=(
            None if bootstrap_result is None else bootstrap_result.get("bootstrap_reps_requested")
        ),
        bootstrap_reps_used=bootstrap_reps_used,
        **matching_precision_metadata(bootstrap_result, selected_levels, shrinkage_policy),
    )


def _northrop_profile_fit(
    statistics: np.ndarray,
    *,
    adjusted: bool,
) -> tuple[float, tuple[float, float], float, str]:
    """Fit Northrop with a profile interval and local-information ``theta`` SE."""
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
    standard_error = float(theta_hat / np.sqrt(stats.size))
    try:
        if adjusted:
            hessian = float(-stats.size / (theta_hat**2))
            scores = (1.0 / theta_hat) - stats
            empirical_variance = float(np.sum(scores**2))
            adjusted_standard_error = float(np.sqrt(empirical_variance) / abs(hessian))
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
                1.0,
                alpha=EI_ALPHA,
            )
            standard_error = adjusted_standard_error
            ci_variant = "chandwich_adjusted"
        else:
            interval = find_1d_profile_likelihood_intervals(
                loglik,
                theta_hat,
                EI_TINY,
                1.0,
                alpha=EI_ALPHA,
            )
    except Exception as exc:  # pragma: no cover - fallback safety
        warnings.warn(
            f"Falling back to unadjusted Northrop profile likelihood: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )
        interval = find_1d_profile_likelihood_intervals(
            loglik,
            theta_hat,
            EI_TINY,
            1.0,
            alpha=EI_ALPHA,
        )
        ci_variant = "profile_fallback"
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
    """Estimate ``theta`` with a native single-block-size BM estimator.

    Northrop fits report an observed-information SE, or the matching sandwich
    SE when Chandler--Bate adjustment is requested. BB fits report their native
    delta-method SE. All are on the ``theta`` scale.
    """
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
    bootstrap_result: dict[str, Any] | None = None,
    covariance_shrinkage: float = EI_DEFAULT_COVARIANCE_SHRINKAGE,
) -> ExtremalIndexEstimate:
    """Estimate ``theta`` by pooling an observed BM path over a stable window.

    The fitted intercept is constrained to ``z = log(1 / theta) >= 0``.
    ``standard_error`` is delta-transformed to the ``theta`` scale, while
    ``z_standard_error`` retains the regression-scale uncertainty.

    ``regression`` must be OLS or FGLS. OLS rejects a bootstrap result; FGLS
    requires covariance from the matching base path and sliding/disjoint scheme
    and never falls back to OLS. Full-grid covariance is subset by block-size
    labels. The default diagonal shrinkage is fixed at 0.37.

    Adaptive precision metadata is retained only when the bootstrap's checked
    window and shrinkage match this fit. The log-scale interval is clipped to
    the legal theta upper boundary of 1 and remains conditional on the selected
    window; it is not a post-selection or bootstrap-percentile interval.
    """
    if regression not in {"OLS", "FGLS"}:
        raise ValueError("regression must be 'OLS' or 'FGLS'.")
    if regression == "OLS" and bootstrap_result is not None:
        raise ValueError("OLS does not accept bootstrap_result.")
    covariance_shrinkage = validate_covariance_shrinkage(covariance_shrinkage)
    path = bundle.paths[(base_path, sliding)]
    method = f"{base_path}_{'sliding' if sliding else 'disjoint'}_{regression.lower()}"
    return _build_bm_estimate(
        method,
        path,
        regression=regression,
        bootstrap_result=bootstrap_result,
        covariance_shrinkage=covariance_shrinkage,
    )
