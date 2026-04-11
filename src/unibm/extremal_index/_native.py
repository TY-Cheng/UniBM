"""Native and pooled BM-based extremal-index estimators."""

from __future__ import annotations

import warnings

import numpy as np

from ._paths import extract_stable_path_window
from ._shared import (
    EI_ALPHA,
    EI_TINY,
    Z_CRIT_95,
    ExtremalIndexEstimate,
    EiPathBundle,
    EiPreparedBundle,
    _central_wald_interval,
    _log_scale_theta_interval,
    find_1d_profile_likelihood_intervals,
    scale_1d_pseudo_likelihood,
)

EI_DEFAULT_COVARIANCE_SHRINKAGE = 0.35


def _regularize_ei_covariance(
    covariance: np.ndarray,
    *,
    covariance_shrinkage: float = EI_DEFAULT_COVARIANCE_SHRINKAGE,
) -> np.ndarray:
    """Shrink and ridge-regularize an EI bootstrap covariance matrix."""
    cov = np.asarray(covariance, dtype=float).copy()
    shrinkage = float(np.clip(covariance_shrinkage, 0.0, 1.0))
    if shrinkage > 0:
        diagonal = np.diag(np.diag(cov))
        cov = (1.0 - shrinkage) * cov + shrinkage * diagonal
    scale = np.trace(cov) / max(cov.shape[0], 1)
    ridge = max(abs(float(scale)) * 1e-8, 1e-12)
    return cov + np.eye(cov.shape[0]) * ridge


def _fit_pooled_z_model(
    z_values: np.ndarray,
    *,
    covariance: np.ndarray | None = None,
    covariance_shrinkage: float = EI_DEFAULT_COVARIANCE_SHRINKAGE,
) -> dict[str, float | np.ndarray]:
    """Fit the pooled intercept-only model on the z-scale."""
    z = np.asarray(z_values, dtype=float)
    X = np.ones((z.size, 1), dtype=float)

    if covariance is not None and covariance.shape == (z.size, z.size):
        regularized = _regularize_ei_covariance(
            covariance,
            covariance_shrinkage=covariance_shrinkage,
        )
        inv_cov = np.linalg.pinv(regularized)
        normal_matrix = X.T @ inv_cov @ X
        beta = np.linalg.pinv(normal_matrix) @ (X.T @ inv_cov @ z)
        cov_beta = np.linalg.pinv(normal_matrix)
        fitted = X @ beta
        resid = z - fitted
        objective = float(resid @ inv_cov @ resid)
    else:
        normal_matrix = X.T @ X
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        fitted = X @ beta
        resid = z - fitted
        objective = float(resid @ resid)
        dof = max(z.size - X.shape[1], 1)
        sigma2 = objective / float(dof) if z.size > X.shape[1] else 0.0
        cov_beta = sigma2 * np.linalg.pinv(normal_matrix)
    try:
        condition_number = float(np.linalg.cond(normal_matrix))
    except np.linalg.LinAlgError:
        condition_number = float("inf")

    result: dict[str, float | np.ndarray] = {
        "intercept": float(beta[0]),
        "standard_error": float(np.sqrt(max(float(cov_beta[0, 0]), 0.0))),
        "objective": objective,
        "condition_number": condition_number,
        "fitted": fitted,
        "cov_beta": cov_beta,
    }
    return result


def _pooled_z_fit(
    z_values: np.ndarray,
    *,
    covariance: np.ndarray | None = None,
    covariance_shrinkage: float = EI_DEFAULT_COVARIANCE_SHRINKAGE,
) -> tuple[float, float, str]:
    """Return the pooled estimate, its SE, and the fit variant."""
    z = np.asarray(z_values, dtype=float)
    use_gls = covariance is not None and covariance.shape == (z.size, z.size)
    model = _fit_pooled_z_model(
        z,
        covariance=covariance if use_gls else None,
        covariance_shrinkage=covariance_shrinkage,
    )
    variant = "bootstrap_cov" if use_gls else "ols"
    return float(model["intercept"]), float(model["standard_error"]), variant


def _build_bm_estimate(
    method: str,
    path: EiPathBundle,
    *,
    regression: str,
    bootstrap_result: dict[str, np.ndarray | None] | None = None,
    covariance_shrinkage: float = EI_DEFAULT_COVARIANCE_SHRINKAGE,
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
    z_hat, se, ci_variant = _pooled_z_fit(
        selected_z,
        covariance=covariance if regression == "FGLS" else None,
        covariance_shrinkage=covariance_shrinkage,
    )
    theta_hat = float(np.exp(-z_hat))
    if regression == "FGLS" and not used_gls:
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
    """Estimate ``theta`` with a native single-block-size BM estimator.

    Parameters
    ----------
    bundle
        Prepared observed-data bundle returned by
        :func:`unibm.extremal_index.prepare_ei_bundle`.
    base_path
        BM path family to use. ``"northrop"`` applies the Northrop
        pseudo-likelihood on the selected block size; ``"bb"`` applies the BB
        fixed-``b`` Wald fit.
    sliding
        If ``True``, use the sliding-block path prepared in ``bundle``.
        Otherwise use the disjoint-block path.
    use_adjusted_chandwich
        If ``True`` and ``base_path="northrop"``, use the Chandler-Bate
        scale-adjusted profile likelihood when building the confidence interval.
        This flag has no effect for ``base_path="bb"``.

    Returns
    -------
    unibm.extremal_index.ExtremalIndexEstimate
        Native fixed-``b`` EI estimate. The headline fields are ``theta_hat``,
        ``confidence_interval``, ``selected_level``, and ``stable_window``.
        ``path_level`` and ``path_theta`` are included only as supporting path
        diagnostics.

    Notes
    -----
    This native fit uses one selected block size from the observed BM path. If
    you want to pool several stable block sizes together instead, use
    :func:`estimate_pooled_bm_ei`.
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
    bootstrap_result: dict[str, np.ndarray | None] | None = None,
    covariance_shrinkage: float = EI_DEFAULT_COVARIANCE_SHRINKAGE,
) -> ExtremalIndexEstimate:
    """Estimate ``theta`` by pooling an observed BM path over a stable window.

    Parameters
    ----------
    bundle
        Prepared observed-data bundle returned by
        :func:`unibm.extremal_index.prepare_ei_bundle`.
    base_path
        BM path family to pool. ``"northrop"`` uses the
        ``-log(Fhat(X_t))``-based path and ``"bb"`` uses the ``1 - Fhat(X_t)``
        path.
    sliding
        If ``True``, pool the sliding-block path prepared in ``bundle``.
        Otherwise pool the disjoint-block path.
    regression
        Pooling rule on the transformed stable-window path. Use ``"OLS"`` to
        ignore cross-block covariance and ``"FGLS"`` to use a covariance matrix
        estimated from bootstrap path draws.
    bootstrap_result
        Optional bootstrap output aligned to the same stable block-size levels.
        When supplied with ``regression="FGLS"``, its covariance matrix is used
        for the FGLS fit. The observed path itself is still the quantity being
        pooled; bootstrap draws only provide covariance information.
    covariance_shrinkage
        Diagonal shrinkage applied when regularizing the bootstrap covariance
        before FGLS fitting.

    Returns
    -------
    unibm.extremal_index.ExtremalIndexEstimate
        Pooled formal EI estimate. The headline fields are ``theta_hat``,
        ``confidence_interval``, ``stable_window``, ``regression``, and
        ``base_path``. ``path_level`` and ``path_theta`` remain available for
        diagnostic inspection of the observed path.

    Notes
    -----
    Pooling is performed on the observed transformed path inside the selected
    stable window. Bootstrap paths do not replace that observed path; they are
    used only to estimate the cross-block covariance needed by the FGLS
    weighting step.
    """
    path = bundle.paths[(base_path, sliding)]
    method = f"{base_path}_{'sliding' if sliding else 'disjoint'}_{regression.lower()}"
    return _build_bm_estimate(
        method,
        path,
        regression=regression,
        bootstrap_result=bootstrap_result,
        covariance_shrinkage=covariance_shrinkage,
    )
