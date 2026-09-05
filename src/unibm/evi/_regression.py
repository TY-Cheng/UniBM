"""Internal regression orchestration for canonical EVI estimators."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .._block_grid import generate_block_sizes
from .._bootstrap_precision import matching_precision_metadata
from .._validation import (
    as_1d_float_array,
    matrix_condition_number,
    regularize_covariance,
    subset_covariance_by_labels,
    validate_covariance_shrinkage,
)
from .blocks import block_summary_curve
from .bootstrap import _adaptive_block_summary_bootstrap, circular_block_summary_bootstrap
from .models import BlockSummaryCurve, PlateauWindow, ScalingFit
from .selection import select_penultimate_window
from .summaries import _validate_quantile


Z_CRIT_95 = 1.96
DEFAULT_COVARIANCE_SHRINKAGE = 0.37
DEFAULT_CURVATURE_PENALTY = 2.0


def _validate_curve_identity(
    curve: BlockSummaryCurve,
    *,
    target: str,
    quantile: float,
    sliding: bool,
) -> None:
    """Require a reused curve to describe the estimator requested by the caller."""
    if curve.target != target:
        raise ValueError("curve target does not match the requested target.")
    if not isinstance(curve.sliding, (bool, np.bool_)) or bool(curve.sliding) != bool(sliding):
        raise ValueError("curve sliding metadata does not match the requested block scheme.")
    if target == "quantile":
        expected_quantile = _validate_quantile(quantile)
        if curve.quantile is None or float(curve.quantile) != expected_quantile:
            raise ValueError("curve quantile does not match the requested quantile.")


def _validate_plateau_slice(curve: BlockSummaryCurve, plateau: PlateauWindow) -> None:
    """Require a reused plateau to be the declared contiguous slice of its curve."""
    n_points = int(curve.log_block_sizes.size)
    if (
        isinstance(plateau.start, (bool, np.bool_))
        or isinstance(plateau.stop, (bool, np.bool_))
        or not isinstance(plateau.start, (int, np.integer))
        or not isinstance(plateau.stop, (int, np.integer))
        or not 0 <= int(plateau.start) < int(plateau.stop) <= n_points
    ):
        raise ValueError("plateau start/stop must identify a non-empty curve slice.")
    start = int(plateau.start)
    stop = int(plateau.stop)
    mask = np.asarray(plateau.mask)
    expected_mask = np.zeros(n_points, dtype=bool)
    expected_mask[start:stop] = True
    if (
        mask.dtype.kind != "b"
        or mask.shape != (n_points,)
        or not np.array_equal(mask, expected_mask)
    ):
        raise ValueError("plateau mask must match its declared curve slice.")
    expected_x = curve.log_block_sizes[start:stop]
    expected_y = curve.log_values[start:stop]
    try:
        x = np.asarray(plateau.x, dtype=float)
        y = np.asarray(plateau.y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("plateau x/y must match its declared curve slice.") from exc
    if (
        x.shape != expected_x.shape
        or y.shape != expected_y.shape
        or not np.allclose(x, expected_x, rtol=1e-12, atol=1e-12)
        or not np.allclose(y, expected_y, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("plateau x/y must match its declared curve slice.")


def _validate_bootstrap_identity(
    bootstrap: dict[str, Any],
    curve: BlockSummaryCurve,
) -> None:
    """Require bootstrap covariance metadata to identify the fitted curve estimator."""
    for field in ("target", "sliding"):
        if field not in bootstrap:
            raise ValueError(f"bootstrap_result must include {field} metadata.")
    if bootstrap["target"] != curve.target:
        raise ValueError("bootstrap_result target does not match the fitted curve.")
    raw_sliding = bootstrap["sliding"]
    if not isinstance(raw_sliding, (bool, np.bool_)) or bool(raw_sliding) != bool(curve.sliding):
        raise ValueError("bootstrap_result sliding metadata does not match the fitted curve.")
    if curve.target == "quantile":
        if "quantile" not in bootstrap:
            raise ValueError("bootstrap_result must include quantile metadata.")
        raw_quantile = bootstrap["quantile"]
        if isinstance(raw_quantile, (bool, np.bool_)):
            raise ValueError("bootstrap_result quantile does not match the fitted curve.")
        try:
            bootstrap_quantile = float(raw_quantile)
        except (TypeError, ValueError) as exc:
            raise ValueError("bootstrap_result quantile does not match the fitted curve.") from exc
        if curve.quantile is None or bootstrap_quantile != float(curve.quantile):
            raise ValueError("bootstrap_result quantile does not match the fitted curve.")


def _fit_linear_model(
    x: np.ndarray,
    y: np.ndarray,
    covariance: np.ndarray | None = None,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
) -> dict[str, Any]:
    """Fit the log-log scaling regression with optional covariance-aware weighting."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    if covariance is not None:
        if np.asarray(covariance).shape != (x.size, x.size):
            raise ValueError("bootstrap covariance shape must match the regression window.")
        regularized = regularize_covariance(
            covariance,
            covariance_shrinkage=covariance_shrinkage,
            context="bootstrap covariance",
        )
        inv_cov = np.linalg.pinv(regularized)
        normal_matrix = X.T @ inv_cov @ X
        beta = np.linalg.pinv(normal_matrix) @ (X.T @ inv_cov @ y)
        cov_beta = np.linalg.pinv(normal_matrix)
        fitted = X @ beta
        resid = y - fitted
        objective = float(resid @ inv_cov @ resid)
        covariance_condition_number_raw = matrix_condition_number(covariance)
        covariance_condition_number_regularized = matrix_condition_number(regularized)
    else:
        normal_matrix = X.T @ X
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ beta
        resid = y - fitted
        xtx_inv = np.linalg.pinv(normal_matrix)
        meat = X.T @ np.diag(resid**2) @ X
        cov_beta = xtx_inv @ meat @ xtx_inv
        objective = float(resid @ resid)
        covariance_condition_number_raw = None
        covariance_condition_number_regularized = None
    try:
        condition_number = float(np.linalg.cond(normal_matrix))
    except np.linalg.LinAlgError:
        condition_number = float("inf")
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "fitted": fitted,
        "cov_beta": cov_beta,
        "standard_error": float(np.sqrt(max(cov_beta[1, 1], 0.0))),
        "objective": objective,
        "condition_number": condition_number,
        "covariance_condition_number_raw": covariance_condition_number_raw,
        "covariance_condition_number_regularized": covariance_condition_number_regularized,
        "n_obs": int(x.size),
        "n_params": int(X.shape[1]),
    }


def _aligned_bootstrap_covariance(
    bootstrap: dict[str, Any] | None,
    curve: BlockSummaryCurve,
    plateau: PlateauWindow,
) -> np.ndarray | None:
    """Align a bootstrap covariance matrix to the selected positive plateau."""
    if bootstrap is None:
        return None
    _validate_bootstrap_identity(bootstrap, curve)
    raw_covariance = bootstrap.get("covariance")
    if raw_covariance is None:
        return None
    if "block_sizes" not in bootstrap:
        raise ValueError("bootstrap_result must include block_sizes labels.")
    selected_block_sizes = curve.positive_block_sizes[plateau.start : plateau.stop]
    return subset_covariance_by_labels(
        raw_covariance,
        bootstrap["block_sizes"],
        selected_block_sizes,
        context="bootstrap covariance",
    )


def _fit_scaling_model(
    vec: np.ndarray | list[float],
    *,
    target: str,
    regression: Literal["OLS", "FGLS", "AUTO"],
    quantile: float = 0.5,
    sliding: bool = True,
    block_sizes: np.ndarray | None = None,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    plateau_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    bootstrap_reps: int | Literal["adaptive"] | None = None,
    super_block_size: int | None = None,
    random_state: int | None = 0,
    curve: BlockSummaryCurve | None = None,
    plateau: PlateauWindow | None = None,
    bootstrap_result: dict[str, Any] | None = None,
) -> ScalingFit:
    """Internal scaling-model implementation shared by all EVI targets."""
    if regression not in {"OLS", "FGLS", "AUTO"}:
        raise ValueError("regression must be 'OLS', 'FGLS', or 'AUTO'.")
    shrinkage_policy = validate_covariance_shrinkage(covariance_shrinkage)
    if regression == "OLS":
        if bootstrap_result is not None:
            raise ValueError("OLS does not accept bootstrap_result.")
        if bootstrap_reps is not None and (
            isinstance(bootstrap_reps, bool)
            or not isinstance(bootstrap_reps, (int, np.integer))
            or bootstrap_reps != 0
        ):
            raise ValueError("OLS does not use bootstrap; bootstrap_reps must be 0 or None.")
        resolved_bootstrap_reps = 0
    elif bootstrap_result is not None:
        if bootstrap_reps is not None:
            raise ValueError("bootstrap_reps must be None when bootstrap_result is supplied.")
        resolved_bootstrap_reps = 0
    else:
        if bootstrap_reps is None or bootstrap_reps == "adaptive":
            resolved_bootstrap_reps = "adaptive"
        elif (
            isinstance(bootstrap_reps, bool)
            or not isinstance(bootstrap_reps, (int, np.integer))
            or bootstrap_reps < 2
        ):
            raise ValueError(
                "bootstrap_reps must be an integer at least 2 or 'adaptive' for FGLS or AUTO."
            )
        else:
            resolved_bootstrap_reps = int(bootstrap_reps)
    arr = as_1d_float_array(vec)
    finite_count = int(np.sum(np.isfinite(arr)))
    if finite_count < 32:
        raise ValueError("At least 32 finite observations are required for block-size selection.")
    if curve is None and block_sizes is None:
        block_sizes = generate_block_sizes(
            n_obs=arr.size,
            num_step=num_step,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            geom=True,
        )
    if curve is None:
        curve = block_summary_curve(
            arr,
            block_sizes,
            sliding=sliding,
            quantile=quantile,
            target=target,
        )
    _validate_curve_identity(
        curve,
        target=target,
        quantile=quantile,
        sliding=sliding,
    )
    if curve.log_block_sizes.size < plateau_points:
        raise ValueError("Not enough positive block summaries for regression.")
    if plateau is None:
        plateau = select_penultimate_window(
            curve.log_block_sizes,
            curve.log_values,
            min_points=plateau_points,
            trim_fraction=trim_fraction,
            curvature_penalty=curvature_penalty,
        )
    _validate_plateau_slice(curve, plateau)
    bootstrap = bootstrap_result
    if bootstrap is None and resolved_bootstrap_reps == "adaptive":
        levels = curve.positive_block_sizes[plateau.start : plateau.stop]

        def evaluate(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            selected_cov = subset_covariance_by_labels(
                cov, curve.positive_block_sizes, levels, context="bootstrap covariance"
            )
            fit = _fit_linear_model(plateau.x, plateau.y, selected_cov, shrinkage_policy)
            xi, se = fit["slope"], fit["standard_error"]
            return np.asarray([xi, xi - Z_CRIT_95 * se, xi + Z_CRIT_95 * se]), np.full(3, se)

        bootstrap = _adaptive_block_summary_bootstrap(
            arr,
            curve.positive_block_sizes,
            target=target,
            quantile=quantile,
            sliding=sliding,
            super_block_size=super_block_size,
            random_state=random_state,
            evaluate=evaluate,
        )
        bootstrap.update(
            bootstrap_mcse_targets=("xi", "xi_ci_lo", "xi_ci_hi"),
            bootstrap_precision_levels=levels.copy(),
            bootstrap_precision_shrinkage=shrinkage_policy,
        )
    elif bootstrap is None and resolved_bootstrap_reps > 1:
        bootstrap = circular_block_summary_bootstrap(
            vec=arr,
            block_sizes=curve.positive_block_sizes,
            target=target,
            quantile=quantile,
            sliding=sliding,
            reps=resolved_bootstrap_reps,
            super_block_size=super_block_size,
            random_state=random_state,
        )
        bootstrap["bootstrap_reps_policy"] = "fixed"
        bootstrap["bootstrap_reps_requested"] = resolved_bootstrap_reps
        bootstrap["bootstrap_reps_used"] = int(np.asarray(bootstrap["samples"]).shape[0])
    if bootstrap is not None and bootstrap_result is None:
        bootstrap["bootstrap_block_length_policy"] = (
            "default" if super_block_size is None else "fixed"
        )
    covariance = _aligned_bootstrap_covariance(bootstrap, curve, plateau)
    realized_shrinkage: float | None = None
    bootstrap_reps_used: int | None = None
    if covariance is not None:
        realized_shrinkage = float(shrinkage_policy)
        if bootstrap is not None and "samples" in bootstrap:
            samples = np.asarray(bootstrap["samples"])
            if samples.ndim == 2:
                bootstrap_reps_used = int(samples.shape[0])
    if covariance is None:
        if bootstrap_result is not None:
            raise ValueError("supplied bootstrap_result must contain usable covariance.")
        if regression == "FGLS":
            raise ValueError("FGLS requires usable bootstrap covariance.")
    model = _fit_linear_model(
        plateau.x,
        plateau.y,
        covariance=covariance,
        covariance_shrinkage=0.0 if realized_shrinkage is None else realized_shrinkage,
    )
    slope = model["slope"]
    standard_error = model["standard_error"]
    return ScalingFit(
        target=target,
        quantile=float(quantile),
        sliding=bool(sliding),
        regression_policy=regression,
        regression="FGLS" if covariance is not None else "OLS",
        ci_variant="bootstrap_cov" if covariance is not None else "hc0",
        intercept=model["intercept"],
        slope=slope,
        standard_error=standard_error,
        confidence_interval=(
            float(slope - Z_CRIT_95 * standard_error),
            float(slope + Z_CRIT_95 * standard_error),
        ),
        curve=curve,
        plateau=plateau,
        cov_beta=model["cov_beta"],
        bootstrap=bootstrap,
        covariance_shrinkage_policy=(None if covariance is None else "fixed"),
        covariance_shrinkage=realized_shrinkage,
        covariance_condition_number_raw=model["covariance_condition_number_raw"],
        covariance_condition_number_regularized=model["covariance_condition_number_regularized"],
        bootstrap_block_length_policy=(
            None if bootstrap is None else bootstrap.get("bootstrap_block_length_policy")
        ),
        bootstrap_block_length=(None if bootstrap is None else bootstrap.get("super_block_size")),
        bootstrap_reps_requested=(
            None if bootstrap is None else bootstrap.get("bootstrap_reps_requested")
        ),
        bootstrap_reps_used=bootstrap_reps_used,
        **matching_precision_metadata(
            bootstrap,
            curve.positive_block_sizes[plateau.start : plateau.stop],
            shrinkage_policy,
        ),
    )
