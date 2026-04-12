"""Internal regression orchestration for canonical EVI estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._block_grid import generate_block_sizes
from .._validation import as_1d_float_array
from .blocks import block_summary_curve
from .bootstrap import circular_block_summary_bootstrap
from .models import BlockSummaryCurve, PlateauWindow, ScalingFit
from .selection import select_penultimate_window


Z_CRIT_95 = 1.96
DEFAULT_COVARIANCE_SHRINKAGE = 0.35
DEFAULT_CURVATURE_PENALTY = 2.0


def _fit_linear_model(
    x: np.ndarray,
    y: np.ndarray,
    covariance: np.ndarray | None = None,
    covariance_shrinkage: float = 0.35,
) -> dict[str, Any]:
    """Fit the log-log scaling regression with optional covariance-aware weighting."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    if covariance is not None and covariance.shape == (x.size, x.size):
        shrinkage = float(np.clip(covariance_shrinkage, 0.0, 1.0))
        if shrinkage > 0:
            diagonal = np.diag(np.diag(covariance))
            covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
        scale = np.trace(covariance) / max(covariance.shape[0], 1)
        ridge = max(abs(float(scale)) * 1e-8, 1e-12)
        regularized = covariance + np.eye(covariance.shape[0]) * ridge
        inv_cov = np.linalg.pinv(regularized)
        normal_matrix = X.T @ inv_cov @ X
        beta = np.linalg.pinv(normal_matrix) @ (X.T @ inv_cov @ y)
        cov_beta = np.linalg.pinv(normal_matrix)
        fitted = X @ beta
        resid = y - fitted
        objective = float(resid @ inv_cov @ resid)
    else:
        normal_matrix = X.T @ X
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ beta
        resid = y - fitted
        xtx_inv = np.linalg.pinv(normal_matrix)
        meat = X.T @ np.diag(resid**2) @ X
        cov_beta = xtx_inv @ meat @ xtx_inv
        objective = float(resid @ resid)
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
    raw_covariance = bootstrap.get("covariance")
    if raw_covariance is None:
        return None
    covariance = np.atleast_2d(np.asarray(raw_covariance, dtype=float))
    if covariance.size == 0:
        return None
    bootstrap_block_sizes = np.asarray(
        bootstrap.get("block_sizes", curve.positive_block_sizes),
        dtype=int,
    )
    if covariance.shape[0] != bootstrap_block_sizes.size:
        return None
    if np.array_equal(bootstrap_block_sizes, curve.block_sizes):
        positive_idx = np.flatnonzero(curve.positive_mask)
        covariance = covariance[np.ix_(positive_idx, positive_idx)]
    elif not np.array_equal(bootstrap_block_sizes, curve.positive_block_sizes):
        positive_idx = []
        lookup = {int(block_size): idx for idx, block_size in enumerate(bootstrap_block_sizes)}
        for block_size in curve.positive_block_sizes:
            idx = lookup.get(int(block_size))
            if idx is None:
                return None
            positive_idx.append(idx)
        covariance = covariance[np.ix_(positive_idx, positive_idx)]
    if covariance.shape[0] != curve.log_block_sizes.size:
        return None
    return covariance[plateau.start : plateau.stop, plateau.start : plateau.stop]


def _fit_scaling_model(
    vec: np.ndarray | list[float],
    *,
    target: str,
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
    bootstrap_reps: int = 0,
    super_block_size: int | None = None,
    random_state: int | None = 0,
    curve: BlockSummaryCurve | None = None,
    plateau: PlateauWindow | None = None,
    bootstrap_result: dict[str, Any] | None = None,
) -> ScalingFit:
    """Internal scaling-model implementation shared by all EVI targets."""
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
            np.asarray(block_sizes, dtype=int),
            sliding=sliding,
            quantile=quantile,
            target=target,
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
    bootstrap = bootstrap_result
    if bootstrap is None and bootstrap_reps > 1:
        bootstrap = circular_block_summary_bootstrap(
            vec=arr,
            block_sizes=curve.positive_block_sizes,
            target=target,
            quantile=quantile,
            sliding=sliding,
            reps=bootstrap_reps,
            super_block_size=super_block_size,
            random_state=random_state,
        )
    covariance = _aligned_bootstrap_covariance(bootstrap, curve, plateau)
    model = _fit_linear_model(
        plateau.x,
        plateau.y,
        covariance=covariance,
        covariance_shrinkage=covariance_shrinkage,
    )
    slope = model["slope"]
    standard_error = model["standard_error"]
    return ScalingFit(
        target=target,
        quantile=float(quantile),
        sliding=bool(sliding),
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
    )
