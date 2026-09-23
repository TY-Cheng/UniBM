"""Internal linear-model and covariance-alignment helpers for EVI estimators."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._validation import (
    matrix_condition_number,
    regularize_covariance,
    subset_covariance_by_labels,
)
from .models import BlockSummaryCurve, PlateauWindow
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
    """Check reused target, quantile, and block-scheme metadata.

    Raise ValueError on a mismatch. This checks estimator identity, not
    whether the curve came from the same observations as the current input.
    """
    if curve.target != target:
        raise ValueError("curve target does not match the requested target.")
    if not isinstance(curve.sliding, (bool, np.bool_)) or bool(curve.sliding) != bool(sliding):
        raise ValueError("curve sliding metadata does not match the requested block scheme.")
    if target == "quantile":
        expected_quantile = _validate_quantile(quantile)
        if curve.quantile is None or float(curve.quantile) != expected_quantile:
            raise ValueError("curve quantile does not match the requested quantile.")


def _validate_plateau_slice(curve: BlockSummaryCurve, plateau: PlateauWindow) -> None:
    """Check that a reused plateau matches its slice of the positive curve.

    Validate inclusive start, exclusive stop, boolean mask, and matching
    log coordinates; raise ValueError instead of silently realigning them.
    """
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
    """Check target and block-scheme metadata before covariance reuse.

    Quantile targets also require an exactly matching quantile. Raise
    ValueError for absent or inconsistent metadata; data identity remains
    the caller's responsibility.
    """
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
    """Regress paired log summaries on an intercept and log block size.

    With an ``n x n`` response covariance, use regularized GLS and return
    ``(X.T @ precision @ X)^+`` as coefficient covariance. Without it, use
    OLS and the HC0 residual sandwich, which does not adjust for dependence
    between scales. The result includes coefficients, a 2x2 covariance in
    (intercept, slope) order, slope SE, fitted values, and diagnostics.
    """
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
    """Extract covariance rows and columns for the selected positive block sizes.

    Return None when no covariance exists. Otherwise require estimator
    metadata and block-size labels, raising ValueError for inconsistent
    input rather than assuming positional alignment.
    """
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
