"""Post-fit design-life and quantile mapping for EVI fits."""

from __future__ import annotations

import numpy as np

from .models import ScalingFit
from .estimation import Z_CRIT_95


def predict_block_quantile(fit: ScalingFit, block_size: float) -> float:
    """Predict a block quantile from the fitted scaling law."""
    if fit.target != "quantile":
        raise ValueError(
            "predict_block_quantile requires a quantile-based ScalingFit. "
            f"Received target={fit.target!r}."
        )
    if block_size <= 0:
        raise ValueError("Block size must be positive.")
    log_b = np.log(block_size)
    return float(np.exp(fit.intercept + fit.slope * log_b))


def estimate_design_life_level(
    fit: ScalingFit,
    years: float | np.ndarray,
    *,
    observations_per_year: float = 365.25,
    tau: float | None = None,
) -> float | np.ndarray:
    """Map a fitted quantile-scaling law to design-life levels."""
    if fit.target != "quantile":
        raise ValueError(
            "estimate_design_life_level requires a quantile-based ScalingFit. "
            f"Received target={fit.target!r}."
        )
    fit_tau = float(fit.quantile)
    tau_value = fit_tau if tau is None else float(tau)
    if not np.isclose(tau_value, fit_tau):
        raise ValueError(
            "estimate_design_life_level must use the same tau as the fitted ScalingFit. "
            f"Received tau={tau_value:.4f}, fit.quantile={fit_tau:.4f}."
        )
    years_arr = np.atleast_1d(np.asarray(years, dtype=float))
    block_sizes = observations_per_year * years_arr
    estimates = np.asarray(
        [predict_block_quantile(fit, block_size=float(size)) for size in block_sizes]
    )
    return float(estimates[0]) if np.ndim(years) == 0 else estimates


def estimate_design_life_level_interval(
    fit: ScalingFit,
    years: float | np.ndarray,
    *,
    observations_per_year: float = 365.25,
    tau: float | None = None,
    z_crit: float = Z_CRIT_95,
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """Return delta-method design-life intervals on the original response scale.

    The fitted scaling law implies

    ``log D(T) = alpha + xi * log(b_T)``,

    so the log-scale variance follows from the fitted 2x2 coefficient covariance
    matrix ``cov_beta``. The returned interval is pointwise and does not include
    any post-selection or model-class uncertainty beyond that covariance matrix.
    """
    if fit.target != "quantile":
        raise ValueError(
            "estimate_design_life_level_interval requires a quantile-based ScalingFit. "
            f"Received target={fit.target!r}."
        )
    fit_tau = float(fit.quantile)
    tau_value = fit_tau if tau is None else float(tau)
    if not np.isclose(tau_value, fit_tau):
        raise ValueError(
            "estimate_design_life_level_interval must use the same tau as the fitted ScalingFit. "
            f"Received tau={tau_value:.4f}, fit.quantile={fit_tau:.4f}."
        )
    cov_beta = np.asarray(fit.cov_beta, dtype=float)
    if cov_beta.shape != (2, 2):
        raise ValueError(
            "estimate_design_life_level_interval requires a 2x2 coefficient covariance matrix. "
            f"Received cov_beta.shape={cov_beta.shape}."
        )
    years_arr = np.atleast_1d(np.asarray(years, dtype=float))
    if np.any(years_arr <= 0.0):
        raise ValueError("Design-life years must be positive.")
    block_sizes = observations_per_year * years_arr
    if np.any(block_sizes <= 0.0):
        raise ValueError("Implied design-life block sizes must be positive.")
    log_block_sizes = np.log(block_sizes)
    design = np.column_stack([np.ones_like(log_block_sizes), log_block_sizes])
    log_mean = fit.intercept + fit.slope * log_block_sizes
    log_var = np.einsum("ij,jk,ik->i", design, cov_beta, design)
    log_se = np.sqrt(np.maximum(log_var, 0.0))
    lower = np.exp(log_mean - float(z_crit) * log_se)
    upper = np.exp(log_mean + float(z_crit) * log_se)
    if np.ndim(years) == 0:
        return (float(lower[0]), float(upper[0]))
    return (lower, upper)
