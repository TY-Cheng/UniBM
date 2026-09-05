"""Post-fit design-life and quantile mapping for EVI fits."""

from __future__ import annotations

import numpy as np

from .models import ScalingFit
from .estimation import Z_CRIT_95


def _design_block_sizes(
    years: float | np.ndarray,
    observations_per_year: float,
) -> np.ndarray:
    """Return validated integer design-life block sizes."""
    years_arr = np.atleast_1d(np.asarray(years, dtype=float))
    if not np.all(np.isfinite(years_arr)) or np.any(years_arr <= 0.0):
        raise ValueError("Design-life years must be positive and finite.")
    observations_per_year = float(observations_per_year)
    if not np.isfinite(observations_per_year) or observations_per_year <= 0.0:
        raise ValueError("observations_per_year must be finite and positive.")
    return np.ceil(observations_per_year * years_arr)


def predict_block_quantile(fit: ScalingFit, block_size: float) -> float:
    """Predict a block quantile from the fitted scaling law."""
    if fit.target != "quantile":
        raise ValueError(
            "predict_block_quantile requires a quantile-based ScalingFit. "
            f"Received target={fit.target!r}."
        )
    block_size = float(block_size)
    if not np.isfinite(block_size) or block_size <= 0:
        raise ValueError("Block size must be positive and finite.")
    log_b = np.log(block_size)
    return float(np.exp(fit.intercept + fit.slope * log_b))


def estimate_design_life_level(
    fit: ScalingFit,
    years: float | np.ndarray,
    *,
    observations_per_year: float = 365.25,
    tau: float | None = None,
) -> float | np.ndarray:
    """Map one fitted quantile-scaling law to horizon-maximum quantiles.

    The design block size is ``ceil(years * observations_per_year)`` on the
    caller's observation clock. This does not split observations by calendar
    year or refit annual maxima. ``tau`` must match the fit's quantile; changing
    it does not construct an application-style shared-slope companion curve.
    A median fit therefore gives a median design-life level, not a return level
    whose waiting time equals ``years``.
    """
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
    block_sizes = _design_block_sizes(years, observations_per_year)
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
    if not np.all(np.isfinite(cov_beta)):
        raise ValueError("Design-life intervals require a finite coefficient covariance matrix.")
    block_sizes = _design_block_sizes(years, observations_per_year)
    z_crit = float(z_crit)
    if not np.isfinite(z_crit) or z_crit < 0.0:
        raise ValueError("z_crit must be finite and non-negative.")
    log_block_sizes = np.log(block_sizes)
    design = np.column_stack([np.ones_like(log_block_sizes), log_block_sizes])
    log_mean = fit.intercept + fit.slope * log_block_sizes
    log_var = np.einsum("ij,jk,ik->i", design, cov_beta, design)
    log_se = np.sqrt(np.maximum(log_var, 0.0))
    lower = np.exp(log_mean - z_crit * log_se)
    upper = np.exp(log_mean + z_crit * log_se)
    if np.ndim(years) == 0:
        return (float(lower[0]), float(upper[0]))
    return (lower, upper)
