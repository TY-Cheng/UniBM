"""Post-fit design-life and quantile mapping for EVI fits."""

from __future__ import annotations

import numpy as np

from .models import ScalingFit


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
