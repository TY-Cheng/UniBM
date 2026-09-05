"""Canonical public estimation entrypoints for the EVI workflow."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ._regression import (
    DEFAULT_COVARIANCE_SHRINKAGE,
    DEFAULT_CURVATURE_PENALTY,
    Z_CRIT_95,
    _fit_scaling_model,
)
from .models import BlockSummaryCurve, PlateauWindow, ScalingFit


def estimate_evi_quantile(
    vec: np.ndarray | list[float],
    *,
    regression: Literal["OLS", "FGLS", "AUTO"],
    quantile: float = 0.5,
    sliding: bool = True,
    block_sizes: np.ndarray | None = None,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    bootstrap_reps: int | Literal["adaptive"] | None = None,
    super_block_size: int | None = None,
    random_state: int | None = 0,
    plateau_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    curve: BlockSummaryCurve | None = None,
    plateau: PlateauWindow | None = None,
    bootstrap_result: dict[str, object] | None = None,
) -> ScalingFit:
    """Estimate EVI from a quantile of block maxima on a selected log-log plateau.

    ``regression`` is explicit: OLS uses HC0 uncertainty; strict FGLS requires
    usable bootstrap covariance. AUTO permits an internally generated missing
    covariance to fall back to OLS, but never accepts malformed supplied covariance.
    The result records both requested policy and actual regression.

    For FGLS/AUTO without a supplied bootstrap, omitted or ``"adaptive"`` reps
    check 128, 256, 512, 768, and 1024 draws. An integer requests a fixed budget.
    Adaptive precision monitors xi and its CI endpoints, not design-life levels.
    A cap warning retains the fit with ``bootstrap_precision_met=False``.
    Default covariance shrinkage is fixed at 0.37, not automatically tuned.

    Supplied covariance must match the target, quantile, and block scheme;
    block-size labels permit full-grid covariance to serve a selected subset.
    Reuse remains the caller's responsibility for data identity. Intervals are
    conditional on the observed plateau and do not include selection uncertainty.
    """
    return _fit_scaling_model(
        vec=vec,
        target="quantile",
        regression=regression,
        quantile=quantile,
        sliding=sliding,
        block_sizes=block_sizes,
        num_step=num_step,
        min_block_size=min_block_size,
        max_block_size=max_block_size,
        plateau_points=plateau_points,
        trim_fraction=trim_fraction,
        curvature_penalty=curvature_penalty,
        covariance_shrinkage=covariance_shrinkage,
        bootstrap_reps=bootstrap_reps,
        super_block_size=super_block_size,
        random_state=random_state,
        curve=curve,
        plateau=plateau,
        bootstrap_result=bootstrap_result,
    )


def estimate_target_scaling(
    vec: np.ndarray | list[float],
    *,
    regression: Literal["OLS", "FGLS", "AUTO"],
    target: str = "quantile",
    quantile: float = 0.5,
    sliding: bool = True,
    block_sizes: np.ndarray | None = None,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    bootstrap_reps: int | Literal["adaptive"] | None = None,
    super_block_size: int | None = None,
    random_state: int | None = 0,
    plateau_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    curve: BlockSummaryCurve | None = None,
    plateau: PlateauWindow | None = None,
    bootstrap_result: dict[str, object] | None = None,
) -> ScalingFit:
    """Fit the UniBM log-log scaling model for quantile, mean, or mode summaries.

    The regression, covariance-reuse, and adaptive-R contracts are the same as
    ``estimate_evi_quantile``. ``quantile`` is used only for the quantile target.
    Mean/mode fits are not accepted by quantile design-life mapping helpers.
    """
    return _fit_scaling_model(
        vec=vec,
        target=target,
        regression=regression,
        quantile=quantile,
        sliding=sliding,
        block_sizes=block_sizes,
        num_step=num_step,
        min_block_size=min_block_size,
        max_block_size=max_block_size,
        plateau_points=plateau_points,
        trim_fraction=trim_fraction,
        curvature_penalty=curvature_penalty,
        covariance_shrinkage=covariance_shrinkage,
        bootstrap_reps=bootstrap_reps,
        super_block_size=super_block_size,
        random_state=random_state,
        curve=curve,
        plateau=plateau,
        bootstrap_result=bootstrap_result,
    )


__all__ = [
    "DEFAULT_COVARIANCE_SHRINKAGE",
    "DEFAULT_CURVATURE_PENALTY",
    "Z_CRIT_95",
    "estimate_evi_quantile",
    "estimate_target_scaling",
]
