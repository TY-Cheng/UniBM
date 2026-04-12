"""Canonical public estimation entrypoints for the EVI workflow."""

from __future__ import annotations

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
    quantile: float = 0.5,
    sliding: bool = True,
    block_sizes: np.ndarray | None = None,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    bootstrap_reps: int = 200,
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
    """Estimate the extreme value index from a block-quantile scaling law."""
    return _fit_scaling_model(
        vec=vec,
        target="quantile",
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
    target: str = "quantile",
    quantile: float = 0.5,
    sliding: bool = True,
    block_sizes: np.ndarray | None = None,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    bootstrap_reps: int = 0,
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
    """Fit the UniBM log-log scaling model for an arbitrary block summary."""
    return _fit_scaling_model(
        vec=vec,
        target=target,
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
