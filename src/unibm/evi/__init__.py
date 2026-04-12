"""Canonical EVI-facing UniBM subpackage.

This package groups the block-maxima extreme-value-index workflow:

- block-maxima construction and summary curves;
- plateau selection and log-log scaling fits;
- bootstrap covariance backbones for FGLS;
- external xi-estimator baselines used in benchmark comparisons.

Shared kernels such as validation, rolling-window operations, and generic
diagnostics remain at the top-level ``unibm`` package because they are used
across both the EVI and EI layers.
"""

from .._block_grid import DEFAULT_MIN_DISJOINT_BLOCKS
from .bootstrap import (
    BlockSummaryBootstrapBackbone,
    build_block_summary_bootstrap_backbone,
    circular_block_summary_bootstrap,
    circular_block_summary_bootstrap_multi_target,
    evaluate_block_summary_bootstrap_backbone,
)
from .core import (
    DEFAULT_COVARIANCE_SHRINKAGE,
    DEFAULT_CURVATURE_PENALTY,
    Z_CRIT_95,
    block_maxima,
    block_summary_curve,
    estimate_design_life_level,
    estimate_evi_quantile,
    estimate_target_scaling,
    generate_block_sizes,
    predict_block_quantile,
    select_penultimate_window,
)
from .external import (
    ExternalXiEstimate,
    SelectionWindow,
    ThresholdWindow,
    candidate_tail_counts,
    estimate_dedh_moment_evi,
    estimate_hill_evi,
    estimate_max_spectrum_evi,
    estimate_pickands_evi,
    select_stable_integer_window,
    select_stable_tail_window,
    wald_confidence_interval,
)
from .models import BlockSummaryCurve, PlateauWindow, ScalingFit

__all__ = [
    "BlockSummaryBootstrapBackbone",
    "BlockSummaryCurve",
    "DEFAULT_COVARIANCE_SHRINKAGE",
    "DEFAULT_CURVATURE_PENALTY",
    "DEFAULT_MIN_DISJOINT_BLOCKS",
    "ExternalXiEstimate",
    "PlateauWindow",
    "ScalingFit",
    "SelectionWindow",
    "ThresholdWindow",
    "Z_CRIT_95",
    "block_maxima",
    "block_summary_curve",
    "build_block_summary_bootstrap_backbone",
    "candidate_tail_counts",
    "circular_block_summary_bootstrap",
    "circular_block_summary_bootstrap_multi_target",
    "estimate_dedh_moment_evi",
    "estimate_design_life_level",
    "estimate_evi_quantile",
    "estimate_hill_evi",
    "estimate_max_spectrum_evi",
    "estimate_pickands_evi",
    "estimate_target_scaling",
    "evaluate_block_summary_bootstrap_backbone",
    "generate_block_sizes",
    "predict_block_quantile",
    "select_penultimate_window",
    "select_stable_integer_window",
    "select_stable_tail_window",
    "wald_confidence_interval",
]
