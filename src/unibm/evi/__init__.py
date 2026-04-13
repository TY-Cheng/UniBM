"""Canonical EVI-facing UniBM subpackage."""

from .._block_grid import DEFAULT_MIN_DISJOINT_BLOCKS, generate_block_sizes
from .blocks import block_maxima, block_summary_curve
from .bootstrap import (
    BlockSummaryBootstrapBackbone,
    build_block_summary_bootstrap_backbone,
    circular_block_summary_bootstrap,
    circular_block_summary_bootstrap_multi_target,
    evaluate_block_summary_bootstrap_backbone,
)
from .design import (
    estimate_design_life_level,
    estimate_design_life_level_interval,
    predict_block_quantile,
)
from .estimation import (
    DEFAULT_COVARIANCE_SHRINKAGE,
    DEFAULT_CURVATURE_PENALTY,
    Z_CRIT_95,
    estimate_evi_quantile,
    estimate_target_scaling,
)
from .models import BlockSummaryCurve, PlateauWindow, ScalingFit
from .plotting import plot_scaling_fit
from .selection import select_penultimate_window
from .spectrum import candidate_max_spectrum_scales, estimate_max_spectrum_evi
from .summaries import estimate_sample_mode, summarize_block_maxima
from .tail import (
    ExternalXiEstimate,
    SelectionWindow,
    ThresholdWindow,
    candidate_tail_counts,
    estimate_dedh_moment_evi,
    estimate_hill_evi,
    estimate_pickands_evi,
    select_stable_integer_window,
    select_stable_tail_window,
    wald_confidence_interval,
)
from .targets import target_stability_summary

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
    "candidate_max_spectrum_scales",
    "candidate_tail_counts",
    "circular_block_summary_bootstrap",
    "circular_block_summary_bootstrap_multi_target",
    "estimate_sample_mode",
    "estimate_dedh_moment_evi",
    "estimate_design_life_level",
    "estimate_design_life_level_interval",
    "estimate_evi_quantile",
    "estimate_hill_evi",
    "estimate_max_spectrum_evi",
    "estimate_pickands_evi",
    "estimate_target_scaling",
    "evaluate_block_summary_bootstrap_backbone",
    "generate_block_sizes",
    "predict_block_quantile",
    "plot_scaling_fit",
    "select_penultimate_window",
    "select_stable_integer_window",
    "select_stable_tail_window",
    "summarize_block_maxima",
    "target_stability_summary",
    "wald_confidence_interval",
]
