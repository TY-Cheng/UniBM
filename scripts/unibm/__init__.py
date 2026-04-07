"""Public UniBM methods API.

This package exposes the reusable statistical layer only. Repo-local benchmark,
screening, and manuscript orchestration code lives under `scripts/benchmark`,
`scripts/application`, `scripts/shared`, and `scripts/vignette`.
"""

from .bootstrap import (
    BlockSummaryBootstrapBackbone,
    CircularBootstrapSampleBank,
    build_block_summary_bootstrap_backbone,
    circular_block_summary_bootstrap,
    circular_block_summary_bootstrap_multi_target,
    draw_circular_block_bootstrap_samples,
    evaluate_block_summary_bootstrap_backbone,
)
from .core import (
    block_maxima,
    block_summary_curve,
    estimate_evi_quantile,
    estimate_return_level,
    estimate_target_scaling,
    generate_block_sizes,
    predict_block_quantile,
    select_penultimate_window,
)
from .diagnostics import (
    empirical_cdf,
    estimate_extremal_index_reciprocal,
    kernel_cdf,
    marginal_cdf,
    target_stability_summary,
)
from .external import (
    ExternalXiEstimate,
    SelectionWindow,
    ThresholdWindow,
    estimate_dedh_moment_evi,
    estimate_hill_evi,
    estimate_max_spectrum_evi,
    estimate_pickands_evi,
)
from .models import BlockSummaryCurve, ExtremalIndexReciprocalFit, PlateauWindow, ScalingFit

__all__ = [
    "BlockSummaryCurve",
    "BlockSummaryBootstrapBackbone",
    "CircularBootstrapSampleBank",
    "ExternalXiEstimate",
    "ExtremalIndexReciprocalFit",
    "PlateauWindow",
    "ScalingFit",
    "SelectionWindow",
    "ThresholdWindow",
    "block_maxima",
    "block_summary_curve",
    "build_block_summary_bootstrap_backbone",
    "circular_block_summary_bootstrap",
    "circular_block_summary_bootstrap_multi_target",
    "draw_circular_block_bootstrap_samples",
    "empirical_cdf",
    "evaluate_block_summary_bootstrap_backbone",
    "estimate_dedh_moment_evi",
    "estimate_evi_quantile",
    "estimate_extremal_index_reciprocal",
    "estimate_hill_evi",
    "estimate_max_spectrum_evi",
    "estimate_pickands_evi",
    "estimate_return_level",
    "estimate_target_scaling",
    "generate_block_sizes",
    "kernel_cdf",
    "marginal_cdf",
    "plot_extremal_index_reciprocal",
    "plot_scaling_fit",
    "predict_block_quantile",
    "select_penultimate_window",
    "target_stability_summary",
]


def __getattr__(name: str):
    """Lazily expose plotting helpers so importing `unibm` stays lightweight."""
    if name in {"plot_extremal_index_reciprocal", "plot_scaling_fit"}:
        from .plotting import plot_extremal_index_reciprocal, plot_scaling_fit

        return {
            "plot_extremal_index_reciprocal": plot_extremal_index_reciprocal,
            "plot_scaling_fit": plot_scaling_fit,
        }[name]
    raise AttributeError(f"module 'unibm' has no attribute {name!r}")
