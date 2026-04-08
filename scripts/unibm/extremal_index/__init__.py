"""Formal extremal-index benchmark estimators.

This package is separate from ``diagnostics.py`` because it implements the
benchmark-facing EI estimators rather than exploratory diagnostics. The main
entrypoints here cover:

- threshold estimators: Ferro-Segers and K-gaps;
- native BM estimators: Northrop and BB on one selected block size;
- pooled UniBM BM estimators: OLS/FGLS pooling over a stable block-size window.
"""

from ._bootstrap import bootstrap_bm_ei_path, bootstrap_bm_ei_path_draws
from ._native import (
    EI_DEFAULT_COVARIANCE_SHRINKAGE,
    _bb_wald_fit,
    _build_bm_estimate,
    _fit_pooled_z_model,
    _northrop_profile_fit,
    _pooled_z_fit,
    _regularize_ei_covariance,
    estimate_native_bm_ei,
    estimate_pooled_bm_ei,
)
from ._paths import (
    _build_bm_paths_from_values,
    _build_path_from_scores,
    _rolling_window_minima,
    _select_stable_ei_window,
    extract_stable_path_window,
    prepare_ei_bundle,
)
from ._shared import (
    EI_ALPHA,
    EI_CI_LEVEL,
    EI_TINY,
    EiPathBundle,
    EiPreparedBundle,
    EiStableWindow,
    ExtremalIndexEstimate,
    ThresholdCandidate,
    _central_wald_interval,
    _finite_nonnegative_series,
    _finite_positive_series,
    _intervals_overlap,
    _log_scale_theta_interval,
    _select_between_candidates,
    find_1d_profile_likelihood_intervals,
    scale_1d_pseudo_likelihood,
)
from ._threshold import (
    _ferro_segers_from_times,
    _inter_exceedance_times,
    _kgaps_profile_fit,
    estimate_ferro_segers,
    estimate_k_gaps,
)

__all__ = [
    "EI_ALPHA",
    "EI_CI_LEVEL",
    "EI_DEFAULT_COVARIANCE_SHRINKAGE",
    "EI_TINY",
    "EiPathBundle",
    "EiPreparedBundle",
    "EiStableWindow",
    "ExtremalIndexEstimate",
    "ThresholdCandidate",
    "_bb_wald_fit",
    "_build_bm_estimate",
    "_build_bm_paths_from_values",
    "_build_path_from_scores",
    "_central_wald_interval",
    "_ferro_segers_from_times",
    "_finite_nonnegative_series",
    "_finite_positive_series",
    "_fit_pooled_z_model",
    "_inter_exceedance_times",
    "_intervals_overlap",
    "_kgaps_profile_fit",
    "_log_scale_theta_interval",
    "_northrop_profile_fit",
    "_pooled_z_fit",
    "_regularize_ei_covariance",
    "_rolling_window_minima",
    "_select_between_candidates",
    "_select_stable_ei_window",
    "bootstrap_bm_ei_path",
    "bootstrap_bm_ei_path_draws",
    "estimate_ferro_segers",
    "estimate_k_gaps",
    "estimate_native_bm_ei",
    "estimate_pooled_bm_ei",
    "extract_stable_path_window",
    "find_1d_profile_likelihood_intervals",
    "prepare_ei_bundle",
    "scale_1d_pseudo_likelihood",
]
