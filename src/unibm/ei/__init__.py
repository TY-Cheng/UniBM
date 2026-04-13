"""Canonical EI-facing UniBM subpackage."""

from .bootstrap import bootstrap_bm_ei_path, bootstrap_bm_ei_path_draws
from .bm import (
    EI_DEFAULT_COVARIANCE_SHRINKAGE,
    estimate_native_bm_ei,
    estimate_pooled_bm_ei,
)
from .preparation import prepare_ei_bundle
from .models import (
    EiPathBundle,
    EiPreparedBundle,
    EiStableWindow,
    ExtremalIndexEstimate,
    ThresholdCandidate,
)
from .plotting import plot_ei_fit, plot_ei_path
from .selection import extract_stable_path_window, select_stable_path_window
from .threshold import estimate_ferro_segers, estimate_k_gaps
from ._stats import (
    EI_ALPHA,
    EI_CI_LEVEL,
    EI_TINY,
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
    "bootstrap_bm_ei_path",
    "bootstrap_bm_ei_path_draws",
    "estimate_ferro_segers",
    "estimate_k_gaps",
    "estimate_native_bm_ei",
    "estimate_pooled_bm_ei",
    "extract_stable_path_window",
    "plot_ei_fit",
    "plot_ei_path",
    "prepare_ei_bundle",
    "select_stable_path_window",
]
