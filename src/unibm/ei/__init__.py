"""Canonical EI-facing UniBM subpackage.

This is the preferred public import surface for formal extremal-index
estimation. Use the grouped submodules for more specific entrypoints:

- :mod:`unibm.ei.paths` for observed BM-path preparation;
- :mod:`unibm.ei.native` for pooled and native BM estimators;
- :mod:`unibm.ei.threshold` for threshold-side comparators;
- :mod:`unibm.ei.bootstrap` for BM path-bootstrap transforms;
- :mod:`unibm.ei.models` for reusable result types.
"""

from .bootstrap import bootstrap_bm_ei_path, bootstrap_bm_ei_path_draws
from .native import (
    EI_DEFAULT_COVARIANCE_SHRINKAGE,
    estimate_native_bm_ei,
    estimate_pooled_bm_ei,
)
from .paths import extract_stable_path_window, prepare_ei_bundle
from .models import (
    EiPathBundle,
    EiPreparedBundle,
    EiStableWindow,
    ExtremalIndexEstimate,
    ThresholdCandidate,
)
from .threshold import estimate_ferro_segers, estimate_k_gaps
from ._internal import (
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
    "prepare_ei_bundle",
]
