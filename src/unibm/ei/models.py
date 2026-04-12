"""Public result types for formal EI estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EiStableWindow:
    """Selected stable window on an integer tuning axis."""

    lo: int
    hi: int


@dataclass(frozen=True)
class EiPathBundle:
    """Observed BM-EI path ingredients for one base-path and block scheme.

    ``theta_path`` and ``z_path`` are computed from the observed series over the
    candidate block-size grid. ``stable_window`` and ``selected_level`` record
    where that observed path is judged stable, while ``sample_statistics``
    preserves the per-block-size window statistics reused by native fixed-``b``
    estimators.
    """

    base_path: str
    sliding: bool
    block_sizes: np.ndarray
    theta_path: np.ndarray
    eir_path: np.ndarray
    z_path: np.ndarray
    sample_counts: np.ndarray
    sample_statistics: dict[int, np.ndarray]
    stable_window: EiStableWindow
    selected_level: int


@dataclass(frozen=True)
class ExtremalIndexEstimate:
    """Unified formal-EI result container.

    Most users should read ``theta_hat`` and ``confidence_interval`` first, then
    inspect ``stable_window``, ``regression``, and ``base_path`` to understand
    which formal estimator produced the headline result. ``path_level``,
    ``path_theta``, and ``path_eir`` are retained for path diagnostics and
    plotting rather than for headline reporting.
    """

    method: str
    theta_hat: float
    confidence_interval: tuple[float, float]
    standard_error: float = np.nan
    ci_method: str = "wald"
    ci_variant: str = "default"
    tuning_axis: str = "b"
    selected_level: int | None = None
    stable_window: EiStableWindow | None = None
    path_level: tuple[int, ...] = ()
    path_theta: tuple[float, ...] = ()
    path_eir: tuple[float, ...] = ()
    selected_threshold_quantile: float | None = None
    selected_threshold_value: float | None = None
    selected_run_k: int | None = None
    block_scheme: str | None = None
    base_path: str | None = None
    regression: str | None = None


@dataclass(frozen=True)
class ThresholdCandidate:
    """One threshold-side EI fit before cross-threshold selection."""

    threshold_quantile: float
    threshold_value: float
    theta_hat: float
    confidence_interval: tuple[float, float]
    standard_error: float
    ci_method: str
    ci_variant: str
    run_k: int | None = None


@dataclass(frozen=True)
class EiPreparedBundle:
    """Reusable EI preparation outputs derived from one observed series.

    The bundle stores the cleaned observed values, the candidate block-size
    grid, all BM path variants, and threshold-side exceedance candidates so the
    native BM, pooled BM, and threshold estimators can all reuse the same
    preparation step.
    """

    values: np.ndarray
    block_sizes: np.ndarray
    paths: dict[tuple[str, bool], EiPathBundle]
    threshold_candidates: dict[float, np.ndarray]
