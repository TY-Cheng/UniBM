"""EVI result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class BlockSummaryCurve:
    """Aligned block-size summaries, including values excluded from log fitting.

    All array fields share the full grid length. ``positive_mask`` identifies
    finite, strictly positive summaries with nonzero counts; filtered and
    log properties use that mask without changing the stored full arrays.
    """

    target: str
    quantile: float | None
    sliding: bool
    block_sizes: np.ndarray
    counts: np.ndarray
    values: np.ndarray
    positive_mask: np.ndarray

    @property
    def positive_block_sizes(self) -> np.ndarray:
        """Return block sizes whose summaries pass ``positive_mask``."""
        return self.block_sizes[self.positive_mask]

    @property
    def positive_values(self) -> np.ndarray:
        """Return finite positive summaries selected for log-log fitting."""
        return self.values[self.positive_mask]

    @property
    def positive_counts(self) -> np.ndarray:
        """Return maxima counts aligned with the positive summary subset."""
        return self.counts[self.positive_mask]

    @property
    def log_block_sizes(self) -> np.ndarray:
        """Return natural logs of the block sizes in the positive subset."""
        return np.log(self.positive_block_sizes)

    @property
    def log_values(self) -> np.ndarray:
        """Return natural logs of the positive block-summary values."""
        return np.log(self.positive_values)


@dataclass(frozen=True)
class PlateauWindow:
    """Selected contiguous slice of the positive-summary log-log curve.

    ``start`` is inclusive and ``stop`` exclusive. ``mask`` indexes the full
    positive subset, while ``x`` and ``y`` contain only the selected slice.
    A lower ``score`` indicates the preferred window under the selection rule.
    """

    start: int
    stop: int
    score: float
    mask: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class ScalingFit:
    """Fitted log-summary intercept, slope, uncertainty, and selection diagnostics.

    ``cov_beta`` orders coefficients as (intercept, slope). The slope is the
    headline EVI estimate when the chosen summary obeys the assumed scaling
    law. ``confidence_interval`` is its nominal 95% Wald interval conditional
    on the selected plateau; it omits window-selection and model uncertainty.
    ``regression_policy`` records the request, while ``regression`` and
    ``ci_variant`` describe the fit actually used. ``bootstrap_precision_met``
    reports Monte Carlo precision, not statistical interval coverage.
    """

    target: str
    quantile: float
    sliding: bool
    regression_policy: Literal["OLS", "FGLS", "AUTO"]
    regression: Literal["OLS", "FGLS"]
    ci_variant: Literal["hc0", "bootstrap_cov"]
    intercept: float
    slope: float
    standard_error: float
    confidence_interval: tuple[float, float]
    curve: BlockSummaryCurve
    plateau: PlateauWindow
    cov_beta: np.ndarray
    bootstrap: dict[str, Any] | None = None
    covariance_shrinkage_policy: str | None = None
    covariance_shrinkage: float | None = None
    covariance_condition_number_raw: float | None = None
    covariance_condition_number_regularized: float | None = None
    bootstrap_block_length_policy: str | None = None
    bootstrap_block_length: int | None = None
    bootstrap_reps_requested: int | None = None
    bootstrap_reps_used: int | None = None
    bootstrap_reps_policy: str | None = None
    bootstrap_precision_met: bool | None = None
    bootstrap_mcse: tuple[float, ...] = ()
    bootstrap_mcse_max_ratio: float | None = None
    bootstrap_mcse_targets: tuple[str, ...] = ()

    @property
    def block_sizes(self) -> np.ndarray:
        """Return all positive-summary block sizes, including those outside the plateau."""
        return self.curve.positive_block_sizes

    @property
    def counts(self) -> np.ndarray:
        """Return maxima counts for all positive summaries, before plateau selection."""
        return self.curve.positive_counts

    @property
    def values(self) -> np.ndarray:
        """Return all positive summary values, before plateau selection."""
        return self.curve.positive_values

    @property
    def log_block_sizes(self) -> np.ndarray:
        """Return natural-log block sizes for all positive summaries."""
        return self.curve.log_block_sizes

    @property
    def log_values(self) -> np.ndarray:
        """Return natural-log summary values for all positive summaries."""
        return self.curve.log_values

    @property
    def plateau_mask(self) -> np.ndarray:
        """Return the plateau mask aligned with the positive-summary subset."""
        return self.plateau.mask

    @property
    def plateau_block_sizes(self) -> np.ndarray:
        """Return only the block sizes used in the selected regression window."""
        return self.block_sizes[self.plateau_mask]

    @property
    def plateau_bounds(self) -> tuple[int, int]:
        """Return the first and last selected block sizes, both inclusive."""
        return int(self.plateau_block_sizes[0]), int(self.plateau_block_sizes[-1])


__all__ = ["BlockSummaryCurve", "PlateauWindow", "ScalingFit"]
