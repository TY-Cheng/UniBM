"""EVI result containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class BlockSummaryCurve:
    """Block-maxima summaries evaluated over a block-size grid."""

    target: str
    quantile: float | None
    sliding: bool
    block_sizes: np.ndarray
    counts: np.ndarray
    values: np.ndarray
    positive_mask: np.ndarray

    @property
    def positive_block_sizes(self) -> np.ndarray:
        return self.block_sizes[self.positive_mask]

    @property
    def positive_values(self) -> np.ndarray:
        return self.values[self.positive_mask]

    @property
    def positive_counts(self) -> np.ndarray:
        return self.counts[self.positive_mask]

    @property
    def log_block_sizes(self) -> np.ndarray:
        return np.log(self.positive_block_sizes)

    @property
    def log_values(self) -> np.ndarray:
        return np.log(self.positive_values)


@dataclass(frozen=True)
class PlateauWindow:
    """The selected intermediate block-size window used for regression."""

    start: int
    stop: int
    score: float
    mask: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class ScalingFit:
    """Full output of a UniBM log-log block-summary regression.

    Most users should start with ``slope`` as the headline ``xi`` estimate and
    ``confidence_interval`` for uncertainty. ``regression_policy``,
    ``regression``, and ``ci_variant`` distinguish the requested policy from the
    estimator and interval construction actually used.
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
        return self.curve.positive_block_sizes

    @property
    def counts(self) -> np.ndarray:
        return self.curve.positive_counts

    @property
    def values(self) -> np.ndarray:
        return self.curve.positive_values

    @property
    def log_block_sizes(self) -> np.ndarray:
        return self.curve.log_block_sizes

    @property
    def log_values(self) -> np.ndarray:
        return self.curve.log_values

    @property
    def plateau_mask(self) -> np.ndarray:
        return self.plateau.mask

    @property
    def plateau_block_sizes(self) -> np.ndarray:
        return self.block_sizes[self.plateau_mask]

    @property
    def plateau_bounds(self) -> tuple[int, int]:
        return int(self.plateau_block_sizes[0]), int(self.plateau_block_sizes[-1])


__all__ = ["BlockSummaryCurve", "PlateauWindow", "ScalingFit"]
