"""Lightweight result containers shared across the methods and workflow layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BlockSummaryCurve:
    """Block-maxima summaries evaluated over a block-size grid."""

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

    Most users should start with ``slope`` as the headline ``xi`` estimate,
    ``confidence_interval`` for uncertainty, ``plateau_bounds`` for the selected
    regression window, and ``bootstrap`` for any covariance-aware fit metadata.
    The remaining fields retain the full observed curve and fitted window for
    plotting, diagnostics, and workflow-side reuse.
    """

    target: str
    quantile: float
    sliding: bool
    intercept: float
    slope: float
    standard_error: float
    confidence_interval: tuple[float, float]
    curve: BlockSummaryCurve
    plateau: PlateauWindow
    cov_beta: np.ndarray
    bootstrap: dict[str, Any] | None = None

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


@dataclass(frozen=True)
class ExtremalIndexReciprocalFit:
    """Outputs for the reciprocal extremal-index diagnostics."""

    block_sizes: np.ndarray
    log_block_sizes: np.ndarray
    northrop_values: np.ndarray
    northrop_standard_deviations: np.ndarray
    bb_values: np.ndarray
    bb_standard_deviations: np.ndarray
    northrop_block_size: int
    northrop_estimate: float
    bb_block_size: int
    bb_estimate: float
