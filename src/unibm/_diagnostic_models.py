"""Result containers for shared diagnostic utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
