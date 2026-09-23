"""Observed-sample preparation for canonical EI workflows."""

from __future__ import annotations

import numpy as np

from .._block_grid import generate_block_sizes, validate_block_sizes
from ._validation import (
    _validate_ei_series,
    _validate_threshold_quantiles,
)
from .models import EiPreparedBundle
from .paths import _build_bm_paths_from_values


def prepare_ei_bundle(
    vec: np.ndarray | list[float],
    *,
    allow_zeros: bool,
    block_sizes: np.ndarray | None = None,
    threshold_quantiles: tuple[float, ...] = (0.90, 0.95),
) -> EiPreparedBundle:
    """Prepare EI paths and a strictly increasing threshold grid without changing the clock.

    ``vec`` must be a finite 1D series of at least 32 observations. Values must
    be positive unless ``allow_zeros=True``; zeros are then retained at their
    original positions. The caller defines what one observation step represents.

    ``block_sizes`` is an increasing integer grid from 2 through the sample
    size, or a generated intermediate-range grid when omitted. Return all four
    Northrop/BB and sliding/disjoint paths with selected stable windows. Window
    selection requires at least four finite path levels.

    Threshold quantiles must be strictly increasing and in (0, 1), defaulting
    to ``(0.90, 0.95)``. The bundle stores indices strictly above each empirical
    quantile; ties equal to the threshold are excluded. Estimators consume this
    order unless the caller requests a validated subset.
    """
    values = _validate_ei_series(vec, allow_zeros=allow_zeros)
    threshold_quantiles = _validate_threshold_quantiles(threshold_quantiles)
    if block_sizes is None:
        block_sizes = generate_block_sizes(values.size)
    block_sizes = validate_block_sizes(block_sizes, n_obs=values.size)
    paths = _build_bm_paths_from_values(values, block_sizes)
    threshold_candidates = {
        float(q): np.flatnonzero(values > np.quantile(values, float(q)))
        for q in threshold_quantiles
    }
    return EiPreparedBundle(
        values=values,
        block_sizes=block_sizes,
        paths=paths,
        threshold_candidates=threshold_candidates,
    )
