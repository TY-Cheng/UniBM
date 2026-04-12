"""Observed-sample preparation for canonical EI workflows."""

from __future__ import annotations

import numpy as np

from .._block_grid import generate_block_sizes
from ._validation import _finite_nonnegative_series, _finite_positive_series
from .models import EiPreparedBundle
from .paths import _build_bm_paths_from_values


def prepare_ei_bundle(
    vec: np.ndarray | list[float],
    *,
    block_sizes: np.ndarray | None = None,
    threshold_quantiles: tuple[float, ...] = (0.90, 0.95),
    allow_zeros: bool = False,
) -> EiPreparedBundle:
    """Prepare the observed-data ingredients reused across formal EI estimators."""
    values = _finite_nonnegative_series(vec) if allow_zeros else _finite_positive_series(vec)
    if block_sizes is None:
        block_sizes = generate_block_sizes(values.size)
    block_sizes = np.asarray(block_sizes, dtype=int)
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
