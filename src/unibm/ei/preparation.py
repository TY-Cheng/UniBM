"""Observed-sample preparation for canonical EI workflows."""

from __future__ import annotations

import numpy as np

from .._block_grid import DEFAULT_MIN_DISJOINT_BLOCKS, generate_block_sizes, validate_block_sizes
from ._validation import (
    _validate_ei_series,
    _validate_threshold_quantiles,
)
from .models import EiPreparedBundle
from .paths import BM_PATH_KEYS, _build_bm_paths_from_values


def prepare_ei_bundle(
    vec: np.ndarray | list[float],
    *,
    allow_zeros: bool,
    block_sizes: np.ndarray | None = None,
    path_keys: tuple[tuple[str, bool], ...] = BM_PATH_KEYS,
    threshold_quantiles: tuple[float, ...] = (0.90, 0.95),
) -> EiPreparedBundle:
    """Prepare EI paths and a strictly increasing threshold grid without changing the clock.

    ``vec`` must be a finite 1D series of at least 32 observations. Values must
    be positive unless ``allow_zeros=True``; zeros are then retained at their
    original positions. The caller defines what one observation step represents.

    ``block_sizes`` is an increasing integer grid from 2 through the sample
    size, or a generated grid from ``max(5, ceil(n**(1/3)))`` through
    ``min(floor(sqrt(n)), floor(n/17))`` when omitted. The bounds are not
    expanded when too few levels remain for selection. ``path_keys``
    selects unique ``(base_path, sliding)`` pairs; all four Northrop/BB and
    sliding/disjoint pairs are prepared by default. Use ``path_keys=()`` for
    threshold-only preparation, without a block grid or BM computation.
    A single supplied block size fixes native inference at that level without
    selecting a stable window. Otherwise selection requires at least four
    finite path levels.

    Threshold quantiles must be strictly increasing and in (0, 1), defaulting
    to ``(0.90, 0.95)``. The bundle stores indices strictly above each empirical
    quantile; ties equal to the threshold are excluded. Estimators consume this
    order unless the caller requests a validated subset.
    """
    values = _validate_ei_series(vec, allow_zeros=allow_zeros)
    threshold_quantiles = _validate_threshold_quantiles(threshold_quantiles)
    try:
        path_keys = tuple(path_keys)
    except TypeError as exc:
        raise ValueError("path_keys must contain unique (base_path, sliding) pairs.") from exc
    if any(
        not isinstance(key, tuple)
        or len(key) != 2
        or not isinstance(key[0], str)
        or key[0] not in {"northrop", "bb"}
        or not isinstance(key[1], (bool, np.bool_))
        for key in path_keys
    ) or len(set(path_keys)) != len(path_keys):
        raise ValueError("path_keys must contain unique ('northrop' or 'bb', bool) pairs.")
    if path_keys:
        if block_sizes is None:
            block_sizes = generate_block_sizes(
                values.size,
                max_block_size=min(
                    int(np.sqrt(values.size)), values.size // DEFAULT_MIN_DISJOINT_BLOCKS
                ),
            )
        block_sizes = validate_block_sizes(block_sizes, n_obs=values.size)
        paths = _build_bm_paths_from_values(values, block_sizes, path_keys=path_keys)
    else:
        if block_sizes is not None:
            raise ValueError("block_sizes is not used when path_keys is empty.")
        block_sizes = np.asarray([], dtype=int)
        paths = {}
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
