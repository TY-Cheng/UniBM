"""Shared block-size grid helpers used by both EVI and EI workflows."""

from __future__ import annotations

import numpy as np


DEFAULT_MIN_DISJOINT_BLOCKS = 17


def validate_block_sizes(
    block_sizes: np.ndarray | list[int] | list[float],
    *,
    n_obs: int | None = None,
) -> np.ndarray:
    """Return a nonempty, strictly increasing 1D array of integer block sizes.

    Integer-valued floats are accepted. Sizes must be at least two and, when
    ``n_obs`` is supplied, no larger than the series. Invalid input raises
    ``ValueError``; the function does not sort or deduplicate a supplied grid.
    """
    try:
        raw = np.asarray(block_sizes)
    except (TypeError, ValueError) as exc:
        raise ValueError("block_sizes must be a one-dimensional numeric sequence.") from exc
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("block_sizes must be a non-empty one-dimensional sequence.")
    if np.iscomplexobj(raw):
        raise ValueError("block_sizes must contain finite integer values.")
    try:
        numeric = raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("block_sizes must contain finite integer values.") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("block_sizes must contain finite integer values.")
    if np.any(numeric < 2):
        raise ValueError("block_sizes must be at least 2.")
    if n_obs is not None and np.any(numeric > int(n_obs)):
        raise ValueError("block_sizes cannot exceed the number of observations.")
    sizes = numeric.astype(int)
    if np.any(np.diff(sizes) <= 0):
        raise ValueError("block_sizes must be strictly increasing with no duplicates.")
    return sizes


def generate_block_sizes(
    n_obs: int,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    geom: bool = True,
    min_disjoint_blocks: int | None = DEFAULT_MIN_DISJOINT_BLOCKS,
) -> np.ndarray:
    """Build a rounded, unique geometric or linear grid for a series of length n.

    At least 32 observations are required. By default the lower bound is
    ``max(5, ceil(n_obs**(1 / 3)))``; the upper bound is
    ``min(floor(n_obs**(1 - 1 / e)), floor(n_obs / 17))``. Set
    ``min_disjoint_blocks=None`` to omit the disjoint-block cap, or supply a
    positive integer to change it. This cap applies only to the automatic
    upper bound, not an explicit ``max_block_size``. Bounds are never expanded to provide
    extra grid points; estimators check their own minimum usable point counts.
    Bounds must increase and fit within the series.

    ``num_step`` counts grid points before rounding and deduplication, so the
    returned 1D integer array can be shorter. Size bounds are integers of at
    least two; supplied ``num_step`` and ``min_disjoint_blocks`` are positive integers.
    """
    if (
        isinstance(n_obs, (bool, np.bool_))
        or not isinstance(n_obs, (int, np.integer))
        or n_obs < 1
    ):
        raise ValueError("n_obs must be a positive integer.")
    if n_obs < 32:
        raise ValueError("At least 32 observations are required for block-size selection.")
    for name, value, minimum in (
        ("min_block_size", min_block_size, 2),
        ("max_block_size", max_block_size, 2),
        ("num_step", num_step, 1),
        ("min_disjoint_blocks", min_disjoint_blocks, 1),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer at least {minimum}.")
    for name, value in (("min_block_size", min_block_size), ("max_block_size", max_block_size)):
        if value is not None and value > n_obs:
            raise ValueError(f"{name} cannot exceed the number of observations.")
    if min_block_size is None:
        min_block_size = max(5, int(np.ceil(n_obs ** (1.0 / 3.0))))
    if max_block_size is None:
        max_block_size = int(np.floor(n_obs ** (1.0 - 1.0 / np.e)))
        if min_disjoint_blocks is not None:
            max_block_size = min(max_block_size, n_obs // min_disjoint_blocks)
    if max_block_size <= min_block_size:
        raise ValueError("max_block_size must be greater than min_block_size.")
    if num_step is None:
        num_step = min(32, max(10, max_block_size - min_block_size + 1))
    if geom:
        block_sizes = np.geomspace(min_block_size, max_block_size, num=num_step)
    else:
        block_sizes = np.linspace(min_block_size, max_block_size, num=num_step)
    block_sizes = np.unique(np.clip(np.rint(block_sizes).astype(int), min_block_size, None))
    return validate_block_sizes(block_sizes[block_sizes > 1], n_obs=n_obs)
