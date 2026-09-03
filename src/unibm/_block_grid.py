"""Shared block-size grid helpers used by both EVI and EI workflows."""

from __future__ import annotations

import numpy as np


DEFAULT_MIN_DISJOINT_BLOCKS = 15


def validate_block_sizes(
    block_sizes: np.ndarray | list[int] | list[float],
    *,
    n_obs: int | None = None,
) -> np.ndarray:
    """Return a validated, strictly increasing integer block-size grid."""
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
    min_disjoint_blocks: int = DEFAULT_MIN_DISJOINT_BLOCKS,
) -> np.ndarray:
    """Generate an intermediate-range grid of block sizes."""
    if n_obs < 32:
        raise ValueError("At least 32 observations are required for block-size selection.")
    if min_block_size is None:
        min_block_size = max(5, int(np.ceil(n_obs**0.2)))
    if max_block_size is None:
        exponent_cap = int(np.floor(n_obs**0.55))
        disjoint_cap = int(np.floor(n_obs / max(min_disjoint_blocks, 1)))
        max_block_size = min(exponent_cap, disjoint_cap)
        max_block_size = max(min_block_size + 4, max_block_size)
    if max_block_size <= min_block_size:
        max_block_size = min_block_size + 4
    if num_step is None:
        num_step = min(32, max(10, max_block_size - min_block_size + 1))
    if geom:
        block_sizes = np.geomspace(min_block_size, max_block_size, num=num_step)
    else:
        block_sizes = np.linspace(min_block_size, max_block_size, num=num_step)
    block_sizes = np.unique(np.clip(np.rint(block_sizes).astype(int), min_block_size, None))
    return validate_block_sizes(block_sizes[block_sizes > 1], n_obs=n_obs)
