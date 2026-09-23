"""Shared raw circular bootstrap sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircularBootstrapSampleBank:
    """A reusable bank of circular block-bootstrap time-series resamples."""

    block_size: int
    samples: np.ndarray


def default_circular_bootstrap_block_size(
    n_obs: int,
    *,
    minimum: int = 16,
) -> int:
    """Return ``round(sqrt(n_obs))``, bounded below by minimum and above by n_obs.

    This heuristic retains short runs of temporal dependence; it does not
    estimate the dependence range. A nonpositive series length raises
    ``ValueError``.
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    return int(min(max(minimum, round(np.sqrt(n_obs))), n_obs))


def draw_circular_block_bootstrap_sample(
    vec: np.ndarray | list[float],
    *,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Join randomly started circular blocks and truncate to the input length.

    The input is flattened to floats and ``block_size`` is clamped to [1, n].
    Block starts are sampled independently using, and advancing, ``rng``.
    Observations keep their order within each block, wrapping at the end of
    the series. Non-finite values are retained if sampled; callers that need
    entirely finite data must validate it first. Empty or wholly non-finite
    input raises ``ValueError``.
    """
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        raise ValueError("Cannot bootstrap a series without any finite observations.")
    block_size = int(min(max(block_size, 1), arr.size))
    wrapped = np.concatenate([arr, arr[: block_size - 1]]) if block_size > 1 else arr
    n_blocks = int(np.ceil(arr.size / block_size))
    starts = rng.integers(0, arr.size, size=n_blocks)
    sampled = np.concatenate([wrapped[start : start + block_size] for start in starts])
    return sampled[: arr.size]


def draw_circular_block_bootstrap_samples(
    vec: np.ndarray | list[float],
    *,
    reps: int,
    block_size: int | None = None,
    random_state: int | None = None,
) -> CircularBootstrapSampleBank:
    """Return a bank with one length-n circular bootstrap sample per row.

    ``reps`` must be positive. A missing block size uses the square-root
    heuristic, and ``random_state`` seeds a local generator. The returned
    ``samples`` array has shape ``(reps, n)``. The bank records the requested
    block size; each draw clamps it to [1, n]. Input is flattened, and any
    non-finite observations are retained rather than silently removed.
    """
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        raise ValueError("Cannot bootstrap a series without any finite observations.")
    if reps < 1:
        raise ValueError("reps must be at least 1.")
    if block_size is None:
        block_size = default_circular_bootstrap_block_size(arr.size)
    rng = np.random.default_rng(random_state)
    samples = np.empty((reps, arr.size), dtype=float)
    for rep in range(reps):
        samples[rep] = draw_circular_block_bootstrap_sample(
            arr,
            block_size=block_size,
            rng=rng,
        )
    return CircularBootstrapSampleBank(block_size=int(block_size), samples=samples)
