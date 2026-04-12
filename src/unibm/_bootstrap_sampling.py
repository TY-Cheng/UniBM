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
    """Choose a simple dependence-preserving block length for raw-series bootstrap."""
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    return int(min(max(minimum, round(np.sqrt(n_obs))), n_obs))


def draw_circular_block_bootstrap_sample(
    vec: np.ndarray | list[float],
    *,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one circular block-bootstrap sample of the same length as the input series."""
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
    """Draw a reusable bank of circular block-bootstrap samples."""
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
