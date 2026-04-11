"""Bootstrap helpers for covariance-aware block-summary regression.

This module has two related responsibilities:

1. generate raw-series circular block-bootstrap resamples that preserve
   short-range serial dependence;
2. build reusable super-block bootstrap backbones for the log block-summary
   curve used by UniBM's FGLS regression.

The benchmark now reuses one super-block backbone per block scheme
(`sliding`/`disjoint`) so median/mean/mode FGLS fits can share the same
resampling work instead of rebuilding it target-by-target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from .summaries import summarize_block_maxima
from ._validation import warn_on_negative_values
from .window_ops import circular_sliding_window_maximum


def _sliding_block_maxima(segment: np.ndarray, block_size: int) -> np.ndarray:
    """Return circular sliding maxima inside one super-block segment."""
    return circular_sliding_window_maximum(segment, block_size)


def _disjoint_block_maxima(segment: np.ndarray, block_size: int) -> np.ndarray:
    """Return disjoint maxima inside one super-block segment."""
    n_block = segment.size // block_size
    if n_block < 1:
        return np.asarray([], dtype=float)
    trimmed = segment[: n_block * block_size]
    return trimmed.reshape(n_block, block_size).max(axis=1)


def _segment_block_maxima(
    segment: np.ndarray,
    block_size: int,
    *,
    sliding: bool,
) -> np.ndarray:
    if sliding:
        return _sliding_block_maxima(segment, block_size)
    return _disjoint_block_maxima(segment, block_size)


@dataclass(frozen=True)
class CircularBootstrapSampleBank:
    """A reusable bank of circular block-bootstrap time-series resamples.

    The benchmark reuses the same bootstrap resamples across multiple
    estimators so the runtime and the uncertainty comparison are both cleaner.
    """

    block_size: int
    samples: np.ndarray


@dataclass(frozen=True)
class BlockSummaryBootstrapBackbone:
    """Reusable super-block bootstrap state for multiple block-summary targets.

    `segment_draws` stores which super-blocks were resampled on each bootstrap
    replicate. `maxima_by_block` stores the per-segment block maxima for every
    requested block size. Median/mean/mode summaries can then be evaluated from
    the same backbone without regenerating the dependence-preserving resamples.
    """

    block_sizes: np.ndarray
    sliding: bool
    super_block_size: int
    segment_draws: np.ndarray
    maxima_by_block: dict[int, np.ndarray]


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
    """Draw a reusable bank of circular block-bootstrap samples.

    Parameters
    ----------
    vec
        One-dimensional raw series to resample.
    reps
        Number of bootstrap replicates.
    block_size
        Circular bootstrap block length. If omitted, a default dependence-
        preserving length is chosen from the sample size.
    random_state
        Optional random seed.

    Returns
    -------
    CircularBootstrapSampleBank
        Immutable container with the chosen block size and the matrix of raw
        bootstrap resamples.

    Notes
    -----
    This is the shared entrypoint used by the external-estimator benchmark so
    different estimators can reuse the same raw bootstrap bank.
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


def build_block_summary_bootstrap_backbone(
    vec: np.ndarray | list[float],
    block_sizes: np.ndarray,
    *,
    sliding: bool = True,
    reps: int = 200,
    super_block_size: int | None = None,
    random_state: int | None = 0,
) -> BlockSummaryBootstrapBackbone | None:
    """Precompute the reusable super-block state for UniBM bootstrap fitting.

    Parameters
    ----------
    vec
        One-dimensional raw series.
    block_sizes
        Candidate block-size grid.
    sliding
        If ``True``, store sliding-block maxima within each super-block.
        Otherwise store disjoint maxima.
    reps
        Number of bootstrap replicates to prepare.
    super_block_size
        Optional super-block size. If omitted, the function chooses a size large
        enough to preserve local dependence while retaining multiple segments.
    random_state
        Optional random seed.

    Returns
    -------
    BlockSummaryBootstrapBackbone or None
        Reusable backbone object if the sample is large enough; otherwise
        ``None``.

    Notes
    -----
    The backbone can be evaluated repeatedly for different block-summary
    targets such as the median, mean, or mode without resampling the time
    series from scratch. This is the main benchmark speed-up path.
    """
    warn_on_negative_values(vec, context="build_block_summary_bootstrap_backbone", stacklevel=3)
    arr = np.asarray(vec, dtype=float).reshape(-1)
    block_sizes = np.asarray(block_sizes, dtype=int)
    if np.sum(np.isfinite(arr)) < 64 or block_sizes.size == 0 or reps < 2:
        return None
    max_block_size = int(block_sizes.max())
    if super_block_size is None:
        super_block_size = max(max_block_size * 4, int(np.sqrt(arr.size)))
    super_block_size = min(max(super_block_size, max_block_size + 1), arr.size)
    n_super = arr.size // super_block_size
    if n_super < 4:
        super_block_size = max(max_block_size + 1, arr.size // 4)
        n_super = arr.size // super_block_size
    if n_super < 2:
        return None
    trimmed = arr[: n_super * super_block_size]
    segments = trimmed.reshape(n_super, super_block_size)
    rng = np.random.default_rng(random_state)
    maxima_by_block: dict[int, np.ndarray] = {}
    for block_size in block_sizes:
        maxima_by_block[int(block_size)] = np.stack(
            [
                _segment_block_maxima(segment, int(block_size), sliding=sliding)
                for segment in segments
            ],
            axis=0,
        )
    segment_draws = rng.integers(0, n_super, size=(reps, n_super))
    return BlockSummaryBootstrapBackbone(
        block_sizes=block_sizes.copy(),
        sliding=bool(sliding),
        super_block_size=int(super_block_size),
        segment_draws=segment_draws,
        maxima_by_block=maxima_by_block,
    )


def _selected_bootstrap_maxima(
    backbone: BlockSummaryBootstrapBackbone,
    *,
    block_size: int,
) -> np.ndarray:
    """Return all replicate maxima for one block size as a 2D matrix."""
    maxima_bank = np.asarray(backbone.maxima_by_block[int(block_size)], dtype=float)
    selected = maxima_bank[backbone.segment_draws]
    return selected.reshape(backbone.segment_draws.shape[0], -1)


def _evaluate_quantile_bootstrap_column(
    backbone: BlockSummaryBootstrapBackbone,
    *,
    block_size: int,
    quantile: float,
) -> np.ndarray:
    """Evaluate one quantile bootstrap column across all replicates at once."""
    selected = _selected_bootstrap_maxima(backbone, block_size=block_size)
    return np.quantile(selected, quantile, axis=1, method="median_unbiased")


def _evaluate_mean_bootstrap_column(
    backbone: BlockSummaryBootstrapBackbone,
    *,
    block_size: int,
) -> np.ndarray:
    """Evaluate one mean bootstrap column across all replicates at once."""
    selected = _selected_bootstrap_maxima(backbone, block_size=block_size)
    return np.mean(selected, axis=1)


def _evaluate_mode_bootstrap_column(
    backbone: BlockSummaryBootstrapBackbone,
    *,
    block_size: int,
    quantile: float,
) -> np.ndarray:
    """Evaluate one mode bootstrap column via the existing per-replicate path."""
    selected = _selected_bootstrap_maxima(backbone, block_size=block_size)
    summaries = np.full(selected.shape[0], np.nan, dtype=float)
    for rep, maxima in enumerate(selected):
        summaries[rep] = summarize_block_maxima(maxima, target="mode", quantile=quantile)
    return summaries


def evaluate_block_summary_bootstrap_backbone(
    backbone: BlockSummaryBootstrapBackbone | None,
    *,
    target: str = "quantile",
    quantile: float = 0.5,
) -> dict[str, Any]:
    """Evaluate one block-summary target on a reusable bootstrap backbone.

    Parameters
    ----------
    backbone
        Reusable backbone returned by
        :func:`build_block_summary_bootstrap_backbone`.
    target
        Block-summary functional to evaluate on each bootstrap replicate.
    quantile
        Quantile level used only when ``target="quantile"``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the block-size grid, valid log-summary samples,
        covariance matrix, and bookkeeping metadata.

    Notes
    -----
    The returned covariance matrix is aligned to the full positive block-size
    grid and is later subset to the selected plateau window by the EVI fit.
    """
    if backbone is None:
        return {
            "block_sizes": np.asarray([], dtype=int),
            "samples": np.empty((0, 0)),
            "covariance": None,
        }
    reps = backbone.segment_draws.shape[0]
    block_sizes = np.asarray(backbone.block_sizes, dtype=int)
    samples = np.full((reps, block_sizes.size), np.nan, dtype=float)
    invalid_summary_count = 0
    for idx, block_size in enumerate(block_sizes):
        if target == "quantile":
            summaries = _evaluate_quantile_bootstrap_column(
                backbone,
                block_size=int(block_size),
                quantile=quantile,
            )
        elif target == "mean":
            summaries = _evaluate_mean_bootstrap_column(
                backbone,
                block_size=int(block_size),
            )
        elif target == "mode":
            summaries = _evaluate_mode_bootstrap_column(
                backbone,
                block_size=int(block_size),
                quantile=quantile,
            )
        else:
            raise ValueError(f"Unsupported target: {target}")
        valid_mask = np.isfinite(summaries) & (summaries > 0)
        samples[valid_mask, idx] = np.log(summaries[valid_mask])
        invalid_summary_count += int(np.size(summaries) - np.sum(valid_mask))
    if invalid_summary_count:
        warnings.warn(
            (
                "evaluate_block_summary_bootstrap_backbone excluded "
                f"{invalid_summary_count} non-positive bootstrap block summaries. "
                "This step requires strictly positive inputs."
            ),
            RuntimeWarning,
            stacklevel=3,
        )
    valid_rows = np.all(np.isfinite(samples), axis=1)
    valid_samples = samples[valid_rows]
    covariance = None
    if valid_samples.shape[0] >= 2:
        covariance = np.atleast_2d(np.cov(valid_samples, rowvar=False))
    return {
        "block_sizes": block_sizes,
        "samples": valid_samples,
        "covariance": covariance,
        "super_block_size": backbone.super_block_size,
        "sliding": backbone.sliding,
        "target": target,
        "invalid_replicates": int(np.sum(~valid_rows)),
    }


def circular_block_summary_bootstrap_multi_target(
    vec: np.ndarray | list[float],
    block_sizes: np.ndarray,
    *,
    targets: tuple[str, ...] = ("quantile", "mean", "mode"),
    quantile: float = 0.5,
    sliding: bool = True,
    reps: int = 200,
    super_block_size: int | None = None,
    random_state: int | None = 0,
) -> dict[str, dict[str, Any]]:
    """Bootstrap multiple block-summary targets from one shared backbone.

    Parameters
    ----------
    vec
        One-dimensional raw series.
    block_sizes
        Candidate block-size grid.
    targets
        Block-summary targets to evaluate, for example ``("quantile", "mean",
        "mode")``.
    quantile
        Quantile level used when one of the requested targets is
        ``"quantile"``.
    sliding
        If ``True``, use sliding blocks; otherwise use disjoint blocks.
    reps
        Number of bootstrap replicates.
    super_block_size
        Optional super-block size.
    random_state
        Optional random seed.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from target name to the evaluated bootstrap result dictionary.
    """
    backbone = build_block_summary_bootstrap_backbone(
        vec=vec,
        block_sizes=block_sizes,
        sliding=sliding,
        reps=reps,
        super_block_size=super_block_size,
        random_state=random_state,
    )
    return {
        target: evaluate_block_summary_bootstrap_backbone(
            backbone,
            target=target,
            quantile=quantile,
        )
        for target in targets
    }


def circular_block_summary_bootstrap(
    vec: np.ndarray | list[float],
    block_sizes: np.ndarray,
    *,
    target: str = "quantile",
    quantile: float = 0.5,
    sliding: bool = True,
    reps: int = 200,
    super_block_size: int | None = None,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Bootstrap one block-summary target by resampling time-series super-blocks.

    Parameters
    ----------
    vec
        One-dimensional raw series.
    block_sizes
        Candidate block-size grid.
    target
        Block-summary target to evaluate on each replicate.
    quantile
        Quantile level used only when ``target="quantile"``.
    sliding
        If ``True``, use sliding blocks; otherwise use disjoint blocks.
    reps
        Number of bootstrap replicates.
    super_block_size
        Optional super-block size.
    random_state
        Optional random seed.

    Returns
    -------
    dict[str, Any]
        Bootstrap result dictionary with block sizes, valid log-summary draws,
        and covariance matrix.

    Notes
    -----
    The key point is that the bootstrap resamples the original time series in
    long contiguous chunks and then recomputes the block summaries. This
    preserves local serial dependence and yields a covariance estimate across
    block sizes.
    """
    block_sizes = np.asarray(block_sizes, dtype=int)
    if block_sizes.size == 0 or reps < 2:
        return {
            "block_sizes": block_sizes,
            "samples": np.empty((0, block_sizes.size)),
            "covariance": None,
        }
    backbone = build_block_summary_bootstrap_backbone(
        vec=vec,
        block_sizes=block_sizes,
        sliding=sliding,
        reps=reps,
        super_block_size=super_block_size,
        random_state=random_state,
    )
    if backbone is None:
        return {
            "block_sizes": block_sizes,
            "samples": np.empty((0, block_sizes.size)),
            "covariance": None,
        }
    return evaluate_block_summary_bootstrap_backbone(
        backbone,
        target=target,
        quantile=quantile,
    )
