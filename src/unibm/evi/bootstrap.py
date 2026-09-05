"""Bootstrap helpers for covariance-aware EVI block-summary regression."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any
import warnings

import numpy as np

from .._block_grid import validate_block_sizes
from .._bootstrap_precision import ADAPTIVE_REPS, adaptive_covariance
from .._validation import as_1d_float_array, warn_on_negative_values
from .._window_ops import circular_sliding_window_maximum
from .summaries import _validate_quantile


_MODE_BOOTSTRAP_GRID_POINTS = 256
_MODE_BOOTSTRAP_MAX_WORKING_BYTES = 64 * 1024 * 1024


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
    """Return the requested block-maxima scheme inside one super-block segment."""
    if sliding:
        return _sliding_block_maxima(segment, block_size)
    return _disjoint_block_maxima(segment, block_size)


@dataclass(frozen=True)
class BlockSummaryBootstrapBackbone:
    """Reusable super-block bootstrap state for multiple block-summary targets."""

    block_sizes: np.ndarray
    sliding: bool
    super_block_size: int
    segment_draws: np.ndarray
    maxima_by_block: dict[int, np.ndarray]


def _adaptive_block_summary_bootstrap(
    vec: np.ndarray,
    block_sizes: np.ndarray,
    *,
    target: str,
    quantile: float,
    sliding: bool,
    super_block_size: int | None,
    random_state: int | None,
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Append path rows while reusing the existing maxima backbone."""
    backbone = build_block_summary_bootstrap_backbone(
        vec, block_sizes, sliding=sliding, reps=2, super_block_size=super_block_size
    )
    identity = {
        "block_sizes": block_sizes,
        "target": target,
        "quantile": quantile if target == "quantile" else None,
        "sliding": sliding,
        "bootstrap_reps_policy": "adaptive",
        "bootstrap_reps_requested": ADAPTIVE_REPS[-1],
        "bootstrap_reps_used": 0,
    }
    if backbone is None:
        return {**identity, "covariance": None, "samples": np.empty((0, len(block_sizes)))}

    def draw(reps: int, rng: np.random.Generator) -> np.ndarray:
        n_super = backbone.segment_draws.shape[1]
        rows = []
        for offset in range(0, reps, 32):
            draws = rng.integers(0, n_super, (min(32, reps - offset), n_super))
            result = evaluate_block_summary_bootstrap_backbone(
                replace(backbone, segment_draws=draws), target=target, quantile=quantile
            )
            rows.append(result["samples"])
        return np.concatenate(rows)

    return {
        **identity,
        **adaptive_covariance(draw, evaluate, random_state=random_state),
        "super_block_size": backbone.super_block_size,
    }


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


def _rowwise_linear_quantile(
    sorted_rows: np.ndarray,
    counts: np.ndarray,
    *,
    quantile: float,
) -> np.ndarray:
    """Return one linear-interpolation quantile for every row of a sorted matrix."""
    result = np.full(sorted_rows.shape[0], np.nan, dtype=float)
    valid = counts > 0
    if not np.any(valid):
        return result
    active_rows = sorted_rows[valid]
    active_counts = counts[valid].astype(float)
    positions = (active_counts - 1.0) * float(quantile)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    weight = positions - lower
    row_index = np.arange(active_rows.shape[0])
    lower_values = active_rows[row_index, lower]
    upper_values = active_rows[row_index, upper]
    result[valid] = lower_values + weight * (upper_values - lower_values)
    return result


def _evaluate_mode_bootstrap_column_batched(
    selected: np.ndarray,
    *,
    max_kernel_bytes: int = _MODE_BOOTSTRAP_MAX_WORKING_BYTES,
) -> np.ndarray:
    """Evaluate modes while limiting each three-dimensional KDE kernel temporary."""
    selected = np.asarray(selected, dtype=float)
    if selected.ndim != 2:
        raise ValueError("selected maxima must be a 2D matrix.")
    minimum_working_bytes = _MODE_BOOTSTRAP_GRID_POINTS * np.dtype(float).itemsize
    if max_kernel_bytes < minimum_working_bytes:
        raise ValueError(f"max_kernel_bytes must be at least {minimum_working_bytes}.")
    summaries = np.full(selected.shape[0], np.nan, dtype=float)
    valid = np.isfinite(selected) & (selected > 0)
    counts = np.sum(valid, axis=1)
    if not np.any(counts):
        return summaries

    single_mask = counts == 1
    if np.any(single_mask):
        single_selected = np.where(valid[single_mask], selected[single_mask], -np.inf)
        summaries[single_mask] = np.max(single_selected, axis=1)

    multi_mask = counts > 1
    if not np.any(multi_mask):
        return summaries

    active_selected = selected[multi_mask]
    active_valid = valid[multi_mask]
    active_counts = counts[multi_mask].astype(int, copy=False)
    log_values = np.zeros_like(active_selected, dtype=float)
    np.log1p(active_selected, out=log_values, where=active_valid)
    sorted_logs = np.sort(np.where(active_valid, log_values, np.inf), axis=1)

    q75 = _rowwise_linear_quantile(sorted_logs, active_counts, quantile=0.75)
    q25 = _rowwise_linear_quantile(sorted_logs, active_counts, quantile=0.25)
    iqr = q75 - q25

    sum_logs = np.sum(log_values, axis=1)
    sum_sq_logs = np.sum(log_values * log_values, axis=1)
    active_counts_f = active_counts.astype(float)
    means = sum_logs / active_counts_f
    variances = np.maximum(
        (sum_sq_logs - active_counts_f * means * means) / np.maximum(active_counts_f - 1.0, 1.0),
        0.0,
    )
    std = np.sqrt(variances)
    sigma = np.minimum(std, np.where(iqr > 0.0, iqr / 1.349, std))
    sigma = np.where(~np.isfinite(sigma) | (sigma <= 0.0), np.maximum(std, 1e-3), sigma)
    bandwidth = np.maximum(1.059 * sigma * active_counts_f ** (-0.2), 1e-3)

    row_index = np.arange(active_selected.shape[0])
    log_min = sorted_logs[:, 0]
    log_max = sorted_logs[row_index, active_counts - 1]
    grid = np.linspace(0.0, 1.0, _MODE_BOOTSTRAP_GRID_POINTS, dtype=float)[None, :]
    grid = log_min[:, None] + (log_max - log_min)[:, None] * grid

    density = np.zeros_like(grid)
    bytes_per_row_column = grid.shape[1] * np.dtype(float).itemsize
    row_chunk_size = max(1, min(grid.shape[0], max_kernel_bytes // bytes_per_row_column))
    for row_start in range(0, grid.shape[0], row_chunk_size):
        row_stop = min(row_start + row_chunk_size, grid.shape[0])
        row_slice = slice(row_start, row_stop)
        bytes_per_column = (row_stop - row_start) * bytes_per_row_column
        column_chunk_size = max(
            1,
            min(log_values.shape[1], max_kernel_bytes // bytes_per_column),
        )
        for start in range(0, log_values.shape[1], column_chunk_size):
            stop = start + column_chunk_size
            chunk_values = log_values[row_slice, start:stop]
            chunk_valid = active_valid[row_slice, start:stop]
            if not np.any(chunk_valid):
                continue
            kernel = grid[row_slice, :, None] - chunk_values[:, None, :]
            kernel /= bandwidth[row_slice, None, None]
            np.square(kernel, out=kernel)
            kernel *= -0.5
            np.exp(kernel, out=kernel)
            kernel *= chunk_valid[:, None, :]
            density[row_slice] += kernel.sum(axis=2)
    density /= active_counts_f[:, None]
    density_on_original_scale = density * np.exp(-grid)
    mode_index = np.argmax(density_on_original_scale, axis=1)
    summaries[multi_mask] = np.expm1(grid[row_index, mode_index])
    return summaries


def _evaluate_mode_bootstrap_column(
    backbone: BlockSummaryBootstrapBackbone,
    *,
    block_size: int,
    quantile: float,
) -> np.ndarray:
    """Evaluate one mode bootstrap column with an exact batched surrogate."""
    del quantile
    selected = _selected_bootstrap_maxima(backbone, block_size=block_size)
    return _evaluate_mode_bootstrap_column_batched(selected)


def evaluate_block_summary_bootstrap_backbone(
    backbone: BlockSummaryBootstrapBackbone | None,
    *,
    target: str = "quantile",
    quantile: float = 0.5,
) -> dict[str, Any]:
    """Evaluate one block-summary target on a precomputed bootstrap backbone."""
    if target not in {"quantile", "mean", "mode"}:
        raise ValueError(f"Unsupported target: {target}")
    resolved_quantile = _validate_quantile(quantile) if target == "quantile" else None
    if backbone is None:
        return {
            "block_sizes": np.asarray([], dtype=int),
            "samples": np.empty((0, 0)),
            "covariance": None,
            "target": target,
            "quantile": resolved_quantile,
            "sliding": None,
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
        "quantile": resolved_quantile,
        "invalid_replicates": int(np.sum(~valid_rows)),
    }


def build_block_summary_bootstrap_backbone(
    vec: np.ndarray | list[float],
    block_sizes: np.ndarray,
    *,
    sliding: bool = True,
    reps: int = 200,
    super_block_size: int | None = None,
    random_state: int | None = 0,
) -> BlockSummaryBootstrapBackbone | None:
    """Precompute the shared super-block state for UniBM bootstrap fitting."""
    warn_on_negative_values(vec, context="build_block_summary_bootstrap_backbone", stacklevel=3)
    arr = as_1d_float_array(vec)
    block_sizes = validate_block_sizes(block_sizes, n_obs=arr.size)
    if reps < 2:
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
    """Bootstrap multiple block-summary targets from one shared backbone."""
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
    """Bootstrap one block-summary target by resampling time-series super-blocks."""
    arr = as_1d_float_array(vec)
    block_sizes = validate_block_sizes(block_sizes, n_obs=arr.size)
    if target not in {"quantile", "mean", "mode"}:
        raise ValueError(f"Unsupported target: {target}")
    resolved_quantile = _validate_quantile(quantile) if target == "quantile" else None
    if reps < 2:
        return {
            "block_sizes": block_sizes,
            "samples": np.empty((0, block_sizes.size)),
            "covariance": None,
            "target": target,
            "quantile": resolved_quantile,
            "sliding": bool(sliding),
        }
    backbone = build_block_summary_bootstrap_backbone(
        vec=arr,
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
            "target": target,
            "quantile": resolved_quantile,
            "sliding": bool(sliding),
        }
    return evaluate_block_summary_bootstrap_backbone(
        backbone,
        target=target,
        quantile=quantile,
    )


__all__ = [
    "BlockSummaryBootstrapBackbone",
    "_evaluate_mode_bootstrap_column_batched",
    "build_block_summary_bootstrap_backbone",
    "circular_block_summary_bootstrap",
    "circular_block_summary_bootstrap_multi_target",
    "evaluate_block_summary_bootstrap_backbone",
]
