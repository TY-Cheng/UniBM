"""Bootstrap helpers for covariance-aware EVI block-summary regression."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from .._block_grid import validate_block_sizes
from .._bootstrap_precision import ADAPTIVE_REPS, adaptive_covariance
from .._validation import as_1d_float_array, warn_on_negative_values
from .._window_ops import circular_sliding_window_maximum
from .._parallel import (
    BOOTSTRAP_WORKING_BYTES,
    bootstrap_executor,
    resolve_n_threads,
    validate_n_threads,
)
from ._quantile_bootstrap import (
    prepare_quantile_counts,
    quantile_from_counts,
    segment_multiplicities,
)
from .summaries import _validate_quantile
from ._mode import prepare_mode_counts, mode_from_counts


_MODE_BOOTSTRAP_GRID_POINTS = 256
_MODE_BOOTSTRAP_MAX_WORKING_BYTES = 64 * 1024 * 1024


def _disjoint_block_maxima(segment: np.ndarray, block_size: int) -> np.ndarray:
    """Return nonoverlapping segment maxima, discarding an incomplete final block."""
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
    """Return circular sliding or complete disjoint maxima within one segment.

    Circular windows wrap inside this segment, never across the boundary
    between independently resampled super-blocks.
    """
    if sliding:
        return circular_sliding_window_maximum(segment, block_size)
    return _disjoint_block_maxima(segment, block_size)


@dataclass(frozen=True)
class BlockSummaryBootstrapBackbone:
    """Cached maxima and common segment draws for resampling summary targets.

    ``segment_draws`` has shape (replicates, segments). Each maxima bank has
    shape (segments, maxima_per_segment) for its block size. Reusing the
    backbone makes different summary targets share the same resampled data.
    """

    block_sizes: np.ndarray
    sliding: bool
    super_block_size: int
    segment_draws: np.ndarray
    maxima_by_block: dict[int, np.ndarray]


@contextmanager
def _summary_evaluator(backbone, *, target, quantile, n_threads, mode_batch=None):
    """Own one call's count tables and pool; map deterministic segment draws.

    NumPy/SciPy work and optional GIL-free kernels run in threads. The caller
    generates every draw before dispatch, so results and adaptive stopping
    do not depend on scheduling. Tables share a bounded budget; temporaries are batched
    per worker. One complete maxima row is the irreducible working set.
    """
    banks = [
        np.asarray(backbone.maxima_by_block[int(b)], dtype=float) for b in backbone.block_sizes
    ]
    n_obs = backbone.segment_draws.shape[1] * backbone.super_block_size
    threads = resolve_n_threads(n_threads, n_tasks=len(banks), n_obs=n_obs)
    tables = []
    remaining = BOOTSTRAP_WORKING_BYTES
    for bank in banks:
        # NumPy's rank arithmetic retains lower-precision q dtypes; keep that
        # established behavior in the fallback instead of promoting those q's.
        table = (
            prepare_quantile_counts(bank, max_bytes=remaining)
            if target == "quantile" and np.asarray(quantile).dtype == np.dtype(float)
            else None
        )
        if target == "mode":
            table = prepare_mode_counts(bank, max_bytes=remaining)
        tables.append(table)
        if table is not None:
            remaining -= table[0].nbytes + table[1].nbytes
    with bootstrap_executor(threads) as pool:

        def evaluate(draws):
            """Return one summary per draw/level without computing covariance."""
            weights = segment_multiplicities(draws) if any(t is not None for t in tables) else None

            def column(index):
                """Limit expanded maxima and mode temporaries independently of R."""
                bank, table = banks[index], tables[index]
                if table is not None and target == "mode":
                    values, segment_counts = table
                    rows = max(1, BOOTSTRAP_WORKING_BYTES // max(1, len(values) * 8 * 8))
                    return (
                        np.concatenate(
                            [
                                mode_from_counts(
                                    values, weights[start : start + rows] @ segment_counts
                                )
                                for start in range(0, len(draws), rows)
                            ]
                        )
                        if len(draws)
                        else np.empty(0)
                    )
                if table is not None:
                    return quantile_from_counts(
                        table,
                        weights,
                        size=bank.size,
                        quantile=quantile,
                        max_bytes=BOOTSTRAP_WORKING_BYTES,
                    )
                # Quantile may copy its input; mode retains several work arrays.
                rows = max(1, BOOTSTRAP_WORKING_BYTES // max(1, bank.size * 8 * 8))
                pieces = []
                group_rows = (mode_batch or max(1, len(draws))) if target == "mode" else rows
                for group_start in range(0, len(draws), group_rows):
                    group = draws[group_start : group_start + group_rows]
                    if target == "mode":
                        # Preserve the old KDE sum grouping even when selections
                        # are split into smaller batches. Singletons do not enter KDE.
                        counts = np.sum(np.isfinite(bank) & (bank > 0), axis=1)
                        active = np.sum(counts[group], axis=1) > 1
                        total_active = int(np.sum(active))
                    for offset in range(0, len(group), rows):
                        selected = bank[group[offset : offset + rows]].reshape(-1, bank.size)
                        if target == "quantile":
                            values = np.quantile(
                                selected,
                                quantile,
                                axis=1,
                                method="median_unbiased",
                                overwrite_input=True,
                            )
                        elif target == "mean":
                            values = np.mean(selected, axis=1)
                        else:
                            values = _evaluate_mode_bootstrap_column_batched(
                                selected,
                                reduction_rows=total_active,
                                reduction_offset=int(np.sum(active[:offset])),
                            )
                        pieces.append(values)
                return np.concatenate(pieces) if pieces else np.empty(0)

            columns = (
                pool.map(column, range(len(banks))) if pool else map(column, range(len(banks)))
            )
            return np.column_stack(list(columns))

        yield evaluate


def _log_bootstrap_summaries(summaries, *, warning_batch=None):
    """Log positive finite summaries and omit incomplete rows, retaining warnings."""
    valid = np.isfinite(summaries) & (summaries > 0)
    samples = np.full(summaries.shape, np.nan)
    np.log(summaries, out=samples, where=valid)
    step = max(1, len(samples)) if warning_batch is None else warning_batch
    for offset in range(0, len(samples), step):
        invalid_count = int(np.sum(~valid[offset : offset + step]))
        if invalid_count:
            warnings.warn(
                "evaluate_block_summary_bootstrap_backbone excluded "
                f"{invalid_count} non-positive bootstrap block summaries. "
                "This step requires strictly positive inputs.",
                RuntimeWarning,
                stacklevel=3,
            )
    return samples[np.all(valid, axis=1)]


def _adaptive_block_summary_bootstrap(
    vec: np.ndarray,
    block_sizes: np.ndarray,
    *,
    target: str,
    quantile: float,
    sliding: bool,
    super_block_size: int | None,
    random_state: int | None,
    evaluate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    n_threads: int | None = None,
) -> dict[str, Any]:
    """Grow log-summary bootstrap samples until MC precision or the cap is reached.

    Cache segment maxima once, draw additional rows in batches, and pass
    covariance estimates and their paired rows to ``evaluate`` for the monitored statistics and
    SE scales. If too few super-blocks exist, return empty samples and no
    covariance; adaptive precision does not certify statistical coverage.
    """
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

    with _summary_evaluator(
        backbone, target=target, quantile=quantile, n_threads=n_threads, mode_batch=32
    ) as summaries:

        def draw(reps: int, rng: np.random.Generator) -> np.ndarray:
            """Keep the original 32-row RNG requests and invalid-row policy."""
            n_super = backbone.segment_draws.shape[1]
            draws = np.concatenate(
                [
                    rng.integers(0, n_super, (min(32, reps - offset), n_super))
                    for offset in range(0, reps, 32)
                ]
            )
            return _log_bootstrap_summaries(summaries(draws), warning_batch=32)

        result = adaptive_covariance(draw, evaluate, random_state=random_state)
    return {**identity, **result, "super_block_size": backbone.super_block_size}


def _rowwise_linear_quantile(
    sorted_rows: np.ndarray,
    counts: np.ndarray,
    *,
    quantile: float,
) -> np.ndarray:
    """Interpolate a quantile within each sorted row's first ``counts`` entries.

    Trailing padding is ignored; a zero count returns NaN. This uses linear
    interpolation for KDE bandwidth estimation, not the median-unbiased
    quantiles used by the block-quantile estimator.
    """
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
    reduction_rows: int | None = None,
    reduction_offset: int = 0,
) -> np.ndarray:
    """Compute a positive-maxima KDE mode surrogate separately for each matrix row.

    Use the same log1p transform, bandwidth rule, 256-point grid, and
    original-scale Jacobian as ``estimate_sample_mode``. Empty positive rows
    return NaN; singleton rows return their observation. ``max_kernel_bytes``
    bounds each KDE kernel temporary, not total process memory. Internal
    ``reduction_rows`` / ``reduction_offset`` preserve the original column-sum
    grouping when the caller splits a larger selection into row batches.
    """
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
    original_rows = grid.shape[0] if reduction_rows is None else reduction_rows
    original_chunk = max(1, min(original_rows, max_kernel_bytes // bytes_per_row_column))
    row_start = 0
    while row_start < grid.shape[0]:
        position = reduction_offset + row_start
        group_start = position // original_chunk * original_chunk
        group_rows = min(original_chunk, original_rows - group_start)
        row_stop = min(row_start + group_start + group_rows - position, grid.shape[0])
        row_slice = slice(row_start, row_stop)
        bytes_per_column = group_rows * bytes_per_row_column
        column_chunk_size = max(1, min(log_values.shape[1], max_kernel_bytes // bytes_per_column))
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
        row_start = row_stop
    density /= active_counts_f[:, None]
    # Convert transformed KDE density back to the original response scale.
    density_on_original_scale = density * np.exp(-grid)
    mode_index = np.argmax(density_on_original_scale, axis=1)
    summaries[multi_mask] = np.expm1(grid[row_index, mode_index])
    return summaries


def evaluate_block_summary_bootstrap_backbone(
    backbone: BlockSummaryBootstrapBackbone | None,
    *,
    target: str = "quantile",
    quantile: float = 0.5,
    n_threads: int | None = None,
) -> dict[str, Any]:
    """Return log-summary samples and covariance from cached segment draws.

    For ``target`` quantile, mean, or mode, evaluate every block-size column
    on the same replicates. Drop a whole replicate if any summary is
    nonfinite or nonpositive. ``samples`` is (valid_replicates, block_sizes);
    ``covariance`` is its sample covariance, or None with fewer than two
    valid rows. A None backbone returns empty arrays and no covariance.
    ``n_threads=None`` selects up to eight threads from CPUs and workload;
    a positive integer caps the pool, and 1 is serial. BLAS is not reconfigured.
    """
    validate_n_threads(n_threads)
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
    with _summary_evaluator(
        backbone, target=target, quantile=quantile, n_threads=n_threads
    ) as summaries:
        valid_samples = _log_bootstrap_summaries(summaries(backbone.segment_draws))
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
        "invalid_replicates": int(reps - len(valid_samples)),
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
    """Cache segment maxima and draw segment indices with replacement.

    Split the 1D series into equal complete super-blocks, discarding the
    incomplete tail. Sliding maxima wrap within each segment; disjoint
    maxima discard each segment's incomplete block. The default length is
    ``max(2 * B, floor(sqrt(N)))``, where B is the largest supplied block size
    and N is the series length. This length is used without adjustment. An explicit
    ``super_block_size`` must be an integer above B, no greater than N, and
    allow at least two complete segments; it is never adjusted.
    Return None if an automatic length yields fewer than two segments or
    ``reps < 2``. A fixed ``random_state`` reproduces segment draws.
    """
    warn_on_negative_values(vec, context="build_block_summary_bootstrap_backbone", stacklevel=3)
    arr = as_1d_float_array(vec)
    block_sizes = validate_block_sizes(block_sizes, n_obs=arr.size)
    max_block_size = int(block_sizes.max())
    if super_block_size is None:
        super_block_size = max(max_block_size * 2, int(np.sqrt(arr.size)))
        n_super = arr.size // super_block_size
    else:
        if (
            isinstance(super_block_size, (bool, np.bool_))
            or not isinstance(super_block_size, (int, np.integer))
            or not max_block_size < int(super_block_size) <= arr.size
        ):
            raise ValueError(
                "super_block_size must be an integer above the largest block size "
                "and no greater than n_obs."
            )
        super_block_size = int(super_block_size)
        n_super = arr.size // super_block_size
        if n_super < 2:
            raise ValueError("super_block_size must allow at least two complete super-blocks.")
    if n_super < 2 or reps < 2:
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
    n_threads: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Bootstrap targets using identical segment draws and cached maxima.

    Return a dictionary keyed by target (quantile, mean, or mode), with each
    value following ``evaluate_block_summary_bootstrap_backbone``. Targets
    may retain different replicate counts because invalid log summaries
    are removed separately for each target.
    ``n_threads`` follows the backbone evaluator's per-call thread budget.
    """
    validate_n_threads(n_threads)
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
            n_threads=n_threads,
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
    n_threads: int | None = None,
) -> dict[str, Any]:
    """Estimate log-summary covariance by resampling time-series super-blocks.

    Return valid log-summary rows, covariance, block-size labels, and target
    metadata. ``sliding=True`` wraps windows within each original segment;
    False uses disjoint maxima. Invalid log-summary rows are removed jointly
    across scales. Fewer than two requested draws or usable segments gives
    empty samples and no covariance. See the backbone builder for how the
    automatic super-block lengths are chosen and explicit lengths validated.
    ``n_threads`` follows the backbone evaluator's per-call thread budget.
    """
    validate_n_threads(n_threads)
    arr = as_1d_float_array(vec)
    block_sizes = validate_block_sizes(block_sizes, n_obs=arr.size)
    if target not in {"quantile", "mean", "mode"}:
        raise ValueError(f"Unsupported target: {target}")
    resolved_quantile = _validate_quantile(quantile) if target == "quantile" else None
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
        n_threads=n_threads,
    )


__all__ = [
    "BlockSummaryBootstrapBackbone",
    "_evaluate_mode_bootstrap_column_batched",
    "build_block_summary_bootstrap_backbone",
    "circular_block_summary_bootstrap",
    "circular_block_summary_bootstrap_multi_target",
    "evaluate_block_summary_bootstrap_backbone",
]
