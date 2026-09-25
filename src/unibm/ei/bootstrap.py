"""Bootstrap helpers for BM-based extremal-index path estimation."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.stats import rankdata

from .._block_grid import validate_block_sizes
from .._bootstrap_sampling import (
    _validate_circular_bootstrap_block_size,
    default_circular_bootstrap_block_size,
)
from .._bootstrap_precision import adaptive_covariance
from .._parallel import (
    BOOTSTRAP_WORKING_BYTES,
    bootstrap_executor,
    resolve_n_threads,
    validate_n_threads,
)
from .._validation import _validated_covariance_matrix, validate_covariance_shrinkage
from ..evi import _accelerator
from ._stats import EI_TINY, Z_CRIT_95, _log_scale_theta_interval
from ._validation import _validate_ei_series
from .bm import EI_DEFAULT_COVARIANCE_SHRINKAGE, _fit_pooled_z_model
from .paths import BM_PATH_KEYS, _build_bm_z_paths_from_values
from .selection import select_stable_path_window


def _resolve_ei_bootstrap_block_length(
    values: np.ndarray,
    *,
    base_path: str,
    bootstrap_block_length: int | None,
) -> tuple[str, int | None]:
    """Validate the base path and return a default or fixed raw resampling length.

    A fixed length must be an integer from 1 through the series length; ``None``
    defers default-length selection to the circular-bootstrap sampler. This is
    distinct from the BM block sizes at which the EI path is evaluated.
    """
    if base_path not in {"bb", "northrop"}:
        raise ValueError("base_path must be 'bb' or 'northrop'.")
    if bootstrap_block_length is None:
        return "default", None
    return "fixed", _validate_circular_bootstrap_block_size(
        bootstrap_block_length, n_obs=values.size, name="bootstrap_block_length"
    )


def _summarize_bm_ei_path_draws(
    z_draws: np.ndarray,
    *,
    block_sizes: np.ndarray,
    base_path: str,
    sliding: bool,
    bootstrap_block_length_policy: str,
    bootstrap_block_length: int,
    reps: int,
) -> dict[str, Any]:
    """Summarize an ``(n_draws, n_levels)`` matrix of log-reciprocal EI paths.

    Drop whole rows containing non-finite values. Return retained samples,
    level labels, path identity, and fixed-replicate metadata. The sample
    covariance is ``None`` if fewer than two complete draws remain.
    """
    valid_draws = np.asarray(z_draws, dtype=float)
    valid_draws = valid_draws[np.all(np.isfinite(valid_draws), axis=1)]
    covariance = (
        np.atleast_2d(np.cov(valid_draws, rowvar=False)) if valid_draws.shape[0] >= 2 else None
    )
    return {
        "block_sizes": np.asarray(block_sizes, dtype=int),
        "samples": valid_draws,
        "covariance": covariance,
        "base_path": base_path,
        "sliding": bool(sliding),
        "bootstrap_block_length_policy": bootstrap_block_length_policy,
        "bootstrap_block_length": int(bootstrap_block_length),
        "bootstrap_reps_requested": int(reps),
        "bootstrap_reps_used": int(valid_draws.shape[0]),
        "bootstrap_reps_policy": "fixed",
    }


def _paths_from_cdf(cdf, block_sizes, path_keys):
    """Batch the same window statistics, retaining NumPy's row reduction order."""
    cdf = np.clip(cdf, EI_TINY, 1.0 - EI_TINY)
    scores = {
        base: -np.log(cdf) if base == "northrop" else 1.0 - cdf
        for base in dict.fromkeys(base for base, _ in path_keys)
    }
    n = cdf.shape[1]
    # Short rows are already cheap in SciPy. Reuse native work buffers only
    # in the long-series regime that benefited in paired full-fit trials.
    native = _accelerator.kernels is not None and n >= 4096 and any(s for _, s in path_keys)
    if native:
        queue = np.empty(cdf.shape, dtype=np.int64)
        scaled_minima = np.empty(cdf.shape, dtype=float)
    draws = {}
    for base, sliding in path_keys:
        score = scores[base]
        path = np.empty((len(cdf), len(block_sizes)))
        for j, block_size in enumerate(block_sizes):
            b = int(block_size)
            if sliding and native:
                _accelerator.kernels.rolling_scaled_minimum(score, b, queue, scaled_minima)
                mean = np.mean(scaled_minima[:, : n - b + 1], axis=1)
            elif sliding:
                start = b // 2
                minima = minimum_filter1d(score, size=b, axis=1)[:, start : start + n - b + 1]
                mean = np.mean(float(b) * minima, axis=1)
            else:
                minima = score[:, : n // b * b].reshape(len(score), -1, b).min(axis=2)
                mean = np.mean(float(b) * minima, axis=1)
            # Both implementations multiply before NumPy's mean to preserve rounding.
            if base == "northrop":
                eir = np.maximum(mean, 1.0)
            else:
                theta = np.maximum(1.0 / np.maximum(mean, EI_TINY) - 1.0 / float(b), EI_TINY)
                eir = 1.0 / np.minimum(theta, 1.0)
            path[:, j] = np.log(eir)
        draws[base, sliding] = path
    return draws


@contextmanager
def _ei_path_sampler(values, block_sizes, *, base_path, sliding, length, n_threads):
    """Prepare observation codes once and own the pool across adaptive checkpoints.

    Integer multiplicities recover each resample's right-inclusive empirical
    ranks without sorting it again. Bootstrap RNG calls stay in the caller
    thread and retain the original per-replicate request size and order.
    """
    unique, inverse = np.unique(values, return_inverse=True)
    n, size = len(values), len(unique)
    threads = resolve_n_threads(n_threads, n_tasks=256, n_obs=n)
    blocks = (n + length - 1) // length
    offsets = np.arange(length)
    # Includes indices/codes, histogram, CDF, scores and window work arrays.
    batch_rows = max(1, BOOTSTRAP_WORKING_BYTES // (80 * n + 16 * size))
    path_key = (base_path, sliding)
    with bootstrap_executor(threads) as pool:

        def transform(codes):
            """Count ties before scoring; reductions stay aligned by resample."""
            row = np.arange(len(codes))[:, None]
            counts = np.bincount((codes + row * size).ravel(), minlength=len(codes) * size)
            counts = counts.reshape(-1, size).cumsum(axis=1)
            cdf = counts[row, codes].astype(float) / float(n + 1)
            return _paths_from_cdf(cdf, block_sizes, (path_key,))[path_key]

        def draw(count, rng):
            """Dispatch bounded row groups without ever drawing future checkpoints."""
            pieces = []
            for offset in range(0, count, batch_rows * threads):
                take = min(batch_rows * threads, count - offset)
                starts = np.stack([rng.integers(0, n, size=blocks) for _ in range(take)])
                indices = ((starts[:, :, None] + offsets) % n).reshape(take, -1)[:, :n]
                codes = inverse[indices]
                groups = np.array_split(codes, min(threads, take))
                results = pool.map(transform, groups) if pool else map(transform, groups)
                pieces.extend(results)
            return np.concatenate(pieces)

        yield draw


def bootstrap_bm_ei_path_draws(
    bootstrap_samples: np.ndarray,
    *,
    block_sizes: np.ndarray,
    allow_zeros: bool,
    path_keys: tuple[tuple[str, bool], ...] = BM_PATH_KEYS,
    n_threads: int | None = None,
) -> dict[tuple[str, bool], np.ndarray]:
    """Transform supplied resamples into BM-EI paths without generating new draws.

    ``bootstrap_samples`` is a 2D array with one series per row and at least 32
    observations per series. Values must be finite and positive, or non-negative
    when ``allow_zeros=True``; zeros retain their positions. ``block_sizes`` is
    an increasing integer grid from 2 through the number of observations.

    Return a dictionary keyed by requested ``(base_path, sliding)`` pairs, where
    ``base_path`` is ``"northrop"`` or ``"bb"``. Each value is an array of shape
    ``(n_draws, n_block_sizes)`` containing ``z = log(1 / theta)``. All four paths
    are returned by default. The caller controls the resampling design and clock.
    ``n_threads=None`` selects up to eight threads from CPUs and workload;
    a positive integer caps the pool, and 1 is serial. BLAS is not reconfigured.
    """
    validate_n_threads(n_threads)
    samples = np.asarray(bootstrap_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("bootstrap_samples must be a two-dimensional matrix.")
    block_sizes = validate_block_sizes(block_sizes, n_obs=samples.shape[1])
    for sample in samples:
        _validate_ei_series(sample, allow_zeros=allow_zeros)
    for base, _ in path_keys:
        if base not in {"northrop", "bb"}:
            raise KeyError(base)
    draws = {key: np.empty((len(samples), len(block_sizes))) for key in path_keys}
    threads = resolve_n_threads(n_threads, n_tasks=len(samples), n_obs=samples.shape[1])
    batch_rows = max(1, BOOTSTRAP_WORKING_BYTES // (samples.shape[1] * 96))

    def transform(rows):
        """Supplied banks may contain arbitrary values; rank each row with ties."""
        cdf = rankdata(rows, method="max", axis=1) / float(rows.shape[1] + 1)
        return _paths_from_cdf(cdf, block_sizes, path_keys)

    with bootstrap_executor(threads) as pool:
        for offset in range(0, len(samples), batch_rows * threads):
            batch = samples[offset : offset + batch_rows * threads]
            groups = np.array_split(batch, min(threads, len(batch)))
            results = pool.map(transform, groups) if pool else map(transform, groups)
            start = offset
            for group, result in zip(groups, results):
                for key in path_keys:
                    draws[key][start : start + len(group)] = result[key]
                start += len(group)
    return draws


def bootstrap_bm_ei_path(
    vec: np.ndarray | list[float],
    *,
    allow_zeros: bool,
    base_path: str,
    sliding: bool,
    block_sizes: np.ndarray,
    reps: int | Literal["adaptive"] = "adaptive",
    random_state: int | None = 0,
    bootstrap_block_length: int | None = None,
    covariance_shrinkage: float | None = None,
    n_threads: int | None = None,
) -> dict[str, Any]:
    """Bootstrap BM-EI covariance; default adaptive precision targets pooled theta and z.

    Resample contiguous circular blocks of the observed series. The raw
    resampling length defaults to ``min(n, max(16, round(sqrt(n))))`` and can be
    set with ``bootstrap_block_length``; it is separate from the increasing
    ``block_sizes`` grid used to evaluate the EI path. ``random_state`` seeds
    NumPy's generator (default 0); ``None`` requests non-reproducible seeding.
    ``n_threads=None`` chooses at most 8 threads from CPU/workload size;
    a positive integer caps this pool and 1 stays serial. It does not modify
    BLAS settings. Outer parallel callers should allocate the inner cap.
    Batch/thread choices preserve draws, path order and adaptive stopping.

    An explicit integer of at least two retains fixed-R sampling and rejects
    an explicit ``covariance_shrinkage``. That parameter only controls the
    pooled fit monitored by adaptive stopping; ``None`` resolves to 0.37.
    Neither mode shrinks the returned sample covariance. Adaptive precision
    is conditional on the original stable window and the monitoring shrinkage.
    Checkpoints are 128, 256, 512, 768, and 1024. The target vector includes theta
    and its CI endpoints plus the unconstrained z fit and endpoints, so the
    theta=1 boundary cannot hide Monte Carlo error. The cap retains the result
    with a warning and ``bootstrap_precision_met=False`` if tolerance is unmet.
    This flag measures Monte Carlo precision, not confidence-interval coverage.

    The returned in-memory dictionary contains full-grid covariance, path draws,
    block-size labels, estimator identity, and sampling/precision metadata. Pass
    it to ``estimate_pooled_bm_ei`` with matching data, path, and block scheme.
    Reusing adaptive precision metadata also requires matching the monitored
    shrinkage (default 0.37). This function does not write intermediate files.
    ``allow_zeros`` declares whether observed zeros are legal; non-finite inputs
    are always rejected rather than removed from the observation clock.
    """
    validate_n_threads(n_threads)
    values = _validate_ei_series(vec, allow_zeros=allow_zeros)
    block_sizes = validate_block_sizes(block_sizes, n_obs=values.size)
    block_length_policy, resolved_block_length = _resolve_ei_bootstrap_block_length(
        values,
        base_path=base_path,
        bootstrap_block_length=bootstrap_block_length,
    )
    length = resolved_block_length or default_circular_bootstrap_block_size(values.size)
    if reps == "adaptive":
        shrinkage = validate_covariance_shrinkage(
            EI_DEFAULT_COVARIANCE_SHRINKAGE
            if covariance_shrinkage is None
            else covariance_shrinkage
        )
        path_key = (base_path, sliding)
        observed_z = _build_bm_z_paths_from_values(values, block_sizes, path_keys=(path_key,))[
            path_key
        ]
        window, _ = select_stable_path_window(block_sizes, observed_z)
        mask = (block_sizes >= window.lo) & (block_sizes <= window.hi)
        levels, z_values = block_sizes[mask], observed_z[mask]
        selected_indices = np.ix_(np.flatnonzero(mask), np.flatnonzero(mask))
        design = np.ones((len(z_values), 1), dtype=float)

        def evaluate(covariance: np.ndarray, _rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return theta/z targets and their SE scales for covariance-precision checks."""
            # The sampler owns the fixed label order; retain full validation
            # and its roundoff correction before extracting the selected levels.
            validated, _ = _validated_covariance_matrix(
                covariance, context="EI bootstrap covariance"
            )
            model = _fit_pooled_z_model(
                z_values,
                covariance=validated[selected_indices],
                covariance_shrinkage=shrinkage,
                design=design,
                diagnostics=False,
            )
            z, se = model["intercept"], model["standard_error"]
            unconstrained = model["unconstrained_intercept"]
            theta = np.exp(-z)
            targets = np.asarray(
                [
                    theta,
                    *_log_scale_theta_interval(z, se),
                    unconstrained,
                    unconstrained - Z_CRIT_95 * se,
                    unconstrained + Z_CRIT_95 * se,
                ]
            )
            return targets, np.asarray([theta * se] * 3 + [se] * 3)

        with _ei_path_sampler(
            values,
            block_sizes,
            base_path=base_path,
            sliding=sliding,
            length=length,
            n_threads=n_threads,
        ) as draw:
            result = adaptive_covariance(draw, evaluate, random_state=random_state)
        return {
            **result,
            "block_sizes": block_sizes,
            "base_path": base_path,
            "sliding": bool(sliding),
            "bootstrap_block_length_policy": block_length_policy,
            "bootstrap_block_length": length,
            "bootstrap_precision_levels": levels.copy(),
            "bootstrap_precision_shrinkage": shrinkage,
            "bootstrap_mcse_targets": (
                "theta",
                "theta_ci_lo",
                "theta_ci_hi",
                "z_unconstrained",
                "z_ci_lo",
                "z_ci_hi",
            ),
        }
    if isinstance(reps, (bool, np.bool_)) or not isinstance(reps, (int, np.integer)) or reps < 2:
        raise ValueError("reps must be an integer at least 2 or 'adaptive'.")
    if covariance_shrinkage is not None:
        raise ValueError(
            "covariance_shrinkage only applies to adaptive monitoring; omit it for fixed reps."
        )
    with _ei_path_sampler(
        values,
        block_sizes,
        base_path=base_path,
        sliding=sliding,
        length=length,
        n_threads=n_threads,
    ) as draw:
        z_draws = draw(int(reps), np.random.default_rng(random_state))
    return _summarize_bm_ei_path_draws(
        z_draws,
        block_sizes=block_sizes,
        base_path=base_path,
        sliding=sliding,
        bootstrap_block_length_policy=block_length_policy,
        bootstrap_block_length=length,
        reps=reps,
    )
