"""Diagnostic helpers that sit beside, not inside, the core estimator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import ndtr

from ._validation import warn_on_negative_values
from ._diagnostic_models import ExtremalIndexReciprocalFit
from .evi.core import block_summary_curve

CdfMethod = Literal["kernel", "empirical"]
KernelName = Literal["gaussian"]
BandwidthSpec = Literal["scott", "silverman"] | float


def _as_finite_1d(vec: np.ndarray | list[float]) -> np.ndarray:
    """Return a one-dimensional finite float array."""
    arr = np.asarray(vec, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _empty_cdf_estimator() -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return a CDF estimator that yields NaN on every query."""

    def estimate_empty(q: float | np.ndarray) -> float | np.ndarray:
        q_arr = np.asarray(q, dtype=float)
        result = np.full(q_arr.shape, np.nan, dtype=float)
        return float(result.item()) if result.ndim == 0 else result

    return estimate_empty


def _singleton_cdf_estimator(point: float) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return the exact CDF of a single-point empirical distribution."""

    def estimate_single(q: float | np.ndarray) -> float | np.ndarray:
        q_arr = np.asarray(q, dtype=float)
        result = np.where(q_arr >= point, 1.0, 0.0)
        return float(result.item()) if np.ndim(result) == 0 else result

    return estimate_single


def _kernel_bandwidth(arr: np.ndarray, bandwidth: BandwidthSpec) -> float:
    """Resolve a kernel-CDF bandwidth from a rule name or explicit value."""
    if isinstance(bandwidth, str):
        if bandwidth == "scott":
            iqr = np.diff(np.nanquantile(arr, q=(0.25, 0.75), method="median_unbiased")) / 1.349
            value = 1.059 * min(np.nanstd(arr), float(iqr[0])) * arr.size ** (-0.2)
        elif bandwidth == "silverman":
            value = np.nanstd(arr) * 0.6973425390765554 * arr.size ** (-0.1111111111111111)
        else:
            raise ValueError(
                "Unsupported bandwidth rule. Expected 'scott', 'silverman', or a positive float."
            )
    else:
        value = float(bandwidth)
    return max(float(value), 1e-6)


def kernel_cdf(
    vec: np.ndarray | list[float],
    *,
    kernel: KernelName = "gaussian",
    bandwidth: BandwidthSpec = "scott",
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return a kernel-smoothed marginal CDF estimator.

    The current implementation supports a Gaussian kernel with either a
    Scott-style or Silverman-style rule-of-thumb bandwidth, or a user-supplied
    positive scalar bandwidth.
    """
    if kernel != "gaussian":
        raise ValueError("Only the Gaussian kernel is currently supported.")
    arr = _as_finite_1d(vec)
    if arr.size == 0:
        return _empty_cdf_estimator()
    if arr.size == 1:
        return _singleton_cdf_estimator(float(arr[0]))
    resolved_bandwidth = _kernel_bandwidth(arr, bandwidth)

    target_matrix_elements = 1_000_000
    arr_chunk_size = min(arr.size, 4_096)

    def estimate(q: float | np.ndarray) -> float | np.ndarray:
        q_arr = np.asarray(q, dtype=float)
        q_flat = q_arr.reshape(-1)
        out = np.empty_like(q_flat, dtype=float)
        q_chunk_size = max(1, min(q_flat.size, target_matrix_elements // max(arr_chunk_size, 1)))
        for q_start in range(0, q_flat.size, q_chunk_size):
            q_chunk = q_flat[q_start : q_start + q_chunk_size]
            cumulative = np.zeros(q_chunk.size, dtype=float)
            for arr_start in range(0, arr.size, arr_chunk_size):
                arr_chunk = arr[arr_start : arr_start + arr_chunk_size]
                cumulative += ndtr(
                    (q_chunk[:, None] - arr_chunk[None, :]) / resolved_bandwidth
                ).sum(axis=1)
            out[q_start : q_start + q_chunk.size] = cumulative / arr.size
        result = out.reshape(q_arr.shape)
        return float(result.item()) if result.ndim == 0 else result

    return estimate


def empirical_cdf(
    vec: np.ndarray | list[float],
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return an empirical marginal CDF estimator based on scaled ranks.

    We use `count / (n + 1)` so evaluations at sample points stay strictly below
    one, which is convenient for extremal-index transforms involving logs.
    """
    arr = np.sort(_as_finite_1d(vec))
    if arr.size == 0:
        return _empty_cdf_estimator()
    if arr.size == 1:
        return _singleton_cdf_estimator(float(arr[0]))
    normalizer = float(arr.size + 1)

    def estimate(q: float | np.ndarray) -> float | np.ndarray:
        q_arr = np.asarray(q, dtype=float)
        q_flat = q_arr.reshape(-1)
        counts = np.searchsorted(arr, q_flat, side="right").astype(float)
        result = (counts / normalizer).reshape(q_arr.shape)
        return float(result.item()) if result.ndim == 0 else result

    return estimate


def marginal_cdf(
    vec: np.ndarray | list[float],
    *,
    method: CdfMethod = "kernel",
    kernel: KernelName = "gaussian",
    bandwidth: BandwidthSpec = "scott",
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Build a marginal CDF estimator for block-based extremal-index diagnostics."""
    if method == "kernel":
        return kernel_cdf(vec, kernel=kernel, bandwidth=bandwidth)
    if method == "empirical":
        return empirical_cdf(vec)
    raise ValueError("Unsupported marginal CDF method. Expected 'kernel' or 'empirical'.")


def _rolling_min_summary(
    scores: pd.Series, block_sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize rolling block minima over a candidate block-size grid."""
    means = np.full(block_sizes.size, np.nan, dtype=float)
    standard_deviations = np.full(block_sizes.size, np.nan, dtype=float)
    for idx, size in enumerate(np.asarray(block_sizes, dtype=int)):
        rolling_min = scores.rolling(
            window=int(size),
            min_periods=int(size),
            step=int(np.ceil(size**0.3)),
        ).min()
        # `min_periods=size` intentionally leaves the first windows undefined;
        # the downstream nan-aware summaries treat those as structural edge effects.
        values = float(size) * rolling_min.to_numpy(dtype=float)
        means[idx] = float(np.nanmean(values))
        standard_deviations[idx] = float(np.nanstd(values, ddof=1))
    return means, standard_deviations


def _select_stable_sd_index(standard_deviations: np.ndarray) -> int:
    """Choose the most stable block size by minimizing finite standard deviation."""
    sd = np.asarray(standard_deviations, dtype=float)
    finite = np.isfinite(sd)
    if not np.any(finite):
        raise ValueError("No finite standard deviations were available for block-size selection.")
    return int(np.flatnonzero(finite)[np.argmin(sd[finite])])


def _quantile_summary_label(quantile: float) -> str:
    """Return a human-readable label for a quantile target column."""
    if np.isclose(float(quantile), 0.5):
        return "median"
    return f"quantile_tau_{float(quantile):.2f}"


def estimate_extremal_index_reciprocal(
    series: pd.Series,
    *,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    geom: bool | None = None,
    cdf_method: CdfMethod = "kernel",
    kernel: KernelName = "gaussian",
    bandwidth: BandwidthSpec = "scott",
) -> ExtremalIndexReciprocalFit:
    """Estimate reciprocal-EI diagnostics over a block-size grid.

    Parameters
    ----------
    series
        One-dimensional time-indexed series. The function keeps finite
        non-negative values, applies a ``log1p`` transform, and then builds
        rolling-minimum diagnostics over a block-size grid.
    num_step, min_block_size, max_block_size
        Controls for the candidate block-size grid used by the diagnostic path.
        If omitted, the grid is chosen adaptively from the sample size.
    geom
        If ``True``, use a geometric block-size grid. If ``False``, use an
        arithmetic grid. If omitted, the function chooses geometrically for
        larger series.
    cdf_method
        Marginal CDF estimator used to transform the series before the EI
        diagnostics. Supported values are ``"kernel"`` and ``"empirical"``.
    kernel, bandwidth
        Kernel-CDF options used only when ``cdf_method="kernel"``.

    Returns
    -------
    ExtremalIndexReciprocalFit
        Diagnostic path object containing Northrop and BB reciprocal-EI curves,
        their empirical standard deviations, and the selected block sizes.

    Notes
    -----
    This function is exploratory: it returns diagnostic reciprocal-EI curves
    rather than the formal benchmark/application EI estimators. For the formal
    threshold or pooled block-maxima EI estimators, see :mod:`unibm.ei`.
    """
    raw = np.asarray(series.values, dtype=float)
    warn_on_negative_values(raw, context="estimate_extremal_index_reciprocal", stacklevel=3)
    valid_mask = np.isfinite(raw) & (raw >= 0)
    filtered_index = series.index[valid_mask]
    filtered_values = raw[valid_mask]
    if filtered_values.size < 2:
        raise ValueError(
            "estimate_extremal_index_reciprocal requires at least two finite non-negative observations."
        )
    vec = np.log1p(filtered_values)
    vec -= np.nanmin(vec)
    n_obs = vec.size + 1
    if min_block_size is None:
        min_block_size = int(max(10, np.exp(np.log(n_obs) * 0.2857142857142857)))
    if max_block_size is None:
        max_block_size = int(np.exp(np.log(n_obs) * 0.6666666666666666))
    max_block_size = max(min_block_size, max_block_size)
    if geom is None:
        geom = n_obs > 5_000
    if geom:
        num_step = int(min(128, np.log(max_block_size) * 100)) if num_step is None else num_step
        block_sizes = np.unique(
            np.geomspace(min_block_size, max_block_size, num=num_step, endpoint=False).astype(int)
        )
    else:
        if num_step is None:
            num_step = max_block_size if max_block_size < 5_000 else max_block_size // 2
        block_sizes = np.arange(
            min_block_size,
            max_block_size + 1,
            max(1, max_block_size // num_step),
        ).astype(int)
    if block_sizes.size == 0:
        raise ValueError(
            "estimate_extremal_index_reciprocal could not construct any admissible block sizes."
        )
    cdf_estimator = marginal_cdf(
        vec,
        method=cdf_method,
        kernel=kernel,
        bandwidth=bandwidth,
    )
    cdf_values = np.asarray(cdf_estimator(vec), dtype=float)
    tiny = np.finfo(float).tiny
    cdf_values = np.clip(cdf_values, tiny, 1.0 - tiny)
    poisson_scores = -pd.Series(np.log(cdf_values), index=filtered_index)

    northrop_values, northrop_sd = _rolling_min_summary(poisson_scores, block_sizes)
    northrop_values = northrop_values.clip(min=1)
    northrop_idx = _select_stable_sd_index(northrop_sd)
    exceedance_scores = pd.Series(1 - cdf_values, index=filtered_index)
    bb_values, bb_sd = _rolling_min_summary(exceedance_scores, block_sizes)
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_values = np.reciprocal(np.reciprocal(bb_values) - 1 / block_sizes).clip(min=1)
    bb_idx = _select_stable_sd_index(bb_sd)
    return ExtremalIndexReciprocalFit(
        block_sizes=block_sizes,
        log_block_sizes=np.log(block_sizes),
        northrop_values=northrop_values,
        northrop_standard_deviations=northrop_sd,
        bb_values=bb_values,
        bb_standard_deviations=bb_sd,
        northrop_block_size=int(block_sizes[northrop_idx]),
        northrop_estimate=float(northrop_values[northrop_idx]),
        bb_block_size=int(block_sizes[bb_idx]),
        bb_estimate=float(bb_values[bb_idx]),
    )


def target_stability_summary(
    vec: np.ndarray | list[float],
    block_sizes: np.ndarray,
    *,
    sliding: bool = True,
    quantile: float = 0.5,
) -> pd.DataFrame:
    """Compare quantile, mean, and mode block-maxima summaries on the same grid."""
    out = {"block_size": np.asarray(block_sizes, dtype=int)}
    for target in ["quantile", "mean", "mode"]:
        curve = block_summary_curve(
            vec,
            block_sizes,
            sliding=sliding,
            quantile=quantile,
            target=target,
        )
        key = _quantile_summary_label(quantile) if target == "quantile" else target
        out[key] = curve.values
    return pd.DataFrame(out)
