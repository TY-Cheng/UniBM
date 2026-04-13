"""Spectrum-style xi comparator estimators for the EVI branch."""

from __future__ import annotations

import numpy as np

from .._validation import positive_finite_values
from .tail import ExternalXiEstimate, _normalize_standard_error, _select_from_path


def candidate_max_spectrum_scales(
    n_obs: int,
    *,
    min_scale: int = 1,
    min_blocks: int = 2,
) -> np.ndarray:
    """Construct dyadic block-size scales for max-spectrum estimation."""
    if n_obs < 2**min_scale:
        return np.empty(0, dtype=int)
    j_max = int(np.floor(np.log2(n_obs)))
    scales = np.arange(min_scale, j_max + 1, dtype=int)
    n_blocks = n_obs // (2**scales)
    return scales[n_blocks >= min_blocks]


def _positive_finite_in_order(sample: np.ndarray) -> np.ndarray:
    """Return positive finite observations in their original time order."""
    return positive_finite_values(
        sample,
        context="spectrum xi estimators",
        minimum_size=8,
        stacklevel=3,
    )


def _weighted_slope_with_se(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Return the weighted slope and HC1-style sandwich SE in one dimension."""
    w = np.asarray(weights, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x = x[mask]
    y = y[mask]
    w = w[mask]
    if x.size != y.size or x.size != w.size or x.size < 3:
        return float("nan"), float("nan")
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    bread = np.linalg.pinv(X.T @ W @ X)
    beta = bread @ (X.T @ W @ y)
    fitted = X @ beta
    resid = y - fitted
    meat = X.T @ W @ np.diag(resid**2) @ W @ X
    cov_beta = bread @ meat @ bread
    if x.size > X.shape[1]:
        cov_beta *= x.size / (x.size - X.shape[1])
    slope = float(beta[1])
    standard_error = _normalize_standard_error(np.sqrt(max(float(cov_beta[1, 1]), 0.0)))
    return slope, standard_error


def _max_spectrum_curve(
    sample: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the max-spectrum ordinates and effective block counts."""
    vec = _positive_finite_in_order(sample)
    y_values: list[float] = []
    n_blocks: list[int] = []
    for scale in np.asarray(scales, dtype=int):
        block_size = 2 ** int(scale)
        block_count = int(vec.size // block_size)
        if block_count <= 0:
            y_values.append(np.nan)
            n_blocks.append(0)
            continue
        trimmed = vec[: block_count * block_size].reshape(block_count, block_size)
        maxima = np.max(trimmed, axis=1)
        y_values.append(float(np.mean(np.log2(maxima))))
        n_blocks.append(block_count)
    return np.asarray(y_values, dtype=float), np.asarray(n_blocks, dtype=int)


def _max_spectrum_path(
    scales: np.ndarray,
    y_values: np.ndarray,
    n_blocks: np.ndarray,
    *,
    min_scale_count: int = 3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Construct the start-scale path for the max-spectrum slope estimator."""
    if scales.size < min_scale_count:
        raise ValueError("Max-spectrum requires at least three usable dyadic scales.")
    j_max = int(scales[-1])
    start_scales: list[int] = []
    xi_path: list[float] = []
    for start_idx in range(0, scales.size - min_scale_count + 1):
        use_scales = scales[start_idx:]
        use_y = y_values[start_idx:]
        use_w = n_blocks[start_idx:]
        slope, _ = _weighted_slope_with_se(use_scales, use_y, use_w)
        start_scales.append(int(use_scales[0]))
        xi_path.append(float(slope))
    return np.asarray(start_scales, dtype=int), np.asarray(xi_path, dtype=float), j_max


def estimate_max_spectrum_evi(
    sample: np.ndarray,
    *,
    scales: np.ndarray | None = None,
    min_scale_count: int = 3,
) -> ExternalXiEstimate:
    """Estimate ``xi`` with the dependent max-spectrum estimator."""
    vec = _positive_finite_in_order(sample)
    if scales is None:
        scales = candidate_max_spectrum_scales(vec.size, min_scale=1, min_blocks=2)
    scales = np.asarray(scales, dtype=int)
    y_values, n_blocks = _max_spectrum_curve(vec, scales)
    start_scales, xi_path, j_max = _max_spectrum_path(
        scales,
        y_values,
        n_blocks,
        min_scale_count=min_scale_count,
    )

    def se_fn(_: float, selected_level: int) -> float:
        matching = np.flatnonzero(scales == selected_level)
        if matching.size != 1:
            return float("nan")
        _, standard_error = _weighted_slope_with_se(
            scales[matching[0] :],
            y_values[matching[0] :],
            n_blocks[matching[0] :],
        )
        return standard_error

    return _select_from_path(
        "max_spectrum_raw",
        start_scales,
        xi_path,
        se_fn=se_fn,
        tuning_axis="scale_start",
        fixed_upper_level=j_max,
        selection_min_window=3,
    )


__all__ = [
    "_max_spectrum_curve",
    "_max_spectrum_path",
    "_positive_finite_in_order",
    "_weighted_slope_with_se",
    "candidate_max_spectrum_scales",
    "estimate_max_spectrum_evi",
]
