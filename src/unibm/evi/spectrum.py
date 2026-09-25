"""Spectrum-style xi comparator estimators for the EVI branch."""

from __future__ import annotations

import numpy as np

from .._validation import as_1d_float_array
from .tail import ExternalXiEstimate, _normalize_standard_error, _select_from_path


def candidate_max_spectrum_scales(
    n_obs: int,
    *,
    min_scale: int = 1,
    min_blocks: int = 2,
) -> np.ndarray:
    """Return integer exponents j with at least ``min_blocks`` blocks of size 2**j.

    The grid starts at ``min_scale`` and is empty when the series is too
    short. These are scale exponents, not block sizes themselves.
    """
    if n_obs < 2**min_scale:
        return np.empty(0, dtype=int)
    j_max = int(np.floor(np.log2(n_obs)))
    scales = np.arange(min_scale, j_max + 1, dtype=int)
    n_blocks = n_obs // (2**scales)
    return scales[n_blocks >= min_blocks]


def _validate_spectrum_series(sample: np.ndarray) -> np.ndarray:
    """Require a 1D finite series with at least eight observations.

    Keep temporal positions and nonpositive observations; positivity is
    checked later on each block maximum rather than on individual values.
    """
    vec = as_1d_float_array(sample)
    if not np.all(np.isfinite(vec)):
        raise ValueError("Max-spectrum requires every observation to be finite.")
    if vec.size < 8:
        raise ValueError("Max-spectrum requires at least eight observations.")
    return vec


def _validate_spectrum_scales(scales: np.ndarray, *, n_obs: int) -> np.ndarray:
    """Validate dyadic scale exponents against the observed series length."""
    try:
        raw = np.asarray(scales)
    except (TypeError, ValueError) as exc:
        raise ValueError("scales must be a one-dimensional numeric sequence.") from exc
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("scales must be a non-empty one-dimensional sequence.")
    if np.iscomplexobj(raw) or raw.dtype.kind == "b":
        raise ValueError("scales must contain finite integer values.")
    try:
        numeric = raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("scales must contain finite integer values.") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("scales must contain finite integer values.")
    if np.any(numeric < 0):
        raise ValueError("scales must be non-negative.")
    if np.any(np.diff(numeric) <= 0):
        raise ValueError("scales must be strictly increasing with no duplicates.")
    max_scale = int(np.floor(np.log2(n_obs // 2)))
    if np.any(numeric > max_scale):
        raise ValueError("scales must each provide at least two complete blocks.")
    return numeric.astype(int)


def _weighted_slope_with_se(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Fit an intercept and weighted slope with an HC1 residual sandwich SE.

    Inputs are aligned 1D arrays; discard nonfinite entries and nonpositive
    weights. Fewer than three retained points gives (NaN, NaN). The sandwich
    uses squared weights and a residual degrees-of-freedom correction;
    it does not model correlations between scale ordinates.
    """
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
    weighted_design = X.T * w
    bread = np.linalg.pinv(weighted_design @ X)
    beta = bread @ (weighted_design @ y)
    fitted = X @ beta
    resid = y - fitted
    meat = (weighted_design * (resid**2) * w) @ X
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
    """Return mean log2 maxima and complete-block counts for each dyadic scale.

    Keep original time order and drop each scale's incomplete tail. Every
    complete-block maximum must be positive, otherwise raise ValueError.
    """
    vec = _validate_spectrum_series(sample)
    scales = _validate_spectrum_scales(scales, n_obs=vec.size)
    y_values: list[float] = []
    n_blocks: list[int] = []
    for scale in scales:
        block_size = 2 ** int(scale)
        block_count = int(vec.size // block_size)
        trimmed = vec[: block_count * block_size].reshape(block_count, block_size)
        maxima = np.max(trimmed, axis=1)
        if np.any(maxima <= 0):
            raise ValueError("Max-spectrum block maxima must be strictly positive.")
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
    """Fit a weighted slope for each eligible suffix of the dyadic scale grid.

    Each suffix contains at least ``min_scale_count`` entries and ends at
    the largest scale; weights are complete-block counts. Return start
    scales, their slopes, and the common largest scale exponent.
    """
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
    """Estimate xi from weighted slopes of mean log2 block maxima versus scale.

    Preserve the finite series' time order and require positive maxima at
    every chosen dyadic scale. ``min_scale_count`` must be an integer at least
    three. Fit suffixes of at least ``min_scale_count`` scales, then choose
    a stable start-scale window and its lower-middle
    observed start. Return the selected slope, path, and a nominal 95% Wald
    interval using a scale-regression HC1 SE. That SE does not explicitly
    adjust for dependence between scales or start-scale selection.
    """
    if (
        isinstance(min_scale_count, (bool, np.bool_))
        or not isinstance(min_scale_count, (int, np.integer))
        or min_scale_count < 3
    ):
        raise ValueError("min_scale_count must be an integer at least 3.")
    min_scale_count = int(min_scale_count)
    vec = _validate_spectrum_series(sample)
    if scales is None:
        scales = candidate_max_spectrum_scales(vec.size, min_scale=1, min_blocks=2)
    scales = _validate_spectrum_scales(scales, n_obs=vec.size)
    y_values, n_blocks = _max_spectrum_curve(vec, scales)
    start_scales, xi_path, j_max = _max_spectrum_path(
        scales,
        y_values,
        n_blocks,
        min_scale_count=min_scale_count,
    )

    def se_fn(_: float, selected_level: int) -> float:
        """Recompute the selected suffix regression SE; the passed xi value is unused."""
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
    "_weighted_slope_with_se",
    "candidate_max_spectrum_scales",
    "estimate_max_spectrum_evi",
]
