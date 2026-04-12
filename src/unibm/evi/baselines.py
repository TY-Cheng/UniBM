"""Canonical public baseline estimators for the xi/EVI branch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from .._validation import positive_finite_values


@dataclass(frozen=True)
class SelectionWindow:
    """Selected stable window on one integer-indexed estimator path."""

    lo: int
    hi: int


ThresholdWindow = SelectionWindow


@dataclass(frozen=True)
class ExternalXiEstimate:
    """Point estimate plus diagnostic path information for baseline estimators."""

    method: str
    xi_hat: float
    selected_level: int | None
    stable_window: SelectionWindow | None
    path_level: tuple[int, ...]
    path_xi: tuple[float, ...]
    standard_error: float = np.nan
    confidence_interval: tuple[float, float] = (np.nan, np.nan)
    ci_method: str = "asymptotic"
    tuning_axis: str = "k"
    fixed_upper_level: int | None = None

    @property
    def selected_k(self) -> int | None:
        """Backward-compatible alias for threshold-based methods."""
        return self.selected_level

    @property
    def path_k(self) -> tuple[int, ...]:
        """Backward-compatible alias for threshold-based methods."""
        return self.path_level


def wald_confidence_interval(
    xi_hat: float,
    standard_error: float,
    *,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Construct a Gaussian/Wald confidence interval around one xi estimate."""
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must lie strictly between 0 and 1.")
    if not np.isfinite(xi_hat) or not np.isfinite(standard_error) or standard_error < 0:
        return (float("nan"), float("nan"))
    z = NormalDist().inv_cdf(0.5 + ci_level / 2.0)
    margin = float(z * standard_error)
    return (float(xi_hat - margin), float(xi_hat + margin))


def candidate_tail_counts(
    n_obs: int,
    *,
    min_count: int = 8,
    max_fraction: float = 0.25,
    num: int = 24,
) -> np.ndarray:
    """Construct a log-spaced threshold grid for raw-sample tail estimators."""
    upper = int(min(max(min_count + 2, int(np.floor(max_fraction * n_obs))), n_obs - 3))
    lower = int(min(min_count, upper))
    if upper <= lower:
        return np.array([lower], dtype=int)
    grid = np.unique(np.round(np.geomspace(lower, upper, num=num)).astype(int))
    return grid[(grid >= lower) & (grid <= upper)]


def _finite_positive(sample: np.ndarray) -> np.ndarray:
    """Return positive finite observations sorted in descending order."""
    vec = positive_finite_values(
        sample,
        context="external tail estimators",
        minimum_size=8,
        stacklevel=3,
    )
    return np.sort(vec)[::-1]


def _positive_finite_in_order(sample: np.ndarray) -> np.ndarray:
    """Return positive finite observations in their original time order."""
    return positive_finite_values(
        sample,
        context="external xi estimators",
        minimum_size=8,
        stacklevel=3,
    )


def _normalize_standard_error(value: float) -> float:
    """Return a non-negative finite standard error or NaN."""
    value = float(value)
    if not np.isfinite(value) or value < 0:
        return float("nan")
    return value


def _hill_standard_error(xi_hat: float, k: int) -> float:
    """Return the classical Hill asymptotic standard error."""
    if k <= 0 or not np.isfinite(xi_hat):
        return float("nan")
    return _normalize_standard_error(abs(float(xi_hat)) / np.sqrt(float(k)))


def _pickands_standard_error(xi_hat: float, k: int) -> float:
    """Return the Pickands asymptotic standard error."""
    if k <= 0 or not np.isfinite(xi_hat):
        return float("nan")
    xi_hat = float(xi_hat)
    log_two = float(np.log(2.0))
    if abs(xi_hat) < 1e-8:
        limit = np.sqrt(3.0) / (2.0 * (log_two**2) * np.sqrt(float(k)))
        return _normalize_standard_error(limit)
    numerator = abs(xi_hat) * np.sqrt(np.power(2.0, 2.0 * xi_hat + 1.0) + 1.0)
    denominator = abs(2.0 * (np.power(2.0, xi_hat) - 1.0) * log_two * np.sqrt(float(k)))
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    return _normalize_standard_error(numerator / denominator)


def _dedh_standard_error(xi_hat: float, k: int) -> float:
    """Return the DEdH asymptotic SE for the Fréchet-domain heavy-tail benchmark."""
    if k <= 0 or not np.isfinite(xi_hat):
        return float("nan")
    return _normalize_standard_error(np.sqrt(1.0 + float(xi_hat) ** 2) / np.sqrt(float(k)))


def _hill_path(ordered: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    """Compute the Hill path from descending positive order statistics."""
    log_ordered = np.log(ordered)
    estimates = []
    for k in k_values:
        threshold = log_ordered[k]
        estimates.append(float(np.mean(log_ordered[:k] - threshold)))
    return np.asarray(estimates, dtype=float)


def _pickands_path(ordered: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    """Compute the Pickands path from descending positive order statistics."""
    n_obs = ordered.size
    estimates = []
    for k in k_values:
        if 4 * k > n_obs:
            estimates.append(np.nan)
            continue
        a = ordered[k - 1] - ordered[2 * k - 1]
        b = ordered[2 * k - 1] - ordered[4 * k - 1]
        if a <= 0 or b <= 0:
            estimates.append(np.nan)
            continue
        estimates.append(float(np.log(a / b) / np.log(2.0)))
    return np.asarray(estimates, dtype=float)


def _dedh_moment_path(ordered: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    """Compute the DEdH moment-estimator path from descending order statistics."""
    log_ordered = np.log(ordered)
    estimates = []
    for k in k_values:
        threshold = log_ordered[k]
        log_excess = log_ordered[:k] - threshold
        m1 = float(np.mean(log_excess))
        m2 = float(np.mean(log_excess**2))
        if m2 <= 0:
            estimates.append(np.nan)
            continue
        denom = 1.0 - (m1 * m1) / m2
        if abs(denom) < 1e-10:
            estimates.append(np.nan)
            continue
        estimates.append(float(m1 + 1.0 - 0.5 / denom))
    return np.asarray(estimates, dtype=float)


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


def select_stable_integer_window(
    levels: np.ndarray,
    path_xi: np.ndarray,
    *,
    min_window: int = 4,
) -> tuple[int, SelectionWindow, np.ndarray]:
    """Pick a stable integer-indexed window using variability and curvature."""
    if levels.size != path_xi.size or levels.size == 0:
        raise ValueError("levels and path_xi must be non-empty and aligned.")
    if levels.size <= min_window:
        center = int(np.median(levels))
        window = SelectionWindow(int(levels[0]), int(levels[-1]))
        return center, window, path_xi

    scores: list[float] = []
    windows: list[slice] = []
    for start in range(0, levels.size - min_window + 1):
        stop = start + min_window
        window_values = path_xi[start:stop]
        local_var = float(np.mean((window_values - window_values.mean()) ** 2))
        if window_values.size >= 3:
            curvature = float(np.mean(np.abs(np.diff(window_values, n=2))))
        else:
            curvature = 0.0
        scores.append(local_var + 0.5 * curvature)
        windows.append(slice(start, stop))

    best = windows[int(np.argmin(scores))]
    best_k = levels[best]
    best_xi = path_xi[best]
    chosen_k = int(np.round(np.median(best_k)))
    window = SelectionWindow(int(best_k[0]), int(best_k[-1]))
    return chosen_k, window, best_xi


def select_stable_tail_window(
    k_values: np.ndarray,
    path_xi: np.ndarray,
    *,
    min_window: int = 4,
) -> tuple[int, SelectionWindow, np.ndarray]:
    """Backward-compatible wrapper for `k`-indexed tail paths."""
    return select_stable_integer_window(k_values, path_xi, min_window=min_window)


def _select_from_path(
    method: str,
    level_values: np.ndarray,
    path_xi: np.ndarray,
    *,
    se_fn: Callable[[float, int], float] | None = None,
    tuning_axis: str = "k",
    fixed_upper_level: int | None = None,
    selection_min_window: int = 4,
) -> ExternalXiEstimate:
    """Filter invalid path values and return the automatically selected estimate."""
    mask = np.isfinite(path_xi)
    if not np.any(mask):
        raise ValueError(f"{method} produced no finite path estimates.")
    k_finite = level_values[mask]
    xi_finite = path_xi[mask]
    selected_k, stable_window, _ = select_stable_integer_window(
        k_finite,
        xi_finite,
        min_window=selection_min_window,
    )
    chosen_idx = int(np.argmin(np.abs(k_finite - selected_k)))
    xi_hat = float(xi_finite[chosen_idx])
    standard_error = (
        _normalize_standard_error(se_fn(xi_hat, selected_k)) if se_fn is not None else float("nan")
    )
    confidence_interval = wald_confidence_interval(xi_hat, standard_error)
    return ExternalXiEstimate(
        method=method,
        xi_hat=xi_hat,
        selected_level=selected_k,
        stable_window=stable_window,
        path_level=tuple(int(k) for k in k_finite),
        path_xi=tuple(float(value) for value in xi_finite),
        standard_error=standard_error,
        confidence_interval=confidence_interval,
        ci_method="asymptotic",
        tuning_axis=tuning_axis,
        fixed_upper_level=fixed_upper_level,
    )


def estimate_hill_evi(
    sample: np.ndarray,
    *,
    k_values: np.ndarray | None = None,
) -> ExternalXiEstimate:
    """Estimate `xi` with Hill's raw-sample tail estimator."""
    ordered = _finite_positive(sample)
    if k_values is None:
        k_values = candidate_tail_counts(ordered.size)
    path_xi = _hill_path(ordered, k_values)
    return _select_from_path("hill_raw", k_values, path_xi, se_fn=_hill_standard_error)


def estimate_pickands_evi(
    sample: np.ndarray,
    *,
    k_values: np.ndarray | None = None,
) -> ExternalXiEstimate:
    """Estimate `xi` with the Pickands raw-sample tail estimator."""
    ordered = _finite_positive(sample)
    if k_values is None:
        k_values = candidate_tail_counts(ordered.size)
    path_xi = _pickands_path(ordered, k_values)
    return _select_from_path("pickands_raw", k_values, path_xi, se_fn=_pickands_standard_error)


def estimate_dedh_moment_evi(
    sample: np.ndarray,
    *,
    k_values: np.ndarray | None = None,
) -> ExternalXiEstimate:
    """Estimate `xi` with the DEdH moment estimator."""
    ordered = _finite_positive(sample)
    if k_values is None:
        k_values = candidate_tail_counts(ordered.size)
    path_xi = _dedh_moment_path(ordered, k_values)
    return _select_from_path(
        "dedh_moment_raw",
        k_values,
        path_xi,
        se_fn=_dedh_standard_error,
    )


def estimate_max_spectrum_evi(
    sample: np.ndarray,
    *,
    scales: np.ndarray | None = None,
    min_scale_count: int = 3,
) -> ExternalXiEstimate:
    """Estimate `xi` with the dependent max-spectrum estimator."""
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
    "ExternalXiEstimate",
    "SelectionWindow",
    "ThresholdWindow",
    "_dedh_moment_path",
    "_dedh_standard_error",
    "_finite_positive",
    "_hill_path",
    "_hill_standard_error",
    "_max_spectrum_curve",
    "_max_spectrum_path",
    "_normalize_standard_error",
    "_pickands_path",
    "_pickands_standard_error",
    "_positive_finite_in_order",
    "_select_from_path",
    "_weighted_slope_with_se",
    "candidate_max_spectrum_scales",
    "candidate_tail_counts",
    "estimate_dedh_moment_evi",
    "estimate_hill_evi",
    "estimate_max_spectrum_evi",
    "estimate_pickands_evi",
    "select_stable_integer_window",
    "select_stable_tail_window",
    "wald_confidence_interval",
]
