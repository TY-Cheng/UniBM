"""Tail-side xi estimator family and shared comparator result models."""

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
    """Point estimate plus diagnostic path information for comparator estimators."""

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
        """Backward-compatible alias for threshold-indexed comparator paths."""
        return self.selected_level

    @property
    def path_k(self) -> tuple[int, ...]:
        """Backward-compatible alias for threshold-indexed comparator paths."""
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
    """Construct a log-spaced tail-count grid or fail if the bounds are infeasible."""
    for name, value, minimum in (
        ("n_obs", n_obs, 2),
        ("min_count", min_count, 1),
        ("num", num, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer at least {minimum}.")
    if isinstance(max_fraction, (bool, np.bool_)):
        raise ValueError("max_fraction must be finite and lie in (0, 1].")
    try:
        max_fraction = float(max_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_fraction must be finite and lie in (0, 1].") from exc
    if not np.isfinite(max_fraction) or not 0.0 < max_fraction <= 1.0:
        raise ValueError("max_fraction must be finite and lie in (0, 1].")
    n_obs = int(n_obs)
    min_count = int(min_count)
    num = int(num)
    lower = int(min_count)
    upper = int(min(np.floor(max_fraction * n_obs), n_obs - 1))
    if upper < lower:
        raise ValueError("No feasible tail-count grid satisfies min_count and max_fraction.")
    if upper == lower:
        return np.array([lower], dtype=int)
    grid = np.unique(np.round(np.geomspace(lower, upper, num=num)).astype(int))
    return grid[(grid >= lower) & (grid <= upper)]


def _validate_tail_counts(
    k_values: np.ndarray,
    *,
    n_obs: int,
    minimum: int = 1,
    maximum: int | None = None,
) -> np.ndarray:
    """Validate a strictly increasing integer tail-count grid."""
    try:
        raw = np.asarray(k_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("k_values must be a one-dimensional numeric sequence.") from exc
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("k_values must be a non-empty one-dimensional sequence.")
    if np.iscomplexobj(raw) or raw.dtype.kind == "b":
        raise ValueError("k_values must contain finite integer values.")
    try:
        numeric = raw.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("k_values must contain finite integer values.") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("k_values must contain finite integer values.")
    if np.any(numeric < minimum):
        raise ValueError(f"k_values must be at least {minimum} for this estimator.")
    if np.any(np.diff(numeric) <= 0):
        raise ValueError("k_values must be strictly increasing with no duplicates.")
    upper = n_obs - 1 if maximum is None else int(maximum)
    if np.any(numeric > upper):
        raise ValueError(f"k_values cannot exceed {upper} for this estimator and sample.")
    return numeric.astype(int)


def _finite_positive(sample: np.ndarray) -> np.ndarray:
    """Return positive finite observations sorted in descending order."""
    vec = positive_finite_values(
        sample,
        context="tail xi estimators",
        minimum_size=8,
        stacklevel=3,
    )
    return np.sort(vec)[::-1]


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


def select_stable_integer_window(
    levels: np.ndarray,
    path_xi: np.ndarray,
    *,
    min_window: int = 4,
) -> tuple[int, SelectionWindow, np.ndarray]:
    """Pick a stable window and return its lower medoid from the observed grid."""
    try:
        raw_levels = np.asarray(levels)
        path_xi = np.asarray(path_xi, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("levels and path_xi must be one-dimensional numeric arrays.") from exc
    if raw_levels.ndim != 1 or path_xi.ndim != 1:
        raise ValueError("levels and path_xi must be one-dimensional arrays.")
    if raw_levels.size != path_xi.size or raw_levels.size == 0:
        raise ValueError("levels and path_xi must be non-empty and aligned.")
    if raw_levels.dtype.kind == "b" or np.iscomplexobj(raw_levels):
        raise ValueError("levels must contain finite strictly increasing integers.")
    try:
        numeric_levels = raw_levels.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("levels must contain finite strictly increasing integers.") from exc
    if (
        not np.all(np.isfinite(numeric_levels))
        or not np.all(numeric_levels == np.floor(numeric_levels))
        or np.any(np.diff(numeric_levels) <= 0)
    ):
        raise ValueError("levels must contain finite strictly increasing integers.")
    if not np.all(np.isfinite(path_xi)):
        raise ValueError("path_xi must contain only finite values.")
    if (
        isinstance(min_window, (bool, np.bool_))
        or not isinstance(min_window, (int, np.integer))
        or min_window < 2
    ):
        raise ValueError("min_window must be an integer at least 2.")
    levels = numeric_levels.astype(int)
    min_window = int(min_window)
    if levels.size <= min_window:
        center = int(levels[(levels.size - 1) // 2])
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
    chosen_k = int(best_k[(best_k.size - 1) // 2])
    window = SelectionWindow(int(best_k[0]), int(best_k[-1]))
    return chosen_k, window, best_xi


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
    level_finite = level_values[mask]
    xi_finite = path_xi[mask]
    selected_level, stable_window, _ = select_stable_integer_window(
        level_finite,
        xi_finite,
        min_window=selection_min_window,
    )
    chosen_idx = int(np.flatnonzero(level_finite == selected_level)[0])
    xi_hat = float(xi_finite[chosen_idx])
    standard_error = (
        _normalize_standard_error(se_fn(xi_hat, selected_level))
        if se_fn is not None
        else float("nan")
    )
    confidence_interval = wald_confidence_interval(xi_hat, standard_error)
    return ExternalXiEstimate(
        method=method,
        xi_hat=xi_hat,
        selected_level=selected_level,
        stable_window=stable_window,
        path_level=tuple(int(level) for level in level_finite),
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
    """Estimate ``xi`` with Hill, using an automatic grid when ``k_values`` is absent."""
    ordered = _finite_positive(sample)
    if k_values is None:
        k_values = candidate_tail_counts(ordered.size)
    k_values = _validate_tail_counts(k_values, n_obs=ordered.size)
    path_xi = _hill_path(ordered, k_values)
    return _select_from_path("hill_raw", k_values, path_xi, se_fn=_hill_standard_error)


def estimate_pickands_evi(
    sample: np.ndarray,
    *,
    k_values: np.ndarray | None = None,
) -> ExternalXiEstimate:
    """Estimate ``xi`` with Pickands over valid tail counts satisfying ``4k <= n``."""
    ordered = _finite_positive(sample)
    if k_values is None:
        k_values = candidate_tail_counts(ordered.size)
    k_values = _validate_tail_counts(
        k_values,
        n_obs=ordered.size,
        maximum=ordered.size // 4,
    )
    path_xi = _pickands_path(ordered, k_values)
    return _select_from_path("pickands_raw", k_values, path_xi, se_fn=_pickands_standard_error)


def estimate_dedh_moment_evi(
    sample: np.ndarray,
    *,
    k_values: np.ndarray | None = None,
) -> ExternalXiEstimate:
    """Estimate ``xi`` with DEdH over an automatic or explicit tail-count grid."""
    ordered = _finite_positive(sample)
    if k_values is None:
        k_values = candidate_tail_counts(ordered.size)
    k_values = _validate_tail_counts(k_values, n_obs=ordered.size, minimum=2)
    path_xi = _dedh_moment_path(ordered, k_values)
    return _select_from_path(
        "dedh_moment_raw",
        k_values,
        path_xi,
        se_fn=_dedh_standard_error,
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
    "_normalize_standard_error",
    "_pickands_path",
    "_pickands_standard_error",
    "_select_from_path",
    "candidate_tail_counts",
    "estimate_dedh_moment_evi",
    "estimate_hill_evi",
    "estimate_pickands_evi",
    "select_stable_integer_window",
    "wald_confidence_interval",
]
