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
    """Selected tail-index estimate with its retained finite diagnostic path.

    ``selected_level`` is a tail count or start-scale exponent according to
    ``tuning_axis``. ``confidence_interval`` is a nominal 95% Wald interval
    using the method's SE; it does not propagate automatic window selection.
    The ``ci_method`` label describes construction, not a coverage guarantee.
    """

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
        """Alias for ``selected_level``; a scale-based estimator still returns a scale."""
        return self.selected_level

    @property
    def path_k(self) -> tuple[int, ...]:
        """Alias for ``path_level`` without converting scale exponents to tail counts."""
        return self.path_level


def wald_confidence_interval(
    xi_hat: float,
    standard_error: float,
    *,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Return the two-sided normal interval ``xi_hat +/- z * standard_error``.

    ``ci_level`` must lie strictly between zero and one. Invalid estimates
    or negative/nonfinite SEs give (NaN, NaN); a zero SE gives equal bounds.
    The caller is responsible for the validity of the normal approximation.
    """
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
    """Return a deduplicated, rounded geometric grid of valid tail counts.

    Counts range from ``min_count`` through
    ``min(floor(max_fraction * n_obs), n_obs - 1)``; rounding may give fewer
    than ``num`` entries. Raise ValueError for invalid arguments or no
    feasible count. Estimators may impose additional restrictions, such as
    Pickands' ``4*k <= n_obs``.
    """
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
    """Require a nonempty 1D strictly increasing grid of integer tail counts.

    Reject booleans, nonfinite values, and counts outside ``minimum`` through
    ``maximum`` (default ``n_obs - 1``). Return an integer array without
    sorting, deduplicating, or silently clipping the caller's grid.
    """
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
    """Filter to at least eight positive finite observations and sort descending.

    Nonpositive finite values trigger a warning; nonfinite values are also
    removed. Sorting discards time order for these marginal tail estimators.
    """
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
    """Return ``abs(xi_hat) / sqrt(k)``, the classical independent-tail Hill SE.

    Invalid xi or nonpositive k gives NaN. This has no serial-dependence,
    threshold-selection, or tail-bias adjustment.
    """
    if k <= 0 or not np.isfinite(xi_hat):
        return float("nan")
    return _normalize_standard_error(abs(float(xi_hat)) / np.sqrt(float(k)))


def _pickands_standard_error(xi_hat: float, k: int) -> float:
    """Return the classical Pickands asymptotic SE, with its limit near xi=0.

    Use the continuous limit when ``abs(xi_hat) < 1e-8`` to avoid numerical
    cancellation. Invalid inputs or an unusable denominator give NaN; the
    formula does not adjust for serial dependence or threshold selection.
    """
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
    """Return ``sqrt(1 + xi_hat**2) / sqrt(k)`` for the heavy-tail DEdH regime.

    This is the independent-sample Fréchet-domain asymptotic formula, not
    a general negative-xi variance formula or a dependence adjustment.
    Invalid xi or nonpositive k gives NaN.
    """
    if k <= 0 or not np.isfinite(xi_hat):
        return float("nan")
    return _normalize_standard_error(np.sqrt(1.0 + float(xi_hat) ** 2) / np.sqrt(float(k)))


def _hill_path(ordered: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    """Average the top-k log excesses above descending order statistic k+1.

    ``ordered`` must already be positive and descending, and each validated
    k must leave a threshold observation. Return estimates aligned with k.
    """
    log_ordered = np.log(ordered)
    estimates = []
    for k in k_values:
        threshold = log_ordered[k]
        estimates.append(float(np.mean(log_ordered[:k] - threshold)))
    return np.asarray(estimates, dtype=float)


def _pickands_path(ordered: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    """Compute log2 ratios of spacings at descending ranks k, 2k, and 4k.

    Return NaN when a required rank is unavailable or ties produce a
    nonpositive spacing. Output positions match ``k_values``.
    """
    estimates = np.full(len(k_values), np.nan)
    available = 4 * k_values <= ordered.size
    k = k_values[available]
    a = ordered[k - 1] - ordered[2 * k - 1]
    b = ordered[2 * k - 1] - ordered[4 * k - 1]
    positive = (a > 0) & (b > 0)
    estimates[np.flatnonzero(available)[positive]] = np.log(a[positive] / b[positive]) / np.log(
        2.0
    )
    return estimates


def _dedh_moment_path(ordered: np.ndarray, k_values: np.ndarray) -> np.ndarray:
    """Compute DEdH xi from the first two moments of the top-k log excesses.

    Use descending observation k+1 as the threshold. Return NaN for a zero
    second moment or a nearly singular moment-ratio correction; the input
    must already be positive, descending, and paired with valid k values.
    """
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
    """Choose a low-variation path window and its lower-middle observed level.

    Require finite paired 1D arrays and strictly increasing integer levels.
    Score contiguous windows of ``min_window`` entries by path variance
    plus half the mean absolute second difference. If the path is shorter,
    use all entries. Return the selected level, inclusive level bounds,
    and selected path values. Levels need not be consecutive integers;
    this is a stability heuristic, not a hypothesis test.
    """
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

    windows = np.lib.stride_tricks.sliding_window_view(path_xi, min_window)
    local_var = np.mean((windows - windows.mean(axis=1, keepdims=True)) ** 2, axis=1)
    curvature = np.mean(np.abs(np.diff(windows, n=2, axis=1)), axis=1) if min_window >= 3 else 0.0
    start = int(np.argmin(local_var + 0.5 * curvature))
    best = slice(start, start + min_window)
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
    """Remove nonfinite path estimates, select a level, and attach a Wald interval.

    Evaluate ``se_fn(xi_hat, selected_level)`` only at the selected point;
    without it the SE and interval are NaN. Selection uses the retained
    finite grid, which can contain gaps. Raise ValueError if no point remains.
    """
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
    """Estimate a positive heavy-tail index with a stability-selected Hill threshold.

    Keep positive finite observations (at least eight) and sort descending.
    ``k_values`` counts upper observations above rank k+1; omit it for the
    rounded geometric grid. Return the chosen estimate and finite path with
    a nominal 95% Wald interval based on ``abs(xi) / sqrt(k)``. This classical
    SE assumes the independent-tail regime and omits serial dependence,
    tail bias, and threshold-selection uncertainty.
    """
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
    """Estimate xi from order-statistic spacings at ranks k, 2k, and 4k.

    This implementation retains only positive finite observations (at
    least eight). Supplied ``k_values`` must satisfy ``4*k <= n`` after
    filtering; otherwise use the default tail grid. Tied spacings produce
    invalid path entries, which are dropped before stability selection.
    Return the selected estimate, retained path, and a classical asymptotic
    95% Wald interval without dependence or selection adjustments.
    """
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
    """Estimate xi from top-tail log moments at a stability-selected threshold.

    Retain at least eight positive finite observations. ``k_values`` must
    satisfy ``2 <= k < n`` after filtering, or is generated automatically.
    Singular moment corrections are removed before path selection. The
    returned nominal 95% Wald interval uses the Fréchet-domain formula
    ``sqrt(1 + xi**2) / sqrt(k)``; it is not a general negative-xi interval
    and does not adjust for serial dependence or threshold selection.
    """
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
