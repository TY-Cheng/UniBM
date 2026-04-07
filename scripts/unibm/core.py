"""Core UniBM estimators.

This module is the main statistical pipeline:
1. build block-maxima summaries across block sizes,
2. choose an intermediate plateau on the log-log scale,
3. regress log summary on log block size,
4. optionally use bootstrap covariance information for FGLS inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .bootstrap import circular_block_summary_bootstrap
from .models import BlockSummaryCurve, PlateauWindow, ScalingFit
from .summaries import summarize_block_maxima
from ._validation import as_1d_float_array, warn_on_negative_values, warn_on_nonpositive_values
from .window_ops import sliding_window_extreme_valid

Z_CRIT_95 = 1.96

# Short-record benchmarks (`n_obs` around a few hundred) make the bootstrap
# cross-block covariance estimate noticeably noisy. Shrinking toward the
# diagonal keeps FGLS from overreacting to unstable off-diagonal entries while
# still preserving overlap/dependence information.
DEFAULT_COVARIANCE_SHRINKAGE = 0.35

# Plateau selection already penalizes curvature; a heavier default penalty helps
# reject windows that only look linear because a few adjacent block sizes happen
# to align in very short samples.
DEFAULT_CURVATURE_PENALTY = 2.0

# Keep the largest candidate block size short enough that the disjoint-block
# baseline still has a minimally meaningful number of maxima. This matters most
# in the short-record benchmark (`n_obs` around 365), where very large blocks
# otherwise leave the right tail of the regression curve dominated by only a
# handful of disjoint maxima.
DEFAULT_MIN_DISJOINT_BLOCKS = 15


def generate_block_sizes(
    n_obs: int,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    geom: bool = True,
    min_disjoint_blocks: int = DEFAULT_MIN_DISJOINT_BLOCKS,
) -> np.ndarray:
    """Generate an intermediate-range grid of block sizes.

    The default exponents deliberately avoid the smallest blocks, where finite-block
    behavior is dominated by local dependence, and the largest blocks, where too few
    maxima remain for stable regression.
    """
    if n_obs < 32:
        raise ValueError("At least 32 observations are required for block-size selection.")
    if min_block_size is None:
        min_block_size = max(5, int(np.ceil(n_obs**0.2)))
    if max_block_size is None:
        exponent_cap = int(np.floor(n_obs**0.55))
        disjoint_cap = int(np.floor(n_obs / max(min_disjoint_blocks, 1)))
        max_block_size = min(exponent_cap, disjoint_cap)
        max_block_size = max(min_block_size + 4, max_block_size)
    if max_block_size <= min_block_size:
        max_block_size = min_block_size + 4
    if num_step is None:
        num_step = min(32, max(10, max_block_size - min_block_size + 1))
    if geom:
        block_sizes = np.geomspace(min_block_size, max_block_size, num=num_step)
    else:
        block_sizes = np.linspace(min_block_size, max_block_size, num=num_step)
    block_sizes = np.unique(np.clip(np.rint(block_sizes).astype(int), min_block_size, None))
    return block_sizes[block_sizes > 1]


def block_maxima(
    vec: np.ndarray | list[float],
    block_size: int,
    sliding: bool = True,
) -> np.ndarray:
    """Compute sliding or disjoint block maxima.

    Sliding blocks use every overlapping window of a given size; disjoint blocks
    only use non-overlapping windows. The workflow compares them explicitly because
    sliding gives a smoother summary curve at the cost of stronger dependence.
    """
    arr = as_1d_float_array(vec)
    if block_size < 2 or arr.size < block_size:
        return np.asarray([], dtype=float)
    if sliding:
        return sliding_window_extreme_valid(arr, block_size, reducer="max")
    n_block = arr.size // block_size
    if n_block < 1:
        return np.asarray([], dtype=float)
    windows = arr[: n_block * block_size].reshape(n_block, block_size)
    maxima = windows.max(axis=1)
    valid = np.all(np.isfinite(windows), axis=1)
    return maxima[valid]


def block_summary_curve(
    vec: np.ndarray | list[float],
    block_sizes: np.ndarray,
    *,
    sliding: bool = True,
    quantile: float = 0.5,
    target: str = "quantile",
) -> BlockSummaryCurve:
    """Summarize block maxima over multiple block sizes."""
    warn_on_negative_values(vec, context="block_summary_curve", stacklevel=3)
    block_sizes = np.asarray(block_sizes, dtype=int)
    values = np.empty(block_sizes.size, dtype=float)
    counts = np.empty(block_sizes.size, dtype=int)
    for idx, block_size in enumerate(block_sizes):
        maxima = block_maxima(vec=vec, block_size=int(block_size), sliding=sliding)
        counts[idx] = maxima.size
        values[idx] = summarize_block_maxima(maxima, target=target, quantile=quantile)
    excluded = values[np.isfinite(values) & (values <= 0) & (counts > 0)]
    if excluded.size:
        warn_on_nonpositive_values(
            excluded,
            context="block_summary_curve",
            noun="block summaries excluded from the log-log regression",
            stacklevel=3,
        )
    positive_mask = np.isfinite(values) & (values > 0) & (counts > 0)
    return BlockSummaryCurve(
        block_sizes=block_sizes,
        counts=counts,
        values=values,
        positive_mask=positive_mask,
    )


def _fit_linear_model(
    x: np.ndarray,
    y: np.ndarray,
    covariance: np.ndarray | None = None,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
) -> dict[str, Any]:
    """Fit the log-log scaling regression with optional covariance-aware weighting."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    if covariance is not None and covariance.shape == (x.size, x.size):
        shrinkage = float(np.clip(covariance_shrinkage, 0.0, 1.0))
        if shrinkage > 0:
            diagonal = np.diag(np.diag(covariance))
            covariance = (1.0 - shrinkage) * covariance + shrinkage * diagonal
        # A tiny diagonal ridge keeps the pseudo-inverse well behaved when the
        # bootstrap covariance is nearly singular on short plateau windows.
        scale = np.trace(covariance) / max(covariance.shape[0], 1)
        ridge = max(abs(float(scale)) * 1e-8, 1e-12)
        regularized = covariance + np.eye(covariance.shape[0]) * ridge
        inv_cov = np.linalg.pinv(regularized)
        normal_matrix = X.T @ inv_cov @ X
        beta = np.linalg.pinv(normal_matrix) @ (X.T @ inv_cov @ y)
        cov_beta = np.linalg.pinv(normal_matrix)
        fitted = X @ beta
        resid = y - fitted
        objective = float(resid @ inv_cov @ resid)
    else:
        normal_matrix = X.T @ X
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ beta
        resid = y - fitted
        xtx_inv = np.linalg.pinv(normal_matrix)
        meat = X.T @ np.diag(resid**2) @ X
        cov_beta = xtx_inv @ meat @ xtx_inv
        objective = float(resid @ resid)
    try:
        condition_number = float(np.linalg.cond(normal_matrix))
    except np.linalg.LinAlgError:
        condition_number = float("inf")
    result = {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "fitted": fitted,
        "cov_beta": cov_beta,
        "standard_error": float(np.sqrt(max(cov_beta[1, 1], 0.0))),
        "objective": objective,
        "condition_number": condition_number,
        "n_obs": int(x.size),
        "n_params": int(X.shape[1]),
    }
    return result


def _aligned_bootstrap_covariance(
    bootstrap: dict[str, Any] | None,
    curve: BlockSummaryCurve,
    plateau: PlateauWindow,
) -> np.ndarray | None:
    """Align a bootstrap covariance matrix to the selected positive plateau.

    Shared benchmark backbones may be evaluated over the full block-size grid,
    while the fitted curve only keeps positive summaries. This helper trims the
    covariance matrix to the same positive block sizes and then to the selected
    plateau window.
    """
    if bootstrap is None:
        return None
    raw_covariance = bootstrap.get("covariance")
    if raw_covariance is None:
        return None
    covariance = np.atleast_2d(np.asarray(raw_covariance, dtype=float))
    if covariance.size == 0:
        return None
    bootstrap_block_sizes = np.asarray(
        bootstrap.get("block_sizes", curve.positive_block_sizes),
        dtype=int,
    )
    if covariance.shape[0] != bootstrap_block_sizes.size:
        return None
    if np.array_equal(bootstrap_block_sizes, curve.block_sizes):
        positive_idx = np.flatnonzero(curve.positive_mask)
        covariance = covariance[np.ix_(positive_idx, positive_idx)]
    elif not np.array_equal(bootstrap_block_sizes, curve.positive_block_sizes):
        positive_idx = []
        lookup = {int(block_size): idx for idx, block_size in enumerate(bootstrap_block_sizes)}
        for block_size in curve.positive_block_sizes:
            idx = lookup.get(int(block_size))
            if idx is None:
                return None
            positive_idx.append(idx)
        covariance = covariance[np.ix_(positive_idx, positive_idx)]
    if covariance.shape[0] != curve.log_block_sizes.size:
        return None
    return covariance[plateau.start : plateau.stop, plateau.start : plateau.stop]


def select_penultimate_window(
    log_block_sizes: np.ndarray,
    log_values: np.ndarray,
    *,
    min_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
) -> PlateauWindow:
    """Choose an intermediate block-size window by balancing linearity and curvature."""
    x = np.asarray(log_block_sizes, dtype=float)
    y = np.asarray(log_values, dtype=float)
    n = x.size
    if n < min_points:
        raise ValueError("Not enough positive block summaries to select a plateau.")
    lo = int(np.floor(n * trim_fraction))
    hi = n - lo
    lo = min(lo, max(n - min_points, 0))
    if hi - lo < min_points:
        lo = 0
        hi = n
    best: tuple[float, int, int] | None = None
    for start in range(lo, hi - min_points + 1):
        for stop in range(start + min_points, hi + 1):
            model = _fit_linear_model(x[start:stop], y[start:stop])
            resid = y[start:stop] - model["fitted"]
            mse = float(np.mean(resid**2))
            # Penalize windows that are only locally linear because of sharp curvature.
            slopes = np.diff(y[start:stop]) / np.diff(x[start:stop])
            curvature = float(np.mean(np.abs(np.diff(slopes)))) if slopes.size > 1 else 0.0
            score = (mse + float(curvature_penalty) * curvature) / np.sqrt(stop - start)
            if best is None or score < best[0]:
                best = (score, start, stop)
    assert best is not None
    _, start, stop = best
    mask = np.zeros(n, dtype=bool)
    mask[start:stop] = True
    return PlateauWindow(
        start=start,
        stop=stop,
        score=float(best[0]),
        mask=mask,
        x=x[start:stop],
        y=y[start:stop],
    )


def _fit_scaling_model(
    vec: np.ndarray | list[float],
    *,
    target: str,
    quantile: float = 0.5,
    sliding: bool = True,
    block_sizes: np.ndarray | None = None,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    plateau_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    bootstrap_reps: int = 0,
    super_block_size: int | None = None,
    random_state: int | None = 0,
    curve: BlockSummaryCurve | None = None,
    plateau: PlateauWindow | None = None,
    bootstrap_result: dict[str, Any] | None = None,
) -> ScalingFit:
    """Internal scaling-model implementation shared by median/mean/mode targets."""
    arr = as_1d_float_array(vec)
    finite_count = int(np.sum(np.isfinite(arr)))
    if finite_count < 32:
        raise ValueError("At least 32 finite observations are required for block-size selection.")
    if curve is None and block_sizes is None:
        block_sizes = generate_block_sizes(
            n_obs=arr.size,
            num_step=num_step,
            min_block_size=min_block_size,
            max_block_size=max_block_size,
            geom=True,
        )
    if curve is None:
        curve = block_summary_curve(
            arr,
            np.asarray(block_sizes, dtype=int),
            sliding=sliding,
            quantile=quantile,
            target=target,
        )
    if curve.log_block_sizes.size < plateau_points:
        raise ValueError("Not enough positive block summaries for regression.")
    if plateau is None:
        plateau = select_penultimate_window(
            curve.log_block_sizes,
            curve.log_values,
            min_points=plateau_points,
            trim_fraction=trim_fraction,
            curvature_penalty=curvature_penalty,
        )
    bootstrap = bootstrap_result
    if bootstrap is None and bootstrap_reps > 1:
        # The bootstrap operates on the original time series, not on the block
        # summaries themselves, so the covariance matrix reflects overlap and
        # serial dependence across block sizes.
        bootstrap = circular_block_summary_bootstrap(
            vec=arr,
            block_sizes=curve.positive_block_sizes,
            target=target,
            quantile=quantile,
            sliding=sliding,
            reps=bootstrap_reps,
            super_block_size=super_block_size,
            random_state=random_state,
        )
    covariance = _aligned_bootstrap_covariance(bootstrap, curve, plateau)

    model = _fit_linear_model(
        plateau.x,
        plateau.y,
        covariance=covariance,
        covariance_shrinkage=covariance_shrinkage,
    )

    slope = model["slope"]
    standard_error = model["standard_error"]
    return ScalingFit(
        target=target,
        quantile=float(quantile),
        sliding=bool(sliding),
        intercept=model["intercept"],
        slope=slope,
        standard_error=standard_error,
        confidence_interval=(
            float(slope - Z_CRIT_95 * standard_error),
            float(slope + Z_CRIT_95 * standard_error),
        ),
        curve=curve,
        plateau=plateau,
        cov_beta=model["cov_beta"],
        bootstrap=bootstrap,
    )


def estimate_evi_quantile(
    vec: np.ndarray | list[float],
    *,
    quantile: float = 0.5,
    sliding: bool = True,
    block_sizes: np.ndarray | None = None,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    bootstrap_reps: int = 200,
    super_block_size: int | None = None,
    random_state: int | None = 0,
    plateau_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    curve: BlockSummaryCurve | None = None,
    plateau: PlateauWindow | None = None,
    bootstrap_result: dict[str, Any] | None = None,
) -> ScalingFit:
    """Estimate the extreme value index from a block-quantile scaling law.

    The estimator builds block maxima over a grid of block sizes, summarizes
    each block-maxima sample by a chosen quantile, selects an intermediate
    approximately linear plateau on the log-log scale, and regresses
    ``log(summary)`` on ``log(block size)``. The fitted slope is the UniBM EVI
    estimate.

    Parameters
    ----------
    vec
        One-dimensional raw series. The series may contain missing values, but
        at least 32 finite observations are required after filtering.
    quantile
        Block-summary quantile in ``(0, 1)``. The benchmark and application
        workflows typically use ``0.5`` so this becomes a median-based fit.
    sliding
        If ``True``, use overlapping block maxima. If ``False``, use disjoint
        non-overlapping blocks.
    block_sizes
        Optional explicit block-size grid. If omitted, the function constructs
        an intermediate-range grid via :func:`generate_block_sizes`.
    num_step, min_block_size, max_block_size
        Optional controls for the automatically generated block-size grid.
    bootstrap_reps
        Number of circular block-bootstrap replicates used to estimate a
        cross-block covariance matrix for FGLS. Set to ``0`` or ``1`` to skip
        bootstrap covariance estimation.
    super_block_size
        Optional super-block size used by the bootstrap backbone.
    random_state
        Seed for bootstrap resampling.
    plateau_points
        Minimum number of positive block summaries required inside the selected
        plateau.
    trim_fraction
        Fraction of the left and right ends of the positive block-size grid
        excluded before plateau search.
    curvature_penalty
        Penalty applied to local curvature during plateau selection.
    covariance_shrinkage
        Diagonal shrinkage applied to the bootstrap covariance matrix before
        FGLS fitting.
    curve, plateau, bootstrap_result
        Optional precomputed intermediate objects. These are mainly useful for
        benchmark workflows that reuse summaries or bootstrap backbones across
        repeated fits.

    Returns
    -------
    unibm.models.ScalingFit
        Immutable result object containing the fitted slope, confidence
        interval, selected plateau, and the underlying block-summary curve.

    Notes
    -----
    The returned slope is the EVI estimate ``xi``. Return-level extrapolation
    is then handled by :func:`estimate_return_level`.
    """
    return _fit_scaling_model(
        vec=vec,
        target="quantile",
        quantile=quantile,
        sliding=sliding,
        block_sizes=block_sizes,
        num_step=num_step,
        min_block_size=min_block_size,
        max_block_size=max_block_size,
        plateau_points=plateau_points,
        trim_fraction=trim_fraction,
        curvature_penalty=curvature_penalty,
        covariance_shrinkage=covariance_shrinkage,
        bootstrap_reps=bootstrap_reps,
        super_block_size=super_block_size,
        random_state=random_state,
        curve=curve,
        plateau=plateau,
        bootstrap_result=bootstrap_result,
    )


def estimate_target_scaling(
    vec: np.ndarray | list[float],
    *,
    target: str = "quantile",
    quantile: float = 0.5,
    sliding: bool = True,
    block_sizes: np.ndarray | None = None,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    bootstrap_reps: int = 0,
    super_block_size: int | None = None,
    random_state: int | None = 0,
    plateau_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    curve: BlockSummaryCurve | None = None,
    plateau: PlateauWindow | None = None,
    bootstrap_result: dict[str, Any] | None = None,
) -> ScalingFit:
    """Fit the UniBM log-log scaling model for an arbitrary block summary.

    Parameters
    ----------
    vec
        One-dimensional raw series.
    target
        Block-summary functional. Supported values are those implemented by
        :func:`unibm.summaries.summarize_block_maxima`, such as ``"quantile"``,
        ``"mean"``, and ``"mode"``.
    quantile
        Quantile level used only when ``target="quantile"``.
    sliding
        If ``True``, use overlapping block maxima. Otherwise use disjoint
        blocks.
    block_sizes, num_step, min_block_size, max_block_size
        Controls for the candidate block-size grid.
    bootstrap_reps, super_block_size, random_state
        Controls for the circular block bootstrap used for covariance-aware
        fitting.
    plateau_points, trim_fraction, curvature_penalty, covariance_shrinkage
        Plateau-selection and FGLS regularization controls.
    curve, plateau, bootstrap_result
        Optional precomputed intermediate objects reused by benchmark code.

    Returns
    -------
    unibm.models.ScalingFit
        The fitted scaling model for the requested block-summary target.

    Notes
    -----
    This is the generic version of :func:`estimate_evi_quantile`. For the
    canonical EVI workflow, prefer :func:`estimate_evi_quantile`.
    """
    return _fit_scaling_model(
        vec=vec,
        target=target,
        quantile=quantile,
        sliding=sliding,
        block_sizes=block_sizes,
        num_step=num_step,
        min_block_size=min_block_size,
        max_block_size=max_block_size,
        plateau_points=plateau_points,
        trim_fraction=trim_fraction,
        curvature_penalty=curvature_penalty,
        covariance_shrinkage=covariance_shrinkage,
        bootstrap_reps=bootstrap_reps,
        super_block_size=super_block_size,
        random_state=random_state,
        curve=curve,
        plateau=plateau,
        bootstrap_result=bootstrap_result,
    )


def predict_block_quantile(fit: ScalingFit, block_size: float) -> float:
    """Predict a block quantile from the fitted scaling law."""
    if fit.target != "quantile":
        raise ValueError(
            "predict_block_quantile requires a quantile-based ScalingFit. "
            f"Received target={fit.target!r}."
        )
    if block_size <= 0:
        raise ValueError("Block size must be positive.")
    log_b = np.log(block_size)
    return float(np.exp(fit.intercept + fit.slope * log_b))


def estimate_return_level(
    fit: ScalingFit,
    years: float | np.ndarray,
    *,
    observations_per_year: float = 365.25,
    extremal_index: float | None = None,
) -> float | np.ndarray:
    """Map a fitted quantile-scaling law to return-horizon levels.

    Parameters
    ----------
    fit
        Quantile-based :class:`~unibm.models.ScalingFit`. The fit should usually
        come from :func:`estimate_evi_quantile`.
    years
        Return horizons in years. May be a scalar or an array.
    observations_per_year
        Effective observation frequency used to convert return horizons into
        block sizes. Daily environmental applications typically use ``365`` or
        ``365.25``.
    extremal_index
        Optional extremal index ``theta`` in ``(0, 1]``. When supplied, the
        return horizon is adjusted by the effective number of independent
        extremes. Pass ``theta`` itself, not ``1/theta``.

    Returns
    -------
    float or ndarray
        Return-level estimate(s) on the original data scale.

    Notes
    -----
    This function assumes the fitted scaling law was built from block
    quantiles. It does **not** fit a separate annual-maxima or GEV model.
    Instead, it reuses the same UniBM scaling law from the fitted block-summary
    curve and evaluates that law at larger block sizes.

    Concretely, if the fitted block quantile satisfies

    ``Q_b \approx exp(alpha) * b**xi``,

    then a ``T``-year return-horizon estimate is obtained by setting
    ``b = observations_per_year * T`` and evaluating the same scaling law at
    that horizon.

    If dependence adjustment is needed, pass the extremal index ``theta`` from
    an EI estimator. The current mapping uses
    ``b = observations_per_year * theta * T``. In this interpretation,
    ``theta < 1`` reduces the effective number of independent extreme episodes
    per calendar year, so the EI-adjusted return level is typically lower than
    the unadjusted single-day calendar-time return level.

    The EVI plateau that supports ``xi`` and the EI stable window that supports
    ``theta`` are selected from different statistical paths and therefore need
    not coincide.
    """
    years_arr = np.atleast_1d(np.asarray(years, dtype=float))
    effective_years = years_arr if extremal_index is None else years_arr * float(extremal_index)
    block_sizes = observations_per_year * effective_years
    estimates = np.asarray(
        [predict_block_quantile(fit, block_size=float(size)) for size in block_sizes]
    )
    return float(estimates[0]) if np.ndim(years) == 0 else estimates
