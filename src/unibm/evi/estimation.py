"""Canonical public estimation entrypoints for the EVI workflow."""

from __future__ import annotations

from typing import Literal

import numpy as np

from .._block_grid import generate_block_sizes
from .._parallel import validate_n_threads
from .._bootstrap_precision import matching_precision_metadata
from .._validation import (
    as_1d_float_array,
    subset_covariance_by_labels,
    validate_covariance_shrinkage,
)
from ._regression import (
    DEFAULT_COVARIANCE_SHRINKAGE,
    DEFAULT_CURVATURE_PENALTY,
    Z_CRIT_95,
    _aligned_bootstrap_covariance,
    _fit_linear_model,
    _validate_curve_identity,
    _validate_plateau_slice,
)
from .blocks import block_summary_curve
from .bootstrap import _adaptive_block_summary_bootstrap, circular_block_summary_bootstrap
from .models import BlockSummaryCurve, PlateauWindow, ScalingFit
from .selection import select_penultimate_window


def estimate_evi_quantile(
    vec: np.ndarray | list[float],
    *,
    regression: Literal["OLS", "FGLS", "AUTO"],
    quantile: float = 0.5,
    sliding: bool = True,
    block_sizes: np.ndarray | None = None,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    bootstrap_reps: int | Literal["adaptive"] | None = None,
    super_block_size: int | None = None,
    random_state: int | None = 0,
    n_threads: int | None = None,
    plateau_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    curve: BlockSummaryCurve | None = None,
    plateau: PlateauWindow | None = None,
    bootstrap_result: dict[str, object] | None = None,
) -> ScalingFit:
    """Estimate EVI from a quantile of block maxima on a selected log-log plateau.

    ``vec`` is a one-dimensional series with at least 32 finite observations.
    ``quantile`` lies strictly between zero and one. Zeros enter the block
    quantiles; only positive summaries enter the log regression. Sliding
    windows overlap, while disjoint windows discard the incomplete tail.
    Supply an increasing integer ``block_sizes`` grid or let ``num_step`` and
    the size bounds control its generation. ``plateau_points`` is the minimum
    window length; trimming and curvature scoring select the fitted interval.

    ``regression`` is explicit: OLS uses HC0 uncertainty; strict FGLS requires
    usable bootstrap covariance. AUTO permits an internally generated missing
    covariance to fall back to OLS, but never accepts malformed supplied covariance.
    The result records both requested policy and actual regression.

    For FGLS/AUTO without a supplied bootstrap, omitted or ``"adaptive"`` reps
    check 128, 256, 512, 768, and 1024 draws. An integer requests a fixed budget.
    Adaptive precision monitors xi and its CI endpoints, not design-life levels.
    A cap warning retains the fit with ``bootstrap_precision_met=False``.
    Default covariance shrinkage is fixed at 0.37, not automatically tuned.
    ``n_threads=None`` chooses a CPU/workload-aware bootstrap pool (at most 8);
    a positive integer caps it, and 1 stays serial. This does not change BLAS
    settings. Callers with an outer process pool should allocate the inner cap.
    Random draws and adaptive stopping are independent of the thread count.

    Supplied covariance must match the target, quantile, and block scheme;
    block-size labels permit full-grid covariance to serve a selected subset.
    Reuse remains the caller's responsibility for data identity. Intervals are
    conditional on the observed plateau and do not include selection uncertainty.
    Return a ``ScalingFit`` with xi in ``slope``, a nominal 95% Wald interval,
    the full summary curve, selected window, and bootstrap diagnostics.
    """
    return estimate_target_scaling(
        vec=vec,
        target="quantile",
        regression=regression,
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
        n_threads=n_threads,
        curve=curve,
        plateau=plateau,
        bootstrap_result=bootstrap_result,
    )


def estimate_target_scaling(
    vec: np.ndarray | list[float],
    *,
    regression: Literal["OLS", "FGLS", "AUTO"],
    target: str = "quantile",
    quantile: float = 0.5,
    sliding: bool = True,
    block_sizes: np.ndarray | None = None,
    num_step: int | None = None,
    min_block_size: int | None = None,
    max_block_size: int | None = None,
    bootstrap_reps: int | Literal["adaptive"] | None = None,
    super_block_size: int | None = None,
    random_state: int | None = 0,
    n_threads: int | None = None,
    plateau_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    curve: BlockSummaryCurve | None = None,
    plateau: PlateauWindow | None = None,
    bootstrap_result: dict[str, object] | None = None,
) -> ScalingFit:
    """Fit the UniBM log-log scaling model for quantile, mean, or mode summaries.

    The regression, covariance-reuse, and adaptive-R contracts are the same as
    ``estimate_evi_quantile``. ``quantile`` is used only for the quantile target.
    Mean/mode fits are not accepted by quantile design-life mapping helpers.
    Means use all finite maxima; the KDE mode surrogate uses only positive
    maxima. Interpreting the fitted slope as xi requires the selected summary
    to obey the assumed power law, which is a separate modeling assumption.
    ``n_threads`` has the same per-call bootstrap budget as ``estimate_evi_quantile``.
    """
    validate_n_threads(n_threads)
    if regression not in {"OLS", "FGLS", "AUTO"}:
        raise ValueError("regression must be 'OLS', 'FGLS', or 'AUTO'.")
    shrinkage_policy = validate_covariance_shrinkage(covariance_shrinkage)
    if regression == "OLS":
        if bootstrap_result is not None:
            raise ValueError("OLS does not accept bootstrap_result.")
        if bootstrap_reps is not None and (
            isinstance(bootstrap_reps, bool)
            or not isinstance(bootstrap_reps, (int, np.integer))
            or bootstrap_reps != 0
        ):
            raise ValueError("OLS does not use bootstrap; bootstrap_reps must be 0 or None.")
        resolved_bootstrap_reps = 0
    elif bootstrap_result is not None:
        if bootstrap_reps is not None:
            raise ValueError("bootstrap_reps must be None when bootstrap_result is supplied.")
        resolved_bootstrap_reps = 0
    else:
        if bootstrap_reps is None or bootstrap_reps == "adaptive":
            resolved_bootstrap_reps = "adaptive"
        elif (
            isinstance(bootstrap_reps, bool)
            or not isinstance(bootstrap_reps, (int, np.integer))
            or bootstrap_reps < 2
        ):
            raise ValueError(
                "bootstrap_reps must be an integer at least 2 or 'adaptive' for FGLS or AUTO."
            )
        else:
            resolved_bootstrap_reps = int(bootstrap_reps)
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
            block_sizes,
            sliding=sliding,
            quantile=quantile,
            target=target,
        )
    _validate_curve_identity(
        curve,
        target=target,
        quantile=quantile,
        sliding=sliding,
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
    _validate_plateau_slice(curve, plateau)
    bootstrap = bootstrap_result
    if bootstrap is None and resolved_bootstrap_reps == "adaptive":
        levels = curve.positive_block_sizes[plateau.start : plateau.stop]

        def evaluate(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return xi and its Wald endpoints plus SE scales for MC precision checks."""
            selected_cov = subset_covariance_by_labels(
                cov, curve.positive_block_sizes, levels, context="bootstrap covariance"
            )
            fit = _fit_linear_model(plateau.x, plateau.y, selected_cov, shrinkage_policy)
            xi, se = fit["slope"], fit["standard_error"]
            return np.asarray([xi, xi - Z_CRIT_95 * se, xi + Z_CRIT_95 * se]), np.full(3, se)

        bootstrap = _adaptive_block_summary_bootstrap(
            arr,
            curve.positive_block_sizes,
            target=target,
            quantile=quantile,
            sliding=sliding,
            super_block_size=super_block_size,
            random_state=random_state,
            n_threads=n_threads,
            evaluate=evaluate,
        )
        bootstrap.update(
            bootstrap_mcse_targets=("xi", "xi_ci_lo", "xi_ci_hi"),
            bootstrap_precision_levels=levels.copy(),
            bootstrap_precision_shrinkage=shrinkage_policy,
        )
    elif bootstrap is None and resolved_bootstrap_reps > 1:
        bootstrap = circular_block_summary_bootstrap(
            vec=arr,
            block_sizes=curve.positive_block_sizes,
            target=target,
            quantile=quantile,
            sliding=sliding,
            reps=resolved_bootstrap_reps,
            super_block_size=super_block_size,
            random_state=random_state,
            n_threads=n_threads,
        )
        bootstrap["bootstrap_reps_policy"] = "fixed"
        bootstrap["bootstrap_reps_requested"] = resolved_bootstrap_reps
        bootstrap["bootstrap_reps_used"] = int(np.asarray(bootstrap["samples"]).shape[0])
    if bootstrap is not None and bootstrap_result is None:
        bootstrap["bootstrap_block_length_policy"] = (
            "default" if super_block_size is None else "fixed"
        )
    covariance = _aligned_bootstrap_covariance(bootstrap, curve, plateau)
    realized_shrinkage: float | None = None
    bootstrap_reps_used: int | None = None
    if covariance is not None:
        realized_shrinkage = float(shrinkage_policy)
        if bootstrap is not None and "samples" in bootstrap:
            samples = np.asarray(bootstrap["samples"])
            if samples.ndim == 2:
                bootstrap_reps_used = int(samples.shape[0])
    if covariance is None:
        if bootstrap_result is not None:
            raise ValueError("supplied bootstrap_result must contain usable covariance.")
        if regression == "FGLS":
            raise ValueError("FGLS requires usable bootstrap covariance.")
    model = _fit_linear_model(
        plateau.x,
        plateau.y,
        covariance=covariance,
        covariance_shrinkage=0.0 if realized_shrinkage is None else realized_shrinkage,
    )
    slope = model["slope"]
    standard_error = model["standard_error"]
    return ScalingFit(
        target=target,
        quantile=float(quantile),
        sliding=bool(sliding),
        regression_policy=regression,
        regression="FGLS" if covariance is not None else "OLS",
        ci_variant="bootstrap_cov" if covariance is not None else "hc0",
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
        covariance_shrinkage_policy=(None if covariance is None else "fixed"),
        covariance_shrinkage=realized_shrinkage,
        covariance_condition_number_raw=model["covariance_condition_number_raw"],
        covariance_condition_number_regularized=model["covariance_condition_number_regularized"],
        bootstrap_block_length_policy=(
            None if bootstrap is None else bootstrap.get("bootstrap_block_length_policy")
        ),
        bootstrap_block_length=(None if bootstrap is None else bootstrap.get("super_block_size")),
        bootstrap_reps_requested=(
            None if bootstrap is None else bootstrap.get("bootstrap_reps_requested")
        ),
        bootstrap_reps_used=bootstrap_reps_used,
        **matching_precision_metadata(
            bootstrap,
            curve.positive_block_sizes[plateau.start : plateau.stop],
            shrinkage_policy,
        ),
    )


__all__ = [
    "DEFAULT_COVARIANCE_SHRINKAGE",
    "DEFAULT_CURVATURE_PENALTY",
    "Z_CRIT_95",
    "estimate_evi_quantile",
    "estimate_target_scaling",
]
