"""Bootstrap helpers for BM-based extremal-index path estimation."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .._block_grid import validate_block_sizes
from .._bootstrap_sampling import (
    default_circular_bootstrap_block_size,
    draw_circular_block_bootstrap_sample,
    draw_circular_block_bootstrap_samples,
)
from .._bootstrap_precision import adaptive_covariance
from .._validation import subset_covariance_by_labels, validate_covariance_shrinkage
from ._stats import Z_CRIT_95, _log_scale_theta_interval
from ._validation import _finite_nonnegative_series, _finite_positive_series
from .bm import EI_DEFAULT_COVARIANCE_SHRINKAGE, _fit_pooled_z_model
from .paths import BM_PATH_KEYS, _build_bm_z_paths_from_values
from .selection import select_stable_path_window


def _resolve_ei_bootstrap_block_length(
    values: np.ndarray,
    *,
    base_path: str,
    bootstrap_block_length: int | None,
) -> tuple[str, int | None]:
    """Resolve one EI bootstrap block-length policy before drawing samples."""
    if base_path not in {"bb", "northrop"}:
        raise ValueError("base_path must be 'bb' or 'northrop'.")
    if bootstrap_block_length is None:
        return "default", None
    if (
        isinstance(bootstrap_block_length, (bool, np.bool_))
        or not isinstance(bootstrap_block_length, (int, np.integer))
        or not 1 <= int(bootstrap_block_length) <= values.size
    ):
        raise ValueError("bootstrap_block_length must be None or an integer between 1 and n_obs.")
    return "fixed", int(bootstrap_block_length)


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
    """Summarize transformed EI bootstrap path draws and their sampling metadata."""
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


def _build_bm_ei_path_draws(
    bootstrap_samples: np.ndarray,
    *,
    block_sizes: np.ndarray,
    path_keys: tuple[tuple[str, bool], ...],
    allow_zeros: bool,
) -> dict[tuple[str, bool], np.ndarray]:
    """Transform one raw bootstrap bank into selected BM-EI z-path draw matrices."""
    samples = np.asarray(bootstrap_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("bootstrap_samples must be a two-dimensional matrix.")
    block_sizes = validate_block_sizes(block_sizes, n_obs=samples.shape[1])
    draws = {
        key: np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float)
        for key in path_keys
    }
    for rep, sample in enumerate(samples):
        sample_values = (
            _finite_nonnegative_series(sample) if allow_zeros else _finite_positive_series(sample)
        )
        sample_paths = _build_bm_z_paths_from_values(
            sample_values,
            block_sizes,
            path_keys=path_keys,
        )
        for key, z_path in sample_paths.items():
            draws[key][rep] = z_path
    return draws


def bootstrap_bm_ei_path_draws(
    bootstrap_samples: np.ndarray,
    *,
    block_sizes: np.ndarray,
    allow_zeros: bool,
    path_keys: tuple[tuple[str, bool], ...] = BM_PATH_KEYS,
) -> dict[tuple[str, bool], np.ndarray]:
    """Transform a bootstrap bank while preserving its caller-chosen EI clock."""
    return _build_bm_ei_path_draws(
        bootstrap_samples,
        block_sizes=block_sizes,
        path_keys=path_keys,
        allow_zeros=allow_zeros,
    )


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
    covariance_shrinkage: float = EI_DEFAULT_COVARIANCE_SHRINKAGE,
) -> dict[str, Any]:
    """Bootstrap BM-EI covariance; default adaptive precision targets pooled theta and z.

    An explicit integer retains fixed-R sampling. Adaptive precision is conditional
    on the original stable window and the declared covariance shrinkage.
    Checkpoints are 128, 256, 512, 768, and 1024. The target vector includes theta
    and its CI endpoints plus the unconstrained z fit and endpoints, so the
    theta=1 boundary cannot hide Monte Carlo error. The cap retains the result
    with a warning and ``bootstrap_precision_met=False`` if tolerance is unmet.

    The returned in-memory dictionary contains full-grid covariance, path draws,
    block-size labels, estimator identity, and sampling/precision metadata. Pass
    it to ``estimate_pooled_bm_ei`` with matching data, path, block scheme, and
    shrinkage (default 0.37). This function does not write intermediate files.
    ``allow_zeros`` declares whether observed zeros are legal; non-finite inputs
    are always rejected rather than removed from the observation clock.
    """
    values = _finite_nonnegative_series(vec) if allow_zeros else _finite_positive_series(vec)
    block_sizes = validate_block_sizes(block_sizes, n_obs=values.size)
    block_length_policy, resolved_block_length = _resolve_ei_bootstrap_block_length(
        values,
        base_path=base_path,
        bootstrap_block_length=bootstrap_block_length,
    )
    if reps == "adaptive":
        shrinkage = validate_covariance_shrinkage(covariance_shrinkage)
        length = resolved_block_length or default_circular_bootstrap_block_size(values.size)
        path_key = (base_path, sliding)
        observed_z = _build_bm_z_paths_from_values(values, block_sizes, path_keys=(path_key,))[
            path_key
        ]
        window, _ = select_stable_path_window(block_sizes, observed_z)
        mask = (block_sizes >= window.lo) & (block_sizes <= window.hi)
        levels, z_values = block_sizes[mask], observed_z[mask]

        def draw(count: int, rng: np.random.Generator) -> np.ndarray:
            # Process each raw series immediately; do not retain an R x n_obs bank.
            rows = []
            for _ in range(count):
                raw = draw_circular_block_bootstrap_sample(values, block_size=length, rng=rng)
                rows.append(
                    _build_bm_z_paths_from_values(raw, block_sizes, path_keys=(path_key,))[
                        path_key
                    ]
                )
            return np.asarray(rows)

        def evaluate(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            selected = subset_covariance_by_labels(
                covariance, block_sizes, levels, context="EI bootstrap covariance"
            )
            model = _fit_pooled_z_model(
                z_values, covariance=selected, covariance_shrinkage=shrinkage
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

        return {
            **adaptive_covariance(draw, evaluate, random_state=random_state),
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
    bank = draw_circular_block_bootstrap_samples(
        values,
        reps=reps,
        block_size=resolved_block_length,
        random_state=random_state,
    )
    samples = bank.samples
    z_draws = bootstrap_bm_ei_path_draws(
        samples,
        block_sizes=block_sizes,
        path_keys=((base_path, sliding),),
        allow_zeros=allow_zeros,
    )[(base_path, sliding)]
    return _summarize_bm_ei_path_draws(
        z_draws,
        block_sizes=block_sizes,
        base_path=base_path,
        sliding=sliding,
        bootstrap_block_length_policy=block_length_policy,
        bootstrap_block_length=bank.block_size,
        reps=reps,
    )
