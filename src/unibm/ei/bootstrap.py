"""Bootstrap helpers for BM-based extremal-index path estimation."""

from __future__ import annotations

import numpy as np

from .._block_grid import validate_block_sizes
from .._bootstrap_sampling import draw_circular_block_bootstrap_samples
from ._validation import _finite_nonnegative_series, _finite_positive_series
from .paths import BM_PATH_KEYS, _build_bm_z_paths_from_values


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
    reps: int,
    random_state: int,
) -> dict[str, np.ndarray | None]:
    """Bootstrap the transformed BM-EI path on the full block-size grid."""
    values = _finite_nonnegative_series(vec) if allow_zeros else _finite_positive_series(vec)
    block_sizes = validate_block_sizes(block_sizes, n_obs=values.size)
    samples = draw_circular_block_bootstrap_samples(
        values,
        reps=reps,
        random_state=random_state,
    ).samples
    z_draws = bootstrap_bm_ei_path_draws(
        samples,
        block_sizes=block_sizes,
        path_keys=((base_path, sliding),),
        allow_zeros=allow_zeros,
    )[(base_path, sliding)]
    valid_draws = z_draws[np.all(np.isfinite(z_draws), axis=1)]
    covariance = (
        np.atleast_2d(np.cov(valid_draws, rowvar=False)) if valid_draws.shape[0] >= 2 else None
    )
    return {
        "block_sizes": block_sizes,
        "samples": valid_draws,
        "covariance": covariance,
    }
