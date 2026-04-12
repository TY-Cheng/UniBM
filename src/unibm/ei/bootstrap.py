"""Bootstrap helpers for BM-based extremal-index path estimation."""

from __future__ import annotations

import numpy as np

from .._bootstrap_sampling import draw_circular_block_bootstrap_samples
from ._validation import _finite_nonnegative_series, _finite_positive_series
from .paths import BM_PATH_KEYS, _build_bm_z_paths_from_values


def _build_bm_ei_path_draws(
    bootstrap_samples: np.ndarray,
    *,
    block_sizes: np.ndarray,
    path_keys: tuple[tuple[str, bool], ...],
    allow_zeros: bool = False,
) -> dict[tuple[str, bool], np.ndarray]:
    """Transform one raw bootstrap bank into selected BM-EI z-path draw matrices."""
    samples = np.asarray(bootstrap_samples, dtype=float)
    block_sizes = np.asarray(block_sizes, dtype=int)
    draws = {
        key: np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float)
        for key in path_keys
    }
    for rep, sample in enumerate(samples):
        try:
            sample_values = (
                _finite_nonnegative_series(sample)
                if allow_zeros
                else _finite_positive_series(sample)
            )
            sample_paths = _build_bm_z_paths_from_values(
                sample_values,
                block_sizes,
                path_keys=path_keys,
            )
        except ValueError:
            continue
        for key, z_path in sample_paths.items():
            draws[key][rep] = z_path
    return draws


def bootstrap_bm_ei_path_draws(
    bootstrap_samples: np.ndarray,
    *,
    block_sizes: np.ndarray,
    path_keys: tuple[tuple[str, bool], ...] = BM_PATH_KEYS,
    allow_zeros: bool = False,
) -> dict[tuple[str, bool], np.ndarray]:
    """Transform one raw bootstrap bank into selected BM-EI z-path draw matrices."""
    return _build_bm_ei_path_draws(
        bootstrap_samples,
        block_sizes=block_sizes,
        path_keys=path_keys,
        allow_zeros=allow_zeros,
    )


def bootstrap_bm_ei_path(
    vec: np.ndarray | list[float],
    *,
    base_path: str,
    sliding: bool,
    block_sizes: np.ndarray,
    reps: int,
    random_state: int,
    allow_zeros: bool = False,
) -> dict[str, np.ndarray | None]:
    """Bootstrap the transformed BM-EI path on the full block-size grid."""
    values = _finite_nonnegative_series(vec) if allow_zeros else _finite_positive_series(vec)
    samples = draw_circular_block_bootstrap_samples(
        values,
        reps=reps,
        random_state=random_state,
    ).samples
    z_draws = bootstrap_bm_ei_path_draws(
        samples,
        block_sizes=np.asarray(block_sizes, dtype=int),
        allow_zeros=allow_zeros,
    )[(base_path, sliding)]
    return {
        "block_sizes": np.asarray(block_sizes, dtype=int),
        "samples": z_draws,
        "covariance": None,
    }
