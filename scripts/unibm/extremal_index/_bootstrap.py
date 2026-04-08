"""Bootstrap helpers for BM-based extremal-index path estimation."""

from __future__ import annotations

import numpy as np

from ..bootstrap import draw_circular_block_bootstrap_samples
from ._paths import _build_bm_paths_from_values
from ._shared import _finite_nonnegative_series, _finite_positive_series


def bootstrap_bm_ei_path_draws(
    bootstrap_samples: np.ndarray,
    *,
    block_sizes: np.ndarray,
    allow_zeros: bool = False,
) -> dict[tuple[str, bool], np.ndarray]:
    """Transform one raw bootstrap bank into all four BM-EI z-path draw matrices."""
    samples = np.asarray(bootstrap_samples, dtype=float)
    block_sizes = np.asarray(block_sizes, dtype=int)
    draws = {
        ("northrop", True): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
        ("northrop", False): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
        ("bb", True): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
        ("bb", False): np.full((samples.shape[0], block_sizes.size), np.nan, dtype=float),
    }
    for rep, sample in enumerate(samples):
        try:
            sample_values = (
                _finite_nonnegative_series(sample)
                if allow_zeros
                else _finite_positive_series(sample)
            )
            sample_paths = _build_bm_paths_from_values(sample_values, block_sizes)
        except ValueError:
            continue
        for key, path in sample_paths.items():
            draws[key][rep] = path.z_path
    return draws


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
