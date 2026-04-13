"""BM-path construction for canonical extremal-index workflows."""

from __future__ import annotations

import numpy as np

from ..cdf import empirical_cdf
from .._window_ops import sliding_window_extreme_valid
from ._stats import EI_TINY
from .models import EiPathBundle
from .selection import select_stable_path_window


BM_PATH_KEYS = (
    ("northrop", True),
    ("northrop", False),
    ("bb", True),
    ("bb", False),
)


def _rolling_window_minima(
    scores: np.ndarray,
    block_size: int,
    *,
    sliding: bool,
) -> np.ndarray:
    """Return sliding or disjoint window minima for one score series."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size < block_size or block_size < 2:
        return np.asarray([], dtype=float)
    if sliding:
        return sliding_window_extreme_valid(scores, block_size, reducer="min")
    n_block = scores.size // block_size
    if n_block < 1:
        return np.asarray([], dtype=float)
    windows = scores[: n_block * block_size].reshape(n_block, block_size)
    valid = np.all(np.isfinite(windows), axis=1)
    return windows.min(axis=1)[valid]


def _path_point_from_statistics(
    base_path: str,
    statistics: np.ndarray,
    *,
    block_size: int,
) -> tuple[float, float, float]:
    """Convert one block-size statistics sample into theta/eir/z path coordinates."""
    mean_stat = float(np.mean(statistics))
    if base_path == "northrop":
        eir = max(mean_stat, 1.0)
        theta = float(1.0 / eir)
    elif base_path == "bb":
        theta = float(max((1.0 / max(mean_stat, EI_TINY)) - 1.0 / float(block_size), EI_TINY))
        theta = min(theta, 1.0)
        eir = float(1.0 / theta)
    else:
        raise ValueError(f"Unknown BM EI base path: {base_path}")
    return theta, eir, float(np.log(eir))


def _compute_path_arrays_from_scores(
    base_path: str,
    scores: np.ndarray,
    block_sizes: np.ndarray,
    *,
    sliding: bool,
    collect_statistics: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray] | None]:
    """Compute BM-EI path arrays with optional per-level statistics retention."""
    theta_path = np.full(block_sizes.size, np.nan, dtype=float)
    eir_path = np.full(block_sizes.size, np.nan, dtype=float)
    z_path = np.full(block_sizes.size, np.nan, dtype=float)
    sample_counts = np.zeros(block_sizes.size, dtype=int)
    sample_statistics = {} if collect_statistics else None
    for idx, block_size in enumerate(np.asarray(block_sizes, dtype=int)):
        minima = _rolling_window_minima(scores, int(block_size), sliding=sliding)
        if minima.size == 0:
            continue
        statistics = float(block_size) * minima
        if sample_statistics is not None:
            sample_statistics[int(block_size)] = statistics
        sample_counts[idx] = statistics.size
        theta, eir, z = _path_point_from_statistics(
            base_path,
            statistics,
            block_size=int(block_size),
        )
        theta_path[idx] = theta
        eir_path[idx] = eir
        z_path[idx] = z
    return theta_path, eir_path, z_path, sample_counts, sample_statistics


def _build_path_from_scores(
    base_path: str,
    scores: np.ndarray,
    block_sizes: np.ndarray,
    *,
    sliding: bool,
) -> EiPathBundle:
    """Construct the full EI path for one BM base estimator."""
    theta_path, eir_path, z_path, sample_counts, sample_statistics = (
        _compute_path_arrays_from_scores(
            base_path,
            np.asarray(scores, dtype=float),
            np.asarray(block_sizes, dtype=int),
            sliding=sliding,
            collect_statistics=True,
        )
    )
    stable_window, stable_mask = select_stable_path_window(block_sizes, z_path)
    selected_level = int(block_sizes[np.isfinite(z_path)][stable_mask][0])
    return EiPathBundle(
        base_path=base_path,
        sliding=bool(sliding),
        block_sizes=np.asarray(block_sizes, dtype=int),
        theta_path=theta_path,
        eir_path=eir_path,
        z_path=z_path,
        sample_counts=sample_counts,
        sample_statistics=sample_statistics or {},
        stable_window=stable_window,
        selected_level=selected_level,
    )


def _build_bm_z_paths_from_values(
    values: np.ndarray,
    block_sizes: np.ndarray,
    *,
    path_keys: tuple[tuple[str, bool], ...] = BM_PATH_KEYS,
) -> dict[tuple[str, bool], np.ndarray]:
    """Build only the transformed BM z-paths needed by bootstrap workflows."""
    cdf_values = np.asarray(empirical_cdf(values)(values), dtype=float)
    cdf_values = np.clip(cdf_values, EI_TINY, 1.0 - EI_TINY)
    score_lookup = {
        "northrop": -np.log(cdf_values),
        "bb": 1.0 - cdf_values,
    }
    draws: dict[tuple[str, bool], np.ndarray] = {}
    for base_path, sliding in path_keys:
        _, _, z_path, _, _ = _compute_path_arrays_from_scores(
            base_path,
            score_lookup[base_path],
            np.asarray(block_sizes, dtype=int),
            sliding=sliding,
            collect_statistics=False,
        )
        draws[(base_path, sliding)] = z_path
    return draws


def _build_bm_paths_from_values(
    values: np.ndarray,
    block_sizes: np.ndarray,
) -> dict[tuple[str, bool], EiPathBundle]:
    """Build the four BM EI paths reused across benchmark methods."""
    cdf_values = np.asarray(empirical_cdf(values)(values), dtype=float)
    cdf_values = np.clip(cdf_values, EI_TINY, 1.0 - EI_TINY)
    score_lookup = {
        "northrop": -np.log(cdf_values),
        "bb": 1.0 - cdf_values,
    }
    return {
        (base_path, sliding): _build_path_from_scores(
            base_path,
            score_lookup[base_path],
            block_sizes,
            sliding=sliding,
        )
        for base_path, sliding in BM_PATH_KEYS
    }
