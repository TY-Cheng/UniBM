"""BM-path construction for canonical extremal-index workflows."""

from __future__ import annotations

import numpy as np

from ..cdf import empirical_cdf
from .._numeric import prefix_sum
from .._window_ops import sliding_window_extreme_valid
from ._stats import EI_TINY
from .models import EiPathBundle, EiStableWindow


BM_PATH_KEYS = (
    ("northrop", True),
    ("northrop", False),
    ("bb", True),
    ("bb", False),
)


def _select_stable_ei_window(
    block_sizes: np.ndarray,
    z_path: np.ndarray,
    *,
    min_points: int = 4,
    trim_fraction: float = 0.15,
    roughness_penalty: float = 0.75,
    curvature_penalty: float = 0.5,
) -> tuple[EiStableWindow, np.ndarray]:
    """Choose the most stable contiguous block-size window on the transformed EI path."""
    levels = np.asarray(block_sizes, dtype=int)
    z = np.asarray(z_path, dtype=float)
    mask = np.isfinite(z)
    levels = levels[mask]
    z = z[mask]
    if levels.size < min_points:
        raise ValueError("Not enough finite EI path values to select a stable window.")
    lo = int(np.floor(levels.size * trim_fraction))
    hi = levels.size - lo
    if hi - lo < min_points:
        lo = 0
        hi = levels.size
    prefix_z = prefix_sum(z)
    prefix_z2 = prefix_sum(z * z)
    abs_diff1_prefix = prefix_sum(np.abs(np.diff(z)))
    abs_diff2_prefix = prefix_sum(np.abs(np.diff(np.diff(z))))
    best: tuple[float, int, int] | None = None
    for start in range(lo, hi - min_points + 1):
        for stop in range(start + min_points, hi + 1):
            window_len = stop - start
            sum_z = prefix_z[stop] - prefix_z[start]
            sum_z2 = prefix_z2[stop] - prefix_z2[start]
            mean_z = float(sum_z / window_len)
            variance = max(float(sum_z2 / window_len - mean_z * mean_z), 0.0)
            if window_len > 1:
                roughness_total = abs_diff1_prefix[stop - 1] - abs_diff1_prefix[start]
                roughness = float(roughness_total / (window_len - 1))
            else:
                roughness = 0.0
            if window_len > 2:
                curvature_total = abs_diff2_prefix[stop - 2] - abs_diff2_prefix[start]
                curvature = float(curvature_total / (window_len - 2))
            else:
                curvature = 0.0
            score = (
                variance
                + float(roughness_penalty) * roughness
                + float(curvature_penalty) * curvature
            ) / np.sqrt(stop - start)
            if best is None or score < best[0]:
                best = (score, start, stop)
    assert best is not None
    _, start, stop = best
    selected_mask = np.zeros(mask.sum(), dtype=bool)
    selected_mask[start:stop] = True
    window = EiStableWindow(int(levels[start]), int(levels[stop - 1]))
    return window, selected_mask


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
    stable_window, stable_mask = _select_stable_ei_window(block_sizes, z_path)
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


def extract_stable_path_window(path: EiPathBundle) -> tuple[np.ndarray, np.ndarray]:
    """Return the selected stable block levels and transformed values for one path."""
    finite_mask = np.isfinite(path.z_path)
    finite_levels = path.block_sizes[finite_mask]
    finite_z = path.z_path[finite_mask]
    window_mask = (finite_levels >= path.stable_window.lo) & (
        finite_levels <= path.stable_window.hi
    )
    selected_levels = finite_levels[window_mask]
    selected_z = finite_z[window_mask]
    if selected_levels.size == 0:
        raise ValueError("Stable EI window did not retain any finite transformed path values.")
    return selected_levels, selected_z
