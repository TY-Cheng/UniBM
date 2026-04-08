"""BM-path construction and stable-window routing for EI estimators."""

from __future__ import annotations

import numpy as np

from ..core import generate_block_sizes
from ..diagnostics import empirical_cdf
from ..window_ops import sliding_window_extreme_valid
from ._shared import (
    EI_TINY,
    EiPathBundle,
    EiPreparedBundle,
    EiStableWindow,
    _finite_nonnegative_series,
    _finite_positive_series,
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
    best: tuple[float, int, int] | None = None
    for start in range(lo, hi - min_points + 1):
        for stop in range(start + min_points, hi + 1):
            window = z[start:stop]
            variance = float(np.mean((window - window.mean()) ** 2))
            first_diff = np.diff(window)
            roughness = float(np.mean(np.abs(first_diff))) if first_diff.size else 0.0
            curvature = float(np.mean(np.abs(np.diff(first_diff)))) if first_diff.size > 1 else 0.0
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


def _build_path_from_scores(
    base_path: str,
    scores: np.ndarray,
    block_sizes: np.ndarray,
    *,
    sliding: bool,
) -> EiPathBundle:
    """Construct the full EI path for one BM base estimator."""
    theta_path = np.full(block_sizes.size, np.nan, dtype=float)
    eir_path = np.full(block_sizes.size, np.nan, dtype=float)
    z_path = np.full(block_sizes.size, np.nan, dtype=float)
    sample_counts = np.zeros(block_sizes.size, dtype=int)
    sample_statistics: dict[int, np.ndarray] = {}
    for idx, block_size in enumerate(np.asarray(block_sizes, dtype=int)):
        minima = _rolling_window_minima(scores, int(block_size), sliding=sliding)
        if minima.size == 0:
            continue
        statistics = float(block_size) * minima
        sample_statistics[int(block_size)] = statistics
        sample_counts[idx] = statistics.size
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
        theta_path[idx] = theta
        eir_path[idx] = eir
        z_path[idx] = float(np.log(eir))
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
        sample_statistics=sample_statistics,
        stable_window=stable_window,
        selected_level=selected_level,
    )


def _build_bm_paths_from_values(
    values: np.ndarray,
    block_sizes: np.ndarray,
) -> dict[tuple[str, bool], EiPathBundle]:
    """Build the four BM EI paths reused across benchmark methods."""
    cdf_values = np.asarray(empirical_cdf(values)(values), dtype=float)
    cdf_values = np.clip(cdf_values, EI_TINY, 1.0 - EI_TINY)
    northrop_scores = -np.log(cdf_values)
    bb_scores = 1.0 - cdf_values
    return {
        ("northrop", True): _build_path_from_scores(
            "northrop", northrop_scores, block_sizes, sliding=True
        ),
        ("northrop", False): _build_path_from_scores(
            "northrop", northrop_scores, block_sizes, sliding=False
        ),
        ("bb", True): _build_path_from_scores("bb", bb_scores, block_sizes, sliding=True),
        ("bb", False): _build_path_from_scores("bb", bb_scores, block_sizes, sliding=False),
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


def prepare_ei_bundle(
    vec: np.ndarray | list[float],
    *,
    block_sizes: np.ndarray | None = None,
    threshold_quantiles: tuple[float, ...] = (0.90, 0.95),
    allow_zeros: bool = False,
) -> EiPreparedBundle:
    """Build all reusable EI ingredients for one series."""
    values = _finite_nonnegative_series(vec) if allow_zeros else _finite_positive_series(vec)
    if block_sizes is None:
        block_sizes = generate_block_sizes(values.size)
    block_sizes = np.asarray(block_sizes, dtype=int)
    paths = _build_bm_paths_from_values(values, block_sizes)
    threshold_candidates = {
        float(q): np.flatnonzero(values > np.quantile(values, float(q)))
        for q in threshold_quantiles
    }
    return EiPreparedBundle(
        values=values,
        block_sizes=block_sizes,
        paths=paths,
        threshold_candidates=threshold_candidates,
    )
